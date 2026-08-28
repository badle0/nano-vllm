from __future__ import annotations

import atexit
import gc
import warnings
import weakref
from collections import deque
from collections.abc import Iterator
from dataclasses import fields
from threading import Event, Lock
from time import perf_counter
from typing import NamedTuple


class AdmissionLimits(NamedTuple):
    """Static request bounds checked before tokenized work enters the engine.

    Snapshotted after the model runner sizes the KV cache, so ``kvcache_blocks``
    is the real pool. A fake test engine may omit the attribute entirely, which
    skips these GPU-derived checks (capacity and emptiness checks still run).
    """

    max_model_len: int | None
    vocab_size: int | None
    block_size: int | None
    kvcache_blocks: int | None

from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, StreamOutput
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.metrics import compute_metrics
from nanovllm.layers.sampler import require_flashinfer_sampling
from nanovllm.utils.errors import record_cleanup_failure
from nanovllm.utils.tokenizer_identity import require_same_token_id_space


_PYTHON_GC_LEASE_LOCK = Lock()
_PYTHON_GC_LEASE_COUNT = 0
_PYTHON_GC_PRE_FIRST_ENABLED: bool | None = None


def _acquire_python_gc_lease() -> bool:
    """Disable cyclic GC and return the state before the first active lease."""
    global _PYTHON_GC_LEASE_COUNT, _PYTHON_GC_PRE_FIRST_ENABLED
    with _PYTHON_GC_LEASE_LOCK:
        first = _PYTHON_GC_LEASE_COUNT == 0
        if first:
            _PYTHON_GC_PRE_FIRST_ENABLED = gc.isenabled()
        try:
            # Reassert the lease invariant on overlapping acquisition in case
            # unrelated process code changed the global setting.
            gc.disable()
        except BaseException:
            if first:
                restore_enabled = _PYTHON_GC_PRE_FIRST_ENABLED
                _PYTHON_GC_PRE_FIRST_ENABLED = None
                if restore_enabled:
                    gc.enable()
                else:
                    gc.disable()
            raise
        if _PYTHON_GC_PRE_FIRST_ENABLED is None:
            raise RuntimeError("Python GC lease baseline is missing")
        _PYTHON_GC_LEASE_COUNT += 1
        return _PYTHON_GC_PRE_FIRST_ENABLED


def _release_python_gc_lease() -> None:
    """Release one lease and restore pre-first state after the final owner."""
    global _PYTHON_GC_LEASE_COUNT, _PYTHON_GC_PRE_FIRST_ENABLED
    with _PYTHON_GC_LEASE_LOCK:
        if _PYTHON_GC_LEASE_COUNT <= 0:
            raise RuntimeError("Python GC lease underflow")
        _PYTHON_GC_LEASE_COUNT -= 1
        if _PYTHON_GC_LEASE_COUNT:
            gc.disable()
            return
        restore_enabled = _PYTHON_GC_PRE_FIRST_ENABLED
        _PYTHON_GC_PRE_FIRST_ENABLED = None
        if restore_enabled is None:
            raise RuntimeError("Python GC lease baseline is missing")
        if restore_enabled:
            gc.enable()
        else:
            gc.disable()


class StepOutput(NamedTuple):
    events: list[StreamOutput]
    finished: list[Sequence]
    num_prefill_tokens: int    # chunk tokens scheduled this step (0 for a pure-decode step)
    num_decode_tokens: int     # decode rows this step (0 for a pure-prefill step)


class StreamSession(Iterator[StreamOutput]):
    """One request-scoped, synchronously backpressured stream session."""

    def __init__(
        self,
        engine: LLMEngine,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        submission_time: float,
    ):
        self._engine = engine
        self._lease = None
        self._pending: deque[StreamOutput] = deque()
        self._sequences: dict[int, Sequence] = {}
        self._remaining: set[int] = set()
        self.seq_ids: tuple[int, ...] = ()
        self.metrics: dict[int, dict] = {}
        self._closed = False
        self._finalizer = None

        params = engine._normalize_batch(prompts, sampling_params)
        sequences = ()
        lease = engine._acquire_session("stream")
        self._lease = lease
        try:
            sequences = engine._admit_batch(
                prompts,
                params,
                submission_time=submission_time,
            )
            self.seq_ids = tuple(seq.seq_id for seq in sequences)
            self._sequences = {seq.seq_id: seq for seq in sequences}
            self._remaining = set(self.seq_ids)
            if self.seq_ids:
                # The callback retains the engine, not this session. The
                # engine reference is required to cancel work reliably.
                self._finalizer = weakref.finalize(
                    self,
                    StreamSession._finalize_abandoned,
                    engine,
                    lease,
                    self.seq_ids,
                )
                # LLMEngine owns process teardown through its own atexit hook.
                self._finalizer.atexit = False
        except BaseException as error:
            self._closed = True
            finalizer, self._finalizer = self._finalizer, None
            lease, self._lease = self._lease, None
            self._engine = None
            try:
                if finalizer is not None:
                    finalizer.detach()
            except BaseException as cleanup_error:
                record_cleanup_failure(
                    error, "StreamSession finalizer rollback", cleanup_error
                )
            try:
                engine.scheduler.cancel(seq.seq_id for seq in sequences)
            except BaseException as cleanup_error:
                record_cleanup_failure(
                    error, "StreamSession admission rollback", cleanup_error
                )
            if lease is not None:
                try:
                    engine._release_session(lease)
                except BaseException as cleanup_error:
                    record_cleanup_failure(
                        error, "StreamSession lease rollback", cleanup_error
                    )
            raise

        if not self.seq_ids:
            self.close()

    @staticmethod
    def _warn_abandoned(count: int) -> None:
        try:
            warnings.warn(
                "a StreamSession was garbage-collected without close(); "
                f"automatic cleanup was attempted for its {count} request(s); "
                "use the context manager or call close() explicitly",
                RuntimeWarning,
                stacklevel=1,
            )
        except Exception:
            # Warning filters must never control cancellation or lease release.
            pass

    @staticmethod
    def _finalize_abandoned(engine: "LLMEngine", lease: object, seq_ids):
        first_error = None
        try:
            engine.scheduler.cancel(seq_ids)
        except BaseException as error:
            first_error = error
        try:
            engine._release_session(lease)
        except BaseException as error:
            if first_error is None:
                first_error = error
            else:
                record_cleanup_failure(
                    first_error, "StreamSession finalizer lease release", error
                )
        StreamSession._warn_abandoned(len(seq_ids))
        if first_error is not None:
            raise first_error

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self):
        return self

    def __next__(self) -> StreamOutput:
        if self._closed:
            raise StopIteration

        try:
            while not self._pending:
                if not self._remaining:
                    self.close()
                    raise StopIteration
                if self._engine.is_finished():
                    raise RuntimeError(
                        "stream scheduler finished before every owned request"
                    )
                step_output = self._engine._step()
                foreign_ids = {
                    event.seq_id for event in step_output.events
                } - self._remaining
                if foreign_ids:
                    raise RuntimeError(
                        "stream received events owned by another request session: "
                        f"{sorted(foreign_ids)}"
                    )
                self._pending.extend(step_output.events)

            event = self._pending.popleft()
            sequence = self._sequences[event.seq_id]
            delivered_at = self._engine._clock()
            if sequence.first_delivery_time is None:
                sequence.first_delivery_time = delivered_at
            if event.finished:
                if not sequence.is_finished:
                    raise RuntimeError(
                        "finished stream event did not reference a finished sequence"
                    )
                sequence.delivery_time = delivered_at
                self.metrics[event.seq_id] = compute_metrics(
                    sequence,
                    delivery_time=delivered_at,
                    first_delivery_time=sequence.first_delivery_time,
                )
                self._remaining.discard(event.seq_id)

            if not self._remaining and not self._pending:
                self.close()
            return event
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                record_cleanup_failure(
                    error, "StreamSession iteration cleanup", cleanup_error
                )
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        finalizer, self._finalizer = self._finalizer, None
        engine, self._engine = self._engine, None
        lease, self._lease = self._lease, None
        first_error = None
        try:
            if finalizer is not None:
                finalizer.detach()
        except BaseException as error:
            first_error = error
        self._pending.clear()
        self._remaining.clear()
        self._sequences.clear()
        try:
            engine.scheduler.cancel(self.seq_ids)
        except BaseException as error:
            if first_error is None:
                first_error = error
            else:
                record_cleanup_failure(
                    first_error, "StreamSession request cancellation", error
                )
        if lease is not None:
            try:
                engine._release_session(lease)
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    record_cleanup_failure(
                        first_error, "StreamSession lease release", error
                    )
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException as cleanup_error:
                record_cleanup_failure(
                    exc_value, "StreamSession context cleanup", cleanup_error
                )
        return False


class LLMEngine:

    def __init__(self, model, *, _clock=None, **kwargs):
        # GC is process-global, so it is touched only after every fallible
        # initialization step succeeds. None means this engine does not own a
        # GC-state restoration obligation.
        self._python_gc_was_enabled: bool | None = None
        self._python_gc_lease_active = False
        self._exit_started = False
        self._exit_lock = Lock()
        self._exit_complete = Event()
        self._atexit_callback = None
        self._atexit_registered = False
        self.model_runner = None
        self.ps = []
        self.events = []
        self._clock = perf_counter if _clock is None else _clock
        config_fields = {
            field.name for field in fields(Config) if field.init
        }
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        if config.top_p_backend == "flashinfer":
            require_flashinfer_sampling()
        # Tokenizer failure must precede GPU/process-group ownership.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        if config.speculation_enabled:
            draft_tokenizer = AutoTokenizer.from_pretrained(
                config.draft_model,
                use_fast=True,
            )
            self.speculative_tokenizer_fingerprint = (
                require_same_token_id_space(
                    self.tokenizer,
                    draft_tokenizer,
                    vocab_size=config.hf_config.vocab_size,
                )
            )
        config.eos = self.tokenizer.eos_token_id
        self._session_lock = Lock()
        self._active_session: tuple[object, str] | None = None
        self._atexit_callback = self.exit
        previous_block_size = Sequence.block_size

        try:
            Sequence.block_size = config.kvcache_block_size
            ctx = mp.get_context("spawn")
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = ModelRunner(config, 0, self.events)
            # allocate_kv_cache writes the sized pool back into config.
            self._admission_limits = self._snapshot_admission_limits(config)
            self.scheduler = Scheduler(config, clock=self._clock)
            atexit.register(self._atexit_callback)
            self._atexit_registered = True
            if config.disable_python_gc:
                lease_acquired = False
                try:
                    baseline = _acquire_python_gc_lease()
                    lease_acquired = True
                    self._python_gc_was_enabled = baseline
                    self._python_gc_lease_active = True
                except BaseException:
                    if lease_acquired:
                        _release_python_gc_lease()
                        self._python_gc_was_enabled = None
                    raise
        except BaseException as error:
            if self.model_runner is not None:
                try:
                    self.exit()
                except BaseException as cleanup_error:
                    record_cleanup_failure(
                        error, "LLMEngine cleanup", cleanup_error
                    )
            else:
                cleanup_error = self._join_workers(abort=True)
                if cleanup_error is not None:
                    record_cleanup_failure(
                        error, "LLMEngine worker cleanup", cleanup_error
                    )
            Sequence.block_size = previous_block_size
            raise

    def exit(self):
        # Explicit exit and the registered atexit callback can both run. Claim
        # cleanup once so model/process teardown and GC restoration are
        # idempotent even when exit is called repeatedly or concurrently.
        with self._exit_lock:
            if self._exit_started:
                owns_cleanup = False
            else:
                self._exit_started = True
                owns_cleanup = True
        if not owns_cleanup:
            self._exit_complete.wait()
            return

        first_error = None
        try:
            runner = getattr(self, "model_runner", None)
            if runner is not None:
                try:
                    runner.call("exit")
                except BaseException as error:
                    first_error = error
                finally:
                    self.model_runner = None
            process_error = self._join_workers(abort=False)
            if first_error is None:
                first_error = process_error
        finally:
            try:
                self._restore_python_gc()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            try:
                if self._atexit_registered:
                    atexit.unregister(self._atexit_callback)
                    self._atexit_registered = False
            except BaseException as error:
                if first_error is None:
                    first_error = error
            finally:
                self._exit_complete.set()
        if first_error is not None:
            raise first_error

    def _join_workers(self, *, abort: bool) -> BaseException | None:
        """Bounded cleanup for every worker owned by this engine."""
        first_error = None

        def record(error):
            nonlocal first_error
            if first_error is None:
                first_error = error

        def probe_alive(process) -> bool:
            is_alive = getattr(process, "is_alive", None)
            if not callable(is_alive):
                return False
            try:
                return bool(is_alive())
            except BaseException as error:
                record(error)
                # A failed probe cannot establish that termination is safe to
                # skip, so continue conservatively as though it were alive.
                return True

        processes, self.ps = self.ps, []
        self.events.clear()
        for process in processes:
            is_alive = getattr(process, "is_alive", None)
            terminate = getattr(process, "terminate", None)
            kill = getattr(process, "kill", None)
            if abort and callable(terminate):
                try:
                    if not callable(is_alive) or probe_alive(process):
                        terminate()
                except BaseException as error:
                    record(error)
            try:
                try:
                    process.join(timeout=5.0)
                except TypeError:
                    # Compatibility with simple process doubles in unit tests.
                    process.join()
            except BaseException as error:
                record(error)

            alive = probe_alive(process)
            if alive and not abort and callable(terminate):
                try:
                    terminate()
                    process.join(timeout=5.0)
                except BaseException as error:
                    record(error)
                alive = probe_alive(process)
                if first_error is None:
                    record(RuntimeError("tensor-parallel worker did not exit cooperatively"))
            if alive and callable(kill):
                try:
                    kill()
                    process.join(timeout=5.0)
                except BaseException as error:
                    record(error)
                alive = probe_alive(process)
            if alive:
                record(RuntimeError("tensor-parallel worker could not be stopped"))
                self.ps.append(process)
            else:
                close = getattr(process, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as error:
                        record(error)
        return first_error

    def _restore_python_gc(self):
        if not self._python_gc_lease_active:
            return
        self._python_gc_lease_active = False
        try:
            _release_python_gc_lease()
        finally:
            self._python_gc_was_enabled = None

    @staticmethod
    def _normalize_batch(
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[SamplingParams]:
        if not isinstance(prompts, list):
            raise TypeError("prompts must be a list of strings or token-id lists")
        if not isinstance(sampling_params, list):
            return [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError(
                "prompts and sampling_params must have the same length "
                "(the same number of items)"
            )
        return sampling_params

    @staticmethod
    def _snapshot_admission_limits(config) -> AdmissionLimits:
        """Snapshot each independently available request-admission bound."""
        def positive_int(value):
            return value if type(value) is int and value > 0 else None

        hf_config = getattr(config, "hf_config", None)
        return AdmissionLimits(
            positive_int(getattr(config, "max_model_len", None)),
            positive_int(getattr(hf_config, "vocab_size", None)),
            positive_int(getattr(config, "kvcache_block_size", None)),
            positive_int(getattr(config, "num_kvcache_blocks", None)),
        )

    def _acquire_session(self, kind: str) -> object:
        with self._session_lock:
            if self._active_session is not None:
                active_kind = self._active_session[1]
                raise RuntimeError(
                    f"cannot start {kind}: an active {active_kind} session owns the engine"
                )
            if not self.scheduler.is_finished():
                raise RuntimeError(
                    f"cannot start {kind}: the engine has manually queued requests"
                )
            lease = object()
            self._active_session = (lease, kind)
            return lease

    def _release_session(self, lease: object):
        with self._session_lock:
            if self._active_session is None:
                return
            if self._active_session[0] is not lease:
                raise RuntimeError("attempted to release a stream session owned elsewhere")
            self._active_session = None

    def _assert_public_step_access(self, operation: str):
        with self._session_lock:
            if self._active_session is not None:
                active_kind = self._active_session[1]
                raise RuntimeError(
                    f"cannot {operation}: an active {active_kind} session owns the engine"
                )

    def _admit_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        submission_time: float | None = None,
    ) -> Sequence:
        if submission_time is None:
            submission_time = self._clock()
        # Fail before tokenization or sequence-id allocation when the bounded
        # scheduler cannot own another request.
        self.scheduler.require_capacity()
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("sampling_params must be a SamplingParams instance")
        # Snapshot into the exact base class so a subclass cannot bypass
        # validation by overriding __init__ or validate().
        sampling_params = SamplingParams(
            temperature=sampling_params.temperature,
            max_tokens=sampling_params.max_tokens,
            ignore_eos=sampling_params.ignore_eos,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )
        if isinstance(prompt, str):
            prompt_ids = list(self.tokenizer.encode(prompt))
        elif isinstance(prompt, list):
            prompt_ids = list(prompt)
        else:
            raise TypeError("prompt must be a string or a list of token ids")
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")
        limits = getattr(self, "_admission_limits", None)
        vocab_size = limits.vocab_size if limits is not None else None
        for token in prompt_ids:
            if type(token) is not int:
                raise TypeError(
                    "prompt token ids must be integers, got "
                    f"{type(token).__name__}"
                )
            if token < 0:
                raise ValueError("prompt token ids must be non-negative")
            if vocab_size is not None and token >= vocab_size:
                raise ValueError(
                    f"prompt token id {token} is outside the model vocabulary "
                    f"[0, {vocab_size})"
                )

        max_model_len = limits.max_model_len if limits is not None else None
        processed_tokens = len(prompt_ids) + sampling_params.max_tokens - 1
        if max_model_len is not None:
            if len(prompt_ids) > max_model_len:
                raise ValueError(
                    f"prompt length {len(prompt_ids)} exceeds max_model_len "
                    f"{max_model_len}"
                )
            if processed_tokens > max_model_len:
                raise ValueError(
                    f"prompt length {len(prompt_ids)} + max_tokens "
                    f"{sampling_params.max_tokens} - 1 = {processed_tokens} "
                    f"model-processed tokens exceeds max_model_len "
                    f"{max_model_len}; reduce max_tokens or raise max_model_len"
                )

        block_size = limits.block_size if limits is not None else None
        kvcache_blocks = limits.kvcache_blocks if limits is not None else None
        if block_size is not None and kvcache_blocks is not None:
            blocks_needed = (
                processed_tokens + block_size - 1
            ) // block_size
            if blocks_needed > kvcache_blocks:
                raise ValueError(
                    f"request needs {blocks_needed} KV-cache blocks for "
                    f"{processed_tokens} model-processed tokens but the pool "
                    f"has only {kvcache_blocks}; reduce prompt length or "
                    "max_tokens, or raise gpu_memory_utilization"
                )
        seq = Sequence(
            prompt_ids,
            sampling_params,
            submission_time=submission_time,
            engine_arrival_time=self._clock(),
        )
        self.scheduler.add(seq)
        return seq

    def _admit_batch(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: list[SamplingParams],
        *,
        submission_time: float,
    ) -> list[Sequence]:
        # Reserve the whole batch logically before doing any tokenization so an
        # oversized batch cannot be partially admitted.
        self.scheduler.require_capacity(len(prompts))
        admitted = []
        try:
            for prompt, params in zip(prompts, sampling_params, strict=True):
                admitted.append(
                    self._admit_request(
                        prompt,
                        params,
                        submission_time=submission_time,
                    )
                )
        except BaseException as error:
            try:
                self.scheduler.cancel(seq.seq_id for seq in admitted)
            except BaseException as cleanup_error:
                record_cleanup_failure(
                    error, "batch admission rollback", cleanup_error
                )
            raise
        return admitted

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        submission_time: float | None = None,
    ) -> int:
        self._assert_public_step_access("add a request")
        return self._admit_request(
            prompt,
            sampling_params,
            submission_time=submission_time,
        ).seq_id

    def _step(self) -> StepOutput:
        seqs, is_prefill = self.scheduler.schedule()
        # must precede postprocess: it zeroes num_scheduled_tokens
        num_prefill_tokens = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        num_decode_tokens = sum(1 for seq in seqs if not seq.is_prefill)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        events = self.scheduler.postprocess(seqs, token_ids)
        finished = [seq for seq in seqs if seq.is_finished]
        return StepOutput(
            events,
            finished,
            num_prefill_tokens,
            num_decode_tokens,
        )

    def _execute_step(self):
        step_output = self._step()
        num_tokens = (
            step_output.num_prefill_tokens
            if step_output.num_prefill_tokens
            else -step_output.num_decode_tokens
        )
        return step_output.finished, num_tokens

    def step(self):
        """Advance the engine and preserve the legacy pair-valued output API."""
        self._assert_public_step_access("step")
        seqs, num_tokens = self._execute_step()
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs]
        return outputs, num_tokens

    def step_with_metrics(self):
        """Advance the engine and opt in to metrics on completed sequences."""
        self._assert_public_step_access("step with metrics")
        seqs, num_tokens = self._execute_step()
        delivery_time = self._clock()
        outputs = [
            (
                seq.seq_id,
                seq.completion_token_ids,
                compute_metrics(seq, delivery_time=delivery_time),
            )
            for seq in seqs
        ]
        return outputs, num_tokens

    def stream(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> StreamSession:
        """Start one capacity-bounded synchronous stream session.

        Raises SchedulerCapacityError before tokenization when the prompt count
        exceeds the scheduler's available sequence slots.
        """
        params = self._normalize_batch(prompts, sampling_params)
        submission_time = self._clock()
        return StreamSession(
            self,
            prompts,
            params,
            submission_time=submission_time,
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """Generate one capacity-bounded batch in input order.

        Raises SchedulerCapacityError before tokenization when the prompt count
        exceeds the scheduler's available sequence slots.
        """
        params = self._normalize_batch(prompts, sampling_params)
        submission_time = self._clock()
        lease = self._acquire_session("generate")
        sequences = []
        pbar = None
        primary_error = None
        try:
            sequences = self._admit_batch(
                prompts,
                params,
                submission_time=submission_time,
            )
            pbar = tqdm(
                total=len(prompts),
                desc="Generating",
                dynamic_ncols=True,
                disable=not use_tqdm,
            )
            outputs = {}
            prefill_throughput = decode_throughput = 0.0
            while not self.is_finished():
                t = self._clock()
                step_output = self._step()
                dt = self._clock() - t
                if step_output.num_prefill_tokens:
                    prefill_throughput = step_output.num_prefill_tokens / dt
                if step_output.num_decode_tokens:
                    decode_throughput = step_output.num_decode_tokens / dt
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
                for seq in step_output.finished:
                    outputs[seq.seq_id] = seq
                    pbar.update(1)

            ordered = [outputs[seq.seq_id] for seq in sequences]
            results = [
                {
                    "text": self.tokenizer.decode(seq.completion_token_ids),
                    "token_ids": seq.completion_token_ids,
                }
                for seq in ordered
            ]
            delivery_time = self._clock()
            for result, seq in zip(results, ordered, strict=True):
                result["metrics"] = compute_metrics(
                    seq,
                    delivery_time=delivery_time,
                )
            return results
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_error = None
            if pbar is not None:
                try:
                    pbar.close()
                except BaseException as error:
                    cleanup_error = error
            try:
                self.scheduler.cancel(seq.seq_id for seq in sequences)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    record_cleanup_failure(
                        cleanup_error, "generate request cancellation", error
                    )
            try:
                self._release_session(lease)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    record_cleanup_failure(
                        cleanup_error, "generate lease release", error
                    )
            if cleanup_error is not None:
                if primary_error is None:
                    raise cleanup_error
                record_cleanup_failure(
                    primary_error, "generate cleanup", cleanup_error
                )
