from __future__ import annotations

import atexit
import gc
import warnings
import weakref
from collections import deque
from collections.abc import Iterator
from dataclasses import fields
from threading import Event, Lock, RLock
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
from nanovllm.engine.speculative_routes import DraftRouteKey
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


class SpeculativeDiscardResultError(RuntimeError):
    """A V3 draft result does not match its immutable scheduler plan."""


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
        # Iteration and explicit close may be called from different threads.
        # Keep session-owned collections stable across one delivered event;
        # RLock is required because the iterator closes itself on completion.
        self._iteration_lock = RLock()

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
                engine._cancel_requests(seq.seq_id for seq in sequences)
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
    def _cancel_engine_requests(engine: "LLMEngine", seq_ids):
        cancel = getattr(engine, "_cancel_requests", None)
        if callable(cancel):
            return cancel(seq_ids)
        # Compatibility for narrow lifecycle doubles that predate the engine
        # execution mutex. Production LLMEngine always takes the locked path.
        return engine.scheduler.cancel(seq_ids)

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
            StreamSession._cancel_engine_requests(engine, seq_ids)
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
        with self._iteration_lock:
            return self._next_locked()

    def _next_locked(self) -> StreamOutput:
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
        with self._iteration_lock:
            return self._close_locked()

    def _close_locked(self):
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
            engine._cancel_requests(self.seq_ids)
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
        # One engine owns one mutable scheduler and one pair of CUDA contexts.
        # Cancellation must never deallocate KV slots while a target or draft
        # kernel can still write them.
        self._execution_lock = RLock()
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

    def _cancel_requests(self, seq_ids):
        """Cancel requests only after any in-flight engine cycle completes."""

        requested = tuple(seq_ids)
        execution_lock = getattr(self, "_execution_lock", None)
        if execution_lock is None:
            # Compatibility for deliberately minimal engine doubles.
            return self.scheduler.cancel(requested)
        with execution_lock:
            return self.scheduler.cancel(requested)

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
                self._cancel_requests(seq.seq_id for seq in admitted)
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

    def _draft_workspace_route_cap(self, batch_size: int) -> int:
        """Return the largest K covered by V2's fixed workspace reservation."""

        runner = self.model_runner
        if not bool(getattr(runner, "speculation_enabled", False)):
            return 0
        memory_plan = getattr(runner, "speculative_memory_plan", None)
        if memory_plan is None:
            raise RuntimeError(
                "speculation is enabled without a speculative memory plan"
            )
        planned_batch = getattr(memory_plan, "batch_size", None)
        planned_k = getattr(memory_plan, "max_effective_k", None)
        if type(planned_batch) is not int or planned_batch < 0:
            raise RuntimeError("speculative memory plan has an invalid batch cap")
        if type(planned_k) is not int or planned_k < 0:
            raise RuntimeError("speculative memory plan has an invalid K cap")
        return planned_k if 0 < batch_size <= planned_batch else 0

    def _resolve_draft_route_admission(self, seqs):
        """Finish V3 route readiness lookup before scheduler reservation."""

        runner = self.model_runner
        if not bool(getattr(runner, "speculation_enabled", False)):
            return None
        resolver = getattr(runner, "resolve_draft_route_admission", None)
        if not callable(resolver):
            raise RuntimeError(
                "speculation is enabled without a draft route registry resolver"
            )
        return resolver(seqs)

    @staticmethod
    def _validate_draft_discard_result(plan, result, seqs, vocab_size: int):
        """Validate host-only draft diagnostics before target state can mutate."""

        if type(vocab_size) is not int or vocab_size < 1:
            raise SpeculativeDiscardResultError(
                "draft result validation requires a positive integer vocabulary"
            )
        effective_k = getattr(plan, "effective_k", None)
        rows = getattr(plan, "rows", None)
        if type(effective_k) is not int or effective_k < 1:
            raise SpeculativeDiscardResultError(
                "executed draft plan must have a positive effective_k"
            )
        if not isinstance(rows, tuple) or len(rows) != len(seqs):
            raise SpeculativeDiscardResultError(
                "draft plan rows do not match the live decode batch"
            )
        result_effective_k = getattr(result, "effective_k", None)
        if type(result_effective_k) is not int or result_effective_k != effective_k:
            raise SpeculativeDiscardResultError(
                "draft result effective_k does not match its plan"
            )
        route_key = getattr(plan, "route_key", None)
        result_route_key = getattr(result, "route_key", None)
        if not isinstance(route_key, DraftRouteKey) or result_route_key != route_key:
            raise SpeculativeDiscardResultError(
                "draft result route key does not match its registered plan"
            )
        result_rows = getattr(result, "rows", None)
        if not isinstance(result_rows, tuple) or len(result_rows) != len(rows):
            raise SpeculativeDiscardResultError(
                "draft result rows do not match its plan"
            )
        draft_positions = getattr(result, "draft_positions", None)
        if (
            type(draft_positions) is not int
            or draft_positions != len(rows) * effective_k
        ):
            raise SpeculativeDiscardResultError(
                "draft result reports an invalid position count"
            )
        graph_steps = getattr(result, "graph_decode_steps", None)
        eager_steps = getattr(result, "eager_decode_steps", None)
        if (
            type(graph_steps) is not int
            or type(eager_steps) is not int
            or graph_steps < 0
            or eager_steps < 0
            or graph_steps + eager_steps != effective_k
        ):
            raise SpeculativeDiscardResultError(
                "draft result reports invalid eager/graph step counts"
            )
        expected_catchup = sum(
            row.committed_tokens - 1 - row.draft_cached_tokens
            for row in rows
        )
        planned_catchup = getattr(plan, "draft_catchup_tokens", None)
        if type(planned_catchup) is not int or planned_catchup != expected_catchup:
            raise SpeculativeDiscardResultError(
                "draft plan reports invalid catch-up work"
            )
        expected_step_counts = (len(rows),) * effective_k
        if getattr(plan, "draft_step_token_counts", None) != expected_step_counts:
            raise SpeculativeDiscardResultError(
                "draft plan reports invalid proposal-step counts"
            )
        target_query_tokens = getattr(plan, "target_query_tokens", None)
        total_scheduled_tokens = getattr(plan, "total_scheduled_tokens", None)
        expected_total = expected_catchup + len(rows) * (effective_k + 1)
        if (
            type(target_query_tokens) is not int
            or target_query_tokens != len(rows)
            or type(total_scheduled_tokens) is not int
            or total_scheduled_tokens != expected_total
        ):
            raise SpeculativeDiscardResultError(
                "draft plan reports invalid total work"
            )
        catchup_positions = getattr(result, "catchup_positions", None)
        if (
            type(catchup_positions) is not int
            or catchup_positions != expected_catchup
        ):
            raise SpeculativeDiscardResultError(
                "draft result reports invalid catch-up work"
            )
        expected_q_shape = (
            len(rows),
            effective_k,
            vocab_size,
        )
        q_shape = getattr(result, "q_shape", None)
        if (
            not isinstance(q_shape, tuple)
            or any(type(dimension) is not int for dimension in q_shape)
            or q_shape != expected_q_shape
        ):
            raise SpeculativeDiscardResultError(
                "draft result reports an invalid retained-q shape"
            )
        # q is a zero-copy [B,K,V] view over one contiguous [K,B,V] allocation.
        expected_q_stride = (vocab_size, len(rows) * vocab_size, 1)
        q_stride = getattr(result, "q_stride", None)
        if (
            not isinstance(q_stride, tuple)
            or any(type(stride) is not int for stride in q_stride)
            or q_stride != expected_q_stride
        ):
            raise SpeculativeDiscardResultError(
                "draft result reports an invalid retained-q stride"
            )
        if getattr(result, "q_dtype", None) != "torch.float32":
            raise SpeculativeDiscardResultError(
                "draft result did not retain canonical FP32 probabilities"
            )
        if (
            getattr(result, "q_storage_contiguous", None) is not True
            or getattr(result, "q_view_zero_copy", None) is not True
        ):
            raise SpeculativeDiscardResultError(
                "draft result violates the direct retained-q storage contract"
            )

        coverage_by_seq_id = {}
        for index, (plan_row, result_row, seq) in enumerate(
            zip(rows, result_rows, seqs, strict=True)
        ):
            plan_seq_id = getattr(plan_row, "seq_id", None)
            result_seq_id = getattr(result_row, "seq_id", None)
            if (
                type(plan_seq_id) is not int
                or type(result_seq_id) is not int
                or type(getattr(seq, "seq_id", None)) is not int
                or plan_seq_id != seq.seq_id
                or result_seq_id != plan_seq_id
            ):
                raise SpeculativeDiscardResultError(
                    f"draft result row {index} has a stale sequence ID"
                )
            planned_coverage = getattr(plan_row, "committed_tokens", None)
            result_coverage = getattr(result_row, "coverage_after_commit", None)
            if (
                type(planned_coverage) is not int
                or type(result_coverage) is not int
                or result_coverage != planned_coverage
            ):
                raise SpeculativeDiscardResultError(
                    f"draft result row {index} has invalid cache coverage"
                )
            proposed = result_row.proposed_token_ids
            proposal_count = getattr(result_row, "proposal_count", None)
            if (
                type(proposal_count) is not int
                or proposal_count != effective_k
                or not isinstance(proposed, tuple)
                or len(proposed) != effective_k
                or any(
                    type(token_id) is not int
                    or token_id < 0
                    or token_id >= vocab_size
                    for token_id in proposed
                )
            ):
                raise SpeculativeDiscardResultError(
                    f"draft result row {index} has invalid proposal tokens"
                )
            if plan_seq_id in coverage_by_seq_id:
                raise SpeculativeDiscardResultError(
                    "draft result contains duplicate sequence IDs"
                )
            coverage_by_seq_id[plan_seq_id] = planned_coverage
        return coverage_by_seq_id

    def _rollback_failed_draft_state(self, plan, error) -> None:
        """Release V3-only scheduler state while preserving ``error``."""

        cleanup_steps = (
            (
                "V3 draft reservation rollback",
                lambda: self.scheduler.rollback_draft_discard(plan),
            ),
            (
                "V3 draft coverage handoff rollback",
                lambda: self.scheduler.abort_draft_coverage(plan),
            ),
        )
        for label, cleanup in cleanup_steps:
            try:
                cleanup()
            except BaseException as cleanup_error:
                record_cleanup_failure(error, label, cleanup_error)

    def _execute_draft_discard(self, seqs, is_prefill):
        """Run one transactional V3 shadow cycle and return commit metadata."""

        runner = self.model_runner
        if is_prefill or not bool(getattr(runner, "speculation_enabled", False)):
            return None
        route_admission = self._resolve_draft_route_admission(seqs)
        if route_admission is None:
            plan = self.scheduler.plan_draft_discard(
                seqs,
                workspace_route_cap=0,
            )
        else:
            plan = self.scheduler.plan_draft_discard(
                seqs,
                route_admission=route_admission,
            )
        if not plan.uses_draft:
            return None

        try:
            result = runner.call("run_speculative_discard", plan, seqs)
            vocab_size = self._admission_limits.vocab_size
            if vocab_size is None:
                raise RuntimeError(
                    "speculative execution requires a known vocabulary"
                )
            coverage = self._validate_draft_discard_result(
                plan,
                result,
                seqs,
                vocab_size,
            )
            # Temporary proposal blocks must not be visible to target decode,
            # target prefix hashing, or ordinary postprocessing.
            self.scheduler.handoff_draft_discard(plan)
            # Validate every coverage field before target execution.  Successful
            # postprocess consumes this staging record as part of its sole public
            # token-commit operation; there is no fallible post-commit callback.
            self.scheduler.stage_draft_coverage(plan, seqs, coverage)
        except BaseException as error:
            self._rollback_failed_draft_state(plan, error)
            raise
        return plan, coverage

    def _step(self) -> StepOutput:
        execution_lock = getattr(self, "_execution_lock", None)
        if execution_lock is None:
            # Compatibility for deliberately minimal engine doubles.
            return self._step_unlocked()
        with execution_lock:
            return self._step_unlocked()

    def _step_unlocked(self) -> StepOutput:
        seqs, is_prefill = self.scheduler.schedule()
        # must precede postprocess: it zeroes num_scheduled_tokens
        num_prefill_tokens = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        num_decode_tokens = sum(1 for seq in seqs if not seq.is_prefill)
        runner = self.model_runner
        schedule_rollback = None
        if (
            not is_prefill
            and bool(getattr(runner, "speculation_enabled", False))
        ):
            # Capture this before route-cap calculation/planning: either may
            # fail before a DraftDiscardPlan exists, while schedule() has
            # already allocated the next ordinary target block.
            schedule_rollback = self.scheduler.capture_decode_schedule_rollback(
                seqs,
                is_prefill=is_prefill,
            )
        draft_cycle = None
        try:
            draft_cycle = self._execute_draft_discard(seqs, is_prefill)
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            events = self.scheduler.postprocess(seqs, token_ids)
        except BaseException as error:
            if draft_cycle is not None:
                plan, _ = draft_cycle
                self._rollback_failed_draft_state(plan, error)
            if schedule_rollback is not None:
                try:
                    self.scheduler.rollback_failed_decode_schedule(
                        schedule_rollback
                    )
                except BaseException as cleanup_error:
                    record_cleanup_failure(
                        error,
                        "V3 ordinary decode schedule rollback",
                        cleanup_error,
                    )
            raise
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
                self._cancel_requests(seq.seq_id for seq in sequences)
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
