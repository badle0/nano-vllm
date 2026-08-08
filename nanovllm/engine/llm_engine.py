from __future__ import annotations

import atexit
from collections import deque
from collections.abc import Iterator
from dataclasses import fields
from threading import Lock
from time import perf_counter
from typing import NamedTuple

from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, StreamOutput
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.metrics import compute_metrics

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

        params = engine._normalize_batch(prompts, sampling_params)
        lease = engine._acquire_session("stream")
        self._lease = lease
        try:
            sequences = engine._admit_batch(
                prompts,
                params,
                submission_time=submission_time,
            )
        except BaseException:
            self._closed = True
            self._lease = None
            engine._release_session(lease)
            raise

        self.seq_ids = tuple(seq.seq_id for seq in sequences)
        self._sequences = {seq.seq_id: seq for seq in sequences}
        self._remaining = set(self.seq_ids)
        if not self.seq_ids:
            self.close()

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
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        self._remaining.clear()
        lease, self._lease = self._lease, None
        try:
            self._engine.scheduler.cancel(self.seq_ids)
        finally:
            if lease is not None:
                self._engine._release_session(lease)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class LLMEngine:

    def __init__(self, model, *, _clock=None, **kwargs):
        self._clock = perf_counter if _clock is None else _clock
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config, clock=self._clock)
        self._session_lock = Lock()
        self._active_session: tuple[object, str] | None = None
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    @staticmethod
    def _normalize_batch(
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[SamplingParams]:
        if not isinstance(sampling_params, list):
            return [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError(
                "prompts and sampling_params must have the same length "
                "(the same number of items)"
            )
        return sampling_params

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
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(
            prompt,
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
        except BaseException:
            self.scheduler.cancel(seq.seq_id for seq in admitted)
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
        params = self._normalize_batch(prompts, sampling_params)
        submission_time = self._clock()
        lease = self._acquire_session("generate")
        sequences = []
        pbar = None
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
        finally:
            if pbar is not None:
                pbar.close()
            try:
                self.scheduler.cancel(seq.seq_id for seq in sequences)
            finally:
                self._release_session(lease)
