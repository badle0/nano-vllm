import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from typing import NamedTuple
from collections.abc import Iterator
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, StreamOutput
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.metrics import compute_metrics

class StepOutput(NamedTuple):
    events: list[StreamOutput]
    finished: list[tuple]      # (seq_id, completion_token_ids, metrics) — shape unchanged
    num_prefill_tokens: int    # chunk tokens scheduled this step (0 for a pure-decode step)
    num_decode_tokens: int     # decode rows this step (0 for a pure-prefill step)


class LLMEngine:

    def __init__(self, model, **kwargs):
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
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def _step(self) -> StepOutput:
        seqs, is_prefill = self.scheduler.schedule()
        # must precede postprocess: it zeroes num_scheduled_tokens
        num_prefill_tokens = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        num_decode_tokens = sum(1 for seq in seqs if not seq.is_prefill)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        events = self.scheduler.postprocess(seqs, token_ids)
        finished = [(seq.seq_id, seq.completion_token_ids, compute_metrics(seq))
                    for seq in seqs if seq.is_finished]
        return StepOutput(events, finished, num_prefill_tokens, num_decode_tokens)

    def step(self):
        # Public shim: keeps the legacy (finished, num_tokens) shape. Pure steps keep
        # the legacy signed value (+prefill tokens / -decode count); a MIXED step
        # reports +num_prefill_tokens — its sign remains a valid "step did prefill
        # work" predicate, and the pinned bench_latency.py ignores this element.
        step_output = self._step()
        num_tokens = (step_output.num_prefill_tokens if step_output.num_prefill_tokens
                      else -step_output.num_decode_tokens)
        return step_output.finished, num_tokens

    def _run_engine(self) -> Iterator[StepOutput]:
        while not self.is_finished():
            yield self._step()

    def stream(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> Iterator[StreamOutput]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        try:
            for step_output in self._run_engine():
                yield from step_output.events
        finally:
            self.scheduler.cancel_all()

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        t = perf_counter()
        for step_output in self._run_engine():
            dt = perf_counter() - t
            if step_output.num_prefill_tokens:      # mixed steps update both rates (F4)
                prefill_throughput = step_output.num_prefill_tokens / dt
            if step_output.num_decode_tokens:
                decode_throughput = step_output.num_decode_tokens / dt
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids, metrics in step_output.finished:
                outputs[seq_id] = (token_ids, metrics)
                pbar.update(1)
            t = perf_counter()
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids, "metrics": metrics} for token_ids, metrics in outputs]
        return outputs
