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
    finished: list[Sequence]
    num_tokens: int


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
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        submission_time: float | None = None,
    ):
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

    def _step(self) -> StepOutput:
        seqs, is_prefill = self.scheduler.schedule()
        # must precede postprocess: it zeroes num_scheduled_tokens
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        events = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        finished = [seq for seq in seqs if seq.is_finished]
        return StepOutput(events, finished, num_tokens)

    def _execute_step(self):
        step_output = self._step()
        return step_output.finished, step_output.num_tokens

    def step(self):
        """Advance the engine and preserve the legacy pair-valued output API."""
        seqs, num_tokens = self._execute_step()
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs]
        return outputs, num_tokens

    def step_with_metrics(self):
        """Advance the engine and opt in to metrics on completed sequences."""
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

    def _run_engine(self) -> Iterator[StepOutput]:
        while not self.is_finished():
            yield self._step()

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        submission_time = self._clock()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        elif len(sampling_params) != len(prompts):
            raise ValueError(
                "prompts and sampling_params must contain the same number of items"
            )
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp, submission_time=submission_time)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = self._clock()
            output, num_tokens = self._execute_step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (self._clock() - t)
            else:
                decode_throughput = -num_tokens / (self._clock() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq in output:
                outputs[seq.seq_id] = seq
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        results = [
            {
                "text": self.tokenizer.decode(seq.completion_token_ids),
                "token_ids": seq.completion_token_ids,
            }
            for seq in outputs
        ]
        delivery_time = self._clock()
        for result, seq in zip(results, outputs):
            result["metrics"] = compute_metrics(seq, delivery_time=delivery_time)
        return results
