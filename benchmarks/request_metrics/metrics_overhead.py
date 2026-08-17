#!/usr/bin/env python3

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def make_prompts(batch, prompt_len, seed):
    rng = random.Random(seed)
    prompts = []
    for row in range(batch):
        prompt = [100 + row]
        prompt.extend(rng.randrange(1000, 30_000) for _ in range(prompt_len - 1))
        prompts.append(prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="64,256")
    parser.add_argument("--repetitions", type=int, default=4)
    args = parser.parse_args()

    batches = [int(value) for value in args.batches.split(",")]
    llm = LLM(
        args.model,
        enforce_eager=False,
        max_model_len=1024,
        max_num_seqs=max(batches),
        gpu_memory_utilization=0.8,
    )
    llm.generate(
        [[17, 19, 23]],
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2),
        use_tqdm=False,
    )

    result = {
        "label": args.label,
        "seed": args.seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "batches": {},
    }
    params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=32)
    for batch in batches:
        prompts = make_prompts(batch, 128, args.seed + batch)
        observations = []
        for repetition in range(args.repetitions):
            run_seed = args.seed * 1000 + batch * 10 + repetition
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert len(outputs) == batch
            assert all(len(output["token_ids"]) == 32 for output in outputs)
            observations.append(
                {
                    "repetition": repetition,
                    "elapsed_s": elapsed,
                    "output_tokens_per_s": batch * 32 / elapsed,
                }
            )
        steady = observations[1:]
        result["batches"][str(batch)] = {
            "cold": observations[0],
            "steady": steady,
            "median_output_tokens_per_s": statistics.median(
                item["output_tokens_per_s"] for item in steady
            ),
        }

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
