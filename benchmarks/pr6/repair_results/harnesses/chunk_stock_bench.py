#!/usr/bin/env python3
"""Fresh-process stock throughput gate with a configurable model path."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    num_seqs = 256
    llm = LLM(args.model, enforce_eager=False, max_model_len=4096)
    prompts = [
        [rng.randint(0, 10_000) for _ in range(rng.randint(100, 1024))]
        for _ in range(num_seqs)
    ]
    params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=rng.randint(100, 1024),
        )
        for _ in range(num_seqs)
    ]
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    torch.cuda.synchronize()
    started = time.perf_counter()
    llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    total_tokens = sum(item.max_tokens for item in params)
    result = {
        "commit": args.commit,
        "seed": args.seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "throughput_tokens_per_s": total_tokens / elapsed,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
