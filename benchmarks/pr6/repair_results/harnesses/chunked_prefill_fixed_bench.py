#!/usr/bin/env python3
"""Post-fix mixed-workload benchmark using the compatible metrics API."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def summarize(rows):
    return {
        "count": len(rows),
        "ttft_p50_ms": statistics.median(
            row["engine_ttft"] for row in rows
        ) * 1e3,
        "ttft_max_ms": max(row["engine_ttft"] for row in rows) * 1e3,
        "mean_itl_p50_ms": statistics.median(
            row["engine_mean_itl"] for row in rows
        ) * 1e3,
        "max_itl_p50_ms": statistics.median(
            row["engine_max_itl"] for row in rows
        ) * 1e3,
        "max_itl_max_ms": max(row["engine_max_itl"] for row in rows) * 1e3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tau", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    max_num_seqs = min(512, args.tau)
    llm = LLM(
        args.model,
        enforce_eager=False,
        max_model_len=4096,
        max_num_batched_tokens=args.tau,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=0.8,
    )
    llm.generate(
        [[rng.randrange(100, 10_000) for _ in range(64)] for _ in range(16)],
        SamplingParams(ignore_eos=True, max_tokens=4),
        use_tqdm=False,
    )

    short = [
        [rng.randrange(100, 10_000) for _ in range(64)] for _ in range(16)
    ]
    long_prompts = [
        [rng.randrange(100, 10_000) for _ in range(2048)] for _ in range(2)
    ]
    params = SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=256,
    )
    collected = {}

    def drain(step_outputs):
        for seq_id, _token_ids, metrics in step_outputs:
            collected[seq_id] = metrics

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for prompt in short:
        llm.add_request(prompt, params)
    for _ in range(40):
        output, _ = llm.step_with_metrics()
        drain(output)
    long_admitted = time.perf_counter()
    for prompt in long_prompts:
        llm.add_request(prompt, params)
    while not llm.is_finished():
        output, _ = llm.step_with_metrics()
        drain(output)
    torch.cuda.synchronize()
    ended = time.perf_counter()

    groups = {"interactive": [], "long": []}
    for metrics in collected.values():
        key = "interactive" if metrics["num_prompt_tokens"] == 64 else "long"
        groups[key].append(metrics)
    assert len(groups["interactive"]) == 16
    assert len(groups["long"]) == 2
    result = {
        "commit": args.commit,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "model": args.model,
        "tau": args.tau,
        "max_num_seqs": max_num_seqs,
        "seed": args.seed,
        "whole_run_s": ended - started,
        "after_long_admission_s": ended - long_admitted,
        "completion_tokens_per_s": (18 * 256) / (ended - started),
        "groups": {
            name: summarize(rows) for name, rows in groups.items()
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
