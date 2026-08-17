#!/usr/bin/env python3
"""Fresh-process paired B=256 end-to-end benchmark for repaired top-k."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers

from nanovllm import LLM, SamplingParams


EXPECTED_COMMIT = "8759c877382f11ea16b40fdbd0dace7000b5e9ba"


def command_output(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def make_prompts(batch: int, prompt_len: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    prompts = []
    for row in range(batch):
        prompt = [100 + row]
        prompt.extend(rng.randrange(1000, 30_000) for _ in range(prompt_len - 1))
        prompts.append(prompt)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--first", choices=("disabled", "enabled"), required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if min(args.batch, args.prompt_len, args.output_len, args.repetitions) < 1:
        parser.error("batch, lengths, and repetitions must be positive")
    if args.repetitions < 4:
        parser.error("at least four repetitions are required (one cold plus three steady)")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    commit = command_output("git", "rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise SystemExit(
            f"wrong checkout for repaired top-k: {commit}; expected {EXPECTED_COMMIT}"
        )

    llm = LLM(
        args.model,
        enforce_eager=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch,
        gpu_memory_utilization=0.8,
    )
    llm.generate(
        [[17, 19, 23]],
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2),
        use_tqdm=False,
    )

    prompts = make_prompts(args.batch, args.prompt_len, args.seed + args.batch)
    scenarios = {
        "disabled": SamplingParams(
            temperature=0.6,
            top_k=-1,
            ignore_eos=True,
            max_tokens=args.output_len,
        ),
        "all_active_top_k_50": SamplingParams(
            temperature=0.6,
            top_k=50,
            ignore_eos=True,
            max_tokens=args.output_len,
        ),
    }
    first_order = (
        ("disabled", "all_active_top_k_50")
        if args.first == "disabled"
        else ("all_active_top_k_50", "disabled")
    )
    observations = {name: [] for name in scenarios}
    for repetition in range(args.repetitions):
        order = first_order if repetition % 2 == 0 else tuple(reversed(first_order))
        for name in order:
            run_seed = args.seed * 1000 + args.batch * 10 + repetition
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.generate(prompts, scenarios[name], use_tqdm=False)
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - started
            if len(outputs) != args.batch:
                raise RuntimeError(f"expected {args.batch} outputs; got {len(outputs)}")
            if not all(len(output["token_ids"]) == args.output_len for output in outputs):
                raise RuntimeError("an output did not contain the requested token count")
            observations[name].append(
                {
                    "repetition": repetition,
                    "order": list(order),
                    "elapsed_s": elapsed_s,
                    "output_tokens_per_s": args.batch * args.output_len / elapsed_s,
                }
            )

    summarized = {}
    for name, rows in observations.items():
        steady = rows[1:]
        summarized[name] = {
            "cold": rows[0],
            "steady": steady,
            "median_elapsed_s": statistics.median(row["elapsed_s"] for row in steady),
            "median_output_tokens_per_s": statistics.median(
                row["output_tokens_per_s"] for row in steady
            ),
        }
    disabled_tps = summarized["disabled"]["median_output_tokens_per_s"]
    enabled_tps = summarized["all_active_top_k_50"]["median_output_tokens_per_s"]

    result = {
        "schema_version": 1,
        "benchmark": "repaired_topk_e2e_b256",
        "commit": commit,
        "seed": args.seed,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "execution": {
            "cwd": str(Path.cwd()),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "nvidia_driver": command_output(
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ),
        },
        "configuration": {
            "model": args.model,
            "batch": args.batch,
            "prompt_tokens": args.prompt_len,
            "output_tokens": args.output_len,
            "temperature": 0.6,
            "top_k": 50,
            "ignore_eos": True,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.batch,
            "gpu_memory_utilization": 0.8,
            "repetitions": args.repetitions,
            "cold_observations_excluded": 1,
            "first_scenario": args.first,
        },
        "scenarios": summarized,
        "paired_steady_throughput_change_percent": (enabled_tps / disabled_tps - 1.0)
        * 100.0,
    }

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
