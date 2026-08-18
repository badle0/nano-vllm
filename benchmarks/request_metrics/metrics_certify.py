#!/usr/bin/env python3
"""Run one self-describing request-metrics certification process."""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from certification_common import (
    BENCHMARK_NAME,
    SCHEMA_VERSION,
    canonical_sha256,
    case_key,
    current_rss_kib,
    derive_observation_seed,
    environment_identity,
    exclusive_json_dump,
    model_identity,
    parse_positive_int_csv,
    peak_rss_kib,
    sha256_file,
    source_identity,
    summarize_observations,
)

from nanovllm import LLM, SamplingParams


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_prompts(batch: int, prompt_length: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    prompts = []
    for row in range(batch):
        prompt = [100 + row]
        prompt.extend(
            rng.randrange(1000, 30_000) for _ in range(prompt_length - 1)
        )
        prompts.append(prompt)
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument("--side", choices=("baseline", "repair"), required=True)
    parser.add_argument("--order", choices=("baseline-repair", "repair-baseline"), required=True)
    parser.add_argument("--order-position", type=int, choices=(1, 2), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="64,256")
    parser.add_argument("--output-lengths", default="32")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    if args.pair_id <= 0 or args.prompt_length <= 0 or args.repetitions < 2:
        parser.error("pair-id/prompt-length must be positive and repetitions must be >= 2")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"output parent does not exist: {args.output.parent}")
    return args


def main() -> None:
    args = parse_args()
    batches = parse_positive_int_csv(args.batches, name="batches")
    output_lengths = parse_positive_int_csv(
        args.output_lengths, name="output-lengths"
    )
    if args.max_model_len < args.prompt_length + max(output_lengths):
        raise SystemExit("max-model-len is smaller than prompt + output length")
    expected_side = args.order.split("-")[args.order_position - 1]
    if expected_side != args.side:
        raise SystemExit(
            f"order position says {expected_side}, but this run says {args.side}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for request-metrics certification")

    started_utc = utc_now()
    source = source_identity(args.source_root, args.expected_head)
    model = model_identity(args.model)
    harness_path = Path(__file__).resolve()
    environment = environment_identity(torch)
    configuration = {
        "batches": batches,
        "output_lengths": output_lengths,
        "prompt_length": args.prompt_length,
        "repetitions": args.repetitions,
        "cold_observations": 1,
        "temperature": 0.6,
        "ignore_eos": True,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(batches),
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }

    llm = LLM(
        str(args.model),
        enforce_eager=False,
        max_model_len=args.max_model_len,
        max_num_seqs=max(batches),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    llm.generate(
        [[17, 19, 23]],
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2),
        use_tqdm=False,
    )

    cases = {}
    for batch in batches:
        prompts = make_prompts(batch, args.prompt_length, args.seed + batch)
        prompt_sha256 = canonical_sha256(prompts)
        for output_length in output_lengths:
            observations = []
            params = SamplingParams(
                temperature=0.6,
                ignore_eos=True,
                max_tokens=output_length,
            )
            for repetition in range(args.repetitions):
                observation_seed = derive_observation_seed(
                    args.seed,
                    args.pair_id,
                    batch,
                    output_length,
                    repetition,
                )
                torch.manual_seed(observation_seed)
                torch.cuda.manual_seed_all(observation_seed)
                torch.cuda.synchronize()
                cuda_allocated_before = torch.cuda.memory_allocated()
                cuda_reserved_before = torch.cuda.memory_reserved()
                host_rss_before = current_rss_kib()
                torch.cuda.reset_peak_memory_stats()

                started = time.perf_counter()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started

                if len(outputs) != batch:
                    raise RuntimeError(f"expected {batch} outputs, got {len(outputs)}")
                token_ids = [output["token_ids"] for output in outputs]
                if any(len(tokens) != output_length for tokens in token_ids):
                    raise RuntimeError("generation returned an unexpected token count")
                total_output_tokens = batch * output_length
                observations.append(
                    {
                        "repetition": repetition,
                        "cold": repetition == 0,
                        "seed": observation_seed,
                        "elapsed_s": elapsed,
                        "output_tokens": total_output_tokens,
                        "output_tokens_per_s": total_output_tokens / elapsed,
                        "token_ids_sha256": canonical_sha256(token_ids),
                        "resources": {
                            "host_rss_before_kib": host_rss_before,
                            "host_rss_after_kib": current_rss_kib(),
                            "host_peak_rss_kib": peak_rss_kib(),
                            "cuda_allocated_before_bytes": cuda_allocated_before,
                            "cuda_reserved_before_bytes": cuda_reserved_before,
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        },
                    }
                )
                del outputs, token_ids

            cases[case_key(batch, output_length)] = {
                "batch": batch,
                "output_length": output_length,
                "prompt_sha256": prompt_sha256,
                "observations": observations,
                "summary": summarize_observations(observations),
            }

    source_after = source_identity(args.source_root, args.expected_head)
    if source_after != source:
        raise RuntimeError("source identity changed during the benchmark")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "identity": {
            "run_id": args.run_id,
            "pair_id": args.pair_id,
            "side": args.side,
            "order": args.order,
            "order_position": args.order_position,
            "label": args.label,
            "seed": args.seed,
        },
        "invocation": {
            "argv": sys.argv,
            "cwd": str(Path.cwd().resolve()),
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "harness_path": str(harness_path),
            "harness_sha256": sha256_file(harness_path),
        },
        "source": source,
        "model": model,
        "environment": environment,
        "configuration": configuration,
        "cases": cases,
    }
    exclusive_json_dump(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
