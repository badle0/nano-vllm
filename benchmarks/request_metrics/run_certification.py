#!/usr/bin/env python3
"""Run eight serialized, balanced baseline/repair process pairs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from certification_common import (
    BENCHMARK_NAME,
    SCHEMA_VERSION,
    exclusive_json_dump,
    model_identity,
    sha256_file,
    source_identity,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-head", required=True)
    parser.add_argument("--expected-repair-head", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=20260821)
    parser.add_argument("--batches", default="64,256")
    parser.add_argument("--output-lengths", default="32")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    if args.pairs <= 0 or args.pairs % 2:
        parser.error("pairs must be a positive even number")
    if args.output_dir.exists():
        parser.error(f"refusing to reuse existing output directory: {args.output_dir}")
    return args


def compare_pair(baseline: dict, repair: dict) -> dict:
    if baseline["configuration"] != repair["configuration"]:
        raise RuntimeError("paired configurations differ")
    if baseline["model"]["fingerprint_sha256"] != repair["model"]["fingerprint_sha256"]:
        raise RuntimeError("paired model fingerprints differ")
    comparisons = {}
    if baseline["cases"].keys() != repair["cases"].keys():
        raise RuntimeError("paired case sets differ")
    for key in baseline["cases"]:
        base_case = baseline["cases"][key]
        fix_case = repair["cases"][key]
        if base_case["prompt_sha256"] != fix_case["prompt_sha256"]:
            raise RuntimeError(f"paired prompt hashes differ for {key}")
        base_observations = base_case["observations"]
        fix_observations = fix_case["observations"]
        if len(base_observations) != len(fix_observations):
            raise RuntimeError(f"paired observation counts differ for {key}")
        for base_item, fix_item in zip(base_observations, fix_observations):
            if base_item["seed"] != fix_item["seed"]:
                raise RuntimeError(f"paired observation seeds differ for {key}")
            if base_item["token_ids_sha256"] != fix_item["token_ids_sha256"]:
                raise RuntimeError(f"paired token hashes differ for {key}")
        base_tps = base_case["summary"]["steady"]["median_output_tokens_per_s"]
        fix_tps = fix_case["summary"]["steady"]["median_output_tokens_per_s"]
        comparisons[key] = {
            "baseline_median_output_tokens_per_s": base_tps,
            "repair_median_output_tokens_per_s": fix_tps,
            "efficiency": fix_tps / base_tps,
            "throughput_delta_fraction": fix_tps / base_tps - 1.0,
            "time_overhead_fraction": base_tps / fix_tps - 1.0,
            "cold_efficiency": (
                fix_case["summary"]["cold"]["output_tokens_per_s"]
                / base_case["summary"]["cold"]["output_tokens_per_s"]
            ),
            "p95_elapsed_ratio": (
                fix_case["summary"]["steady"]["p95_elapsed_s"]
                / base_case["summary"]["steady"]["p95_elapsed_s"]
            ),
        }
    return comparisons


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.resolve(strict=True)
    repair_root = args.repair_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    for source_root in (baseline_root, repair_root):
        if output_dir == source_root or output_dir.is_relative_to(source_root):
            raise SystemExit("output directory must be outside both source worktrees")
    baseline_source = source_identity(baseline_root, args.expected_baseline_head)
    repair_source = source_identity(repair_root, args.expected_repair_head)
    harness = Path(__file__).with_name("metrics_certify.py").resolve()
    runner = Path(__file__).resolve()
    model = model_identity(args.model)
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "artifact_kind": "balanced_pair_manifest",
        "run_id": args.run_id,
        "created_utc": utc_now(),
        "invocation": {"argv": sys.argv, "cwd": str(Path.cwd().resolve())},
        "harness": {"path": str(harness), "sha256": sha256_file(harness)},
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "sources": {"baseline": baseline_source, "repair": repair_source},
        "model": model,
        "protocol": {
            "pairs": args.pairs,
            "order": "odd baseline-repair; even repair-baseline",
            "seed_base": args.seed_base,
            "batches": args.batches,
            "output_lengths": args.output_lengths,
            "prompt_length": args.prompt_length,
            "repetitions": args.repetitions,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "median_efficiency_floor": 0.98,
            "individual_efficiency_floor": 0.95,
        },
        "pair_runs": [],
    }

    raw_payloads = {}
    for pair_id in range(1, args.pairs + 1):
        order = (
            ["baseline", "repair"] if pair_id % 2 else ["repair", "baseline"]
        )
        order_name = "-".join(order)
        pair_record = {
            "pair_id": pair_id,
            "seed": args.seed_base + pair_id - 1,
            "order": order_name,
            "runs": {},
        }
        for position, side in enumerate(order, start=1):
            source_root = baseline_root if side == "baseline" else repair_root
            expected_head = (
                args.expected_baseline_head
                if side == "baseline"
                else args.expected_repair_head
            )
            label = "main" if side == "baseline" else "fix-request-metrics"
            filename = f"pair{pair_id:02d}_pos{position}_{side}.json"
            output = output_dir / filename
            command = [
                sys.executable,
                str(harness),
                "--run-id", args.run_id,
                "--pair-id", str(pair_id),
                "--side", side,
                "--order", order_name,
                "--order-position", str(position),
                "--label", label,
                "--source-root", str(source_root),
                "--expected-head", expected_head,
                "--model", str(args.model.resolve()),
                "--seed", str(pair_record["seed"]),
                "--output", str(output),
                "--batches", args.batches,
                "--output-lengths", args.output_lengths,
                "--prompt-length", str(args.prompt_length),
                "--repetitions", str(args.repetitions),
                "--max-model-len", str(args.max_model_len),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(source_root)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            print(f"pair {pair_id}/{args.pairs} position {position}: {side}", flush=True)
            subprocess.run(command, cwd=source_root, env=environment, check=True)
            payload = json.loads(output.read_text())
            raw_payloads[(pair_id, side)] = payload
            pair_record["runs"][side] = {
                "path": filename,
                "sha256": sha256_file(output),
                "command": command,
            }
        pair_record["comparisons"] = compare_pair(
            raw_payloads[(pair_id, "baseline")],
            raw_payloads[(pair_id, "repair")],
        )
        manifest["pair_runs"].append(pair_record)

    case_names = manifest["pair_runs"][0]["comparisons"].keys()
    aggregate = {}
    for key in case_names:
        efficiencies = [
            pair["comparisons"][key]["efficiency"]
            for pair in manifest["pair_runs"]
        ]
        aggregate[key] = {
            "pair_efficiencies": efficiencies,
            "median_efficiency": statistics.median(efficiencies),
            "minimum_efficiency": min(efficiencies),
            "median_gate_pass": statistics.median(efficiencies) >= 0.98,
            "individual_gate_pass": min(efficiencies) >= 0.95,
        }
    manifest["aggregate"] = aggregate
    manifest["gate_pass"] = all(
        item["median_gate_pass"] and item["individual_gate_pass"]
        for item in aggregate.values()
    )
    manifest["finished_utc"] = utc_now()
    exclusive_json_dump(output_dir / "manifest.json", manifest)
    print(f"wrote {output_dir / 'manifest.json'}; gate_pass={manifest['gate_pass']}")


if __name__ == "__main__":
    main()
