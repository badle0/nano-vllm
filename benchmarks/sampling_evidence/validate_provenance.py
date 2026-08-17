#!/usr/bin/env python3
"""Validate repaired sampling evidence, hashes, pins, and derived summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("provenance.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise SystemExit(f"manifest path escapes repository: {relative}")
    return path


def require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit(f"{label}: {actual} != {expected}")


def validate_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="also hash the pinned local model files",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    environment = manifest["environment"]
    expected_environment = {
        "python": environment["python"],
        "torch": environment["pytorch"],
        "cuda_build": environment["cuda_build"],
        "transformers": environment["transformers"],
        "gpu": environment["gpu"],
        "gpu_total_memory_bytes": environment["gpu_total_memory_bytes"],
        "nvidia_driver": environment["nvidia_driver"],
    }

    for harness in manifest["harnesses"]:
        validate_hash(checked_repo_path(harness["path"]), harness["sha256"])

    micro_by_scenario = defaultdict(list)
    e2e_results = []
    release_paths = set()
    cache_dirs = set()
    for artifact in manifest["release_raw_results"]:
        path = checked_repo_path(artifact["path"])
        release_paths.add(path)
        validate_hash(path, artifact["sha256"])
        result = json.loads(path.read_text())

        if result["commit"] != artifact["commit"]:
            raise SystemExit(f"commit mismatch for {path}")
        if result["seed"] != artifact["seed"]:
            raise SystemExit(f"seed mismatch for {path}")
        for key, expected in expected_environment.items():
            if result["environment"][key] != expected:
                raise SystemExit(f"{key} mismatch for {path}")
        cache_dir = result["execution"]["torchinductor_cache_dir"]
        if not cache_dir or cache_dir in cache_dirs:
            raise SystemExit(f"missing or reused Inductor cache for {path}")
        cache_dirs.add(cache_dir)

        if artifact["kind"] == "micro":
            if result["benchmark"] != "repaired_sampling_microbenchmark":
                raise SystemExit(f"micro benchmark tag mismatch for {path}")
            if result["feature"] != artifact["feature"]:
                raise SystemExit(f"feature mismatch for {path}")
            if result["scenario"] != artifact["scenario"]:
                raise SystemExit(f"scenario mismatch for {path}")
            if len(result["steady"]["samples_cuda_ms"]) != 25:
                raise SystemExit(f"steady sample count mismatch for {path}")
            micro_by_scenario[result["scenario"]].append(result)
        elif artifact["kind"] == "e2e":
            if result["benchmark"] != "repaired_topk_e2e_b256":
                raise SystemExit(f"E2E benchmark tag mismatch for {path}")
            if result["configuration"]["first_scenario"] != artifact["first_scenario"]:
                raise SystemExit(f"first-scenario mismatch for {path}")
            e2e_results.append(result)
        else:
            raise SystemExit(f"unknown artifact kind for {path}")

    expected_scenarios = {
        "greedy_all",
        "disabled",
        "one_active_top_k_50",
        "all_active_top_k_50",
    }
    if set(micro_by_scenario) != expected_scenarios:
        raise SystemExit("micro scenario set mismatch")
    if any(len(rows) != 3 for rows in micro_by_scenario.values()):
        raise SystemExit("each micro scenario must have three fresh processes")
    if len(e2e_results) != 4:
        raise SystemExit("expected four fresh E2E processes")

    chronological_micro = sorted(
        (result for rows in micro_by_scenario.values() for result in rows),
        key=lambda result: result["started_at_utc"],
    )
    observed_micro_order = [
        (result["seed"], result["scenario"]) for result in chronological_micro
    ]
    expected_micro_order = [
        (round_["seed"], scenario)
        for round_ in manifest["protocol"]["micro"]["process_rounds"]
        for scenario in round_["order"]
    ]
    if observed_micro_order != expected_micro_order:
        raise SystemExit("micro process chronology does not match declared rotation")

    for scenario, expected in manifest["release_results"]["micro"].items():
        rows = micro_by_scenario[scenario]
        derived = {
            "cold_wall_ms": statistics.median(row["cold"]["wall_ms"] for row in rows),
            "steady_median_cuda_ms": statistics.median(
                row["steady"]["median_cuda_ms"] for row in rows
            ),
            "steady_p95_cuda_ms": statistics.median(
                row["steady"]["p95_cuda_ms"] for row in rows
            ),
            "peak_incremental_allocated_mib": max(
                row["steady"]["peak_incremental_allocated_mib"] for row in rows
            ),
        }
        for field, actual in derived.items():
            require_close(actual, expected[field], f"{scenario}.{field}")

    e2e_results.sort(key=lambda result: result["started_at_utc"])
    expected_first_order = manifest["protocol"]["e2e"]["first_scenario_order"]
    observed_first_order = [
        result["configuration"]["first_scenario"] for result in e2e_results
    ]
    if observed_first_order != expected_first_order:
        raise SystemExit("E2E first-scenario order is not the declared balanced order")

    deltas = [
        result["paired_steady_throughput_change_percent"] for result in e2e_results
    ]
    disabled = [
        result["scenarios"]["disabled"]["median_output_tokens_per_s"]
        for result in e2e_results
    ]
    enabled = [
        result["scenarios"]["all_active_top_k_50"]["median_output_tokens_per_s"]
        for result in e2e_results
    ]
    expected_e2e = manifest["release_results"]["e2e_topk_b256"]
    require_close(
        statistics.median(deltas),
        expected_e2e["median_paired_throughput_change_percent"],
        "E2E median paired change",
    )
    require_close(
        min(deltas),
        expected_e2e["worst_paired_throughput_change_percent"],
        "E2E worst paired change",
    )
    require_close(
        statistics.median(disabled),
        expected_e2e["disabled_median_output_tokens_per_s"],
        "E2E disabled median",
    )
    require_close(
        statistics.median(enabled),
        expected_e2e["all_active_top_k_50_median_output_tokens_per_s"],
        "E2E enabled median",
    )
    ratio_change = (statistics.median(enabled) / statistics.median(disabled) - 1) * 100
    require_close(
        ratio_change,
        expected_e2e["ratio_of_median_throughputs_change_percent"],
        "E2E ratio-of-medians change",
    )
    budget = expected_e2e["suggested_maximum_loss_percent"]
    if expected_e2e["passes_suggested_budget"] != (
        statistics.median(deltas) >= -budget
    ):
        raise SystemExit("E2E suggested-budget status mismatch")

    rejected = manifest["rejected_prototype_evidence"]
    rejected_path = checked_repo_path(rejected["path"])
    if rejected_path in release_paths or rejected["release_aggregate_member"]:
        raise SystemExit("rejected prototype leaked into release evidence")
    if rejected["status"] != "rejected_non_release":
        raise SystemExit("rejected prototype status mismatch")
    validate_hash(rejected_path, rejected["sha256"])
    candidate = json.loads(rejected_path.read_text())
    candidate_disabled = candidate["batches"]["256"]["top_p_disabled"][
        "median_output_tokens_per_s"
    ]
    candidate_enabled = candidate["batches"]["256"]["top_p_0.9"][
        "median_output_tokens_per_s"
    ]
    candidate_change = (candidate_enabled / candidate_disabled - 1) * 100
    require_close(
        candidate_change,
        rejected["steady_throughput_change_percent"],
        "rejected candidate throughput change",
    )

    model_status = "model hashes skipped"
    if args.check_model:
        model_root = Path(manifest["model"]["path"])
        for model_file in manifest["model"]["files"]:
            validate_hash(model_root / model_file["path"], model_file["sha256"])
        model_status = f"{len(manifest['model']['files'])} model files"

    print(
        "validated "
        f"{len(manifest['release_raw_results'])} release raw files, "
        f"{len(manifest['harnesses'])} harnesses, 1 rejected artifact, and "
        f"{model_status}"
    )


if __name__ == "__main__":
    main()
