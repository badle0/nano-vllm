#!/usr/bin/env python3
"""Recompute and validate request-metrics certification evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from certification_common import (
    BENCHMARK_NAME,
    SCHEMA_VERSION,
    derive_observation_seed,
    model_identity,
    sha256_file,
    source_identity,
    summarize_observations,
)


LOG_TOST_PAIR_COUNT = 8
LOG_TOST_T_CRITICAL_90_DF7 = 1.894578605061305
LOG_TOST_BOUNDS = (0.98, 1.02)


def require_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"{context}: {actual} != {expected}")


def require_nested_close(actual, expected, context: str) -> None:
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{context}: dictionary keys differ")
        for key in expected:
            require_nested_close(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{context}: list lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            require_nested_close(left, right, f"{context}[{index}]")
    elif isinstance(expected, float):
        require_close(actual, expected, context)
    elif actual != expected:
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def recompute_pair(baseline: dict, repair: dict) -> dict:
    comparisons = {}
    for key in baseline["cases"]:
        base_case = baseline["cases"][key]
        fix_case = repair["cases"][key]
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


def paired_log_efficiency_tost90(efficiencies: list[float]) -> dict:
    """Return the 90% Student-t CI used by a 5% two-one-sided equivalence test."""
    if len(efficiencies) != LOG_TOST_PAIR_COUNT:
        raise ValueError(
            f"paired log-efficiency TOST requires {LOG_TOST_PAIR_COUNT} pairs"
        )
    if any(efficiency <= 0 for efficiency in efficiencies):
        raise ValueError("efficiencies must be positive before log transformation")
    log_efficiencies = [math.log(efficiency) for efficiency in efficiencies]
    mean_log = statistics.mean(log_efficiencies)
    sample_sd_log = statistics.stdev(log_efficiencies)
    standard_error_log = sample_sd_log / math.sqrt(len(log_efficiencies))
    half_width_log = LOG_TOST_T_CRITICAL_90_DF7 * standard_error_log
    lower = math.exp(mean_log - half_width_log)
    upper = math.exp(mean_log + half_width_log)
    return {
        "method": "paired log-efficiency Student-t 90% CI (TOST alpha=0.05)",
        "pairs": len(efficiencies),
        "degrees_of_freedom": len(efficiencies) - 1,
        "t_critical": LOG_TOST_T_CRITICAL_90_DF7,
        "equivalence_bounds": list(LOG_TOST_BOUNDS),
        "median_efficiency": statistics.median(efficiencies),
        "geometric_mean_efficiency": math.exp(mean_log),
        "sample_sd_log_efficiency": sample_sd_log,
        "standard_error_log_efficiency": standard_error_log,
        "ci90_lower": lower,
        "ci90_upper": upper,
        "equivalence_gate_pass": (
            lower >= LOG_TOST_BOUNDS[0] and upper <= LOG_TOST_BOUNDS[1]
        ),
    }


def validate_archive_provenance(root: Path, statistics_result: dict) -> bool:
    provenance_path = root / "archive_provenance.json"
    if not provenance_path.exists():
        return False
    provenance = json.loads(provenance_path.read_text())
    if provenance["copied_byte_for_byte"] is not True:
        raise ValueError("archive does not declare byte-for-byte preservation")
    files = provenance["files"]
    if provenance["artifact_count"] != len(files) or len(files) != 17:
        raise ValueError("archive provenance must cover exactly 17 run artifacts")
    for relative_path, expected_digest in files.items():
        path = (root / relative_path).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("archived artifact escapes the archive directory")
        if sha256_file(path) != expected_digest:
            raise ValueError(f"archived artifact hash mismatch: {relative_path}")
    if files.get("manifest.json") != provenance["source_manifest_sha256"]:
        raise ValueError("source manifest hash and archived manifest hash differ")
    require_nested_close(
        statistics_result,
        provenance["expected_statistics"],
        "archive.expected_statistics",
    )
    return True


def validate_run(payload: dict, pair: dict, side: str, source: dict, model: dict) -> None:
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("raw schema version mismatch")
    if payload["benchmark"] != BENCHMARK_NAME:
        raise ValueError("raw benchmark name mismatch")
    identity = payload["identity"]
    if identity["pair_id"] != pair["pair_id"] or identity["seed"] != pair["seed"]:
        raise ValueError("raw pair identity mismatch")
    if identity["side"] != side or identity["order"] != pair["order"]:
        raise ValueError("raw side/order mismatch")
    expected_position = pair["order"].split("-").index(side) + 1
    if identity["order_position"] != expected_position:
        raise ValueError("raw order position mismatch")
    if payload["source"] != source:
        raise ValueError("raw source identity differs from manifest")
    if payload["model"]["fingerprint_sha256"] != model["fingerprint_sha256"]:
        raise ValueError("raw model fingerprint differs from manifest")

    configuration = payload["configuration"]
    expected_cases = {
        f"b{batch}_o{output_length}"
        for batch in configuration["batches"]
        for output_length in configuration["output_lengths"]
    }
    if set(payload["cases"]) != expected_cases:
        raise ValueError("raw case set does not match configuration")
    for key, case in payload["cases"].items():
        batch = case["batch"]
        output_length = case["output_length"]
        if key != f"b{batch}_o{output_length}":
            raise ValueError(f"case key mismatch: {key}")
        observations = case["observations"]
        if len(observations) != configuration["repetitions"]:
            raise ValueError(f"observation count mismatch for {key}")
        for repetition, observation in enumerate(observations):
            if observation["repetition"] != repetition:
                raise ValueError(f"repetition mismatch for {key}")
            if observation["cold"] != (repetition == 0):
                raise ValueError(f"cold marker mismatch for {key}")
            expected_seed = derive_observation_seed(
                identity["seed"],
                identity["pair_id"],
                batch,
                output_length,
                repetition,
            )
            if observation["seed"] != expected_seed:
                raise ValueError(f"observation seed mismatch for {key}")
            expected_tokens = batch * output_length
            if observation["output_tokens"] != expected_tokens:
                raise ValueError(f"output token count mismatch for {key}")
            if observation["elapsed_s"] <= 0:
                raise ValueError(f"non-positive elapsed time for {key}")
            require_close(
                observation["output_tokens_per_s"],
                expected_tokens / observation["elapsed_s"],
                f"{key} throughput",
            )
            digest = observation["token_ids_sha256"]
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"invalid token digest for {key}")
            for resource_name, resource_value in observation["resources"].items():
                if resource_value is not None and resource_value < 0:
                    raise ValueError(f"negative {resource_name} for {key}")
        require_nested_close(
            case["summary"],
            summarize_observations(observations),
            f"{key}.summary",
        )


def validate_manifest(
    manifest_path: Path,
    *,
    check_source: bool = False,
    check_model: bool = False,
) -> dict:
    manifest_path = manifest_path.resolve(strict=True)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("manifest schema version mismatch")
    if manifest["benchmark"] != BENCHMARK_NAME:
        raise ValueError("manifest benchmark name mismatch")
    if manifest["artifact_kind"] != "balanced_pair_manifest":
        raise ValueError("manifest artifact kind mismatch")
    for tool_name in ("harness", "runner"):
        record = manifest[tool_name]
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"{tool_name} hash mismatch")

    pair_runs = manifest["pair_runs"]
    if len(pair_runs) != manifest["protocol"]["pairs"]:
        raise ValueError("pair count mismatch")
    expected_pair_ids = list(range(1, len(pair_runs) + 1))
    if [pair["pair_id"] for pair in pair_runs] != expected_pair_ids:
        raise ValueError("pair IDs are not contiguous")
    orders = [pair["order"] for pair in pair_runs]
    if orders.count("baseline-repair") != orders.count("repair-baseline"):
        raise ValueError("pair order is not balanced")

    raw_payloads = {}
    previous_finished = None
    for pair in pair_runs:
        expected_order = "baseline-repair" if pair["pair_id"] % 2 else "repair-baseline"
        if pair["order"] != expected_order:
            raise ValueError("pair order does not follow the declared alternation")
        position_payloads = []
        for side in pair["order"].split("-"):
            artifact = pair["runs"][side]
            path = (root / artifact["path"]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("raw artifact escapes the manifest directory")
            if sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"raw artifact hash mismatch: {path}")
            payload = json.loads(path.read_text())
            if artifact["command"][1:] != payload["invocation"]["argv"]:
                raise ValueError("captured command and raw argv differ")
            if payload["invocation"]["harness_sha256"] != manifest["harness"]["sha256"]:
                raise ValueError("raw harness hash mismatch")
            validate_run(
                payload,
                pair,
                side,
                manifest["sources"][side],
                manifest["model"],
            )
            raw_payloads[(pair["pair_id"], side)] = payload
            position_payloads.append(payload)
        first, second = position_payloads
        if first["invocation"]["finished_utc"] > second["invocation"]["started_utc"]:
            raise ValueError("paired processes overlap or have inconsistent timestamps")
        if previous_finished and previous_finished > first["invocation"]["started_utc"]:
            raise ValueError("process pairs overlap or have inconsistent timestamps")
        previous_finished = second["invocation"]["finished_utc"]

        baseline = raw_payloads[(pair["pair_id"], "baseline")]
        repair = raw_payloads[(pair["pair_id"], "repair")]
        if baseline["configuration"] != repair["configuration"]:
            raise ValueError("paired configurations differ")
        comparable_environment_fields = (
            "python",
            "python_executable",
            "platform",
            "torch",
            "torch_cuda",
            "cudnn",
            "transformers",
            "flash_attn",
            "gpu",
            "gpu_capability",
            "gpu_total_memory_bytes",
            "gpu_multiprocessor_count",
            "visible_cuda_devices",
            "nvidia_driver_versions",
        )
        for field in comparable_environment_fields:
            if baseline["environment"][field] != repair["environment"][field]:
                raise ValueError(f"paired environment differs for {field}")
        for key in baseline["cases"]:
            base_case = baseline["cases"][key]
            fix_case = repair["cases"][key]
            if base_case["prompt_sha256"] != fix_case["prompt_sha256"]:
                raise ValueError(f"paired prompt hashes differ for {key}")
            for base_item, fix_item in zip(
                base_case["observations"], fix_case["observations"]
            ):
                if base_item["seed"] != fix_item["seed"]:
                    raise ValueError(f"paired seeds differ for {key}")
                if base_item["token_ids_sha256"] != fix_item["token_ids_sha256"]:
                    raise ValueError(f"paired token hashes differ for {key}")
        require_nested_close(
            pair["comparisons"],
            recompute_pair(baseline, repair),
            f"pair{pair['pair_id']}.comparisons",
        )

    case_names = pair_runs[0]["comparisons"].keys()
    recomputed_aggregate = {}
    median_floor = manifest["protocol"]["median_efficiency_floor"]
    individual_floor = manifest["protocol"]["individual_efficiency_floor"]
    for key in case_names:
        efficiencies = [pair["comparisons"][key]["efficiency"] for pair in pair_runs]
        recomputed_aggregate[key] = {
            "pair_efficiencies": efficiencies,
            "median_efficiency": statistics.median(efficiencies),
            "minimum_efficiency": min(efficiencies),
            "median_gate_pass": statistics.median(efficiencies) >= median_floor,
            "individual_gate_pass": min(efficiencies) >= individual_floor,
        }
    require_nested_close(manifest["aggregate"], recomputed_aggregate, "aggregate")
    manifest_gate_pass = all(
        item["median_gate_pass"] and item["individual_gate_pass"]
        for item in recomputed_aggregate.values()
    )
    if manifest["gate_pass"] != manifest_gate_pass:
        raise ValueError("stored aggregate gate result is incorrect")

    equivalence = {}
    if len(pair_runs) == LOG_TOST_PAIR_COUNT:
        equivalence = {
            key: paired_log_efficiency_tost90(
                recomputed_aggregate[key]["pair_efficiencies"]
            )
            for key in case_names
        }
    statistics_result = {
        key: {
            "median_efficiency": recomputed_aggregate[key]["median_efficiency"],
            "log_efficiency_tost90": equivalence[key],
        }
        for key in equivalence
    }
    equivalence_gate_pass = all(
        result["equivalence_gate_pass"] for result in equivalence.values()
    )
    gate_pass = manifest_gate_pass and equivalence_gate_pass
    archive_validated = validate_archive_provenance(root, statistics_result)

    if check_source:
        for side, expected in manifest["sources"].items():
            observed = source_identity(Path(expected["root"]), expected["head"])
            if observed != expected:
                raise ValueError(f"current {side} source identity changed")
    if check_model:
        observed_model = model_identity(Path(manifest["model"]["root"]))
        if observed_model != manifest["model"]:
            raise ValueError("current model identity changed")
    return {
        "pairs": len(pair_runs),
        "raw_files": len(pair_runs) * 2,
        "cases": len(case_names),
        "gate_pass": gate_pass,
        "equivalence": equivalence,
        "archive_validated": archive_validated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-source", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()
    result = validate_manifest(
        args.manifest,
        check_source=args.check_source,
        check_model=args.check_model,
    )
    for key, interval in result["equivalence"].items():
        print(
            f"{key}: median={interval['median_efficiency']:.9f}, "
            f"log-TOST90=[{interval['ci90_lower']:.9f}, "
            f"{interval['ci90_upper']:.9f}], "
            f"equivalent={interval['equivalence_gate_pass']}"
        )
    print(
        f"validated {result['raw_files']} raw files, {result['pairs']} balanced "
        f"pairs, and {result['cases']} cases; archive={result['archive_validated']}; "
        f"gate_pass={result['gate_pass']}"
    )


if __name__ == "__main__":
    main()
