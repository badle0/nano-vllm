#!/usr/bin/env python3
"""Validate the accepted repaired-streaming certificate without CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


ACCEPTED_COMMIT = "14002ae04102eef58aea09fa8a2a78eca0103b5f"
ACCEPTED_EVIDENCE_SCHEMA = 3
ACCEPTED_ARTIFACT_SHA256 = (
    "df662215db769d0c93129b8d29fc9fbcda4998e41cc412d312864c437da7f8c7"
)
SUPERSEDED_COMMIT = "cf6da50ec8fca39588ed7ba0b8be734379bb3765"
SUPERSEDED_ARTIFACT_SHA256 = (
    "e3ac8f95424450f48a6f67a58b3575569562aab95e5557f61b85179c6f4643ec"
)
CORE_PREFIX_CACHE_POLICY = "fresh_block_manager_before_each_timed_route"
T90_DF7 = 1.894579
CORRECTION_WINDOW_SIZE = 32
CORRECTION_BOUNDARY_OVERLAP = 8
CORRECTION_TIME_SLOPE_LIMIT = 1.5
CORRECTION_STATE_SLOPE_LIMIT = 1.125


class CertificateValidationError(ValueError):
    """The supplied evidence is not the accepted streaming certificate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CertificateValidationError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "cannot summarize an empty observation list")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _same(left: Any, right: Any, tolerance: float = 1e-11) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def _paired_peak_ok(stream_peak: int, generate_peak: int) -> bool:
    if stream_peak < 0 or generate_peak < 0:
        return False
    if generate_peak == 0:
        return stream_peak == 0
    return stream_peak <= generate_peak * 1.01


def _correction_gate(document: dict[str, Any]) -> bool:
    correction = document["correction_heavy_cpu_scaling"]
    small, large = correction["results"]
    token_ratio = large["num_tokens"] / small["num_tokens"]
    time_ratio = large["median_process_seconds"] / small["median_process_seconds"]
    state_ratio = (
        large["state_bytes_before_flush"] / small["state_bytes_before_flush"]
    )
    hard_incremental_decode_limit = (
        CORRECTION_WINDOW_SIZE + 2 * CORRECTION_BOUNDARY_OVERLAP
    )
    process_time_ratio_limit = token_ratio * CORRECTION_TIME_SLOPE_LIMIT
    state_bytes_ratio_limit = token_ratio * CORRECTION_STATE_SLOPE_LIMIT
    _require(correction["window_size"] == CORRECTION_WINDOW_SIZE,
             "correction window size mismatch")
    _require(correction["boundary_overlap"] == CORRECTION_BOUNDARY_OVERLAP,
             "correction boundary overlap mismatch")
    _require(
        correction["hard_incremental_decode_limit"]
        == hard_incremental_decode_limit,
        "correction incremental-decode limit mismatch",
    )
    _require(
        _same(correction["scaling"]["process_time_ratio_limit"],
              process_time_ratio_limit),
        "correction time-scaling limit mismatch",
    )
    _require(
        _same(correction["scaling"]["state_bytes_ratio_limit"],
              state_bytes_ratio_limit),
        "correction state-scaling limit mismatch",
    )
    gates = {
        "exact_and_state_released": all(
            item["exact_final_text"] and item["state_released_after_flush"]
            for item in correction["results"]
        ),
        "correction_fraction_at_least_0_49": all(
            item["correction_fraction"] >= 0.49 for item in correction["results"]
        ),
        "incremental_decode_length_bounded": all(
            item["max_feed_decode_tokens"] <= hard_incremental_decode_limit
            for item in correction["results"]
        ),
        "exactly_one_full_length_flush": all(
            item["flush_decode_tokens"] == item["num_tokens"]
            and item["full_length_decode_calls"] == 1
            for item in correction["results"]
        ),
        "process_time_scaling_within_roofline": (
            time_ratio <= process_time_ratio_limit
        ),
        "state_scaling_within_linear_roofline": (
            state_ratio <= state_bytes_ratio_limit
        ),
    }
    _require(_same(token_ratio, correction["scaling"]["token_ratio"]),
             "correction token ratio mismatch")
    _require(_same(time_ratio, correction["scaling"]["process_time_ratio"]),
             "correction time ratio mismatch")
    _require(_same(state_ratio, correction["scaling"]["state_bytes_ratio"]),
             "correction state ratio mismatch")
    _require(gates == correction["gates"], "correction gates were not recomputed")
    _require(all(gates.values()) == correction["all_gates_pass"],
             "correction aggregate gate mismatch")
    return all(gates.values())


def validate_evidence(
    document: dict[str, Any],
    *,
    raw_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute every accepted gate and return compact accepted metrics."""
    repository = document.get("provenance", {}).get("repository", {})
    commit = repository.get("commit")
    schema = document.get("schema_version")
    if (
        raw_sha256 == SUPERSEDED_ARTIFACT_SHA256
        or commit == SUPERSEDED_COMMIT
        or schema == 2
    ):
        raise CertificateValidationError(
            "superseded cf6da50/schema-2 evidence is not an accepted certificate"
        )
    _require(schema == ACCEPTED_EVIDENCE_SCHEMA,
             f"unsupported evidence schema: {schema!r}")
    _require(commit == ACCEPTED_COMMIT, f"unaccepted evidence commit: {commit!r}")
    if raw_sha256 is not None:
        _require(raw_sha256 == ACCEPTED_ARTIFACT_SHA256,
                 "accepted artifact SHA-256 mismatch")

    workers = document["observations"]
    protocol = document["protocol"]
    _require(len(workers) == 8, "accepted certificate requires eight workers")
    _require(protocol["fresh_worker_processes"] == 8, "worker protocol mismatch")
    _require(protocol["timed_rounds_per_worker"] == 4, "round protocol mismatch")
    _require(protocol["raw_timed_route_pairs"] == 32, "raw-pair protocol mismatch")
    rounds = [item for worker in workers for item in worker["core"]["timed_rounds"]]
    _require(len(rounds) == 32, "accepted certificate requires 32 raw pairs")

    worker_deltas = []
    worker_exposure = []
    for worker in workers:
        timed = worker["core"]["timed_rounds"]
        _require(len(timed) == 4, "each worker must contain four timed rounds")
        round_hashes = []
        for item in timed:
            generate = item["generate"]
            stream = item["stream"]
            _require(_canonical_sha256(item["prompts"]) == item["prompt_sha256"],
                     "timed prompt hash mismatch")
            round_hashes.append(item["prompt_sha256"])
            _require(generate["token_sha256"] == stream["token_sha256"]
                     and generate["text_sha256"] == stream["text_sha256"]
                     and item["tokens_equivalent"] and item["texts_equivalent"],
                     "seeded route outputs diverged")
            expected_delta = (
                stream["tokens_per_second"] / generate["tokens_per_second"] - 1.0
            ) * 100.0
            _require(_same(expected_delta,
                           item["paired_stream_throughput_delta_percent"]),
                     "raw paired throughput delta mismatch")
            _require(
                [item[route]["pair_position"] for route in item["measurement_order"]]
                == [0, 1],
                "route position does not match declared order",
            )
            resets = item["prefix_cache_resets"]
            _require([reset["route"] for reset in resets] == item["measurement_order"],
                     "prefix-cache reset order mismatch")
            _require(item["max_prompt_plus_completion_tokens"] < item["block_size"],
                     "timed workload crossed a prefix-cache block")
            _require(all(
                reset["policy"] == CORE_PREFIX_CACHE_POLICY
                and reset["pair_position"] == position
                and reset["used_blocks_before_reset"] == 0
                and reset["used_blocks_after_route"] == 0
                and reset["cached_block_hashes_after_route"] == 0
                for position, reset in enumerate(resets)
            ), "cache-neutral reset invariant failed")

        _require(_canonical_sha256(round_hashes) == worker["prompt_sha256"],
                 "worker prompt-suite hash mismatch")
        orders = [item["measurement_order"] for item in timed]
        _require(orders.count(["generate", "stream"]) == 2
                 and orders.count(["stream", "generate"]) == 2,
                 "worker route order is not 2/2 counterbalanced")
        deltas = [item["paired_stream_throughput_delta_percent"] for item in timed]
        exposure = [item["caller_exposure_factor"] for item in timed]
        worker_delta = statistics.median(deltas)
        worker_factor = statistics.median(exposure)
        _require(_same(worker_delta,
                       worker["core"]["paired_stream_throughput_delta_percent"]),
                 "worker paired-median mismatch")
        _require(_same(worker_factor, worker["core"]["caller_exposure_factor"]),
                 "worker exposure-median mismatch")
        worker_deltas.append(worker_delta)
        worker_exposure.append(worker_factor)

    mean_delta = statistics.mean(worker_deltas)
    margin = T90_DF7 * statistics.stdev(worker_deltas) / math.sqrt(8)
    paired_ci = {
        "mean": mean_delta,
        "low": mean_delta - margin,
        "high": mean_delta + margin,
    }
    recorded_ci = document["aggregate"]["matched_final_decode_consumer"][
        "paired_mean_delta_percent_90ci"
    ]
    _require(_same(paired_ci, recorded_ci), "paired 90% interval mismatch")

    event_delays = [
        event["engine_to_caller_seconds"]
        for item in rounds
        for event in item["stream"]["event_delivery"]["values"]
    ]
    _require(len(event_delays) == 65_536, "raw event count mismatch")
    generate_peak = [
        item["generate"]["memory"]["peak"]["allocated_bytes"] for item in rounds
    ]
    stream_peak = [
        item["stream"]["memory"]["peak"]["allocated_bytes"] for item in rounds
    ]

    sleep_values = sorted({
        item["consumer_sleep_seconds_per_event"]
        for worker in workers
        for item in worker["slow_consumer"]
    })
    slow_batch_sizes = {
        item["num_events"] // item["num_steps"]
        for worker in workers
        for item in worker["slow_consumer"]
        if item["num_steps"] > 0
        and item["num_events"] % item["num_steps"] == 0
    }
    _require(len(slow_batch_sizes) == 1,
             "slow-consumer batch size is not uniquely derivable")
    slow_batch_size = next(iter(slow_batch_sizes))
    _require(slow_batch_size == protocol["slow_batch_size"],
             "derived slow-consumer batch size does not match protocol")
    slow_gap_ms = {}
    for sleep_value in sleep_values:
        values = [
            item["median_inter_step_gap_seconds"] * 1000.0
            for worker in workers
            for item in worker["slow_consumer"]
            if item["consumer_sleep_seconds_per_event"] == sleep_value
        ]
        slow_gap_ms[f"{sleep_value * 1000.0:g}"] = statistics.median(values)
    baseline = slow_gap_ms["0"]
    roofline = {}
    for key, observed in slow_gap_ms.items():
        sleep_ms = float(key)
        expected = baseline + slow_batch_size * sleep_ms
        tolerance = max(2.0, 0.10 * slow_batch_size * sleep_ms)
        residual = observed - expected
        roofline[key] = {
            "observed_gap_ms": observed,
            "expected_gap_ms": expected,
            "residual_ms": residual,
            "tolerance_ms": tolerance,
            "within_tolerance": abs(residual) <= tolerance,
        }
    _require(_same(roofline, document["aggregate"]["backpressure_roofline"]),
             "backpressure roofline mismatch")

    correction_gate = _correction_gate(document)
    stable_keys = (
        "repository", "benchmark_script_sha256", "model", "software", "cpu", "gpu"
    )
    pins = [{key: worker["environment"][key] for key in stable_keys}
            for worker in workers]
    gates = {
        "all_seeded_tokens_equivalent": all(
            worker["core"]["tokens_equivalent"] for worker in workers
        ),
        "all_seeded_text_equivalent": all(
            worker["core"]["texts_equivalent"] for worker in workers
        ),
        "eight_distinct_seeds": len({worker["trial_seed"] for worker in workers}) == 8,
        "eight_distinct_prompt_sets": len({
            worker["prompt_sha256"] for worker in workers
        }) == 8,
        "all_timed_round_seeds_distinct": len({
            item["round_seed"] for item in rounds
        }) == 32,
        "all_timed_prompt_sets_distinct": len({
            item["prompt_sha256"] for item in rounds
        }) == 32,
        "balanced_pair_order": all(
            [item["measurement_order"] for item in worker["core"]["timed_rounds"]]
            .count(["generate", "stream"]) == 2
            and [item["measurement_order"] for item in worker["core"]["timed_rounds"]]
            .count(["stream", "generate"]) == 2
            for worker in workers
        ),
        "route_positions_match_declared_order": all(
            item[route]["pair_position"] == position
            for item in rounds
            for position, route in enumerate(item["measurement_order"])
        ),
        "full_length_warmup_covers_both_routes": all(
            worker["warmup"]["full_length_tokens_per_route"]
            == protocol["full_length_warmup_tokens_per_route"]
            and all(
                item["generate"]["num_tokens"] % len(item["prompts"]) == 0
                and item["generate"]["num_tokens"] // len(item["prompts"])
                == worker["warmup"]["full_length_tokens_per_route"]
                for item in worker["core"]["timed_rounds"]
            )
            and {
                item["route"]
                for item in worker["warmup"]["records"]
                if item["max_tokens"]
                == worker["warmup"]["full_length_tokens_per_route"]
            } == {"generate", "stream"}
            for worker in workers
        ),
        "cache_neutral_core_policy_asserted": all(
            worker["core"]["prefix_cache_policy"] == CORE_PREFIX_CACHE_POLICY
            and worker["warmup"]["prefix_cache_policy"] == CORE_PREFIX_CACHE_POLICY
            for worker in workers
        ) and all(
            item["max_prompt_plus_completion_tokens"] < item["block_size"]
            and [reset["route"] for reset in item["prefix_cache_resets"]]
            == item["measurement_order"]
            and all(
                reset["policy"] == CORE_PREFIX_CACHE_POLICY
                and reset["pair_position"] == position
                and reset["used_blocks_before_reset"] == 0
                and reset["used_blocks_after_route"] == 0
                and reset["cached_block_hashes_after_route"] == 0
                for position, reset in enumerate(item["prefix_cache_resets"])
            )
            for item in rounds
        ),
        "all_worker_pins_identical": all(pin == pins[0] for pin in pins),
        "paired_mean_90ci_inside_plus_or_minus_2_percent": (
            paired_ci["low"] >= -2.0 and paired_ci["high"] <= 2.0
        ),
        "event_delivery_p95_at_most_1ms": _percentile(event_delays, 0.95) <= 0.001,
        "minimum_caller_exposure_at_least_10x": min(worker_exposure) >= 10.0,
        "stream_peak_memory_within_1_percent_of_generate": all(
            _paired_peak_ok(stream, generate)
            for stream, generate in zip(stream_peak, generate_peak, strict=True)
        ),
        "pending_storage_bounded_by_slow_batch_minus_one": all(
            item["max_pending_events_after_delivery"]
            <= item["num_events"] // item["num_steps"] - 1
            for worker in workers
            for item in worker["slow_consumer"]
        ),
        "backpressure_matches_synchronous_roofline": all(
            item["within_tolerance"] for item in roofline.values()
        ),
        "incremental_detokenizer_decode_bounded": all(
            item["max_feed_decode_tokens"] <= 48
            for worker in workers
            for item in worker["detokenizer"]
        ),
        "correction_heavy_cpu_scaling": correction_gate,
    }
    gates["all_required_gates_pass"] = all(gates.values())
    _require(gates == document["aggregate"]["gates"],
             "recorded accepted gates differ from independent recomputation")
    _require(gates["all_required_gates_pass"], "one or more required gates failed")

    raw_deltas = [item["paired_stream_throughput_delta_percent"] for item in rounds]
    worst = min(rounds, key=lambda item: item["paired_stream_throughput_delta_percent"])
    worst_worker = next(
        index for index, worker in enumerate(workers)
        if worst in worker["core"]["timed_rounds"]
    )
    return {
        "fresh_worker_units": 8,
        "raw_timed_pairs": 32,
        "event_count": len(event_delays),
        "worker_paired_delta_percent": {
            "values": worker_deltas,
            "mean": mean_delta,
            "central_90_percent_t_interval": [paired_ci["low"], paired_ci["high"]],
        },
        "raw_paired_delta_percent": _summary(raw_deltas),
        "delivery_ms": _summary([value * 1000.0 for value in event_delays]),
        "exposure_factor": _summary(worker_exposure),
        "maximum_paired_peak_delta_bytes": max(
            stream - generate
            for stream, generate in zip(stream_peak, generate_peak, strict=True)
        ),
        "backpressure_roofline": roofline,
        "correction_scaling": document["correction_heavy_cpu_scaling"]["scaling"],
        "worst_raw_round": {
            "worker": worst_worker,
            "round": worst["round_index"],
            "order": worst["measurement_order"],
            "delta_percent": worst["paired_stream_throughput_delta_percent"],
            "worker_median_percent": worker_deltas[worst_worker],
        },
        "gates": gates,
    }


def validate_archive(artifact: Path, manifest: Path) -> dict[str, Any]:
    raw = artifact.read_bytes()
    raw_sha256 = _sha256_bytes(raw)
    manifest_document = json.loads(manifest.read_text())
    artifact_record = manifest_document["artifact"]
    _require(manifest_document["manifest_schema_version"] == 1,
             "unsupported archive manifest schema")
    _require(manifest_document["status"] == "accepted", "archive is not accepted")
    _require(artifact_record["filename"] == artifact.name, "archive filename mismatch")
    _require(artifact_record["sha256"] == raw_sha256, "manifest SHA-256 mismatch")
    _require(artifact_record["size_bytes"] == len(raw), "manifest size mismatch")
    document = json.loads(raw)
    _require(
        artifact_record["evidence_schema_version"] == document["schema_version"],
        "manifest evidence schema mismatch",
    )
    recomputed = validate_evidence(document, raw_sha256=raw_sha256)
    _require(manifest_document["accepted_gates"] == recomputed["gates"],
             "manifest accepted gates mismatch")
    _require(_same(manifest_document["recomputed_summary"], recomputed),
             "manifest recomputed summary mismatch")
    provenance = manifest_document["provenance"]
    evidence_repository = document["provenance"]["repository"]
    evidence_model = document["provenance"]["model"]
    evidence_environment = document["environment"]
    _require(provenance["commit"] == ACCEPTED_COMMIT, "manifest commit mismatch")
    _require(provenance["branch"] == evidence_repository["branch"],
             "manifest branch mismatch")
    _require(provenance["source_tree"]
             == evidence_repository["source"]["git_tree"],
             "manifest source tree mismatch")
    _require(provenance["benchmark_script_sha256"]
             == document["provenance"]["benchmark_script_sha256"],
             "manifest benchmark script mismatch")
    _require(provenance["source_manifest_sha256"]
             == evidence_repository["source"]["manifest_sha256"],
             "manifest source fingerprint mismatch")
    _require(provenance["model_manifest_sha256"]
             == evidence_model["manifest_sha256"],
             "manifest model fingerprint mismatch")
    _require(provenance["model_size_bytes"] == evidence_model["total_size_bytes"],
             "manifest model size mismatch")
    _require(provenance["recorded_at_utc"] == document["recorded_at_utc"],
             "manifest recording timestamp mismatch")
    _require(manifest_document["certification_date"]
             == document["recorded_at_utc"][:10],
             "manifest certification date mismatch")
    _require(all(
        evidence_environment["software"].get(key) == value
        for key, value in provenance["software"].items()
    ), "manifest software provenance mismatch")
    _require(all(
        evidence_environment["gpu"].get(key) == value
        for key, value in provenance["gpu"].items()
    ), "manifest GPU provenance mismatch")
    superseded = manifest_document["superseded_runs"]
    _require(any(
        item["commit"] == SUPERSEDED_COMMIT
        and item["sha256"] == SUPERSEDED_ARTIFACT_SHA256
        and item["archived"] is False
        for item in superseded
    ), "manifest does not quarantine the superseded cf6da50 run")
    return recomputed


def _parser() -> argparse.ArgumentParser:
    results = Path(__file__).resolve().parents[1] / "pr5_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=results / "repaired_streaming_cert_a100_2026-08-18_14002ae.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=results / "repaired_streaming_cert_a100_2026-08-18_14002ae.manifest.json",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = validate_archive(args.artifact, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
