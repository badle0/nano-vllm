#!/usr/bin/env python3
"""Validate and aggregate five fresh scheduler-roofline process artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from benchmarks.chunked_prefill_tail.common import immutable_write_json
from benchmarks.chunked_prefill_tail.scheduler_roofline import (
    ABSOLUTE_MEDIAN_GATE_US,
    BACKLOG_SIZES,
    RELATIVE_MEDIAN_GATE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"unexpected schema in {path}")
    if payload.get("kind") != "chunk_scheduler_backlog_roofline":
        raise ValueError(f"unexpected artifact kind in {path}")
    if not payload.get("gates", {}).get("all_pass"):
        raise ValueError(f"raw artifact failed a release gate: {path}")
    measurements = payload.get("measurements", [])
    if [item.get("backlog_size") for item in measurements] != list(BACKLOG_SIZES):
        raise ValueError(f"backlog matrix mismatch in {path}")
    for item in measurements:
        raw = item["raw_nanoseconds"]
        if len(raw) != item["sample_count"]:
            raise ValueError(f"raw sample count mismatch in {path}")
        expected_median = statistics.median(raw) / 1_000.0
        if expected_median != item["microseconds"]["median"]:
            raise ValueError(f"median mismatch in {path}")
        if item["post_state"]["waiting"] != item["backlog_size"]:
            raise ValueError(f"timed backlog changed in {path}")
    return payload


def validate(paths: list[Path]) -> dict:
    if len(paths) != 5:
        raise ValueError("scheduler release evidence requires exactly five artifacts")
    resolved = [path.expanduser().resolve(strict=True) for path in paths]
    if len(set(resolved)) != 5:
        raise ValueError("scheduler evidence paths must be distinct")
    payloads = [_load(path) for path in resolved]
    seeds = [payload["seed"] for payload in payloads]
    if len(set(seeds)) != 5:
        raise ValueError("scheduler evidence seeds must be distinct")
    commits = {payload["provenance"]["git"]["commit"] for payload in payloads}
    sources = {
        payload["provenance"]["source"]["aggregate_sha256"]
        for payload in payloads
    }
    environments = {json.dumps(payload["environment"], sort_keys=True) for payload in payloads}
    if len(commits) != 1 or len(sources) != 1 or len(environments) != 1:
        raise ValueError("commit, source, or environment pins differ across artifacts")

    aggregate = {}
    for backlog_size in BACKLOG_SIZES:
        rows = [
            next(
                item for item in payload["measurements"]
                if item["backlog_size"] == backlog_size
            )
            for payload in payloads
        ]
        aggregate[str(backlog_size)] = {
            "per_process_median_us": [row["microseconds"]["median"] for row in rows],
            "median_of_process_medians_us": statistics.median(
                row["microseconds"]["median"] for row in rows
            ),
            "per_process_p95_us": [row["microseconds"]["p95"] for row in rows],
        }
    zero = aggregate["0"]["median_of_process_medians_us"]
    large = aggregate["500000"]["median_of_process_medians_us"]
    threshold = max(ABSOLUTE_MEDIAN_GATE_US, zero * RELATIVE_MEDIAN_GATE)
    gate = large <= threshold
    return {
        "schema_version": 1,
        "kind": "chunk_scheduler_backlog_roofline_manifest",
        "raw_artifacts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "seed": payload["seed"],
                "measurement_order": payload["measurement_order"],
            }
            for path, payload in zip(resolved, payloads, strict=True)
        ],
        "commit": commits.pop(),
        "source_sha256": sources.pop(),
        "environment": payloads[0]["environment"],
        "aggregate": aggregate,
        "roofline": {
            "zero_backlog_median_us": zero,
            "five_hundred_thousand_median_us": large,
            "relative_ratio": large / zero,
            "gate_threshold_us": threshold,
            "passes": gate,
        },
        "gates": {
            "five_fresh_distinct_process_seeds": True,
            "all_raw_gates_pass": True,
            "approximately_constant_schedule_cost": gate,
            "all_pass": gate,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.inputs)
    immutable_write_json(args.output.expanduser().resolve(), result)
    print(json.dumps({"output": str(args.output), **result["roofline"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
