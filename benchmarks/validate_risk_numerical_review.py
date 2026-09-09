"""Validate the sealed risk-review and invariant-qualification archive."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_COMPARISONS = (
    "known_serial_b4.json",
    "mixed_chunk_b257.json",
    "cache_chunk_b257.json",
    "eviction_resume_b257.json",
    "long_chunk_b511.json",
)
EXPECTED_SPECULATIVE = (
    ("spec_graph_review.json", 140),
    ("spec_invariant_review.json", 124),
)


def _fail(message: str) -> None:
    raise RuntimeError(message)


def validate(directory: Path) -> dict:
    checksum_path = directory / "SHA256SUMS"
    if not checksum_path.is_file():
        _fail("missing SHA256SUMS")
    for line in checksum_path.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        artifact = directory / filename
        if not artifact.is_file():
            _fail(f"missing checksummed artifact: {filename}")
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != digest:
            _fail(f"checksum mismatch: {filename}")

    reports = {
        path.name: json.loads(path.read_text())
        for path in directory.glob("*.json")
    }
    summary = reports.get("review_summary.json")
    if not isinstance(summary, dict):
        _fail("missing review_summary.json")
    implementation = summary.get("implementation_sha256")
    bound = {
        report.get("implementation_sha256")
        for report in reports.values()
        if report.get("implementation_sha256") is not None
    }
    if bound != {implementation}:
        _fail("invariant reports do not bind one reviewed implementation")

    kv_states = 0
    logit_states = 0
    for filename in EXPECTED_COMPARISONS:
        comparison = reports[filename].get("comparison", {})
        if (
            comparison.get("passed") is not True
            or comparison.get("kv_mismatches") != []
            or comparison.get("logit_mismatches") != []
        ):
            _fail(f"numerical comparison failed: {filename}")
        kv_states += comparison["corresponding_kv_states"]
        logit_states += comparison["corresponding_logit_states"]
    if (kv_states, logit_states) != (55, 55):
        _fail("unexpected numerical comparison coverage")
    if reports["fp64_primitives.json"].get("passed") is not True:
        _fail("FP64 primitive comparison failed")

    graph = reports["fast_known_graph_b256.json"]["graph17"]["token_ids"]
    eager_report = reports["fast_known_eager_b256.json"]
    eager = eager_report["graph17"]["token_ids"]
    mismatches = [index for index, pair in enumerate(zip(graph, eager)) if pair[0] != pair[1]]
    if mismatches != [9, 12] or eager_report.get("graph_eager_tokens_equal") is not False:
        _fail("fast-mode negative control changed")

    for filename, cycles in EXPECTED_SPECULATIVE:
        report = reports[filename]
        if (
            len(report.get("cycles", ())) != cycles
            or report.get("causality_probe") is not True
            or report.get("failure_retry") is not True
            or not all(row.get("compiler_unchanged") for row in report["cycles"])
            or not all(
                row.get("cache_repeat") and row.get("stream_parity")
                for row in report.get("results", ())
            )
        ):
            _fail(f"speculative smoke failed: {filename}")

    return {
        "passed": True,
        "implementation_sha256": implementation,
        "corresponding_kv_states": kv_states,
        "corresponding_logit_states": logit_states,
        "fast_mode_divergent_requests": mismatches,
        "speculative_cycles": sum(cycles for _, cycles in EXPECTED_SPECULATIVE),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), sort_keys=True))


if __name__ == "__main__":
    main()
