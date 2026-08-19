from pathlib import Path

import pytest

from benchmarks.chunked_prefill_tail.validate_retained_evidence import (
    DIAGNOSTIC_KIND,
    FULL_KIND,
    validate_archive,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "chunked_prefill_tail" / "evidence"


@pytest.mark.parametrize(
    ("directory", "kind"),
    (
        ("2026-08-18-a100-full-completion-ba1bde4", FULL_KIND),
        ("2026-08-18-a100-contract-phase-e50e732", DIAGNOSTIC_KIND),
    ),
)
def test_retained_chunk_evidence_validates(directory: str, kind: str):
    assert validate_archive(EVIDENCE / directory)["kind"] == kind


def test_retained_full_completion_verdicts_are_non_certifying():
    report = validate_archive(
        EVIDENCE / "2026-08-18-a100-full-completion-ba1bde4"
    )
    assert report["archives"] == [
        {"tau": 256, "classification": "latency_not_certified"},
        {
            "tau": 512,
            "classification": "throughput_ttft_only_not_latency_certified",
        },
    ]
