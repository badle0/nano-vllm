from benchmarks.chunked_prefill_tail.validate_decode_jitter_evidence import (
    ARTIFACT_SHA256,
    validate_archive,
)


def test_retained_decode_jitter_evidence_is_exact_and_non_certifying():
    report = validate_archive()
    assert report == {
        "artifact_sha256": ARTIFACT_SHA256,
        "classification": "intrusive_diagnostic_only_non_certifying",
        "stall_count": 8,
        "post_enqueue_sync_residual_count": 7,
        "model_cuda_spike_count": 1,
        "max_stall_thread_cpu_wall_difference_ms": 0.0002089999999999037,
        "latency_certified": False,
        "production_performance_claim": False,
    }
