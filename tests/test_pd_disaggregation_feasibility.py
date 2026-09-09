import pytest

from benchmarks.pd_disaggregation_feasibility import (
    HandoffDescriptor,
    HandoffProgress,
    KVGeometry,
    compare_matched_resources,
    transfer_estimate,
)


def test_recorded_qwen_kv_transfer_geometry():
    target = transfer_estimate(
        KVGeometry(36, 8, 128), prompt_tokens=4096, bandwidth_gbps=200
    )
    draft = transfer_estimate(
        KVGeometry(28, 8, 128), prompt_tokens=4096, bandwidth_gbps=200
    )
    assert target["bytes_per_token"] == 147456
    assert target["total_mib"] == 576
    assert draft["total_mib"] == 448


def test_handoff_uses_logical_order_and_explicit_first_token_owner():
    handoff = HandoffDescriptor(
        model_identity="qwen3-4b:sha256",
        numerical_backend="invariant-v1",
        token_ids=(1, 2, 3),
        processed_tokens=3,
        kv_layout="layers,logical_tokens,kv_heads,head_dim",
        logical_block_order=(0, 1),
        first_token_owner="decode",
    )
    handoff.validate()
    with pytest.raises(ValueError, match="first_token_owner"):
        HandoffDescriptor(
            "model", "fast", (1,), 1, "layout", (99,), "both"
        ).validate()


def test_handoff_acknowledgments_prevent_early_source_release_and_unsafe_retry():
    HandoffProgress(
        receiver_allocation_ack=True,
        transfer_completion_ack=True,
        receiver_install_ack=True,
        source_release_ack=True,
    ).validate()
    HandoffProgress(
        cancellation_requested=True,
        cancellation_ack=True,
        retry_count=1,
        source_release_ack=True,
    ).validate()
    with pytest.raises(ValueError, match="source release"):
        HandoffProgress(source_release_ack=True).validate()
    with pytest.raises(ValueError, match="retry"):
        HandoffProgress(retry_count=1).validate()


def test_comparison_requires_matched_resources_and_uses_slo_goodput():
    base = dict(
        gpu_count=2,
        duration_s=100,
        completed_requests=1000,
        requests_meeting_both_slos=800,
        p95_ttft_ms=100,
        p99_itl_ms=20,
        max_itl_ms=40,
        cost_usd=2,
    )
    disagg = dict(base, requests_meeting_both_slos=880, p95_ttft_ms=80)
    result = compare_matched_resources(disagg, base)
    assert result["goodput_improvement_fraction"] == pytest.approx(0.1)
    assert result["goodput_gate_passed"] is True
    assert result["operational_cost_covered"] is True
    assert result["latency_not_regressed"] is True
    assert result["proceed_to_serving_plan"] is True
    costly = compare_matched_resources(dict(disagg, cost_usd=3), base)
    assert costly["operational_cost_covered"] is False
    assert costly["proceed_to_serving_plan"] is False
    with pytest.raises(ValueError, match="same total GPU"):
        compare_matched_resources(dict(disagg, gpu_count=1), base)
