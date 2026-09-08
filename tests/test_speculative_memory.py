from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.speculative_memory import (
    ALLOCATOR_MARGIN_ALIGNMENT_BYTES,
    ALLOCATOR_MARGIN_MIN_BYTES,
    SpeculativeMemoryPlanningError,
    kv_cache_block_bytes,
    plan_speculative_workspace,
    speculative_route_fits_plan,
)


def _hf_config(**overrides):
    values = dict(
        num_hidden_layers=28,
        num_key_value_heads=8,
        num_attention_heads=16,
        hidden_size=2048,
        head_dim=128,
        dtype=torch.bfloat16,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_kv_block_bytes_matches_qwen_06b_geometry_exactly():
    assert kv_cache_block_bytes(_hf_config(), block_size=256) == 29_360_128


def test_kv_block_bytes_supports_fallback_head_dim_tp_and_dtype_override():
    config = _hf_config(
        num_hidden_layers=2,
        num_key_value_heads=4,
        num_attention_heads=8,
        hidden_size=512,
        head_dim=None,
        dtype=torch.float32,
    )
    assert kv_cache_block_bytes(
        config,
        block_size=16,
        tensor_parallel_size=2,
        dtype=torch.float16,
    ) == 2 * 2 * 16 * 2 * 64 * 2


@pytest.mark.parametrize(
    ("config", "kwargs", "error", "message"),
    [
        (_hf_config(num_hidden_layers=0), {}, SpeculativeMemoryPlanningError, "num_hidden_layers"),
        (_hf_config(num_key_value_heads=True), {}, TypeError, "num_key_value_heads"),
        (_hf_config(num_key_value_heads=3), {"tensor_parallel_size": 2}, SpeculativeMemoryPlanningError, "divisible"),
        (_hf_config(head_dim=0), {}, SpeculativeMemoryPlanningError, "head_dim"),
        (_hf_config(head_dim=None, hidden_size=513), {}, SpeculativeMemoryPlanningError, "hidden_size must be divisible"),
        (_hf_config(dtype=torch.int64), {}, SpeculativeMemoryPlanningError, "floating-point"),
        (_hf_config(dtype="bfloat16"), {}, TypeError, "torch.dtype"),
    ],
)
def test_kv_block_bytes_rejects_invalid_model_geometry(
    config, kwargs, error, message
):
    with pytest.raises(error, match=message):
        kv_cache_block_bytes(config, block_size=256, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"block_size": 0}, SpeculativeMemoryPlanningError, "block_size"),
        ({"block_size": True}, TypeError, "block_size"),
        ({"block_size": 256, "tensor_parallel_size": 0}, SpeculativeMemoryPlanningError, "tensor_parallel_size"),
    ],
)
def test_kv_block_bytes_rejects_invalid_execution_geometry(kwargs, error, message):
    with pytest.raises(error, match=message):
        kv_cache_block_bytes(_hf_config(), **kwargs)


def test_kv_block_bytes_reports_missing_required_fields():
    with pytest.raises(SpeculativeMemoryPlanningError, match="num_hidden_layers"):
        kv_cache_block_bytes(SimpleNamespace(), block_size=256)


def test_workspace_plan_has_exact_small_fixture_components():
    plan = plan_speculative_workspace(
        vocab_size=10,
        configured_k=2,
        max_num_seqs=3,
        max_num_batched_tokens=100,
        max_model_len=100,
        target_logits_dtype=torch.bfloat16,
        draft_logits_dtype=torch.float16,
    )

    assert plan.batch_size == 3
    assert plan.max_effective_k == 2
    assert plan.draft_rows == 3
    assert plan.verifier_rows == 9
    assert plan.draft_probability_bytes == 240
    assert plan.target_probability_bytes == 360
    assert plan.probability_floor_bytes == 600
    assert plan.draft_logits_bytes == 60
    assert plan.verifier_logits_bytes == 180
    assert plan.draft_transform_bytes == 180
    assert plan.verifier_transform_bytes == 540
    assert plan.draft_top_k_workspace_bytes == 366
    assert plan.verifier_top_k_workspace_bytes == 1_098
    assert plan.draft_top_p_workspace_bytes == 1_560
    assert plan.verifier_top_p_workspace_bytes == 4_680
    assert plan.draft_sampling_workspace_bytes == 840
    assert plan.bonus_sampling_workspace_bytes == 840
    assert plan.rejection_correction_workspace_bytes == 2_040
    assert plan.metadata_bytes == 861

    assert plan.draft_filter_phase_bytes == 2_781
    assert plan.draft_softmax_phase_bytes == 1_341
    assert plan.draft_race_phase_bytes == 2_001
    assert plan.draft_phase_bytes == 2_781
    assert plan.verifier_filter_phase_bytes == 6_141
    assert plan.verifier_softmax_phase_bytes == 2_181
    assert plan.verifier_phase_bytes == 6_141
    assert plan.rejection_phase_bytes == 3_501
    assert plan.bonus_phase_bytes == 2_301
    assert plan.modeled_live_peak_bytes == 6_141
    assert plan.allocator_margin_bytes == ALLOCATOR_MARGIN_MIN_BYTES
    assert plan.reservation_bytes == 6_141 + ALLOCATOR_MARGIN_MIN_BYTES


def test_probability_floor_counts_both_q_and_p_at_configured_maximum():
    plan = plan_speculative_workspace(
        vocab_size=151_936,
        configured_k=4,
        max_num_seqs=8,
        max_num_batched_tokens=64,
        max_model_len=512,
        target_logits_dtype=torch.bfloat16,
        draft_logits_dtype=torch.bfloat16,
    )
    expected_q = 8 * 4 * 151_936 * 4
    expected_p = 8 * 5 * 151_936 * 4
    assert plan.draft_probability_bytes == expected_q
    assert plan.target_probability_bytes == expected_p
    assert plan.probability_floor_bytes == expected_q + expected_p
    assert plan.probability_floor_bytes > expected_q


def test_internal_batch_is_limited_by_verification_token_budget():
    plan = plan_speculative_workspace(
        vocab_size=32,
        configured_k=4,
        max_num_seqs=512,
        max_num_batched_tokens=103,
        max_model_len=512,
        target_logits_dtype=torch.float32,
        draft_logits_dtype=torch.float32,
    )
    assert plan.batch_size == 20
    assert plan.verifier_rows == 100
    assert plan.verifier_rows <= 103


def test_route_fit_keeps_configured_batch_cap_when_effective_k_shrinks():
    plan = plan_speculative_workspace(
        vocab_size=32,
        configured_k=4,
        max_num_seqs=512,
        max_num_batched_tokens=103,
        max_model_len=512,
        target_logits_dtype=torch.float32,
        draft_logits_dtype=torch.float32,
    )
    assert plan.batch_size == 20
    for effective_k in range(1, plan.max_effective_k + 1):
        assert speculative_route_fits_plan(
            plan,
            batch_size=plan.batch_size,
            effective_k=effective_k,
        )

    # K=1 would fit 51 rows by token budget alone, but correction/race storage
    # for that larger B was not reserved by the configured-max plan.
    assert not speculative_route_fits_plan(
        plan,
        batch_size=103 // 2,
        effective_k=1,
    )
    assert not speculative_route_fits_plan(
        plan,
        batch_size=plan.batch_size + 1,
        effective_k=plan.max_effective_k,
    )
    assert not speculative_route_fits_plan(
        plan,
        batch_size=plan.batch_size,
        effective_k=plan.max_effective_k + 1,
    )


def test_route_fit_zero_is_fallback_and_invalid_geometry_is_typed():
    plan = plan_speculative_workspace(
        vocab_size=32,
        configured_k=2,
        max_num_seqs=8,
        max_num_batched_tokens=32,
        max_model_len=32,
        target_logits_dtype=torch.float32,
        draft_logits_dtype=torch.float32,
    )
    assert not speculative_route_fits_plan(
        plan, batch_size=0, effective_k=1
    )
    assert not speculative_route_fits_plan(
        plan, batch_size=1, effective_k=0
    )
    with pytest.raises(TypeError, match="plan"):
        speculative_route_fits_plan(
            object(), batch_size=1, effective_k=1
        )
    with pytest.raises(TypeError, match="batch_size"):
        speculative_route_fits_plan(
            plan, batch_size=True, effective_k=1
        )
    with pytest.raises(
        SpeculativeMemoryPlanningError, match="effective_k"
    ):
        speculative_route_fits_plan(
            plan, batch_size=1, effective_k=-1
        )


def test_configured_k_is_clipped_to_the_largest_schedulable_effective_k():
    plan = plan_speculative_workspace(
        vocab_size=32,
        configured_k=100,
        max_num_seqs=8,
        max_num_batched_tokens=16,
        max_model_len=9,
        target_logits_dtype=torch.float32,
        draft_logits_dtype=torch.float32,
    )
    assert plan.configured_k == 100
    assert plan.max_effective_k == 8
    assert plan.batch_size == 1
    assert plan.verifier_rows == 9


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "max_model_len"),
    [(1, 512), (512, 1), (1, 1)],
)
def test_zero_effective_k_returns_an_explicit_zero_route_plan(
    max_num_batched_tokens, max_model_len
):
    plan = plan_speculative_workspace(
        vocab_size=32,
        configured_k=4,
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        target_logits_dtype=torch.float32,
        draft_logits_dtype=torch.float32,
    )
    assert plan.max_effective_k == 0
    assert plan.batch_size == 0
    assert plan.modeled_live_peak_bytes == 0
    assert plan.allocator_margin_bytes == 0
    assert plan.reservation_bytes == 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("vocab_size", 0, SpeculativeMemoryPlanningError),
        ("vocab_size", True, TypeError),
        ("configured_k", 0, SpeculativeMemoryPlanningError),
        ("configured_k", 1.0, TypeError),
        ("max_num_seqs", -1, SpeculativeMemoryPlanningError),
        ("max_num_batched_tokens", "64", TypeError),
        ("max_model_len", 0, SpeculativeMemoryPlanningError),
        ("target_logits_dtype", torch.int64, SpeculativeMemoryPlanningError),
        ("draft_logits_dtype", "float16", TypeError),
    ],
)
def test_workspace_plan_rejects_invalid_inputs(field, value, error):
    kwargs = dict(
        vocab_size=32,
        configured_k=2,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=64,
        target_logits_dtype=torch.float32,
        draft_logits_dtype=torch.float32,
    )
    kwargs[field] = value
    with pytest.raises(error):
        plan_speculative_workspace(**kwargs)


def test_allocator_margin_is_ten_percent_when_larger_than_minimum_and_aligned():
    plan = plan_speculative_workspace(
        vocab_size=200_000,
        configured_k=4,
        max_num_seqs=64,
        max_num_batched_tokens=320,
        max_model_len=512,
        target_logits_dtype=torch.float32,
        draft_logits_dtype=torch.float32,
    )
    unaligned_ten_percent = (plan.modeled_live_peak_bytes + 9) // 10
    expected = (
        (unaligned_ten_percent + ALLOCATOR_MARGIN_ALIGNMENT_BYTES - 1)
        // ALLOCATOR_MARGIN_ALIGNMENT_BYTES
        * ALLOCATOR_MARGIN_ALIGNMENT_BYTES
    )
    assert unaligned_ten_percent > ALLOCATOR_MARGIN_MIN_BYTES
    assert plan.allocator_margin_bytes == expected
    assert plan.allocator_margin_bytes % ALLOCATOR_MARGIN_ALIGNMENT_BYTES == 0


def test_workspace_plan_is_deterministic_and_immutable():
    kwargs = dict(
        vocab_size=128,
        configured_k=3,
        max_num_seqs=7,
        max_num_batched_tokens=64,
        max_model_len=64,
        target_logits_dtype=torch.float32,
        draft_logits_dtype=torch.bfloat16,
    )
    first = plan_speculative_workspace(**kwargs)
    second = plan_speculative_workspace(**kwargs)
    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.reservation_bytes = 0


def test_logits_dtype_changes_logits_and_transform_but_not_probability_floor():
    common = dict(
        vocab_size=128,
        configured_k=3,
        max_num_seqs=7,
        max_num_batched_tokens=64,
        max_model_len=64,
        draft_logits_dtype=torch.bfloat16,
    )
    fp16 = plan_speculative_workspace(
        **common, target_logits_dtype=torch.float16
    )
    fp32 = plan_speculative_workspace(
        **common, target_logits_dtype=torch.float32
    )
    assert fp16.probability_floor_bytes == fp32.probability_floor_bytes
    assert fp32.verifier_logits_bytes == 2 * fp16.verifier_logits_bytes
    assert fp32.verifier_transform_bytes > fp16.verifier_transform_bytes
    assert fp32.reservation_bytes > fp16.reservation_bytes


def test_top_k_and_top_p_are_named_and_only_the_larger_private_peak_is_charged():
    plan = plan_speculative_workspace(
        vocab_size=128,
        configured_k=3,
        max_num_seqs=128,
        max_num_batched_tokens=512,
        max_model_len=512,
        target_logits_dtype=torch.float16,
        draft_logits_dtype=torch.float16,
    )
    assert plan.verifier_rows == 512
    assert plan.verifier_top_k_workspace_bytes > plan.verifier_top_p_workspace_bytes
    expected_filter_phase = (
        plan.draft_probability_bytes
        + 2 * plan.verifier_logits_bytes
        + plan.verifier_top_k_workspace_bytes
        + plan.metadata_bytes
    )
    expected_softmax_phase = (
        plan.draft_probability_bytes
        + 2 * plan.verifier_logits_bytes
        + 2 * plan.target_probability_bytes
        + plan.metadata_bytes
    )
    assert plan.verifier_filter_phase_bytes == expected_filter_phase
    assert plan.verifier_softmax_phase_bytes == expected_softmax_phase
    assert plan.verifier_phase_bytes == max(
        expected_filter_phase, expected_softmax_phase
    )


def test_unmeasured_graph_and_backend_components_are_explicit_not_zero():
    plan = plan_speculative_workspace(
        vocab_size=128,
        configured_k=3,
        max_num_seqs=7,
        max_num_batched_tokens=64,
        max_model_len=64,
        target_logits_dtype=torch.float32,
        draft_logits_dtype=torch.bfloat16,
    )
    assert plan.graph_static_workspace_bytes is None
    assert plan.backend_library_workspace_bytes is None
    assumptions = " ".join(plan.audit_required_components)
    assert "CUDA graph-pool" in assumptions
    assert "backend-internal" in assumptions
