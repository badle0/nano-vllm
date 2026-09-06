import pickle
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from nanovllm.engine.speculative_memory import (
    SpeculativeMemoryPlanningError,
    plan_speculative_workspace,
)
from nanovllm.engine.speculative_plan import (
    SpecPlanRow,
    SpecStepPlan,
    build_speculative_step_plan,
    certify_speculative_workspace,
    derive_speculative_effective_k,
    draft_catchup_tokens,
    plan_exact_speculative_workspace,
    speculative_k_budget,
)
from nanovllm.engine.speculative_routes import (
    DRAFT_ROUTE_SCHEMA,
    DraftCatchupFamily,
    DraftExecutionMode,
    DraftRouteKey,
    DraftSamplerEnvelope,
)


def _configured_workspace(*, batch_size=8, configured_k=4, tokens=256):
    return plan_speculative_workspace(
        vocab_size=32,
        configured_k=configured_k,
        max_num_seqs=batch_size,
        max_num_batched_tokens=tokens,
        max_model_len=128,
        target_logits_dtype=torch.bfloat16,
        draft_logits_dtype=torch.float16,
    )


def _route(
    effective_k,
    *,
    batch_bucket=4,
    catchup_family=DraftCatchupFamily.PAGED_EAGER_DYNAMIC,
    execution_mode=DraftExecutionMode.CUDA_GRAPH,
):
    return DraftRouteKey(
        schema=DRAFT_ROUTE_SCHEMA,
        execution_mode=execution_mode,
        batch_bucket=batch_bucket,
        effective_k=effective_k,
        catchup_family=catchup_family,
        sampler_envelope=DraftSamplerEnvelope.EXACT_WORST_CASE,
    )


def _row(
    seq_id,
    *,
    committed=10,
    catchup=0,
    remaining=16,
    headroom=32,
    highest_draft=None,
    highest_target=None,
):
    target_cached = committed - 1
    assert catchup <= target_cached
    return SpecPlanRow(
        seq_id=seq_id,
        committed_tokens=committed,
        target_cached_tokens=target_cached,
        draft_cached_tokens=target_cached - catchup,
        remaining_completion_tokens=remaining,
        model_position_headroom=headroom,
        highest_draft_write_position=highest_draft,
        highest_target_write_position=highest_target,
        block_table=(seq_id, seq_id + 10),
    )


def _build(*, rows, effective_k, max_tokens, route_key, **overrides):
    kwargs = dict(
        cycle_id=7,
        rows=rows,
        configured_k=4,
        workspace_route_cap=4,
        effective_k=effective_k,
        max_num_batched_tokens=max_tokens,
        configured_workspace=_configured_workspace(),
        route_key=route_key,
        bypass_reason=None,
    )
    kwargs.update(overrides)
    return build_speculative_step_plan(**kwargs)


@pytest.mark.parametrize(
    ("maximum", "batch", "catchup", "expected"),
    [
        (17, 2, 3, 3),
        (16, 2, 3, 2),
        (5, 2, 3, 0),
        (0, 1, 0, 0),
        (100, 0, 99, 0),
    ],
)
def test_k_budget_is_exact_floor_formula_with_zero_clamp(
    maximum, batch, catchup, expected
):
    assert speculative_k_budget(
        max_num_batched_tokens=maximum,
        batch_size=batch,
        draft_catchup_tokens=catchup,
    ) == expected


@pytest.mark.parametrize("bad", [True, 1.0, "1", None])
def test_k_budget_rejects_non_integer_geometry(bad):
    with pytest.raises(TypeError, match="integer"):
        speculative_k_budget(
            max_num_batched_tokens=bad,
            batch_size=1,
            draft_catchup_tokens=0,
        )


def test_effective_k_property_sweep_matches_all_documented_caps():
    for batch_size in range(1, 5):
        for catchup in range(8):
            rows = tuple(
                _row(
                    index,
                    committed=catchup + 2 if index == 0 else 10,
                    catchup=catchup if index == 0 else 0,
                    remaining=7,
                    headroom=6,
                )
                for index in range(batch_size)
            )
            assert draft_catchup_tokens(rows) == catchup
            for maximum in range(31):
                expected = min(
                    5,
                    6,
                    6,
                    max(maximum // batch_size - 1, 0),
                    max(
                        (maximum - catchup - batch_size)
                        // (2 * batch_size),
                        0,
                    ),
                    5,
                )
                actual = derive_speculative_effective_k(
                    rows=rows,
                    configured_k=5,
                    max_num_batched_tokens=maximum,
                    workspace_route_cap=5,
                )
                assert actual == expected


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"configured_k": 0}, 0),
        ({"workspace_route_cap": 0}, 0),
        ({"rows": (_row(0, remaining=1),)}, 0),
        ({"rows": (_row(0, headroom=0),)}, 0),
        ({"max_num_batched_tokens": 4}, 1),
    ],
)
def test_effective_k_fails_closed_at_each_scalar_cap(overrides, expected):
    kwargs = dict(
        rows=(_row(0),),
        configured_k=4,
        max_num_batched_tokens=64,
        workspace_route_cap=4,
    )
    kwargs.update(overrides)
    assert derive_speculative_effective_k(**kwargs) == expected


def test_positive_plan_records_exact_full_v5_geometry_and_v4_shadow_view():
    rows = (
        _row(10, committed=10, catchup=1),
        _row(20, committed=20, catchup=2),
    )
    route = _route(3, batch_bucket=4)
    plan = _build(
        rows=rows,
        effective_k=3,
        max_tokens=17,
        route_key=route,
    )

    assert plan.configured_k == 4
    assert plan.workspace_route_cap == 4
    assert plan.draft_catchup_tokens == 3
    assert plan.draft_query_tokens == 2 * 3
    assert plan.target_query_tokens == 2 * (3 + 1)
    assert plan.total_model_positions == 3 + 2 * 3 + 2 * (3 + 1)
    assert plan.total_model_positions == 17
    assert plan.draft_step_token_counts == (2, 2, 2)
    assert plan.shadow_target_query_tokens == 2
    assert plan.total_scheduled_tokens == 3 + 2 * 3 + 2
    assert plan.fallback_reason is None
    assert plan.uses_speculation

    assert plan.rows[0].highest_draft_write_position == 11
    assert plan.rows[0].highest_proposal_input_position == 11
    assert plan.rows[0].highest_target_write_position == 12
    assert plan.rows[1].highest_draft_write_position == 21
    assert plan.rows[1].highest_target_write_position == 22


def test_workspace_certificate_counts_q_and_p_for_exact_route_bucket():
    configured = _configured_workspace()
    route = _route(3, batch_bucket=4)
    exact = plan_exact_speculative_workspace(
        configured,
        batch_size=route.batch_bucket,
        effective_k=route.effective_k,
    )
    certificate = certify_speculative_workspace(
        configured,
        live_batch_size=2,
        draft_catchup_tokens=3,
        effective_k=3,
        route_key=route,
    )

    assert exact.batch_size == 4
    assert exact.max_effective_k == 3
    assert exact.draft_probability_bytes == 4 * 3 * 32 * 4
    assert exact.target_probability_bytes == 4 * 4 * 32 * 4
    assert exact.probability_floor_bytes == 4 * 32 * (4 * 3 + 4 * 4)
    assert certificate.live_batch_size == 2
    assert certificate.workspace_batch_size == 4
    assert certificate.effective_k == 3
    assert certificate.draft_catchup_tokens == 3
    assert certificate.modeled_live_peak_bytes == exact.modeled_live_peak_bytes
    assert certificate.reservation_bytes == exact.reservation_bytes
    assert certificate.reservation_bytes <= configured.reservation_bytes
    assert not certificate.gpu_certified


def test_plan_certificate_is_independently_recomputed_from_b_c_k_and_route():
    configured = _configured_workspace()
    rows = (_row(0, catchup=1), _row(1, catchup=2))
    route = _route(2, batch_bucket=4)
    plan = _build(
        rows=rows,
        effective_k=2,
        max_tokens=13,
        route_key=route,
        configured_workspace=configured,
    )
    independent = certify_speculative_workspace(
        configured,
        live_batch_size=2,
        draft_catchup_tokens=3,
        effective_k=2,
        route_key=route,
    )
    assert plan.modeled_live_peak_bytes == independent.modeled_live_peak_bytes
    assert plan.reservation_bytes == independent.reservation_bytes
    assert plan.workspace_fingerprint == independent.workspace_fingerprint


def test_fingerprint_binds_live_batch_catchup_and_route():
    configured = _configured_workspace()
    base = certify_speculative_workspace(
        configured,
        live_batch_size=2,
        draft_catchup_tokens=1,
        effective_k=2,
        route_key=_route(2, batch_bucket=4),
    )
    changed_catchup = certify_speculative_workspace(
        configured,
        live_batch_size=2,
        draft_catchup_tokens=2,
        effective_k=2,
        route_key=_route(2, batch_bucket=4),
    )
    changed_batch = certify_speculative_workspace(
        configured,
        live_batch_size=3,
        draft_catchup_tokens=1,
        effective_k=2,
        route_key=_route(2, batch_bucket=4),
    )
    changed_mode = certify_speculative_workspace(
        configured,
        live_batch_size=2,
        draft_catchup_tokens=1,
        effective_k=2,
        route_key=_route(
            2,
            batch_bucket=4,
            execution_mode=DraftExecutionMode.EAGER_DYNAMIC,
        ),
    )
    fingerprints = {
        item.workspace_fingerprint
        for item in (base, changed_catchup, changed_batch, changed_mode)
    }
    assert len(fingerprints) == 4


def test_zero_k_sentinel_has_no_route_writes_catchup_or_speculative_bytes():
    rows = (
        _row(0, catchup=4, highest_draft=99, highest_target=100),
        _row(1, catchup=2, highest_draft=88, highest_target=89),
    )
    plan = _build(
        rows=rows,
        effective_k=0,
        max_tokens=1,
        route_key=None,
        bypass_reason="aggregate_token_budget",
    )
    assert plan.effective_k == 0
    assert plan.route_key is None
    assert plan.bypass_reason == "aggregate_token_budget"
    assert plan.fallback_reason == plan.bypass_reason
    assert plan.draft_catchup_tokens == 0
    assert plan.draft_query_tokens == 0
    assert plan.target_query_tokens == 2
    assert plan.total_model_positions == 2
    assert plan.total_scheduled_tokens == 2
    assert plan.modeled_live_peak_bytes == 0
    assert plan.reservation_bytes == 0
    assert not plan.gpu_certified
    assert all(row.highest_draft_write_position is None for row in plan.rows)
    assert all(row.highest_target_write_position is None for row in plan.rows)


def test_plan_and_certificate_are_immutable_and_pickle_safe():
    plan = _build(
        rows=(_row(0, catchup=1),),
        effective_k=2,
        max_tokens=8,
        route_key=_route(2, batch_bucket=1),
    )
    assert pickle.loads(pickle.dumps(plan, protocol=5)) == plan
    with pytest.raises(FrozenInstanceError):
        plan.effective_k = 1
    with pytest.raises(ValueError, match="cannot be GPU-certified"):
        replace(plan, gpu_certified=True)
    certificate = certify_speculative_workspace(
        _configured_workspace(),
        live_batch_size=1,
        draft_catchup_tokens=1,
        effective_k=2,
        route_key=_route(2, batch_bucket=1),
    )
    assert pickle.loads(pickle.dumps(certificate, protocol=5)) == certificate
    with pytest.raises(ValueError, match="cannot be GPU-certified"):
        replace(certificate, gpu_certified=True)


def test_builder_rejects_first_model_position_above_aggregate_budget():
    rows = (_row(0, catchup=1), _row(1, catchup=2))
    # C + B*K + B*(K+1) = 3 + 4 + 6 = 13.
    with pytest.raises(
        SpeculativeMemoryPlanningError,
        match="aggregate speculative work",
    ):
        _build(
            rows=rows,
            effective_k=2,
            max_tokens=12,
            route_key=_route(2, batch_bucket=2),
        )


def test_builder_enforces_independent_verifier_input_bound_first():
    with pytest.raises(
        SpeculativeMemoryPlanningError,
        match="target verifier rows",
    ):
        _build(
            rows=(_row(0), _row(1)),
            effective_k=2,
            max_tokens=5,
            route_key=_route(
                2,
                batch_bucket=2,
                catchup_family=DraftCatchupFamily.NONE,
            ),
        )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ((_row(0, remaining=2),), "completion headroom"),
        ((_row(0, headroom=1),), "position headroom"),
    ],
)
def test_builder_enforces_k_plus_one_output_and_target_position_headroom(
    rows, message
):
    with pytest.raises(SpeculativeMemoryPlanningError, match=message):
        _build(
            rows=rows,
            effective_k=2,
            max_tokens=8,
            route_key=_route(
                2,
                batch_bucket=1,
                catchup_family=DraftCatchupFamily.NONE,
            ),
        )


@pytest.mark.parametrize(
    ("route", "message"),
    [
        (_route(1, batch_bucket=2), "route K"),
        (_route(2, batch_bucket=1), "batch bucket"),
        (
            _route(
                2,
                batch_bucket=2,
                catchup_family=DraftCatchupFamily.NONE,
            ),
            "catch-up family",
        ),
    ],
)
def test_builder_rejects_route_geometry_mismatches(route, message):
    with pytest.raises(ValueError, match=message):
        _build(
            rows=(_row(0, catchup=1), _row(1)),
            effective_k=2,
            max_tokens=11,
            route_key=route,
        )


def test_builder_rejects_uncertified_route_and_scalar_caps():
    small = _configured_workspace(batch_size=1, configured_k=2)
    with pytest.raises(SpeculativeMemoryPlanningError, match="route geometry"):
        _build(
            rows=(_row(0),),
            effective_k=2,
            max_tokens=8,
            route_key=_route(
                2,
                batch_bucket=2,
                catchup_family=DraftCatchupFamily.NONE,
            ),
            configured_workspace=small,
            configured_k=2,
            workspace_route_cap=2,
        )
    with pytest.raises(SpeculativeMemoryPlanningError, match="effective_k"):
        _build(
            rows=(_row(0),),
            effective_k=2,
            max_tokens=8,
            route_key=_route(
                2,
                batch_bucket=1,
                catchup_family=DraftCatchupFamily.NONE,
            ),
            configured_k=1,
        )


def test_fallback_requires_reason_and_forbids_route_key():
    with pytest.raises(ValueError, match="bypass reason"):
        _build(
            rows=(_row(0),),
            effective_k=0,
            max_tokens=1,
            route_key=None,
        )
    with pytest.raises(ValueError, match="cannot carry a route key"):
        _build(
            rows=(_row(0),),
            effective_k=0,
            max_tokens=1,
            route_key=_route(
                1,
                batch_bucket=1,
                catchup_family=DraftCatchupFamily.NONE,
            ),
            bypass_reason="disabled",
        )


def test_rows_fail_closed_on_nonprimitive_or_inconsistent_cache_facts():
    with pytest.raises(TypeError, match="block_table must be a tuple"):
        replace(_row(0), block_table=[0])
    with pytest.raises(ValueError, match="draft cache coverage"):
        replace(_row(0), draft_cached_tokens=10)
    with pytest.raises(TypeError, match="seq_id must be an integer"):
        replace(_row(0), seq_id=True)


def test_plan_post_init_rejects_tampered_counts_and_fingerprint():
    plan = _build(
        rows=(_row(0),),
        effective_k=2,
        max_tokens=8,
        route_key=_route(
            2,
            batch_bucket=1,
            catchup_family=DraftCatchupFamily.NONE,
        ),
    )
    with pytest.raises(ValueError, match=r"B\*\(K\+1\)"):
        replace(plan, target_query_tokens=2)
    with pytest.raises(ValueError, match="total model-position"):
        replace(plan, total_model_positions=plan.total_model_positions - 1)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(plan, workspace_fingerprint="not-a-digest")
    with pytest.raises(ValueError, match="workspace route cap"):
        replace(plan, workspace_route_cap=1)


def test_spec_step_type_rejects_duplicate_rows():
    plan = _build(
        rows=(_row(0),),
        effective_k=2,
        max_tokens=8,
        route_key=_route(
            2,
            batch_bucket=1,
            catchup_family=DraftCatchupFamily.NONE,
        ),
    )
    with pytest.raises(ValueError, match="unique sequence IDs"):
        replace(plan, rows=(plan.rows[0], plan.rows[0]))


def test_empty_selected_batch_has_a_stable_zero_k_sentinel():
    plan = _build(
        rows=(),
        effective_k=0,
        max_tokens=0,
        route_key=None,
        bypass_reason="no_decode_rows",
    )
    assert isinstance(plan, SpecStepPlan)
    assert plan.rows == ()
    assert plan.target_query_tokens == 0
    assert plan.total_model_positions == 0
    assert len(plan.workspace_fingerprint) == 64
