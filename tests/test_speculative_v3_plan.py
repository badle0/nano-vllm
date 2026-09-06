from collections import deque
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from nanovllm import SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(
    *,
    configured_k=4,
    max_model_len=4096,
    max_num_batched_tokens=1024,
    num_blocks=32,
):
    return Scheduler(
        SimpleNamespace(
            max_num_seqs=32,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            configured_k=configured_k,
            eos=-1,
            kvcache_block_size=256,
            num_kvcache_blocks=num_blocks,
        )
    )


def make_decode_sequence(
    scheduler,
    *,
    committed_tokens=8,
    completion_tokens=0,
    max_tokens=64,
):
    prompt_tokens = committed_tokens - completion_tokens
    assert prompt_tokens >= 1
    seq = Sequence(
        list(range(prompt_tokens)),
        SamplingParams(max_tokens=max_tokens, ignore_eos=True),
    )
    for token_id in range(completion_tokens):
        seq.append_token(10_000 + token_id)
    scheduler.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = len(seq) - 1
    seq.num_scheduled_tokens = 1
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    scheduler.running.append(seq)
    return seq


@pytest.mark.parametrize(
    (
        "configured_k",
        "remaining",
        "model_headroom",
        "budget_cap",
        "workspace_cap",
        "expected",
    ),
    [
        (8, 20, 20, 20, 20, 8),
        (3, 20, 20, 20, 20, 3),
        (8, 4, 20, 20, 20, 3),
        (8, 20, 2, 20, 20, 2),
        (8, 20, 20, 1, 20, 1),
        (8, 20, 20, 20, 5, 5),
        (8, 1, 20, 20, 20, 0),
        (8, 20, 0, 20, 20, 0),
        (8, 20, 20, 0, 20, 0),
        (8, 20, 20, 20, 0, 0),
    ],
)
def test_effective_k_scalar_formula_table(
    configured_k,
    remaining,
    model_headroom,
    budget_cap,
    workspace_cap,
    expected,
):
    completion_tokens = 2
    committed_tokens = 8
    batch_size = 2
    scheduler = make_scheduler(
        configured_k=configured_k,
        max_model_len=committed_tokens + model_headroom,
        max_num_batched_tokens=batch_size * (budget_cap + 1),
    )
    seqs = [
        make_decode_sequence(
            scheduler,
            committed_tokens=committed_tokens,
            completion_tokens=completion_tokens,
            max_tokens=completion_tokens + remaining,
        )
        for _ in range(batch_size)
    ]
    for seq in seqs:
        # Isolate the scalar K bound under test from the independently charged
        # cold draft-cache catch-up term.
        seq.num_draft_cached_tokens = len(seq) - 1

    plan = scheduler.plan_draft_discard(
        seqs,
        workspace_route_cap=workspace_cap,
    )

    assert plan.effective_k == expected
    assert plan.uses_draft is (expected > 0)
    assert plan.target_query_tokens == batch_size
    assert plan.draft_catchup_tokens == 0
    assert plan.total_scheduled_tokens == batch_size * (expected + 1)
    assert plan.draft_step_token_counts == (batch_size,) * expected
    assert plan.total_scheduled_tokens <= scheduler.max_num_batched_tokens
    if expected:
        assert all(
            row.highest_proposal_input_position
            == row.committed_tokens + expected - 2
            for row in plan.rows
        )
        assert scheduler.rollback_draft_discard(plan)
    else:
        assert plan.fallback_reason is not None
        assert all(
            row.highest_proposal_input_position is None
            for row in plan.rows
        )


def test_effective_k_uses_the_minimum_request_bound_across_rows():
    scheduler = make_scheduler(configured_k=8, max_model_len=100)
    long = make_decode_sequence(
        scheduler,
        completion_tokens=2,
        max_tokens=20,
    )
    tail = make_decode_sequence(
        scheduler,
        completion_tokens=3,
        max_tokens=6,
    )

    plan = scheduler.plan_draft_discard(
        [long, tail], workspace_route_cap=8
    )

    assert plan.effective_k == 2
    assert [row.remaining_completion_tokens for row in plan.rows] == [18, 3]
    scheduler.rollback_draft_discard(plan)


def test_effective_k_property_sweep_matches_scalar_minimum():
    for configured_k in (0, 1, 4):
        for remaining in (1, 2, 8):
            for model_headroom in (0, 1, 8):
                for token_budget_cap in (0, 1, 8):
                    for workspace_cap in (0, 1, 8):
                        scheduler = make_scheduler(
                            configured_k=configured_k,
                            max_model_len=1 + model_headroom,
                            max_num_batched_tokens=token_budget_cap + 1,
                            num_blocks=4,
                        )
                        seq = make_decode_sequence(
                            scheduler,
                            committed_tokens=1,
                            max_tokens=remaining,
                        )
                        expected = min(
                            configured_k,
                            remaining - 1,
                            model_headroom,
                            token_budget_cap,
                            workspace_cap,
                        )

                        plan = scheduler.plan_draft_discard(
                            [seq], workspace_route_cap=workspace_cap
                        )

                        assert plan.effective_k == expected
                        if plan.uses_draft:
                            scheduler.rollback_draft_discard(plan)


@pytest.mark.parametrize(
    ("committed_tokens", "effective_k", "expected_blocks"),
    [
        (255, 4, 2),
        (256, 4, 2),
        (257, 4, 2),
        (250, 520, 4),
    ],
)
def test_reservation_covers_proposal_boundaries_and_multiblock_k(
    committed_tokens,
    effective_k,
    expected_blocks,
):
    scheduler = make_scheduler(
        configured_k=effective_k,
        max_model_len=committed_tokens + effective_k + 2,
        max_num_batched_tokens=effective_k + 1,
        num_blocks=16,
    )
    seq = make_decode_sequence(
        scheduler,
        committed_tokens=committed_tokens,
        max_tokens=effective_k + 2,
    )
    seq.num_draft_cached_tokens = len(seq) - 1
    table_before = tuple(seq.block_table)

    plan = scheduler.plan_draft_discard(
        [seq], workspace_route_cap=effective_k
    )

    assert plan.effective_k == effective_k
    assert plan.rows[0].highest_proposal_input_position == (
        committed_tokens + effective_k - 2
    )
    assert len(seq.block_table) == expected_blocks
    assert plan.rows[0].block_table == tuple(seq.block_table)
    assert scheduler.rollback_draft_discard(plan)
    assert tuple(seq.block_table) == table_before


def allocator_snapshot(block_manager):
    return {
        "free": tuple(block_manager.free_block_ids),
        "used": frozenset(block_manager.used_block_ids),
        "hashes": dict(block_manager.hash_to_block_id),
        "blocks": tuple(
            (
                block.ref_count,
                block.hash,
                tuple(block.token_ids),
            )
            for block in block_manager.blocks
        ),
    }


def test_insufficient_capacity_returns_baseline_before_mutation():
    scheduler = make_scheduler(
        configured_k=2,
        max_model_len=300,
        max_num_batched_tokens=3,
        num_blocks=1,
    )
    seq = make_decode_sequence(scheduler, committed_tokens=256)
    seq.num_draft_cached_tokens = len(seq) - 1
    table_before = tuple(seq.block_table)
    before = allocator_snapshot(scheduler.block_manager)

    plan = scheduler.plan_draft_discard([seq], workspace_route_cap=2)

    assert plan.effective_k == 0
    assert plan.fallback_reason == "insufficient_kv_blocks"
    assert not plan.uses_draft
    assert tuple(seq.block_table) == table_before
    assert allocator_snapshot(scheduler.block_manager) == before


def test_exact_reverse_rollback_restores_prefix_metadata_and_is_idempotent():
    scheduler = make_scheduler(
        configured_k=300,
        max_model_len=700,
        max_num_batched_tokens=301,
        num_blocks=8,
    )
    first = make_decode_sequence(scheduler, committed_tokens=256)
    second = make_decode_sequence(scheduler, committed_tokens=256)
    first.num_draft_cached_tokens = len(first) - 1
    second.num_draft_cached_tokens = len(second) - 1
    block_manager = scheduler.block_manager
    first_free = block_manager.free_block_ids[0]
    cached = block_manager.blocks[first_free]
    cached.hash = 12345
    cached.token_ids = [7] * 256
    block_manager.hash_to_block_id[cached.hash] = first_free
    tables_before = (tuple(first.block_table), tuple(second.block_table))
    before = allocator_snapshot(block_manager)

    plan = scheduler.plan_draft_discard(
        [first, second], workspace_route_cap=300
    )

    assert plan.uses_draft
    assert len(first.block_table) > len(tables_before[0])
    assert len(second.block_table) > len(tables_before[1])
    assert 12345 not in block_manager.hash_to_block_id
    assert scheduler.rollback_draft_discard(plan)
    assert not scheduler.rollback_draft_discard(plan)
    assert (tuple(first.block_table), tuple(second.block_table)) == tables_before
    assert allocator_snapshot(block_manager) == before


def test_cancel_unrelated_row_rolls_back_global_draft_lease_before_queue_edit():
    scheduler = make_scheduler(
        configured_k=4,
        max_model_len=300,
        max_num_batched_tokens=16,
    )
    reserved = make_decode_sequence(scheduler, committed_tokens=256)
    unrelated = make_decode_sequence(scheduler, committed_tokens=8)
    reserved.num_draft_cached_tokens = len(reserved) - 1
    unrelated.num_draft_cached_tokens = len(unrelated) - 1
    unrelated_table = tuple(unrelated.block_table)

    plan = scheduler.plan_draft_discard(
        [reserved], workspace_route_cap=4
    )
    assert plan.uses_draft
    assert scheduler._active_draft_discard is not None

    assert scheduler.cancel([unrelated.seq_id]) == [unrelated.seq_id]

    assert scheduler._active_draft_discard is None
    assert list(scheduler.running) == [reserved]
    assert reserved.status is SequenceStatus.RUNNING
    assert unrelated.status is SequenceStatus.CANCELLED
    assert not unrelated.block_table
    assert all(
        block_id not in scheduler.block_manager.used_block_ids
        for block_id in unrelated_table
    )
    assert not scheduler.rollback_draft_discard(plan)


def test_rollback_validates_live_snapshot_before_mutating():
    scheduler = make_scheduler(
        configured_k=4,
        max_model_len=300,
        max_num_batched_tokens=5,
    )
    seq = make_decode_sequence(scheduler, committed_tokens=256)
    seq.num_draft_cached_tokens = len(seq) - 1
    plan = scheduler.plan_draft_discard([seq], workspace_route_cap=4)
    appended = seq.block_table[-1]
    block = scheduler.block_manager.blocks[appended]
    block.ref_count = 2

    with pytest.raises(RuntimeError, match="metadata drifted"):
        scheduler.rollback_draft_discard(plan)

    block.ref_count = 1
    assert scheduler.rollback_draft_discard(plan)


def test_plan_and_rows_are_immutable_and_reservation_is_scheduler_private():
    scheduler = make_scheduler()
    seq = make_decode_sequence(scheduler, committed_tokens=256)
    plan = scheduler.plan_draft_discard([seq], workspace_route_cap=4)

    with pytest.raises(FrozenInstanceError):
        plan.effective_k = 0
    with pytest.raises(FrozenInstanceError):
        plan.rows[0].block_table = ()
    assert not hasattr(plan, "reservation")
    scheduler.rollback_draft_discard(plan)


def test_target_prefix_allocation_never_claims_draft_cache_coverage():
    block_manager = BlockManager(num_blocks=4, block_size=256)
    seed = Sequence(list(range(512)))
    block_manager.allocate(seed, num_cached_blocks=0)
    seed.num_scheduled_tokens = len(seed)
    block_manager.hash_blocks(seed)
    block_manager.deallocate(seed)

    seq = Sequence(list(range(512)))
    seq.num_draft_cached_tokens = 256
    num_cached_blocks = block_manager.can_allocate(seq)
    assert num_cached_blocks == 1

    block_manager.allocate(seq, num_cached_blocks=num_cached_blocks)

    assert seq.num_cached_tokens == 256
    assert seq.num_draft_cached_tokens == 0
    block_manager.deallocate(seq)
    assert seq.num_cached_tokens == 0
    assert seq.num_draft_cached_tokens == 0


def test_preempt_rolls_back_group_reservation_and_clears_draft_coverage():
    scheduler = make_scheduler(configured_k=4, max_model_len=300)
    seq = make_decode_sequence(scheduler, committed_tokens=256)
    seq.num_draft_cached_tokens = 200
    baseline_free = len(scheduler.block_manager.free_block_ids)
    plan = scheduler.plan_draft_discard([seq], workspace_route_cap=4)
    assert plan.uses_draft

    scheduler.preempt(seq)

    assert seq.status is SequenceStatus.WAITING
    assert seq.num_cached_tokens == 0
    assert seq.num_draft_cached_tokens == 0
    assert not seq.block_table
    assert len(scheduler.block_manager.free_block_ids) == baseline_free + 1
    assert not scheduler.rollback_draft_discard(plan)


def test_cancel_without_blocks_still_clears_draft_coverage():
    scheduler = make_scheduler()
    seq = Sequence([1])
    seq.num_draft_cached_tokens = 1
    scheduler.add(seq)

    assert scheduler.cancel([seq.seq_id]) == [seq.seq_id]
    assert seq.status is SequenceStatus.CANCELLED
    assert seq.num_draft_cached_tokens == 0


def test_finish_rolls_back_before_hash_and_clears_draft_coverage():
    scheduler = make_scheduler(configured_k=4, max_model_len=300)
    seq = make_decode_sequence(
        scheduler,
        committed_tokens=256,
        max_tokens=8,
    )
    scheduler.eos = 1234
    seq.ignore_eos = False
    seq.num_draft_cached_tokens = 200
    plan = scheduler.plan_draft_discard([seq], workspace_route_cap=4)
    assert plan.uses_draft
    assert scheduler.handoff_draft_discard(plan)
    scheduler.stage_draft_coverage(
        plan,
        [seq],
        {seq.seq_id: plan.rows[0].committed_tokens},
    )

    events = scheduler.postprocess([seq], [1234])

    assert events[0].finished
    assert seq.status is SequenceStatus.FINISHED
    assert seq.num_cached_tokens == 0
    assert seq.num_draft_cached_tokens == 0
    assert not seq.block_table
    assert not scheduler.rollback_draft_discard(plan)


def test_scheduler_stages_then_postprocess_commits_exact_draft_coverage():
    scheduler = make_scheduler(configured_k=4, max_model_len=300)
    seq = make_decode_sequence(
        scheduler,
        committed_tokens=256,
        max_tokens=8,
    )
    plan = scheduler.plan_draft_discard([seq], workspace_route_cap=4)

    with pytest.raises(RuntimeError, match="still active"):
        scheduler.stage_draft_coverage(
            plan,
            [seq],
            {seq.seq_id: 256},
        )

    scheduler.handoff_draft_discard(plan)
    with pytest.raises(ValueError, match="planned committed prefix"):
        scheduler.stage_draft_coverage(
            plan,
            [seq],
            {seq.seq_id: 255},
        )
    with pytest.raises(RuntimeError, match="before committing"):
        scheduler.schedule()

    scheduler.stage_draft_coverage(
        plan,
        [seq],
        {seq.seq_id: 256},
    )
    scheduler.postprocess([seq], [1234])
    assert len(seq) == 257
    assert seq.num_cached_tokens == 256
    assert seq.num_draft_cached_tokens == 256


def test_draft_coverage_staging_rejects_row_reordering():
    scheduler = make_scheduler(configured_k=2, max_model_len=300)
    first = make_decode_sequence(scheduler, committed_tokens=8)
    second = make_decode_sequence(scheduler, committed_tokens=8)
    plan = scheduler.plan_draft_discard(
        [first, second], workspace_route_cap=2
    )
    scheduler.handoff_draft_discard(plan)
    coverage = {
        row.seq_id: row.committed_tokens
        for row in plan.rows
    }

    with pytest.raises(ValueError, match="IDs/order"):
        scheduler.stage_draft_coverage(
            plan,
            [second, first],
            coverage,
        )

    scheduler.stage_draft_coverage(
        plan,
        [first, second],
        coverage,
    )
    scheduler.postprocess([first, second], [100, 101])


def test_handoff_restores_tables_before_target_and_abort_allows_reschedule():
    scheduler = make_scheduler(configured_k=4, max_model_len=300)
    seq = make_decode_sequence(scheduler, committed_tokens=256)
    table_before = tuple(seq.block_table)
    plan = scheduler.plan_draft_discard([seq], workspace_route_cap=4)
    assert tuple(seq.block_table) != table_before

    assert scheduler.handoff_draft_discard(plan)
    assert tuple(seq.block_table) == table_before
    with pytest.raises(RuntimeError, match="before committing"):
        scheduler.schedule()
    assert scheduler.abort_draft_coverage(plan)
    assert not scheduler.abort_draft_coverage(plan)

    seqs, is_prefill = scheduler.schedule()
    assert seqs == [seq]
    assert not is_prefill


def test_postprocess_rejects_sequence_order_different_from_handoff():
    scheduler = make_scheduler(configured_k=2, max_model_len=300)
    first = make_decode_sequence(scheduler, committed_tokens=8)
    second = make_decode_sequence(scheduler, committed_tokens=8)
    plan = scheduler.plan_draft_discard(
        [first, second], workspace_route_cap=2
    )
    scheduler.handoff_draft_discard(plan)
    scheduler.stage_draft_coverage(
        plan,
        [first, second],
        {row.seq_id: row.committed_tokens for row in plan.rows},
    )

    with pytest.raises(ValueError, match="IDs/order"):
        scheduler.postprocess([second, first], [100, 101])

    assert scheduler.abort_draft_coverage(plan)


def test_short_target_result_is_rejected_before_any_row_mutates():
    scheduler = make_scheduler(configured_k=2, max_model_len=300)
    first = make_decode_sequence(scheduler, committed_tokens=8)
    second = make_decode_sequence(scheduler, committed_tokens=8)
    plan = scheduler.plan_draft_discard(
        [first, second], workspace_route_cap=2
    )
    scheduler.handoff_draft_discard(plan)
    scheduler.stage_draft_coverage(
        plan,
        [first, second],
        {row.seq_id: row.committed_tokens for row in plan.rows},
    )
    before = tuple(
        (
            tuple(seq.token_ids),
            seq.num_cached_tokens,
            seq.num_draft_cached_tokens,
            seq.num_scheduled_tokens,
            tuple(seq.block_table),
        )
        for seq in (first, second)
    )

    with pytest.raises(ValueError, match="token count"):
        scheduler.postprocess([first, second], [123])

    assert tuple(
        (
            tuple(seq.token_ids),
            seq.num_cached_tokens,
            seq.num_draft_cached_tokens,
            seq.num_scheduled_tokens,
            tuple(seq.block_table),
        )
        for seq in (first, second)
    ) == before
    assert scheduler.abort_draft_coverage(plan)


def test_cold_catchup_is_charged_and_over_budget_batch_falls_back():
    scheduler = make_scheduler(
        configured_k=2,
        max_model_len=64,
        max_num_batched_tokens=8,
    )
    seqs = [
        make_decode_sequence(scheduler, committed_tokens=8)
        for _ in range(2)
    ]

    plan = scheduler.plan_draft_discard(seqs, workspace_route_cap=2)

    assert not plan.uses_draft
    assert plan.effective_k == 0
    assert plan.draft_catchup_tokens == 0
    assert plan.total_scheduled_tokens == 2
    assert plan.fallback_reason == "draft_catchup_token_budget"


def test_block_manager_deallocation_resets_both_coverages_without_blocks():
    block_manager = BlockManager(num_blocks=1, block_size=256)
    seq = Sequence([1])
    seq.num_cached_tokens = 1
    seq.num_draft_cached_tokens = 1

    block_manager.deallocate(seq)

    assert seq.num_cached_tokens == 0
    assert seq.num_draft_cached_tokens == 0
    assert seq.block_table == []
    assert block_manager.free_block_ids == deque([0])
