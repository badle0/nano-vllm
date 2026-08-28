from collections import deque

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


def make_allocated_sequence(
    block_manager: BlockManager,
    *,
    num_tokens: int = 1,
) -> Sequence:
    seq = Sequence(list(range(num_tokens)))
    block_manager.allocate(seq, num_cached_blocks=0)
    return seq


def allocator_snapshot(block_manager: BlockManager):
    return (
        tuple(block_manager.free_block_ids),
        frozenset(block_manager.used_block_ids),
        tuple(sorted(block_manager.hash_to_block_id.items())),
        tuple(
            (
                block.block_id,
                block.ref_count,
                block.hash,
                tuple(block.token_ids),
            )
            for block in block_manager.blocks
        ),
        tuple(sorted(block_manager._active_temporary_reservations)),
        tuple(sorted(block_manager._temporary_reservation_by_seq_id.items())),
        block_manager._temporary_reservation_in_progress,
        block_manager._temporary_internal_allocation,
    )


def sequence_snapshot(*seqs: Sequence):
    return tuple(
        (
            tuple(seq.block_table),
            seq.num_cached_tokens,
            seq.num_draft_cached_tokens,
        )
        for seq in seqs
    )


def test_temporary_reservation_restores_late_mutating_allocation_failure(
    monkeypatch,
):
    block_manager = BlockManager(num_blocks=8, block_size=256)
    seq = make_allocated_sequence(block_manager)
    first_candidate = block_manager.free_block_ids[0]
    block_manager.blocks[first_candidate].hash = 12345
    block_manager.blocks[first_candidate].token_ids = [7, 8, 9]
    block_manager.hash_to_block_id[12345] = first_candidate
    before_allocator = allocator_snapshot(block_manager)
    before_sequence = sequence_snapshot(seq)

    original_allocate = block_manager._allocate_block
    calls = 0

    def fail_after_second_mutation():
        nonlocal calls
        block_id = original_allocate()
        calls += 1
        if calls == 2:
            raise RuntimeError("injected late allocation failure")
        return block_id

    monkeypatch.setattr(
        block_manager,
        "_allocate_block",
        fail_after_second_mutation,
    )
    with pytest.raises(RuntimeError, match="injected late allocation failure"):
        block_manager.reserve_temporary_append(((seq, 512),))

    assert allocator_snapshot(block_manager) == before_allocator
    assert sequence_snapshot(seq) == before_sequence

    monkeypatch.setattr(block_manager, "_allocate_block", original_allocate)
    reservation = block_manager.reserve_temporary_append(((seq, 512),))
    assert reservation is not None
    assert block_manager.rollback_temporary_append(reservation)
    assert allocator_snapshot(block_manager) == before_allocator
    assert sequence_snapshot(seq) == before_sequence


def test_temporary_reservation_restores_publication_validation_failure(
    monkeypatch,
):
    block_manager = BlockManager(num_blocks=6, block_size=256)
    seq = make_allocated_sequence(block_manager)
    before_allocator = allocator_snapshot(block_manager)
    before_sequence = sequence_snapshot(seq)
    original_validate = block_manager._validate_live_temporary_reservation

    def fail_after_validation(reservation):
        original_validate(reservation)
        block_manager.blocks[seq.block_table[0]].ref_count += 1
        block_manager._temporary_reservation_by_seq_id[seq.seq_id] = -999
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(
        block_manager,
        "_validate_live_temporary_reservation",
        fail_after_validation,
    )
    with pytest.raises(RuntimeError, match="injected publication failure"):
        block_manager.reserve_temporary_append(((seq, 256),))

    assert allocator_snapshot(block_manager) == before_allocator
    assert sequence_snapshot(seq) == before_sequence

    monkeypatch.setattr(
        block_manager,
        "_validate_live_temporary_reservation",
        original_validate,
    )
    reservation = block_manager.reserve_temporary_append(((seq, 256),))
    assert reservation is not None
    assert block_manager.rollback_temporary_append(reservation)


def test_one_global_reservation_fences_unrelated_allocator_mutations():
    block_manager = BlockManager(num_blocks=8, block_size=256)
    reserved = make_allocated_sequence(block_manager)
    unrelated = make_allocated_sequence(block_manager)
    reservation = block_manager.reserve_temporary_append(((reserved, 256),))
    assert reservation is not None
    live_allocator = allocator_snapshot(block_manager)
    live_sequences = sequence_snapshot(reserved, unrelated)

    with pytest.raises(RuntimeError, match="already active"):
        block_manager.reserve_temporary_append(((unrelated, 256),))
    with pytest.raises(RuntimeError, match="temporary reservation is active"):
        block_manager.allocate(Sequence([99]), num_cached_blocks=0)
    with pytest.raises(RuntimeError, match="temporary reservation is active"):
        block_manager.may_append(unrelated)
    with pytest.raises(RuntimeError, match="temporary reservation is active"):
        block_manager.hash_blocks(unrelated)
    with pytest.raises(RuntimeError, match="temporary reservation is active"):
        block_manager._allocate_block()
    with pytest.raises(RuntimeError, match="unrelated sequence"):
        block_manager.deallocate(unrelated)

    assert allocator_snapshot(block_manager) == live_allocator
    assert sequence_snapshot(reserved, unrelated) == live_sequences
    assert block_manager.rollback_temporary_append(reservation)


def test_reservation_construction_rejects_reentrant_reservation(monkeypatch):
    block_manager = BlockManager(num_blocks=6, block_size=256)
    first = make_allocated_sequence(block_manager)
    second = make_allocated_sequence(block_manager)
    original_allocate = block_manager._allocate_block

    def allocate_after_reentrant_check():
        with pytest.raises(RuntimeError, match="already active"):
            block_manager.reserve_temporary_append(((second, 256),))
        return original_allocate()

    monkeypatch.setattr(
        block_manager,
        "_allocate_block",
        allocate_after_reentrant_check,
    )
    reservation = block_manager.reserve_temporary_append(((first, 256),))

    assert reservation is not None
    assert block_manager.rollback_temporary_append(reservation)


def test_temporary_reservation_detects_owned_shared_prefix_metadata_drift():
    block_manager = BlockManager(num_blocks=6, block_size=256)
    owner = make_allocated_sequence(block_manager, num_tokens=256)
    shared = Sequence(list(range(256)))
    shared.block_table[:] = owner.block_table
    shared_block = block_manager.blocks[owner.block_table[0]]
    shared_block.ref_count = 2
    shared_block.hash = 4242
    shared_block.token_ids = list(range(256))
    block_manager.hash_to_block_id[4242] = shared_block.block_id
    owned_state = (
        shared_block.ref_count,
        shared_block.hash,
        tuple(shared_block.token_ids),
    )

    reservation = block_manager.reserve_temporary_append(((owner, 256),))
    assert reservation is not None

    shared_block.ref_count = 3
    with pytest.raises(RuntimeError, match="non-temporary block metadata drifted"):
        block_manager.rollback_temporary_append(reservation)
    shared_block.ref_count = owned_state[0]

    shared_block.token_ids.append(-1)
    with pytest.raises(RuntimeError, match="non-temporary block metadata drifted"):
        block_manager.rollback_temporary_append(reservation)
    shared_block.token_ids[:] = owned_state[2]

    assert block_manager.rollback_temporary_append(reservation)
    assert (
        shared_block.ref_count,
        shared_block.hash,
        tuple(shared_block.token_ids),
    ) == owned_state
    assert block_manager.hash_to_block_id[4242] == shared_block.block_id


def test_group_uncommitted_append_rollback_restores_exact_allocator_order():
    block_manager = BlockManager(num_blocks=8, block_size=256)
    first = make_allocated_sequence(block_manager)
    second = make_allocated_sequence(block_manager)
    rows = ((first, len(first.block_table)), (second, len(second.block_table)))
    before_allocator = allocator_snapshot(block_manager)
    before_sequences = sequence_snapshot(first, second)

    block_manager.may_append(first)
    block_manager.may_append(second)
    assert block_manager.rollback_uncommitted_appends(rows)

    assert allocator_snapshot(block_manager) == before_allocator
    assert sequence_snapshot(first, second) == before_sequences
    assert not block_manager.rollback_uncommitted_appends(rows)


def test_group_uncommitted_append_prevalidation_is_atomic():
    block_manager = BlockManager(num_blocks=8, block_size=256)
    first = make_allocated_sequence(block_manager)
    second = make_allocated_sequence(block_manager)
    rows = ((first, len(first.block_table)), (second, len(second.block_table)))
    block_manager.may_append(first)
    block_manager.may_append(second)
    second_suffix = block_manager.blocks[second.block_table[-1]]
    second_suffix.hash = 7
    before_allocator = allocator_snapshot(block_manager)
    before_sequences = sequence_snapshot(first, second)

    with pytest.raises(RuntimeError, match="fresh, unshared block"):
        block_manager.rollback_uncommitted_appends(rows)

    assert allocator_snapshot(block_manager) == before_allocator
    assert sequence_snapshot(first, second) == before_sequences
    second_suffix.hash = -1
    assert block_manager.rollback_uncommitted_appends(rows)


def test_group_uncommitted_append_mutation_failure_is_atomic():
    class FailSecondAppendLeft(deque):
        def __init__(self, values):
            super().__init__(values)
            self.calls = 0

        def appendleft(self, value):
            super().appendleft(value)
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected free-list failure")

    block_manager = BlockManager(num_blocks=8, block_size=256)
    first = make_allocated_sequence(block_manager)
    second = make_allocated_sequence(block_manager)
    rows = ((first, len(first.block_table)), (second, len(second.block_table)))
    block_manager.may_append(first)
    block_manager.may_append(second)
    block_manager.free_block_ids = FailSecondAppendLeft(
        block_manager.free_block_ids
    )
    before_allocator = allocator_snapshot(block_manager)
    before_sequences = sequence_snapshot(first, second)

    with pytest.raises(RuntimeError, match="injected free-list failure"):
        block_manager.rollback_uncommitted_appends(rows)

    assert allocator_snapshot(block_manager) == before_allocator
    assert sequence_snapshot(first, second) == before_sequences
    assert block_manager.rollback_uncommitted_appends(rows)
