from collections.abc import Mapping
from dataclasses import replace
from itertools import product

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class _FixedItemsMapping(Mapping):
    """Mapping test double whose items preserve malformed duplicate keys."""

    def __init__(self, pairs):
        self._pairs = tuple(pairs)

    def __getitem__(self, key):
        for candidate, value in self._pairs:
            if candidate is key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._pairs)

    def __len__(self):
        return len(self._pairs)

    def items(self):
        return self._pairs


def _make_allocated_sequence(block_manager: BlockManager) -> Sequence:
    seq = Sequence([11])
    block_manager.allocate(seq, num_cached_blocks=0)
    return seq


def _allocator_snapshot(block_manager: BlockManager):
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


def _sequence_snapshot(*seqs: Sequence):
    return tuple(
        (
            tuple(seq.block_table),
            seq.num_cached_tokens,
            seq.num_draft_cached_tokens,
        )
        for seq in seqs
    )


def _reserve_counts(
    block_manager: BlockManager,
    seqs: tuple[Sequence, ...],
    appended_counts: tuple[int, ...],
):
    rows = tuple(
        (
            seq,
            (len(seq.block_table) + appended_count)
            * block_manager.block_size
            - 1,
        )
        for seq, appended_count in zip(seqs, appended_counts, strict=True)
    )
    reservation = block_manager.reserve_temporary_append(rows)
    assert reservation is not None
    assert tuple(
        len(row.appended_block_ids) for row in reservation.rows
    ) == appended_counts
    return reservation


def test_finalize_exhaustively_keeps_each_row_prefix_and_restores_order():
    """All two-row retain vectors through three appended blocks are exact."""

    for first_appended, second_appended in product(range(4), repeat=2):
        appended_counts = (first_appended, second_appended)
        for first_keep, second_keep in product(
            range(first_appended + 1),
            range(second_appended + 1),
        ):
            block_manager = BlockManager(num_blocks=10, block_size=4)
            seqs = (
                _make_allocated_sequence(block_manager),
                _make_allocated_sequence(block_manager),
            )
            seqs[0].num_cached_tokens = 1
            seqs[0].num_draft_cached_tokens = 2
            seqs[1].num_cached_tokens = 3
            seqs[1].num_draft_cached_tokens = 4
            reservation = _reserve_counts(
                block_manager, seqs, appended_counts
            )
            keep_counts = (first_keep, second_keep)
            kept = {
                block_id
                for row, keep_count in zip(
                    reservation.rows, keep_counts, strict=True
                )
                for block_id in row.appended_block_ids[:keep_count]
            }

            assert block_manager.finalize_temporary_append(
                reservation,
                {
                    seq.seq_id: keep_count
                    for seq, keep_count in zip(
                        seqs, keep_counts, strict=True
                    )
                },
            )

            assert tuple(block_manager.free_block_ids) == tuple(
                block_id
                for block_id in reservation.free_block_ids_before
                if block_id not in kept
            )
            assert block_manager.used_block_ids == (
                set(reservation.used_block_ids_before) | kept
            )
            assert block_manager.hash_to_block_id == {
                block_hash: block_id
                for block_hash, block_id
                in reservation.hash_to_block_id_before
                if block_id not in kept
            }
            for row, keep_count in zip(
                reservation.rows, keep_counts, strict=True
            ):
                assert tuple(row.sequence.block_table) == (
                    row.block_table_before
                    + row.appended_block_ids[:keep_count]
                )
                assert (
                    row.sequence.num_cached_tokens
                    == row.num_cached_tokens_before
                )
                assert (
                    row.sequence.num_draft_cached_tokens
                    == row.num_draft_cached_tokens_before
                )
            for block_id in kept:
                block = block_manager.blocks[block_id]
                assert (block.ref_count, block.hash, block.token_ids) == (
                    1,
                    -1,
                    [],
                )
            assert not block_manager._active_temporary_reservations
            assert not block_manager._temporary_reservation_by_seq_id
            assert not block_manager.finalize_temporary_append(
                reservation,
                {seq.seq_id: 0 for seq in seqs},
            )
            assert not block_manager.rollback_temporary_append(reservation)


def test_finalize_restores_released_cached_free_metadata_but_resets_kept():
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seqs = (
        _make_allocated_sequence(block_manager),
        _make_allocated_sequence(block_manager),
    )
    candidate_ids = tuple(block_manager.free_block_ids)[:4]
    prior_states = {}
    for offset, block_id in enumerate(candidate_ids):
        block = block_manager.blocks[block_id]
        block_hash = 10_000 + offset
        block.hash = block_hash
        block.token_ids = [offset, offset + 1]
        block_manager.hash_to_block_id[block_hash] = block_id
        prior_states[block_id] = (
            block.ref_count,
            block.hash,
            tuple(block.token_ids),
        )

    reservation = _reserve_counts(block_manager, seqs, (2, 2))
    kept = {reservation.rows[0].appended_block_ids[0]}
    released = set(reservation.allocation_order) - kept

    assert block_manager.finalize_temporary_append(
        reservation,
        {seqs[0].seq_id: 1, seqs[1].seq_id: 0},
    )

    for block_id in released:
        block = block_manager.blocks[block_id]
        assert (
            block.ref_count,
            block.hash,
            tuple(block.token_ids),
        ) == prior_states[block_id]
        assert block_manager.hash_to_block_id[block.hash] == block_id
    kept_id = next(iter(kept))
    kept_block = block_manager.blocks[kept_id]
    assert (kept_block.ref_count, kept_block.hash, kept_block.token_ids) == (
        1,
        -1,
        [],
    )
    assert all(
        block_id != kept_id
        for block_id in block_manager.hash_to_block_id.values()
    )
    assert tuple(block_manager.free_block_ids) == tuple(
        block_id
        for block_id in reservation.free_block_ids_before
        if block_id != kept_id
    )


def test_finalize_never_mutates_shared_hashed_prefix_or_logical_coverage():
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seqs = (
        _make_allocated_sequence(block_manager),
        _make_allocated_sequence(block_manager),
    )
    prefix = block_manager.blocks[seqs[0].block_table[0]]
    prefix.ref_count = 2
    prefix.hash = 777
    prefix.token_ids = [1, 2, 3, 4]
    block_manager.hash_to_block_id[prefix.hash] = prefix.block_id
    seqs[0].num_cached_tokens = 4
    seqs[0].num_draft_cached_tokens = 3
    prefix_state = (
        prefix.ref_count,
        prefix.hash,
        tuple(prefix.token_ids),
    )

    reservation = _reserve_counts(block_manager, seqs, (3, 2))
    assert block_manager.finalize_temporary_append(
        reservation,
        {seqs[0].seq_id: 2, seqs[1].seq_id: 1},
    )

    assert (
        prefix.ref_count,
        prefix.hash,
        tuple(prefix.token_ids),
    ) == prefix_state
    assert block_manager.hash_to_block_id[777] == prefix.block_id
    assert seqs[0].num_cached_tokens == 4
    assert seqs[0].num_draft_cached_tokens == 3


@pytest.mark.parametrize(
    ("request_factory", "error_type", "message"),
    [
        (lambda ids: [], TypeError, "must be a mapping"),
        (lambda ids: {}, ValueError, "every reservation row"),
        (
            lambda ids: {ids[0]: 0},
            ValueError,
            "every reservation row",
        ),
        (
            lambda ids: {ids[0]: 0, ids[1]: 0, max(ids) + 1: 0},
            ValueError,
            "every reservation row",
        ),
        (
            lambda ids: _FixedItemsMapping(((True, 0), (ids[1], 0))),
            TypeError,
            "sequence IDs must be integers",
        ),
        (
            lambda ids: _FixedItemsMapping(((ids[0], 0), (ids[0], 0))),
            ValueError,
            "sequence IDs must be unique",
        ),
        (
            lambda ids: {str(ids[0]): 0, ids[1]: 0},
            TypeError,
            "sequence IDs must be integers",
        ),
        (
            lambda ids: {ids[0]: True, ids[1]: 0},
            TypeError,
            "counts must be integers",
        ),
        (
            lambda ids: {ids[0]: -1, ids[1]: 0},
            ValueError,
            "outside its appended suffix",
        ),
        (
            lambda ids: {ids[0]: 3, ids[1]: 0},
            ValueError,
            "outside its appended suffix",
        ),
    ],
)
def test_invalid_finalize_request_is_an_exact_noop_with_live_retryable_lease(
    request_factory,
    error_type,
    message,
):
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seqs = (
        _make_allocated_sequence(block_manager),
        _make_allocated_sequence(block_manager),
    )
    reservation = _reserve_counts(block_manager, seqs, (2, 2))
    before_allocator = _allocator_snapshot(block_manager)
    before_sequences = _sequence_snapshot(*seqs)
    seq_ids = tuple(seq.seq_id for seq in seqs)

    with pytest.raises(error_type, match=message):
        block_manager.finalize_temporary_append(
            reservation,
            request_factory(seq_ids),
        )

    assert _allocator_snapshot(block_manager) == before_allocator
    assert _sequence_snapshot(*seqs) == before_sequences
    assert block_manager.rollback_temporary_append(reservation)


def test_finalize_rejects_forged_identity_without_mutation():
    block_manager = BlockManager(num_blocks=6, block_size=4)
    seq = _make_allocated_sequence(block_manager)
    reservation = _reserve_counts(block_manager, (seq,), (2,))
    forged = replace(reservation)
    before_allocator = _allocator_snapshot(block_manager)
    before_sequence = _sequence_snapshot(seq)

    with pytest.raises(RuntimeError, match="identity mismatch"):
        block_manager.finalize_temporary_append(
            forged,
            {seq.seq_id: 1},
        )

    assert _allocator_snapshot(block_manager) == before_allocator
    assert _sequence_snapshot(seq) == before_sequence
    assert block_manager.rollback_temporary_append(reservation)


def test_finalize_live_drift_validation_is_a_noop():
    block_manager = BlockManager(num_blocks=6, block_size=4)
    seq = _make_allocated_sequence(block_manager)
    reservation = _reserve_counts(block_manager, (seq,), (2,))
    temporary = block_manager.blocks[reservation.allocation_order[0]]
    temporary.hash = 123
    drifted_allocator = _allocator_snapshot(block_manager)
    drifted_sequence = _sequence_snapshot(seq)

    with pytest.raises(RuntimeError, match="temporary block metadata drifted"):
        block_manager.finalize_temporary_append(
            reservation,
            {seq.seq_id: 1},
        )

    assert _allocator_snapshot(block_manager) == drifted_allocator
    assert _sequence_snapshot(seq) == drifted_sequence
    temporary.hash = -1
    assert block_manager.rollback_temporary_append(reservation)


def test_finalize_restores_live_lease_after_partial_unpublish_failure(
    monkeypatch,
):
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seqs = (
        _make_allocated_sequence(block_manager),
        _make_allocated_sequence(block_manager),
    )
    reservation = _reserve_counts(block_manager, seqs, (2, 2))
    live_allocator = _allocator_snapshot(block_manager)
    live_sequences = _sequence_snapshot(*seqs)
    original_unpublish = block_manager._unpublish_temporary_reservation

    def fail_after_partial_unpublish(active_reservation):
        original_unpublish(active_reservation)
        block_manager.blocks[reservation.allocation_order[-1]].ref_count = 99
        raise RuntimeError("injected finalizer failure")

    monkeypatch.setattr(
        block_manager,
        "_unpublish_temporary_reservation",
        fail_after_partial_unpublish,
    )
    with pytest.raises(RuntimeError, match="injected finalizer failure"):
        block_manager.finalize_temporary_append(
            reservation,
            {seqs[0].seq_id: 1, seqs[1].seq_id: 0},
        )

    assert _allocator_snapshot(block_manager) == live_allocator
    assert _sequence_snapshot(*seqs) == live_sequences

    monkeypatch.setattr(
        block_manager,
        "_unpublish_temporary_reservation",
        original_unpublish,
    )
    assert block_manager.finalize_temporary_append(
        reservation,
        {seqs[0].seq_id: 1, seqs[1].seq_id: 0},
    )


def test_finalize_restores_live_lease_after_row_mutation_failure():
    class FailOnceList(list):
        failed = False

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if isinstance(key, slice) and not self.failed:
                self.failed = True
                raise RuntimeError("injected row mutation failure")

    block_manager = BlockManager(num_blocks=6, block_size=4)
    seq = _make_allocated_sequence(block_manager)
    reservation = _reserve_counts(block_manager, (seq,), (3,))
    seq.block_table = FailOnceList(seq.block_table)
    live_allocator = _allocator_snapshot(block_manager)
    live_sequence = _sequence_snapshot(seq)

    with pytest.raises(RuntimeError, match="injected row mutation failure"):
        block_manager.finalize_temporary_append(
            reservation,
            {seq.seq_id: 1},
        )

    assert _allocator_snapshot(block_manager) == live_allocator
    assert _sequence_snapshot(seq) == live_sequence
    assert block_manager.finalize_temporary_append(
        reservation,
        {seq.seq_id: 1},
    )
    unrelated = _make_allocated_sequence(block_manager)
    assert unrelated.block_table
