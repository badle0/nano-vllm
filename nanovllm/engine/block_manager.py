from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Iterable
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


@dataclass(frozen=True, slots=True)
class _TemporaryBlockState:
    block_id: int
    ref_count: int
    hash: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _TemporaryReservationRow:
    sequence: Sequence
    block_table_before: tuple[int, ...]
    appended_block_ids: tuple[int, ...]
    num_cached_tokens_before: int
    num_draft_cached_tokens_before: int


@dataclass(frozen=True, slots=True)
class _TemporaryReservationRowSpec:
    sequence: Sequence
    block_table_before: tuple[int, ...]
    new_blocks: int
    num_cached_tokens_before: int
    num_draft_cached_tokens_before: int


@dataclass(frozen=True, slots=True)
class TemporaryBlockReservation:
    """Immutable lease for cycle-local, not-yet-committed block IDs.

    The record snapshots allocator membership/order/hash state, metadata for
    every candidate block and participating preexisting block, plus the rows'
    block tables and cache coverages.  A manager permits only one such lease at
    a time and fences every other allocator mutation until the lease is rolled
    back.  Callers may retain the record only to return it to
    :meth:`BlockManager.rollback_temporary_append`; they must not interpret
    dirty KV slots as logically cached tokens.
    """

    reservation_id: int
    rows: tuple[_TemporaryReservationRow, ...]
    allocation_order: tuple[int, ...]
    free_block_ids_before: tuple[int, ...]
    used_block_ids_before: frozenset[int]
    hash_to_block_id_before: tuple[tuple[int, int], ...]
    block_states_before: tuple[_TemporaryBlockState, ...]


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self._temporary_reservation_ids = count()
        self._active_temporary_reservations: dict[
            int, TemporaryBlockReservation
        ] = {}
        self._temporary_reservation_by_seq_id: dict[int, int] = {}
        self._temporary_reservation_in_progress = False
        self._temporary_internal_allocation = False

    def _active_temporary_reservation(
        self,
    ) -> TemporaryBlockReservation | None:
        if len(self._active_temporary_reservations) > 1:
            raise RuntimeError("multiple temporary reservations are active")
        return next(iter(self._active_temporary_reservations.values()), None)

    def _assert_allocator_mutation_allowed(
        self,
        operation: str,
        *,
        allow_internal_allocation: bool = False,
    ) -> None:
        active = self._active_temporary_reservation()
        internal = (
            allow_internal_allocation
            and self._temporary_reservation_in_progress
            and self._temporary_internal_allocation
            and active is None
        )
        if (active is not None or self._temporary_reservation_in_progress) \
                and not internal:
            raise RuntimeError(
                f"cannot {operation} while a temporary reservation is active"
            )

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        self._assert_allocator_mutation_allowed(
            "allocate a block",
            allow_internal_allocation=True,
        )
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        self._assert_allocator_mutation_allowed("deallocate a block")
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        self._assert_allocator_mutation_allowed("allocate a sequence")
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        # Prefix-cache lookup applies only to the target model.  V3 catches the
        # draft model up explicitly before proposal execution.
        seq.num_draft_cached_tokens = 0

    def deallocate(self, seq: Sequence):
        active = self._active_temporary_reservation()
        if active is not None:
            reservation_id = self._temporary_reservation_by_seq_id.get(
                seq.seq_id
            )
            if reservation_id != active.reservation_id:
                raise RuntimeError(
                    "cannot deallocate an unrelated sequence while a "
                    "temporary reservation is active"
                )
            self.rollback_temporary_append(active)
        self._assert_allocator_mutation_allowed("deallocate a sequence")
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_draft_cached_tokens = 0
        seq.block_table.clear()

    def rollback_uncommitted_appends(
        self,
        rows: Iterable[tuple[Sequence, int]],
    ) -> bool:
        """Atomically remove fresh ``may_append`` suffixes after decode failure.

        Each integer is the sequence's block-table length captured immediately
        before scheduling.  A row whose length is unchanged is a no-op; every
        changed row must have exactly one newly allocated, unshared, unhashed,
        empty suffix.  The complete group is validated before any mutation and
        removed in reverse scheduling order to restore the free-list prefix.
        """

        requested_rows = tuple(rows)
        self._assert_allocator_mutation_allowed(
            "rollback uncommitted appends"
        )
        seen_seq_ids: set[int] = set()
        validated: list[tuple[Sequence, tuple[int, ...], Block | None]] = []
        appended_block_ids: set[int] = set()
        for seq, expected_block_table_len in requested_rows:
            if not isinstance(seq, Sequence):
                raise TypeError(
                    "uncommitted append rollback rows require Sequence objects"
                )
            if type(expected_block_table_len) is not int:
                raise TypeError(
                    "expected block-table length must be an integer"
                )
            if expected_block_table_len < 0:
                raise ValueError(
                    "expected block-table length must be non-negative"
                )
            if seq.seq_id in seen_seq_ids:
                raise ValueError(
                    "uncommitted append rollback rows must be unique"
                )
            seen_seq_ids.add(seq.seq_id)
            table_before_rollback = tuple(seq.block_table)
            actual_len = len(table_before_rollback)
            if actual_len == expected_block_table_len:
                validated.append((seq, table_before_rollback, None))
                continue
            if actual_len != expected_block_table_len + 1:
                raise RuntimeError(
                    "uncommitted append rollback requires exactly one "
                    "suffix block per changed row"
                )

            block_id = table_before_rollback[-1]
            if type(block_id) is not int \
                    or not 0 <= block_id < len(self.blocks):
                raise RuntimeError("uncommitted suffix has an invalid block ID")
            if block_id in table_before_rollback[:-1] \
                    or block_id in appended_block_ids:
                raise RuntimeError("uncommitted suffix aliases another block")
            block = self.blocks[block_id]
            if block_id not in self.used_block_ids \
                    or block_id in self.free_block_ids:
                raise RuntimeError(
                    "uncommitted suffix is not exclusively allocated"
                )
            if block.ref_count != 1 or block.hash != -1 or block.token_ids:
                raise RuntimeError(
                    "uncommitted suffix is not a fresh, unshared block"
                )
            appended_block_ids.add(block_id)
            validated.append((seq, table_before_rollback, block))

        for _, table, block in validated:
            prefix = table[:-1] if block is not None else table
            if appended_block_ids.intersection(prefix):
                raise RuntimeError(
                    "uncommitted suffix aliases a participating row prefix"
                )

        free_before = tuple(self.free_block_ids)
        used_before = frozenset(self.used_block_ids)
        ref_counts_before = {
            block.block_id: block.ref_count
            for _, _, block in validated
            if block is not None
        }
        try:
            for seq, _, block in reversed(validated):
                if block is None:
                    continue
                seq.block_table.pop()
                block.ref_count = 0
                self.used_block_ids.remove(block.block_id)
                self.free_block_ids.appendleft(block.block_id)
        except BaseException:
            for seq, table_before_rollback, _ in validated:
                seq.block_table[:] = table_before_rollback
            for block_id, ref_count in ref_counts_before.items():
                self.blocks[block_id].ref_count = ref_count
            self.free_block_ids = deque(free_before)
            self.used_block_ids = set(used_before)
            raise
        return bool(appended_block_ids)

    def rollback_uncommitted_append(
        self,
        seq: Sequence,
        expected_block_table_len: int,
    ) -> bool:
        """Compatibility wrapper for a one-row failed-decode rollback."""

        return self.rollback_uncommitted_appends(
            ((seq, expected_block_table_len),)
        )

    def _snapshot_block_states(
        self,
        block_ids: Iterable[int],
    ) -> tuple[_TemporaryBlockState, ...]:
        return tuple(
            _TemporaryBlockState(
                block_id=block_id,
                ref_count=block.ref_count,
                hash=block.hash,
                token_ids=tuple(block.token_ids),
            )
            for block_id in sorted(set(block_ids))
            for block in (self.blocks[block_id],)
        )

    def _restore_temporary_snapshot(
        self,
        *,
        rows: Iterable[_TemporaryReservationRow | _TemporaryReservationRowSpec],
        free_block_ids_before: tuple[int, ...],
        used_block_ids_before: frozenset[int],
        hash_to_block_id_before: tuple[tuple[int, int], ...],
        block_states_before: tuple[_TemporaryBlockState, ...],
    ) -> None:
        for row in rows:
            row.sequence.block_table[:] = row.block_table_before
            row.sequence.num_cached_tokens = row.num_cached_tokens_before
            row.sequence.num_draft_cached_tokens = (
                row.num_draft_cached_tokens_before
            )
        for state in block_states_before:
            block = self.blocks[state.block_id]
            block.ref_count = state.ref_count
            block.hash = state.hash
            block.token_ids = list(state.token_ids)
        self.free_block_ids = deque(free_block_ids_before)
        self.used_block_ids = set(used_block_ids_before)
        self.hash_to_block_id = dict(hash_to_block_id_before)

    def _unpublish_temporary_reservation(
        self,
        reservation: TemporaryBlockReservation,
    ) -> None:
        self._active_temporary_reservations.pop(
            reservation.reservation_id, None
        )
        for row in reservation.rows:
            if self._temporary_reservation_by_seq_id.get(
                row.sequence.seq_id
            ) == reservation.reservation_id:
                self._temporary_reservation_by_seq_id.pop(
                    row.sequence.seq_id, None
                )

    def reserve_temporary_append(
        self,
        rows: Iterable[tuple[Sequence, int]],
    ) -> TemporaryBlockReservation | None:
        """Reserve blocks through each zero-based highest written position.

        Capacity is checked for the complete group before allocator state is
        touched.  The one global lease snapshots allocator membership/order/hash
        state; all candidate and participating-owned block metadata; and each
        participating row's table and cache coverages.  Other allocator
        mutations are fenced until rollback.  ``None`` is the deterministic,
        mutation-free insufficient-capacity result.
        """
        if self._active_temporary_reservation() is not None \
                or self._temporary_reservation_in_progress:
            raise RuntimeError("a temporary reservation is already active")
        if self._temporary_reservation_by_seq_id:
            raise RuntimeError("temporary reservation sequence index is stale")

        self._temporary_reservation_in_progress = True
        try:
            requested_rows = tuple(rows)
            seen_seq_ids: set[int] = set()
            row_specs: list[_TemporaryReservationRowSpec] = []
            total_new_blocks = 0
            for seq, highest_written_position in requested_rows:
                if not isinstance(seq, Sequence):
                    raise TypeError(
                        "temporary reservation rows require Sequence objects"
                    )
                if type(highest_written_position) is not int:
                    raise TypeError(
                        "highest_written_position must be an integer"
                    )
                if highest_written_position < 0:
                    raise ValueError(
                        "highest_written_position must be non-negative"
                    )
                if seq.seq_id in seen_seq_ids:
                    raise ValueError(
                        "temporary reservation rows must be unique"
                    )
                seen_seq_ids.add(seq.seq_id)
                block_table_before = tuple(seq.block_table)
                for block_id in block_table_before:
                    if type(block_id) is not int \
                            or not 0 <= block_id < len(self.blocks) \
                            or block_id not in self.used_block_ids \
                            or self.blocks[block_id].ref_count <= 0:
                        raise RuntimeError(
                            f"sequence {seq.seq_id} block table contains "
                            "an unowned block"
                        )
                required_blocks = (
                    highest_written_position // self.block_size + 1
                )
                required_blocks = max(
                    required_blocks, len(block_table_before)
                )
                new_blocks = required_blocks - len(block_table_before)
                total_new_blocks += new_blocks
                row_specs.append(
                    _TemporaryReservationRowSpec(
                        sequence=seq,
                        block_table_before=block_table_before,
                        new_blocks=new_blocks,
                        num_cached_tokens_before=seq.num_cached_tokens,
                        num_draft_cached_tokens_before=(
                            seq.num_draft_cached_tokens
                        ),
                    )
                )

            if total_new_blocks > len(self.free_block_ids):
                return None

            free_before = tuple(self.free_block_ids)
            used_before = frozenset(self.used_block_ids)
            hash_before = tuple(sorted(self.hash_to_block_id.items()))
            if len(set(free_before)) != len(free_before) \
                    or set(free_before) & set(used_before) \
                    or set(free_before) | set(used_before) \
                    != set(range(len(self.blocks))):
                raise RuntimeError("allocator membership snapshot is invalid")
            candidate_ids = free_before[:total_new_blocks]
            participating_ids = {
                block_id
                for spec in row_specs
                for block_id in spec.block_table_before
            }
            block_states = self._snapshot_block_states(
                (*candidate_ids, *participating_ids)
            )
            state_by_id = {
                state.block_id: state for state in block_states
            }
            for block_id in candidate_ids:
                if state_by_id[block_id].ref_count != 0:
                    raise RuntimeError(
                        "candidate free block has a nonzero reference count"
                    )

            allocated: list[int] = []
            reservation_rows: list[_TemporaryReservationRow] = []
            try:
                for spec in row_specs:
                    appended: list[int] = []
                    for _ in range(spec.new_blocks):
                        self._temporary_internal_allocation = True
                        try:
                            block_id = self._allocate_block()
                        finally:
                            self._temporary_internal_allocation = False
                        seq = spec.sequence
                        seq.block_table.append(block_id)
                        appended.append(block_id)
                        allocated.append(block_id)
                    reservation_rows.append(
                        _TemporaryReservationRow(
                            sequence=spec.sequence,
                            block_table_before=spec.block_table_before,
                            appended_block_ids=tuple(appended),
                            num_cached_tokens_before=(
                                spec.num_cached_tokens_before
                            ),
                            num_draft_cached_tokens_before=(
                                spec.num_draft_cached_tokens_before
                            ),
                        )
                    )
                if tuple(allocated) != candidate_ids:
                    raise RuntimeError(
                        "temporary allocation order differs from free-list order"
                    )
            except BaseException:
                self._restore_temporary_snapshot(
                    rows=row_specs,
                    free_block_ids_before=free_before,
                    used_block_ids_before=used_before,
                    hash_to_block_id_before=hash_before,
                    block_states_before=block_states,
                )
                raise

            active_reservations_before = dict(
                self._active_temporary_reservations
            )
            reservation_index_before = dict(
                self._temporary_reservation_by_seq_id
            )
            reservation = None
            try:
                reservation = TemporaryBlockReservation(
                    reservation_id=next(self._temporary_reservation_ids),
                    rows=tuple(reservation_rows),
                    allocation_order=tuple(allocated),
                    free_block_ids_before=free_before,
                    used_block_ids_before=used_before,
                    hash_to_block_id_before=hash_before,
                    block_states_before=block_states,
                )
                self._active_temporary_reservations[
                    reservation.reservation_id
                ] = reservation
                for row in reservation.rows:
                    self._temporary_reservation_by_seq_id[
                        row.sequence.seq_id
                    ] = reservation.reservation_id
                self._validate_live_temporary_reservation(reservation)
            except BaseException:
                self._active_temporary_reservations = (
                    active_reservations_before
                )
                self._temporary_reservation_by_seq_id = (
                    reservation_index_before
                )
                self._restore_temporary_snapshot(
                    rows=(
                        reservation.rows
                        if reservation is not None
                        else row_specs
                    ),
                    free_block_ids_before=free_before,
                    used_block_ids_before=used_before,
                    hash_to_block_id_before=hash_before,
                    block_states_before=block_states,
                )
                raise
            return reservation
        finally:
            self._temporary_internal_allocation = False
            self._temporary_reservation_in_progress = False

    def _validate_live_temporary_reservation(
        self,
        reservation: TemporaryBlockReservation,
    ) -> None:
        if self._active_temporary_reservation() is not reservation:
            raise RuntimeError("temporary reservation publication drifted")
        expected_sequence_index = {
            row.sequence.seq_id: reservation.reservation_id
            for row in reservation.rows
        }
        if self._temporary_reservation_by_seq_id != expected_sequence_index:
            raise RuntimeError("temporary reservation sequence index drifted")
        expected_state_ids = frozenset(reservation.allocation_order) | {
            block_id
            for row in reservation.rows
            for block_id in row.block_table_before
        }
        if tuple(
            state.block_id
            for state in reservation.block_states_before
        ) != tuple(sorted(expected_state_ids)):
            raise RuntimeError("temporary reservation block snapshot is invalid")
        if len(set(reservation.allocation_order)) \
                != len(reservation.allocation_order):
            raise RuntimeError("temporary reservation repeats an allocated block")

        expected_free = reservation.free_block_ids_before[
            len(reservation.allocation_order):
        ]
        if tuple(self.free_block_ids) != expected_free:
            raise RuntimeError("temporary reservation free-block snapshot drifted")
        expected_used = reservation.used_block_ids_before | frozenset(
            reservation.allocation_order
        )
        if self.used_block_ids != expected_used:
            raise RuntimeError("temporary reservation used-block snapshot drifted")
        expected_hashes = dict(reservation.hash_to_block_id_before)
        state_by_id = {
            state.block_id: state
            for state in reservation.block_states_before
        }
        allocated_ids = frozenset(reservation.allocation_order)
        for block_id in allocated_ids:
            state = state_by_id[block_id]
            if (
                state.hash != -1
                and expected_hashes.get(state.hash) == state.block_id
            ):
                del expected_hashes[state.hash]
        if self.hash_to_block_id != expected_hashes:
            raise RuntimeError("temporary reservation prefix-hash snapshot drifted")
        for state in reservation.block_states_before:
            block = self.blocks[state.block_id]
            if state.block_id in allocated_ids:
                if block.ref_count != 1 \
                        or block.hash != -1 \
                        or block.token_ids:
                    raise RuntimeError("temporary block metadata drifted")
            elif block.ref_count != state.ref_count \
                    or block.hash != state.hash \
                    or tuple(block.token_ids) != state.token_ids:
                raise RuntimeError(
                    "non-temporary block metadata drifted during reservation"
                )

        flattened_appends = tuple(
            block_id
            for row in reservation.rows
            for block_id in row.appended_block_ids
        )
        if flattened_appends != reservation.allocation_order:
            raise RuntimeError("temporary reservation row ownership drifted")
        for row in reservation.rows:
            expected_table = (
                row.block_table_before + row.appended_block_ids
            )
            if tuple(row.sequence.block_table) != expected_table:
                raise RuntimeError(
                    f"sequence {row.sequence.seq_id} temporary block table drifted"
                )
            if row.sequence.num_cached_tokens != row.num_cached_tokens_before:
                raise RuntimeError("target cache coverage changed during reservation")
            if (
                row.sequence.num_draft_cached_tokens
                != row.num_draft_cached_tokens_before
            ):
                raise RuntimeError("draft cache coverage changed during reservation")

    def rollback_temporary_append(
        self,
        reservation: TemporaryBlockReservation,
    ) -> bool:
        """Rollback one temporary lease; return false after it was rolled back."""

        if not isinstance(reservation, TemporaryBlockReservation):
            raise TypeError("reservation must be a TemporaryBlockReservation")
        active = self._active_temporary_reservations.get(
            reservation.reservation_id
        )
        if active is None:
            return False
        if active is not reservation:
            raise RuntimeError("temporary reservation identity mismatch")
        self._validate_live_temporary_reservation(reservation)

        self._restore_temporary_snapshot(
            rows=reservation.rows,
            free_block_ids_before=reservation.free_block_ids_before,
            used_block_ids_before=reservation.used_block_ids_before,
            hash_to_block_id_before=reservation.hash_to_block_id_before,
            block_states_before=reservation.block_states_before,
        )
        self._unpublish_temporary_reservation(reservation)
        return True

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        self._assert_allocator_mutation_allowed("append a sequence block")
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        self._assert_allocator_mutation_allowed("hash sequence blocks")
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
