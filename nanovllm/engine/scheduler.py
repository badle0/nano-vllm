from collections import deque
from dataclasses import dataclass, replace
from itertools import count
from time import perf_counter
from typing import Mapping

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus, StreamOutput
from nanovllm.engine.block_manager import (
    BlockManager,
    TemporaryBlockReservation,
)
from nanovllm.engine.speculative_routes import (
    DraftRouteAdmission,
    DraftRouteKey,
    speculative_plan_fingerprint,
)
from nanovllm.engine.speculative_memory import (
    SpeculativeMemoryPlan, SpeculativeMemoryPlanningError,
)
from nanovllm.engine.speculative_plan import (
    SpecPlanRow, SpecStepPlan, build_speculative_step_plan,
    derive_speculative_effective_k, draft_catchup_tokens,
    speculative_k_budget,
)
from nanovllm.engine.speculative_result import SPEC_METRIC_KEYS, validate_result


@dataclass(frozen=True, slots=True)
class DraftDiscardRow:
    """Immutable per-sequence routing facts for a V3 discard cycle."""

    seq_id: int
    committed_tokens: int
    target_cached_tokens: int
    draft_cached_tokens: int
    remaining_completion_tokens: int
    model_position_headroom: int
    highest_proposal_input_position: int | None
    block_table: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DraftDiscardPlan:
    """Scheduler decision for compute-then-discard draft execution.

    The temporary block lease is deliberately absent: it remains private to
    ``Scheduler`` and is released before ordinary target postprocessing.
    """

    cycle_id: int
    rows: tuple[DraftDiscardRow, ...]
    configured_k: int
    workspace_route_cap: int
    effective_k: int
    draft_catchup_tokens: int
    draft_step_token_counts: tuple[int, ...]
    target_query_tokens: int
    total_scheduled_tokens: int
    fallback_reason: str | None
    route_key: DraftRouteKey | None = None

    @property
    def uses_draft(self) -> bool:
        return self.effective_k > 0 and self.fallback_reason is None


@dataclass(frozen=True, slots=True)
class _DecodeScheduleRollbackRow:
    sequence: Sequence
    seq_id: int
    committed_tokens: int
    target_cached_tokens: int
    draft_cached_tokens: int
    expected_block_count: int


@dataclass(frozen=True, slots=True)
class DecodeScheduleRollback:
    """Plan-independent lease for undoing one scheduled decode batch."""

    rows: tuple[_DecodeScheduleRollbackRow, ...]


@dataclass(slots=True)
class ActiveSpecTransaction:
    """Scheduler-owned V4 undo state, retained until target commit succeeds."""

    plan: SpecStepPlan
    reservation: TemporaryBlockReservation
    baseline_decode_rollback: DecodeScheduleRollback
    target_writes_prepared: bool = False


def as_draft_discard_plan(plan: SpecStepPlan) -> DraftDiscardPlan:
    """Project planned V5 geometry onto the V3 work V4 actually executes."""
    return DraftDiscardPlan(
        cycle_id=plan.cycle_id,
        rows=tuple(DraftDiscardRow(
            seq_id=row.seq_id,
            committed_tokens=row.committed_tokens,
            target_cached_tokens=row.target_cached_tokens,
            draft_cached_tokens=row.draft_cached_tokens,
            remaining_completion_tokens=row.remaining_completion_tokens,
            model_position_headroom=row.model_position_headroom,
            highest_proposal_input_position=row.highest_draft_write_position,
            block_table=row.block_table,
        ) for row in plan.rows),
        configured_k=plan.configured_k,
        workspace_route_cap=plan.workspace_route_cap,
        effective_k=plan.effective_k,
        draft_catchup_tokens=plan.draft_catchup_tokens,
        draft_step_token_counts=plan.draft_step_token_counts,
        target_query_tokens=plan.shadow_target_query_tokens,
        total_scheduled_tokens=plan.total_scheduled_tokens,
        fallback_reason=plan.bypass_reason,
        route_key=plan.route_key,
    )


class SchedulerCapacityError(RuntimeError):

    def __init__(self, requested: int, available: int, capacity: int):
        self.requested = requested
        self.available = available
        self.capacity = capacity
        super().__init__(
            f"cannot admit {requested} request(s): only {available} of "
            f"{capacity} scheduler slots are available; split the batch or "
            "retry after requests finish or are cancelled"
        )


class Scheduler:

    def __init__(self, config: Config, clock=None):
        self._clock = perf_counter if clock is None else clock
        self.max_num_seqs = config.max_num_seqs
        self._max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        hf_config = getattr(config, "hf_config", None)
        vocab_size = getattr(hf_config, "vocab_size", None)
        self.vocab_size = (
            vocab_size if type(vocab_size) is int and vocab_size > 0 else None
        )
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self._configured_k = getattr(config, "configured_k", 0)
        self._max_model_len = getattr(config, "max_model_len", 0)
        self._draft_discard_cycle_ids = count()
        self._active_spec_transaction: ActiveSpecTransaction | None = None
        self._active_draft_discard: tuple[
            DraftDiscardPlan | SpecStepPlan, TemporaryBlockReservation
        ] | None = None
        self._pending_draft_coverage_plan: DraftDiscardPlan | SpecStepPlan | None = None
        self._pending_draft_coverage: tuple[tuple[int, int], ...] | None = None
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.mid_chunk_seq: Sequence | None = None

    @property
    def max_num_batched_tokens(self) -> int:
        """Constructor-configured token budget used by scheduling and graphs."""
        return self._max_num_batched_tokens

    @property
    def available_capacity(self) -> int:
        return self.max_num_seqs - len(self.waiting) - len(self.running)

    def require_capacity(self, requested: int = 1):
        if requested < 0:
            raise ValueError("requested capacity must be non-negative")
        available = self.available_capacity
        if requested > available:
            raise SchedulerCapacityError(
                requested=requested,
                available=max(available, 0),
                capacity=self.max_num_seqs,
            )

    def _check_mid_chunk_invariant(self):
        mid = self.mid_chunk_seq
        if mid is None:
            if self.waiting and self.waiting[0].block_table:
                raise RuntimeError("waiting head owns KV blocks without mid_chunk_seq")
            return
        if not self.waiting or self.waiting[0] is not mid:
            raise RuntimeError("mid_chunk_seq must remain at the waiting head")
        if not mid.block_table:
            raise RuntimeError("mid_chunk_seq must own its allocated KV blocks")

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.require_capacity()
        if self._configured_k > 0:
            seq.spec_metrics = dict.fromkeys(SPEC_METRIC_KEYS, 0)
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self._active_spec_transaction is not None:
            raise RuntimeError("cannot schedule before speculative transaction completes")
        if self._active_draft_discard is not None:
            raise RuntimeError(
                "cannot schedule while a draft-discard reservation is active"
            )
        if self._pending_draft_coverage_plan is not None:
            raise RuntimeError(
                "cannot schedule before committing draft-cache coverage"
            )
        self._check_mid_chunk_invariant()
        scheduled_seqs = []

        # Prefer ongoing decode, including when a partial prefill holds the KV
        # capacity it needs. This is scheduling priority, not a wall-time ITL
        # bound: all rows still wait for the complete mixed forward.
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.mid_chunk_seq is not None:
                    # The invariant above identifies the waiting head. No
                    # prefill work has been scheduled in this synchronous step,
                    # so its blocks can be reclaimed before evicting a decoder.
                    # Retry capacity: reclamation must not assume exclusive KV
                    # ownership or a particular number of released blocks.
                    self.preempt(self.waiting.popleft())
                elif self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        num_decodes = len(scheduled_seqs)
        num_batched_tokens = num_decodes    # decodes charge 1 each (F2 budget accounting)
        self.running.extendleft(reversed(scheduled_seqs))

        # FIFO chunk fill to the remaining budget: each seq takes min(work, remaining),
        # so only the last admitted seq can be partial (<=1 partial per step, F2)
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                self.block_manager.allocate(seq, num_cached_blocks)
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                if seq is self.mid_chunk_seq:
                    self.mid_chunk_seq = None
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = self._clock()
            scheduled_seqs.append(seq)
            if seq.num_scheduled_tokens < num_tokens:   # partial => budget exhausted
                if self.mid_chunk_seq not in (None, seq):
                    raise RuntimeError("more than one sequence is mid-chunk")
                self.mid_chunk_seq = seq
                break

        if not scheduled_seqs:
            if not self.waiting and not self.running:
                raise RuntimeError("cannot schedule: no pending requests")
            # Admission prevents a single request from exceeding the pool.
            # Keep a typed guard for internal corruption or a pool exhausted by
            # work that could not be preempted.
            raise RuntimeError(
                "scheduler could not schedule pending work with the available "
                "KV-cache blocks"
            )
        self._check_mid_chunk_invariant()
        # is_prefill return semantics are now "ragged step": any prefill work present
        return scheduled_seqs, len(scheduled_seqs) > num_decodes

    def _draft_discard_rows(
        self,
        seqs: tuple[Sequence, ...],
        *,
        effective_k: int,
    ) -> tuple[DraftDiscardRow, ...]:
        rows = []
        for seq in seqs:
            remaining = max(
                seq.max_tokens - seq.num_completion_tokens,
                0,
            )
            model_headroom = max(self._max_model_len - len(seq), 0)
            highest_position = (
                len(seq) + effective_k - 2
                if effective_k > 0
                else None
            )
            rows.append(
                DraftDiscardRow(
                    seq_id=seq.seq_id,
                    committed_tokens=len(seq),
                    target_cached_tokens=seq.num_cached_tokens,
                    draft_cached_tokens=seq.num_draft_cached_tokens,
                    remaining_completion_tokens=remaining,
                    model_position_headroom=model_headroom,
                    highest_proposal_input_position=highest_position,
                    block_table=tuple(seq.block_table),
                )
            )
        return tuple(rows)

    def _draft_discard_plan(
        self,
        *,
        cycle_id: int,
        seqs: tuple[Sequence, ...],
        workspace_route_cap: int,
        effective_k: int,
        fallback_reason: str | None,
        draft_catchup_tokens: int = 0,
        route_key: DraftRouteKey | None = None,
    ) -> DraftDiscardPlan:
        batch_size = len(seqs)
        return DraftDiscardPlan(
            cycle_id=cycle_id,
            rows=self._draft_discard_rows(seqs, effective_k=effective_k),
            configured_k=self._configured_k,
            workspace_route_cap=workspace_route_cap,
            effective_k=effective_k,
            draft_catchup_tokens=draft_catchup_tokens,
            draft_step_token_counts=(batch_size,) * effective_k,
            target_query_tokens=batch_size,
            total_scheduled_tokens=(
                draft_catchup_tokens + batch_size * (effective_k + 1)
            ),
            fallback_reason=fallback_reason,
            route_key=route_key,
        )

    def plan_draft_discard(
        self,
        seqs: list[Sequence] | tuple[Sequence, ...],
        *,
        workspace_route_cap: int | None = None,
        route_admission: DraftRouteAdmission | None = None,
    ) -> DraftDiscardPlan:
        """Plan one pure-decode draft cycle and reserve its temporary suffix.

        ``workspace_route_cap`` is the largest K certified by the caller's
        route-specific workspace registry for this exact batch composition.
        A zero bound or any scalar/capacity limit returns a whole-batch baseline
        plan before temporary allocator mutation.
        """

        if self._active_spec_transaction is not None:
            raise RuntimeError("a speculative transaction is already active")
        if self._active_draft_discard is not None:
            raise RuntimeError("a draft-discard reservation is already active")
        if self._pending_draft_coverage_plan is not None:
            raise RuntimeError(
                "draft-cache coverage from the previous cycle is pending"
            )
        if route_admission is not None:
            if not isinstance(route_admission, DraftRouteAdmission):
                raise TypeError("route_admission must be a DraftRouteAdmission")
            if workspace_route_cap is not None:
                raise ValueError(
                    "provide route_admission or workspace_route_cap, not both"
                )
            workspace_route_cap = route_admission.max_effective_k
        elif workspace_route_cap is None:
            workspace_route_cap = 0
        if type(workspace_route_cap) is not int:
            raise TypeError("workspace_route_cap must be an integer")
        if workspace_route_cap < 0:
            raise ValueError("workspace_route_cap must be non-negative")
        if type(self._configured_k) is not int or self._configured_k < 0:
            raise RuntimeError("scheduler configured K is invalid")
        if type(self._max_model_len) is not int or self._max_model_len < 1:
            raise RuntimeError("scheduler max_model_len is invalid")
        selected = tuple(seqs)
        if route_admission is not None \
                and route_admission.batch_size != len(selected):
            raise ValueError("route admission batch size does not match decode rows")
        if len({seq.seq_id for seq in selected}) != len(selected):
            raise ValueError("draft-discard rows must be unique")
        cycle_id = next(self._draft_discard_cycle_ids)
        if not selected:
            return self._draft_discard_plan(
                cycle_id=cycle_id,
                seqs=selected,
                workspace_route_cap=workspace_route_cap,
                effective_k=0,
                fallback_reason="no_decode_rows",
            )

        pure_decode = all(
            seq.status is SequenceStatus.RUNNING
            and not seq.is_prefill
            and seq.num_scheduled_tokens == 1
            and seq.num_cached_tokens == len(seq) - 1
            for seq in selected
        )
        if not pure_decode:
            return self._draft_discard_plan(
                cycle_id=cycle_id,
                seqs=selected,
                workspace_route_cap=workspace_route_cap,
                effective_k=0,
                fallback_reason="not_pure_decode",
            )

        request_cap = min(
            max(seq.max_tokens - seq.num_completion_tokens - 1, 0)
            for seq in selected
        )
        model_cap = min(
            max(self._max_model_len - len(seq), 0)
            for seq in selected
        )
        draft_catchup_tokens = 0
        for seq in selected:
            if (
                type(seq.num_draft_cached_tokens) is not int
                or seq.num_draft_cached_tokens < 0
                or seq.num_draft_cached_tokens > len(seq) - 1
            ):
                raise RuntimeError(
                    f"sequence {seq.seq_id} has invalid draft-cache coverage"
                )
            draft_catchup_tokens += len(seq) - 1 - seq.num_draft_cached_tokens
        if route_admission is not None \
                and route_admission.catchup_tokens != draft_catchup_tokens:
            raise RuntimeError(
                "draft-cache coverage changed after route admission"
            )
        # Count every actual model input in the hard scheduler budget.  Catch-up
        # can be much larger than B*(K+1) after chunked prefill or preemption.
        remaining_after_catchup = (
            self.max_num_batched_tokens - draft_catchup_tokens
        )
        token_budget_cap = max(
            remaining_after_catchup // len(selected) - 1,
            0,
        )
        effective_k = min(
            self._configured_k,
            request_cap,
            model_cap,
            token_budget_cap,
            workspace_route_cap,
        )
        if effective_k == 0:
            bounds = (
                ("speculation_disabled", self._configured_k),
                ("request_tail", request_cap),
                ("model_position_limit", model_cap),
                ("token_budget", token_budget_cap),
                ("workspace_route_cap", workspace_route_cap),
            )
            reason = next(name for name, bound in bounds if bound == 0)
            if (
                reason == "token_budget"
                and draft_catchup_tokens + 2 * len(selected)
                > self.max_num_batched_tokens
            ):
                reason = "draft_catchup_token_budget"
            return self._draft_discard_plan(
                cycle_id=cycle_id,
                seqs=selected,
                workspace_route_cap=workspace_route_cap,
                effective_k=0,
                fallback_reason=reason,
            )

        route_key = (
            route_admission.key_for(effective_k)
            if route_admission is not None
            else None
        )
        if route_admission is not None and route_key is None:
            return self._draft_discard_plan(
                cycle_id=cycle_id,
                seqs=selected,
                workspace_route_cap=workspace_route_cap,
                effective_k=0,
                fallback_reason="route_registry_miss",
            )

        reservation = self.block_manager.reserve_temporary_append(
            (
                (seq, len(seq) + effective_k - 2)
                for seq in selected
            )
        )
        if reservation is None:
            return self._draft_discard_plan(
                cycle_id=cycle_id,
                seqs=selected,
                workspace_route_cap=workspace_route_cap,
                effective_k=0,
                fallback_reason="insufficient_kv_blocks",
            )
        try:
            plan = self._draft_discard_plan(
                cycle_id=cycle_id,
                seqs=selected,
                workspace_route_cap=workspace_route_cap,
                effective_k=effective_k,
                fallback_reason=None,
                draft_catchup_tokens=draft_catchup_tokens,
                route_key=route_key,
            )
            self._active_draft_discard = (plan, reservation)
            return plan
        except BaseException:
            self.block_manager.rollback_temporary_append(reservation)
            raise

    def plan_speculative_step(
        self,
        seqs: list[Sequence] | tuple[Sequence, ...],
        *,
        configured_workspace: SpeculativeMemoryPlan,
        route_admission: DraftRouteAdmission | None = None,
        baseline_decode_rollback: DecodeScheduleRollback | None = None,
    ) -> SpecStepPlan:
        """Promote an ordinary decode batch to a V4 shadow transaction.

        Admission prices the later full verifier cycle. All modeled checks
        precede additional block allocation. The existing schedule has already
        reserved the ordinary target input; this lease reserves only its suffix.
        """
        if (self._active_spec_transaction is not None
                or self._active_draft_discard is not None
                or self._pending_draft_coverage_plan is not None):
            raise RuntimeError("a speculative transaction is already active")
        if not isinstance(configured_workspace, SpeculativeMemoryPlan):
            raise TypeError("configured_workspace must be a SpeculativeMemoryPlan")
        selected = tuple(seqs)
        if any(not isinstance(seq, Sequence) for seq in selected):
            raise TypeError("speculative rows require Sequence objects")
        rows = tuple(SpecPlanRow(
            seq_id=seq.seq_id,
            committed_tokens=len(seq),
            target_cached_tokens=seq.num_cached_tokens,
            draft_cached_tokens=seq.num_draft_cached_tokens,
            remaining_completion_tokens=max(seq.max_tokens - seq.num_completion_tokens, 0),
            model_position_headroom=max(self._max_model_len - len(seq), 0),
            highest_draft_write_position=None,
            highest_target_write_position=None,
            block_table=tuple(seq.block_table),
        ) for seq in selected)
        cap = 0
        if route_admission is not None:
            if not isinstance(route_admission, DraftRouteAdmission):
                raise TypeError("route_admission must be a DraftRouteAdmission")
            if route_admission.batch_size != len(rows):
                raise ValueError("route admission batch size does not match decode rows")
            if route_admission.plan_fingerprint != speculative_plan_fingerprint(configured_workspace):
                raise ValueError("route admission workspace fingerprint is stale")
            cap = route_admission.max_effective_k
        cycle_id = next(self._draft_discard_cycle_ids)

        def build(k, reason=None, key=None):
            return build_speculative_step_plan(
                cycle_id=cycle_id, rows=rows, configured_k=self._configured_k,
                workspace_route_cap=cap, effective_k=k,
                max_num_batched_tokens=self.max_num_batched_tokens,
                configured_workspace=configured_workspace,
                route_key=key, bypass_reason=reason,
            )

        if not selected:
            return build(0, "no_decode_rows")
        rollback = self.capture_decode_schedule_rollback(selected, is_prefill=False)
        if rollback is None:
            return build(0, "not_pure_decode")
        if baseline_decode_rollback is not None and baseline_decode_rollback != rollback:
            raise ValueError("baseline decode rollback does not match selected rows")
        if baseline_decode_rollback is not None:
            rollback = baseline_decode_rollback
        if tuple(self.running) != selected:
            raise ValueError("speculative plan must cover the whole scheduled decode batch")
        catchup = draft_catchup_tokens(rows)
        if route_admission is not None and route_admission.catchup_tokens != catchup:
            raise ValueError("draft-cache coverage changed after route admission")
        k = derive_speculative_effective_k(
            rows=rows, configured_k=self._configured_k,
            max_num_batched_tokens=self.max_num_batched_tokens,
            workspace_route_cap=cap,
        )
        if k == 0:
            batch = len(rows)
            budget = self.max_num_batched_tokens
            bounds = (
                ("speculation_disabled", self._configured_k),
                ("request_tail", min(max(row.remaining_completion_tokens - 1, 0) for row in rows)),
                ("model_position_limit", min(row.model_position_headroom for row in rows)),
                ("verifier_token_budget", max(budget // batch - 1, 0)),
                ("draft_catchup_token_budget" if 3 * batch <= budget < catchup + 3 * batch
                 else "aggregate_token_budget", speculative_k_budget(
                     max_num_batched_tokens=budget, batch_size=batch,
                     draft_catchup_tokens=catchup)),
                ("workspace_route_cap", cap),
            )
            return build(0, next(name for name, bound in bounds if bound == 0))
        key = route_admission.key_for(k)
        if key is None:
            return build(0, "route_registry_miss")
        try:
            plan = build(k, key=key)
        except SpeculativeMemoryPlanningError:
            return build(0, "workspace_route_cap")
        reservation = self.block_manager.reserve_temporary_append(
            (seq, row.highest_target_write_position)
            for seq, row in zip(selected, plan.rows, strict=True)
        )
        if reservation is None:
            return build(0, "insufficient_kv_blocks")
        try:
            plan = replace(plan, rows=tuple(
                replace(row, block_table=tuple(seq.block_table))
                for seq, row in zip(selected, plan.rows, strict=True)
            ))
            transaction = ActiveSpecTransaction(plan, reservation, rollback)
            self._active_draft_discard = (plan, reservation)
            self._active_spec_transaction = transaction
            return plan
        except BaseException:
            self._active_draft_discard = None
            self._active_spec_transaction = None
            self.block_manager.rollback_temporary_append(reservation)
            raise

    def abort_speculative_step(self) -> bool:
        """Undo both reservation layers; retain ownership if cleanup fails."""
        transaction = getattr(self, "_active_spec_transaction", None)
        if transaction is None:
            return False
        self.rollback_draft_discard(transaction.plan)
        self.abort_draft_coverage(transaction.plan)
        self.rollback_failed_decode_schedule(transaction.baseline_decode_rollback)
        self._active_spec_transaction = None
        return True

    def prepare_speculative_target_writes(self, plan):
        transaction = self._active_spec_transaction
        if transaction is None or transaction.plan is not plan:
            raise RuntimeError("speculative target writes require the active transaction")
        if not transaction.target_writes_prepared:
            transaction.reservation = self.block_manager.prepare_temporary_target_writes(transaction.reservation)
            self._active_draft_discard = (plan, transaction.reservation)
            transaction.target_writes_prepared = True

    def commit_speculative(self, plan, result):
        """Atomic batch commit: physical trim, logical state, hashes, then events.

        The live lease/undo record is restored on every failure so the engine's
        ordinary abort path can release both reservation layers. Recycled free
        cache entries were already evicted before target writes and stay evicted.
        """
        transaction = self._active_spec_transaction
        if transaction is None or transaction.plan is not plan or not transaction.target_writes_prepared:
            raise RuntimeError("speculative commit requires a prepared target-write transaction")
        seqs = [row.sequence for row in transaction.reservation.rows]
        if tuple(seqs) != tuple(self.running):
            raise ValueError("speculative commit must cover the scheduled running batch")
        validate_result(plan, result, seqs, self.vocab_size)
        prepared = []
        for seq, row, snapshot in zip(seqs, result.rows, plan.rows, strict=True):
            tokens = []
            for token in row.committed_token_ids:
                if seq.num_completion_tokens + len(tokens) >= seq.max_tokens:
                    break
                tokens.append(token)
                if not seq.ignore_eos and token == self.eos:
                    break
            if not tokens:
                raise ValueError("speculative commit has no completion headroom")
            cached = len(seq) + len(tokens) - 1
            draft_cached = min(cached, len(seq) + plan.effective_k - 1)
            prepared.append((seq, row, snapshot, tokens, cached, draft_cached))

        manager = self.block_manager
        # Host metadata only: never clone model weights or GPU caches here.
        sequence_states = [(seq, {key: value.copy() if isinstance(value, (list, dict)) else value
                                  for key, value in seq.__dict__.items()}) for seq in seqs]
        allocator_state = (
            tuple(manager.free_block_ids), set(manager.used_block_ids), dict(manager.hash_to_block_id),
            [(b.ref_count, b.hash, b.token_ids[:]) for b in manager.blocks],
            dict(manager._active_temporary_reservations), dict(manager._temporary_reservation_by_seq_id),
        )
        running_before, waiting_before = tuple(self.running), tuple(self.waiting)
        now = self._clock()
        try:
            keep = {
                seq.seq_id: (cached + self.block_size - 1) // self.block_size - len(lease_row.block_table_before)
                for (seq, _, _, _, cached, _), lease_row in zip(prepared, transaction.reservation.rows, strict=True)
            }
            if not manager.finalize_temporary_append(transaction.reservation, keep):
                raise RuntimeError("speculative commit did not finalize its lease")
            for seq, row, snapshot, tokens, cached, draft_cached in prepared:
                for token in tokens:
                    seq.append_token(token)
                # Publish only complete blocks covered by actual target KV.
                # hash_blocks uses the OLD coverage plus newly processed count.
                seq.num_scheduled_tokens = cached - seq.num_cached_tokens
                manager.hash_blocks(seq)
                seq.num_cached_tokens, seq.num_draft_cached_tokens = cached, draft_cached
                seq.num_scheduled_tokens = 0
                if seq.first_token_time is None:
                    seq.first_token_time = now
                seq.token_times.extend([now] * len(tokens))
                counters = dict(getattr(seq, "spec_metrics", {}))
                updates = {
                    "spec_cycles": 1, "spec_proposed_draft_tokens": plan.effective_k,
                    "spec_accepted_draft_tokens": min(row.accepted_draft_tokens, len(tokens)),
                    "spec_committed_tokens": len(tokens),
                    "spec_bonus_tokens": int(row.used_bonus and len(tokens) == len(row.committed_token_ids)),
                    "spec_draft_positions": plan.effective_k + snapshot.target_cached_tokens - snapshot.draft_cached_tokens,
                    "spec_target_verification_positions": plan.effective_k + 1,
                    "spec_residual_numerical_fallbacks": row.residual_numerical_fallbacks,
                }
                for name, value in updates.items():
                    counters[name] = counters.get(name, 0) + value
                seq.spec_metrics = counters
                if (not seq.ignore_eos and tokens[-1] == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                    seq.finish_time = now
                    seq.status = SequenceStatus.FINISHED
                    manager.deallocate(seq)
                    self.running.remove(seq)
            events = [StreamOutput(seq.seq_id, token, seq.is_finished and index == len(tokens) - 1)
                      for seq, _, _, tokens, _, _ in prepared for index, token in enumerate(tokens)]
        except BaseException:
            for seq, state in sequence_states:
                seq.__dict__.clear()
                seq.__dict__.update(state)
            free, used, hashes, blocks, reservations, indices = allocator_state
            manager.free_block_ids, manager.used_block_ids, manager.hash_to_block_id = deque(free), used, hashes
            for block, (ref_count, block_hash, tokens) in zip(manager.blocks, blocks, strict=True):
                block.ref_count, block.hash, block.token_ids = ref_count, block_hash, tokens
            manager._active_temporary_reservations = reservations
            manager._temporary_reservation_by_seq_id = indices
            self.running, self.waiting = deque(running_before), deque(waiting_before)
            raise
        self._active_draft_discard = None
        self._active_spec_transaction = None
        return events

    def capture_decode_schedule_rollback(
        self,
        seqs: list[Sequence] | tuple[Sequence, ...],
        *,
        is_prefill: bool,
    ) -> DecodeScheduleRollback | None:
        """Capture retry metadata immediately after an ordinary decode schedule."""

        selected = tuple(seqs)
        if is_prefill or not selected:
            return None
        rows = []
        for seq in selected:
            if (
                seq.status is not SequenceStatus.RUNNING
                or seq.is_prefill
                or seq.num_scheduled_tokens != 1
                or seq.num_cached_tokens != len(seq) - 1
            ):
                return None
            expected_blocks = (
                seq.num_cached_tokens + self.block_size - 1
            ) // self.block_size
            rows.append(
                _DecodeScheduleRollbackRow(
                    sequence=seq,
                    seq_id=seq.seq_id,
                    committed_tokens=len(seq),
                    target_cached_tokens=seq.num_cached_tokens,
                    draft_cached_tokens=seq.num_draft_cached_tokens,
                    expected_block_count=expected_blocks,
                )
            )
        return DecodeScheduleRollback(tuple(rows))

    def rollback_draft_discard(self, plan: DraftDiscardPlan | SpecStepPlan) -> bool:
        """Release the scheduler-owned lease; repeated calls are harmless."""

        if not isinstance(plan, (DraftDiscardPlan, SpecStepPlan)):
            raise TypeError("plan must be a DraftDiscardPlan")
        active = self._active_draft_discard
        if active is None:
            return False
        active_plan, reservation = active
        if active_plan is not plan:
            raise RuntimeError("draft-discard plan identity mismatch")
        rolled_back = self.block_manager.rollback_temporary_append(reservation)
        self._active_draft_discard = None
        return rolled_back

    def handoff_draft_discard(self, plan: DraftDiscardPlan | SpecStepPlan) -> bool:
        """Release proposal blocks and arm coverage commit before target run."""

        if not isinstance(plan, (DraftDiscardPlan, SpecStepPlan)):
            raise TypeError("plan must be a DraftDiscardPlan")
        if not plan.uses_draft:
            raise ValueError("cannot hand off a fallback plan")
        if getattr(self, "_pending_draft_coverage_plan", None) is not None:
            raise RuntimeError("draft coverage handoff is already pending")
        active = getattr(self, "_active_draft_discard", None)
        if active is None or active[0] is not plan:
            raise RuntimeError("draft-discard plan is not the active reservation")
        rolled_back = self.rollback_draft_discard(plan)
        if not rolled_back:
            raise RuntimeError("active draft-discard reservation was not released")
        self._pending_draft_coverage_plan = plan
        self._pending_draft_coverage = None
        return True

    def abort_draft_coverage(self, plan: DraftDiscardPlan | SpecStepPlan) -> bool:
        """Clear an exact pre-target handoff after target/postprocess failure."""

        if not isinstance(plan, (DraftDiscardPlan, SpecStepPlan)):
            raise TypeError("plan must be a DraftDiscardPlan")
        pending = getattr(self, "_pending_draft_coverage_plan", None)
        if pending is None:
            return False
        if pending is not plan:
            raise RuntimeError("pending draft-coverage plan identity mismatch")
        self._pending_draft_coverage_plan = None
        self._pending_draft_coverage = None
        return True

    def rollback_failed_decode_schedule(
        self,
        rollback: DecodeScheduleRollback,
    ) -> bool:
        """Undo the ordinary decode append after a failed V3 shadow cycle.

        ``schedule`` may allocate the next target block before proposal
        execution.  That allocation is not owned by the temporary proposal
        reservation, so it needs a separate, group-atomic rollback before a
        manual caller can retry the same request.
        """

        if not isinstance(rollback, DecodeScheduleRollback):
            raise TypeError("rollback must be a DecodeScheduleRollback")
        if self._active_draft_discard is not None:
            raise RuntimeError("temporary draft reservation is still active")
        if self._pending_draft_coverage_plan is not None:
            raise RuntimeError("draft coverage handoff is still pending")
        rollback_rows = []
        for row in rollback.rows:
            seq = row.sequence
            if type(row.seq_id) is not int or row.seq_id != seq.seq_id:
                raise ValueError("failed schedule sequence ID drifted")
            if (
                seq.status is not SequenceStatus.RUNNING
                or seq.is_prefill
                or seq.num_scheduled_tokens != 1
                or len(seq) != row.committed_tokens
                or seq.num_cached_tokens != row.target_cached_tokens
                or row.target_cached_tokens != row.committed_tokens - 1
                or seq.num_draft_cached_tokens != row.draft_cached_tokens
            ):
                raise RuntimeError(
                    f"sequence {seq.seq_id} changed before schedule rollback"
                )
            rollback_rows.append((seq, row.expected_block_count))

        changed = self.block_manager.rollback_uncommitted_appends(rollback_rows)
        for row in rollback.rows:
            row.sequence.num_scheduled_tokens = 0
        return changed

    def _rollback_active_draft_discard(self):
        active = getattr(self, "_active_draft_discard", None)
        if active is not None:
            self.rollback_draft_discard(active[0])

    def stage_draft_coverage(
        self,
        plan: DraftDiscardPlan | SpecStepPlan,
        seqs: list[Sequence] | tuple[Sequence, ...],
        coverage_by_seq_id: Mapping[int, int],
    ) -> None:
        """Validate and stage draft coverage before authoritative target work.

        Every fallible plan/result check occurs here. ``postprocess`` consumes
        this immutable staging record while it performs the target token commit,
        so no separate fallible draft-coverage operation can occur after a
        public token has been appended but before its event is returned.
        """

        if not isinstance(plan, (DraftDiscardPlan, SpecStepPlan)):
            raise TypeError("plan must be a DraftDiscardPlan")
        if not plan.uses_draft:
            raise ValueError("cannot commit draft coverage for a fallback plan")
        if getattr(self, "_active_draft_discard", None) is not None:
            raise RuntimeError(
                "temporary draft-discard reservation is still active"
            )
        pending = getattr(self, "_pending_draft_coverage_plan", None)
        if pending is not plan:
            raise RuntimeError(
                "draft coverage may be staged only for the handed-off plan"
            )
        if self._pending_draft_coverage is not None:
            raise RuntimeError("draft coverage is already staged")
        selected = tuple(seqs)
        planned_ids = tuple(row.seq_id for row in plan.rows)
        selected_ids = tuple(seq.seq_id for seq in selected)
        if selected_ids != planned_ids:
            raise ValueError(
                "draft coverage sequence IDs/order do not match the plan"
            )
        if not isinstance(coverage_by_seq_id, Mapping):
            raise TypeError("coverage_by_seq_id must be a mapping")
        supplied_ids = set(coverage_by_seq_id)
        if supplied_ids != set(planned_ids):
            raise ValueError("draft coverage IDs do not exactly match the plan")

        rows_by_id = {row.seq_id: row for row in plan.rows}
        for seq in selected:
            row = rows_by_id[seq.seq_id]
            supplied = coverage_by_seq_id[seq.seq_id]
            if type(supplied) is not int:
                raise TypeError("draft cache coverage must be an integer")
            if supplied != row.committed_tokens:
                raise ValueError(
                    f"sequence {seq.seq_id} draft coverage must equal "
                    f"the planned committed prefix {row.committed_tokens}"
                )
            if seq.status is not SequenceStatus.RUNNING:
                raise RuntimeError("coverage sequence is not running")
            if len(seq) != row.committed_tokens:
                raise RuntimeError(
                    "coverage sequence length changed before target execution"
                )
            if seq.num_cached_tokens != row.target_cached_tokens:
                raise RuntimeError(
                    "target cache coverage changed before target execution"
                )
            if seq.num_scheduled_tokens != 1 or seq.is_prefill:
                raise RuntimeError("coverage sequence is not a scheduled decode")
            if seq.num_draft_cached_tokens != row.draft_cached_tokens:
                raise RuntimeError("draft cache coverage changed before staging")

        self._pending_draft_coverage = tuple(
            (seq_id, coverage_by_seq_id[seq_id]) for seq_id in planned_ids
        )

    def preempt(self, seq: Sequence):
        self.abort_speculative_step()
        self._rollback_active_draft_discard()
        pending = getattr(self, "_pending_draft_coverage_plan", None)
        if pending is not None and seq.seq_id in {
            row.seq_id for row in pending.rows
        }:
            self._pending_draft_coverage_plan = None
            self._pending_draft_coverage = None
        if seq is self.mid_chunk_seq:
            self.mid_chunk_seq = None
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        seq.num_draft_cached_tokens = 0
        # A preempted victim must not jump ahead of an already-waiting request.
        self.waiting.append(seq)
        self._check_mid_chunk_invariant()

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[StreamOutput]:
        transaction = getattr(self, "_active_spec_transaction", None)
        if transaction is not None and self._pending_draft_coverage_plan is not transaction.plan:
            raise RuntimeError("speculative transaction has not handed off to target commit")
        # V3 proposal blocks must have been handed off before target execution,
        # so they can never reach target hashing or ordinary postprocessing.
        if getattr(self, "_active_draft_discard", None) is not None:
            raise RuntimeError(
                "draft-discard reservation must be handed off before postprocess"
            )
        pending = getattr(self, "_pending_draft_coverage_plan", None)
        if not isinstance(token_ids, (list, tuple)):
            raise TypeError("target token IDs must be a list or tuple")
        if len(token_ids) != len(seqs):
            raise ValueError(
                "target token count does not match the scheduled batch"
            )
        if any(
            type(token_id) is not int
            or token_id < 0
            or (self.vocab_size is not None and token_id >= self.vocab_size)
            for token_id in token_ids
        ):
            bound = (
                "non-negative integers"
                if self.vocab_size is None
                else f"integers in [0, {self.vocab_size})"
            )
            raise ValueError(f"target token IDs must be {bound}")
        if pending is not None:
            planned_ids = tuple(row.seq_id for row in pending.rows)
            actual_ids = tuple(seq.seq_id for seq in seqs)
            if actual_ids != planned_ids:
                raise ValueError(
                    "postprocess sequence IDs/order do not match draft handoff"
                )
            staged = self._pending_draft_coverage
            if staged is None or tuple(seq_id for seq_id, _ in staged) != planned_ids:
                raise RuntimeError("draft coverage was not staged before target work")
            rows_by_id = {row.seq_id: row for row in pending.rows}
            for seq in seqs:
                row = rows_by_id[seq.seq_id]
                if (
                    seq.status is not SequenceStatus.RUNNING
                    or seq.is_prefill
                    or seq.num_scheduled_tokens != 1
                    or len(seq) != row.committed_tokens
                    or seq.num_cached_tokens != row.target_cached_tokens
                    or seq.num_draft_cached_tokens != row.draft_cached_tokens
                ):
                    raise RuntimeError(
                        f"sequence {seq.seq_id} changed before target commit"
                    )
        now = self._clock()
        events: list[StreamOutput] = []
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # emission gate: skip mid-prefill rows. After the increment above,
            # cached < total is exact — decode rows always arrive at equality
            # (preemption zeroes cached and re-enters via the prefill path)
            if seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if seq.first_token_time is None:
                seq.first_token_time = now
            seq.token_times.append(now)
            finished = (not seq.ignore_eos and token_id == self.eos) \
                    or seq.num_completion_tokens >= seq.max_tokens
            if finished:
                seq.finish_time = now
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                seq.num_draft_cached_tokens = 0
                self.running.remove(seq)
            events.append(StreamOutput(seq.seq_id, token_id, finished))
        if pending is not None:
            coverage = dict(self._pending_draft_coverage)
            for seq in seqs:
                if seq.status is SequenceStatus.RUNNING:
                    seq.num_draft_cached_tokens = coverage[seq.seq_id]
            self._pending_draft_coverage_plan = None
            self._pending_draft_coverage = None
        if transaction is not None:
            self._active_spec_transaction = None
        return events

    def cancel(self, seq_ids) -> list[int]:
        """Cancel only the queued sequences named by ``seq_ids``.

        A partially prefetched sequence can still be in ``waiting`` while it
        owns KV blocks, so both scheduler queues must use the same deallocation
        rule. Unknown and duplicate IDs are harmless.
        """
        targets = set(seq_ids)
        if not targets:
            return []

        # Resolve the actual queue members before touching either queue or the
        # allocator.  V3's temporary append lease is global: even cancelling a
        # sequence outside the draft batch may need to deallocate blocks, which
        # the allocator correctly fences while the lease is live.  Roll the
        # lease back first so cancellation cannot fail after destructive
        # ``popleft`` operations and strand otherwise retained rows.
        queued_targets = tuple(
            seq
            for queue in (self.waiting, self.running)
            for seq in queue
            if seq.seq_id in targets
        )
        if not queued_targets:
            return []

        self.abort_speculative_step()
        active = getattr(self, "_active_draft_discard", None)
        if active is not None:
            self._rollback_active_draft_discard()
        pending = getattr(self, "_pending_draft_coverage_plan", None)
        if pending is not None and targets.intersection(
            row.seq_id for row in pending.rows
        ):
            self._pending_draft_coverage_plan = None
            self._pending_draft_coverage = None

        cancelled = []
        for queue in (self.waiting, self.running):
            # Remove each target only after its allocator cleanup succeeds.
            # Thus an unexpected deallocation error may yield a coherent partial
            # cancellation, but it cannot erase unrelated queue members or leave
            # an already-deallocated sequence marked RUNNING outside the queue.
            for seq in tuple(
                seq for seq in queue if seq.seq_id in targets
            ):
                if seq.block_table:
                    self.block_manager.deallocate(seq)
                queue.remove(seq)
                if seq is self.mid_chunk_seq:
                    self.mid_chunk_seq = None
                seq.num_scheduled_tokens = 0
                seq.num_draft_cached_tokens = 0
                seq.status = SequenceStatus.CANCELLED
                cancelled.append(seq.seq_id)
        self._check_mid_chunk_invariant()
        return cancelled

    def cancel_all(self):
        """Administrative compatibility wrapper; request cleanup uses cancel."""
        return self.cancel(
            seq.seq_id for seq in (*self.running, *self.waiting)
        )
