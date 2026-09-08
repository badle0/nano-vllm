from threading import Event, RLock, Thread
from types import MethodType, SimpleNamespace

import pytest

from nanovllm.engine.llm_engine import (
    AdmissionLimits,
    LLMEngine,
    SpeculativeDiscardResultError,
)
from nanovllm.engine.model_runner import (
    DraftDiscardResult,
    DraftDiscardRowResult,
)
from nanovllm.engine.scheduler import DraftDiscardPlan, DraftDiscardRow
from nanovllm.engine.sequence import StreamOutput
from nanovllm.engine.speculative_routes import (
    DRAFT_ROUTE_SCHEMA,
    DraftCatchupFamily,
    DraftExecutionMode,
    DraftRouteAdmission,
    DraftRouteKey,
    DraftSamplerEnvelope,
)


VOCAB_SIZE = 11


def _sequence(seq_id, *, committed_tokens=4):
    return SimpleNamespace(
        seq_id=seq_id,
        is_prefill=False,
        num_scheduled_tokens=1,
        num_tokens=committed_tokens,
        num_cached_tokens=committed_tokens - 1,
        num_draft_cached_tokens=0,
        is_finished=False,
    )


def _plan(seqs, *, effective_k=2, fallback_reason=None):
    rows = tuple(
        DraftDiscardRow(
            seq_id=seq.seq_id,
            committed_tokens=seq.num_tokens,
            target_cached_tokens=seq.num_cached_tokens,
            draft_cached_tokens=seq.num_draft_cached_tokens,
            remaining_completion_tokens=8,
            model_position_headroom=32,
            highest_proposal_input_position=(
                seq.num_tokens + effective_k - 2
                if effective_k
                else None
            ),
            block_table=(seq.seq_id,),
        )
        for seq in seqs
    )
    catchup = (
        sum(seq.num_tokens - 1 - seq.num_draft_cached_tokens for seq in seqs)
        if effective_k
        else 0
    )
    route_key = (
        DraftRouteKey(
            schema=DRAFT_ROUTE_SCHEMA,
            execution_mode=DraftExecutionMode.EAGER_DYNAMIC,
            batch_bucket=len(seqs),
            effective_k=effective_k,
            catchup_family=(
                DraftCatchupFamily.NONE
                if catchup == 0
                else DraftCatchupFamily.PAGED_EAGER_DYNAMIC
            ),
            sampler_envelope=DraftSamplerEnvelope.EXACT_WORST_CASE,
        )
        if effective_k
        else None
    )
    return DraftDiscardPlan(
        cycle_id=7,
        rows=rows,
        configured_k=2,
        workspace_route_cap=2,
        effective_k=effective_k,
        draft_catchup_tokens=catchup,
        draft_step_token_counts=(len(seqs),) * effective_k,
        target_query_tokens=len(seqs),
        total_scheduled_tokens=catchup + len(seqs) * (effective_k + 1),
        fallback_reason=fallback_reason,
        route_key=route_key,
    )


def _result(plan, **overrides):
    rows = tuple(
        DraftDiscardRowResult(
            seq_id=row.seq_id,
            coverage_after_commit=row.committed_tokens,
            proposed_token_ids=tuple(
                (row.seq_id + step) % VOCAB_SIZE
                for step in range(plan.effective_k)
            ),
            proposal_count=plan.effective_k,
        )
        for row in plan.rows
    )
    values = dict(
        rows=rows,
        route_key=plan.route_key,
        effective_k=plan.effective_k,
        catchup_positions=sum(
            row.committed_tokens - 1 - row.draft_cached_tokens
            for row in plan.rows
        ),
        draft_positions=len(rows) * plan.effective_k,
        graph_decode_steps=0,
        eager_decode_steps=plan.effective_k,
        q_shape=(len(rows), plan.effective_k, VOCAB_SIZE),
        q_stride=(VOCAB_SIZE, len(rows) * VOCAB_SIZE, 1),
        q_dtype="torch.float32",
        q_storage_contiguous=True,
        q_view_zero_copy=True,
    )
    values.update(overrides)
    return DraftDiscardResult(**values)


class _Scheduler:
    def __init__(self, seqs, plan, order):
        self.seqs = seqs
        self.plan = plan
        self.order = order
        self.committed = None
        self.plan_error = None

    def schedule(self):
        self.order.append("schedule")
        return self.seqs, False

    def capture_decode_schedule_rollback(self, seqs, *, is_prefill):
        assert seqs is self.seqs
        assert not is_prefill
        self.schedule_rollback = object()
        return self.schedule_rollback

    def plan_draft_discard(
        self,
        seqs,
        *,
        workspace_route_cap=None,
        route_admission=None,
    ):
        route_cap = (
            route_admission.max_effective_k
            if route_admission is not None
            else workspace_route_cap
        )
        self.order.append(("plan", route_cap))
        assert seqs is self.seqs
        if self.plan_error is not None:
            raise self.plan_error
        return self.plan

    def rollback_draft_discard(self, plan):
        self.order.append("rollback")
        assert plan is self.plan
        return True

    def handoff_draft_discard(self, plan):
        self.order.append("handoff")
        assert plan is self.plan
        return True

    def abort_draft_coverage(self, plan):
        self.order.append("abort_coverage")
        assert plan is self.plan
        return True

    def rollback_failed_decode_schedule(self, rollback):
        self.order.append("rollback_schedule")
        assert rollback is self.schedule_rollback
        for seq in self.seqs:
            seq.num_scheduled_tokens = 0
        return True

    def stage_draft_coverage(self, plan, seqs, coverage):
        self.order.append("stage_coverage")
        assert plan is self.plan
        self.staged = dict(coverage)

    def postprocess(self, seqs, token_ids):
        self.order.append("postprocess")
        events = []
        for seq, token_id in zip(seqs, token_ids, strict=True):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            seq.num_tokens += 1
            if hasattr(self, "staged"):
                seq.num_draft_cached_tokens = self.staged[seq.seq_id]
            events.append(StreamOutput(seq.seq_id, token_id, False))
        return events


class _Runner:
    def __init__(self, result, order, *, speculation_enabled=True):
        self.result = result
        self.order = order
        self.speculation_enabled = speculation_enabled
        if speculation_enabled:
            self.speculative_memory_plan = SimpleNamespace(
                batch_size=2,
                max_effective_k=2,
            )
        self.draft_error = None
        self.target_error = None
        self.route_miss = False

    def resolve_draft_route_admission(self, seqs):
        if self.route_miss:
            return None
        catchup = sum(
            seq.num_tokens - 1 - seq.num_draft_cached_tokens for seq in seqs
        )
        family = (
            DraftCatchupFamily.NONE
            if catchup == 0
            else DraftCatchupFamily.PAGED_EAGER_DYNAMIC
        )
        return DraftRouteAdmission(
            registry_schema=DRAFT_ROUTE_SCHEMA,
            plan_fingerprint="engine-double",
            batch_size=len(seqs),
            catchup_tokens=catchup,
            route_keys=tuple(
                DraftRouteKey(
                    schema=DRAFT_ROUTE_SCHEMA,
                    execution_mode=DraftExecutionMode.EAGER_DYNAMIC,
                    batch_bucket=len(seqs),
                    effective_k=k,
                    catchup_family=family,
                    sampler_envelope=DraftSamplerEnvelope.EXACT_WORST_CASE,
                )
                for k in range(1, 3)
            ),
        )

    def call(self, method, *args):
        self.order.append(method)
        if method == "run_speculative_discard":
            if self.draft_error is not None:
                raise self.draft_error
            return self.result
        if method == "run":
            if self.target_error is not None:
                raise self.target_error
            return [8] * len(args[0])
        raise AssertionError(method)


def _engine(seqs, plan, result, *, speculation_enabled=True):
    order = []
    engine = object.__new__(LLMEngine)
    engine.scheduler = _Scheduler(seqs, plan, order)
    engine.model_runner = _Runner(
        result,
        order,
        speculation_enabled=speculation_enabled,
    )
    engine._admission_limits = AdmissionLimits(None, VOCAB_SIZE, None, None)
    return engine, order


def test_v3_step_orders_discard_transaction_before_authoritative_target():
    seqs = [_sequence(1), _sequence(2)]
    plan = _plan(seqs)
    engine, order = _engine(seqs, plan, _result(plan))

    output = engine._step()

    assert order == [
        "schedule",
        ("plan", 2),
        "run_speculative_discard",
        "handoff",
        "stage_coverage",
        "run",
        "postprocess",
    ]
    assert output.num_prefill_tokens == 0
    assert output.num_decode_tokens == 2
    assert [event.token_id for event in output.events] == [8, 8]
    assert engine.scheduler.staged == {1: 4, 2: 4}
    assert [seq.num_draft_cached_tokens for seq in seqs] == [4, 4]
    assert [seq.num_tokens for seq in seqs] == [5, 5]


def test_v3_fallback_plan_runs_only_the_ordinary_target_path():
    seqs = [_sequence(1)]
    plan = _plan(seqs, effective_k=0, fallback_reason="request_tail")
    engine, order = _engine(seqs, plan, None)

    engine._step()

    assert order == [
        "schedule",
        ("plan", 2),
        "run",
        "postprocess",
    ]


def test_route_miss_falls_back_before_draft_execution():
    seqs = [_sequence(1)]
    plan = _plan(
        seqs,
        effective_k=0,
        fallback_reason="workspace_route_cap",
    )
    engine, order = _engine(seqs, plan, None)
    engine.model_runner.route_miss = True

    engine._step()

    assert order == [
        "schedule",
        ("plan", 0),
        "run",
        "postprocess",
    ]
    assert seqs[0].num_draft_cached_tokens == 0


def test_speculation_off_does_not_touch_the_discard_planner():
    seqs = [_sequence(1)]
    plan = _plan(seqs, effective_k=0, fallback_reason="speculation_disabled")
    engine, order = _engine(
        seqs,
        plan,
        None,
        speculation_enabled=False,
    )

    engine._step()

    assert order == ["schedule", "run", "postprocess"]


def test_draft_failure_rolls_back_and_never_calls_the_target():
    class DraftFailure(RuntimeError):
        pass

    seqs = [_sequence(1)]
    plan = _plan(seqs)
    engine, order = _engine(seqs, plan, _result(plan))
    error = DraftFailure("injected draft failure")
    engine.model_runner.draft_error = error

    with pytest.raises(DraftFailure, match="injected draft failure") as exc_info:
        engine._step()

    assert exc_info.value is error
    assert order == [
        "schedule",
        ("plan", 2),
        "run_speculative_discard",
        "rollback",
        "abort_coverage",
        "rollback_schedule",
    ]
    assert seqs[0].num_tokens == 4
    assert seqs[0].num_draft_cached_tokens == 0


def test_malformed_draft_result_is_rejected_after_rollback_before_target():
    seqs = [_sequence(1)]
    plan = _plan(seqs)
    malformed = _result(plan, q_dtype="torch.bfloat16")
    engine, order = _engine(seqs, plan, malformed)

    with pytest.raises(
        SpeculativeDiscardResultError,
        match="canonical FP32",
    ):
        engine._step()

    assert order == [
        "schedule",
        ("plan", 2),
        "run_speculative_discard",
        "rollback",
        "abort_coverage",
        "rollback_schedule",
    ]
    assert seqs[0].num_tokens == 4
    assert seqs[0].num_draft_cached_tokens == 0


def test_target_failure_after_discard_does_not_advance_draft_coverage():
    class TargetFailure(RuntimeError):
        pass

    seqs = [_sequence(1)]
    plan = _plan(seqs)
    engine, order = _engine(seqs, plan, _result(plan))
    error = TargetFailure("injected target failure")
    engine.model_runner.target_error = error

    with pytest.raises(TargetFailure, match="injected target failure") as exc_info:
        engine._step()

    assert exc_info.value is error
    assert order == [
        "schedule",
        ("plan", 2),
        "run_speculative_discard",
        "handoff",
        "stage_coverage",
        "run",
        "rollback",
        "abort_coverage",
        "rollback_schedule",
    ]
    assert seqs[0].num_tokens == 4
    assert seqs[0].num_draft_cached_tokens == 0


def test_planner_failure_rolls_back_schedule_without_a_draft_plan():
    class PlannerFailure(RuntimeError):
        pass

    seqs = [_sequence(1)]
    plan = _plan(seqs)
    engine, order = _engine(seqs, plan, _result(plan))
    error = PlannerFailure("injected planner failure")
    engine.scheduler.plan_error = error

    with pytest.raises(PlannerFailure, match="injected planner failure") as exc_info:
        engine._step()

    assert exc_info.value is error
    assert order == ["schedule", ("plan", 2), "rollback_schedule"]
    assert seqs[0].num_scheduled_tokens == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("draft_positions", 1, "position count"),
        ("catchup_positions", 0, "catch-up work"),
        ("q_view_zero_copy", False, "direct retained-q"),
        ("graph_decode_steps", 3, "step counts"),
    ],
)
def test_draft_result_semantic_diagnostics_fail_closed(field, value, message):
    seqs = [_sequence(1)]
    plan = _plan(seqs)
    result = _result(plan, **{field: value})

    with pytest.raises(SpeculativeDiscardResultError, match=message):
        LLMEngine._validate_draft_discard_result(
            plan,
            result,
            seqs,
            VOCAB_SIZE,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("effective_k", True, "effective_k"),
        ("draft_positions", True, "position count"),
        ("catchup_positions", True, "catch-up work"),
        ("q_shape", (True, 2, VOCAB_SIZE), "retained-q shape"),
        ("q_stride", (999,), "retained-q stride"),
    ],
)
def test_draft_result_rejects_bool_scalars_and_wrong_stride(
    field, value, message
):
    seqs = [_sequence(1)]
    plan = _plan(seqs)
    with pytest.raises(SpeculativeDiscardResultError, match=message):
        LLMEngine._validate_draft_discard_result(
            plan,
            _result(plan, **{field: value}),
            seqs,
            VOCAB_SIZE,
        )


def test_cancel_waits_until_inflight_step_releases_execution_lock():
    entered = Event()
    release = Event()
    cancelled = Event()
    engine = object.__new__(LLMEngine)
    engine._execution_lock = RLock()
    engine.scheduler = SimpleNamespace(
        cancel=lambda seq_ids: cancelled.set() or list(seq_ids)
    )

    def blocked_step(self):
        entered.set()
        assert release.wait(timeout=5)
        return "done"

    engine._step_unlocked = MethodType(blocked_step, engine)
    step_thread = Thread(target=engine._step)
    cancel_thread = Thread(target=lambda: engine._cancel_requests([7]))
    step_thread.start()
    assert entered.wait(timeout=5)
    cancel_thread.start()
    assert not cancelled.wait(timeout=0.1)
    release.set()
    step_thread.join(timeout=5)
    cancel_thread.join(timeout=5)

    assert not step_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert cancelled.is_set()
