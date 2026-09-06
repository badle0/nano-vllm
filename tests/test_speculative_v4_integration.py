"""V4 CPU integration: real allocator/scheduler/engine, mocked model compute.

These checks do not certify CUDA peaks, attention numerics, or V5 acceptance.
The V3 CPU model fixture deliberately consumes RNG inside draft sampling.
"""

from types import SimpleNamespace

import pytest
import torch

import nanovllm.engine.model_runner as runner_module
from nanovllm.engine.llm_engine import AdmissionLimits, LLMEngine
from nanovllm.engine.model_runner import SpeculativeDraftPlanError
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.speculative_memory import plan_speculative_workspace, speculative_route_fits_plan
from nanovllm.engine.speculative_routes import build_draft_route_registry
from nanovllm.sampling_params import SamplingParams
from test_speculative_v3_runner import _runner, _install_cpu_rng_neutral_seam


def _system(monkeypatch, *, batch=1, k=4, budget=64, blocks=16, remaining=20):
    monkeypatch.setattr(Sequence, "block_size", 4)
    runner = _runner(monkeypatch, k=k, vocab_size=64)
    # Restore the real memory-fit predicate: V4 must not inherit V3's fixture
    # shortcut, which intentionally bypasses that independent subsystem.
    monkeypatch.setattr(runner_module, "speculative_route_fits_plan", speculative_route_fits_plan)
    runner.config.max_num_batched_tokens = budget
    runner.kv_cache = torch.empty(2, 1, blocks, 4, 1, 1)
    runner.draft_kv_cache = torch.empty_like(runner.kv_cache)
    phases = []
    _install_cpu_rng_neutral_seam(runner, phases)
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=8, max_num_batched_tokens=budget, max_model_len=32,
        configured_k=k, eos=63, kvcache_block_size=4,
        num_kvcache_blocks=blocks,
    ))
    seqs = []
    for index in range(batch):
        seq = Sequence([index + 1] * 4, SamplingParams(
            max_tokens=remaining + 1, ignore_eos=True,
        ))
        scheduler.block_manager.allocate(seq, 0)
        seq.append_token(5)
        seq.num_cached_tokens = 4
        seq.num_draft_cached_tokens = 2
        seq.num_scheduled_tokens = 0
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        scheduler.running.append(seq)
        seqs.append(seq)
    engine = object.__new__(LLMEngine)
    engine.scheduler = scheduler
    engine.model_runner = runner
    engine._admission_limits = AdmissionLimits(None, 64, None, None)
    engine._resolve_draft_route_admission = lambda selected: runner.draft_route_registry.resolve(
        batch_size=len(selected),
        catchup_tokens=sum(len(seq) - 1 - seq.num_draft_cached_tokens for seq in selected),
    )
    seen = []

    def call(method, *args):
        seen.append(method)
        if method == "run_speculative_discard":
            assert scheduler._active_spec_transaction is not None
            return runner.run_speculative_discard(*args)
        assert method == "run"
        # Handoff releases the extra verifier suffix, not the undo record.
        assert scheduler._active_draft_discard is None
        selected, _ = args
        return torch.randint(0, 63, (len(selected),)).tolist()

    runner.call = call
    return engine, seqs, phases, seen


def _plan(engine, seqs):
    selected, prefill = engine.scheduler.schedule()
    assert selected == seqs and not prefill
    return engine.scheduler.plan_speculative_step(
        selected, configured_workspace=engine.model_runner.speculative_memory_plan,
        route_admission=engine._resolve_draft_route_admission(selected),
    )


def _snapshot(engine, seqs):
    manager = engine.scheduler.block_manager
    return (
        tuple((tuple(seq.token_ids), seq.num_cached_tokens,
               seq.num_draft_cached_tokens, tuple(seq.block_table)) for seq in seqs),
        frozenset(manager.used_block_ids),
        tuple(manager.free_block_ids),
        tuple(sorted(manager.hash_to_block_id.items())),
        tuple((block.ref_count, block.hash, tuple(block.token_ids)) for block in manager.blocks),
    )


def test_full_geometry_reservation_survives_handoff_until_target_commit(monkeypatch):
    engine, seqs, _, _ = _system(monkeypatch)
    scheduler = engine.scheduler
    plan = _plan(engine, seqs)
    assert plan.effective_k == 4
    assert plan.draft_query_tokens == 4
    assert plan.target_query_tokens == 5
    assert plan.total_model_positions == 11  # catch-up 2 + draft 4 + target 5
    assert plan.total_scheduled_tokens == 7  # actual shadow work only
    assert plan.rows[0].highest_target_write_position == 8
    assert len(seqs[0].block_table) == 3
    assert plan.gpu_certified is False
    scheduler.handoff_draft_discard(plan)
    assert len(seqs[0].block_table) == 2
    assert scheduler._active_spec_transaction.plan is plan
    with pytest.raises(RuntimeError, match="transaction"):
        scheduler.schedule()
    scheduler.stage_draft_coverage(plan, seqs, {seqs[0].seq_id: 5})
    events = scheduler.postprocess(seqs, [9])
    assert len(events) == 1
    assert seqs[0].num_cached_tokens == seqs[0].num_draft_cached_tokens == 5
    assert scheduler._active_spec_transaction is None


@pytest.mark.parametrize("handoff", [False, True])
def test_abort_restores_both_block_reservation_layers(monkeypatch, handoff):
    engine, seqs, _, _ = _system(monkeypatch, batch=2)
    before = _snapshot(engine, seqs)
    plan = _plan(engine, seqs)
    if handoff:
        engine.scheduler.handoff_draft_discard(plan)
        engine.scheduler.stage_draft_coverage(plan, seqs, {s.seq_id: len(s) for s in seqs})
    assert engine.scheduler.abort_speculative_step()
    assert _snapshot(engine, seqs) == before
    assert all(seq.num_scheduled_tokens == 0 for seq in seqs)
    assert not engine.scheduler.abort_speculative_step()


@pytest.mark.parametrize("budget", range(2, 30))
def test_budget_admission_matches_enumerated_feasible_candidates(monkeypatch, budget):
    engine, seqs, _, _ = _system(monkeypatch, batch=2, budget=budget)
    plan = _plan(engine, seqs)
    feasible = [k for k in range(1, 5)
                if 2 * (k + 1) <= budget and 4 + 2 * k + 2 * (k + 1) <= budget]
    assert plan.effective_k == max(feasible, default=0)
    if plan.uses_speculation:
        engine.scheduler.abort_speculative_step()
    else:
        assert engine.scheduler._active_spec_transaction is None
        assert len(seqs[0].block_table) == 2


def test_insufficient_suffix_capacity_falls_back_without_partial_allocation(monkeypatch):
    engine, seqs, _, _ = _system(monkeypatch, batch=2, blocks=5)
    plan = _plan(engine, seqs)
    assert plan.bypass_reason == "insufficient_kv_blocks"
    assert plan.effective_k == 0
    assert len(engine.scheduler.block_manager.used_block_ids) == 4
    assert [len(seq.block_table) for seq in seqs] == [2, 2]
    assert engine.scheduler._active_spec_transaction is None


@pytest.mark.parametrize("point", ["route", "reserve", "draft", "result", "handoff", "stage", "target", "target_result"])
def test_engine_failure_undoes_admission_and_allows_retry(monkeypatch, point):
    engine, seqs, _, _ = _system(monkeypatch, batch=2)
    before = _snapshot(engine, seqs)
    original_call = engine.model_runner.call

    def fail(*args, **kwargs):
        raise RuntimeError("injected V4 failure")

    with monkeypatch.context() as fault:
        if point == "route":
            fault.setattr(engine, "_resolve_draft_route_admission", fail)
        elif point == "reserve":
            fault.setattr(engine.scheduler.block_manager, "reserve_temporary_append", fail)
        elif point in ("handoff", "stage"):
            name = "handoff_draft_discard" if point == "handoff" else "stage_draft_coverage"
            fault.setattr(engine.scheduler, name, fail)
        else:
            def call(method, *args):
                if (point == "draft" and method == "run_speculative_discard"
                        or point == "target" and method == "run"):
                    fail()
                result = original_call(method, *args)
                if point == "result" and method == "run_speculative_discard":
                    return result._replace(effective_k=result.effective_k + 1)
                if point == "target_result" and method == "run":
                    return []
                return result
            fault.setattr(engine.model_runner, "call", call)
        with pytest.raises((RuntimeError, ValueError)):
            engine._step_unlocked()
    assert _snapshot(engine, seqs) == before
    assert engine.scheduler._active_spec_transaction is None
    assert engine.scheduler._pending_draft_coverage_plan is None
    assert all(seq.num_scheduled_tokens == 0 for seq in seqs)
    output = engine._step_unlocked()
    assert len(output.events) == 2
    assert all(len(seq) == 6 for seq in seqs)


def test_shadow_and_disabled_paths_preserve_target_tokens_and_rng(monkeypatch):
    def run(enabled):
        engine, seqs, phases, seen = _system(monkeypatch, batch=2)
        engine.model_runner.speculation_enabled = enabled
        torch.manual_seed(1203)
        events = []
        for _ in range(3):
            output = engine._step_unlocked()
            events.extend((event.token_id, event.finished) for event in output.events)
        return events, torch.random.get_rng_state(), phases, seen

    off_events, off_rng, off_phases, _ = run(False)
    on_events, on_rng, on_phases, on_seen = run(True)
    assert on_events == off_events
    assert torch.equal(on_rng, off_rng)
    assert not off_phases and len(on_phases) == 3
    assert on_seen.count("run_speculative_discard") == 3


@pytest.mark.parametrize("field,value", [
    ("total_model_positions", 0), ("workspace_fingerprint", "0" * 64),
    ("gpu_certified", True), ("target_query_tokens", 1),
])
def test_runner_rejects_forged_certificate_before_model_or_rng(monkeypatch, field, value):
    engine, seqs, phases, _ = _system(monkeypatch)
    plan = _plan(engine, seqs)
    object.__setattr__(plan, field, value)
    with pytest.raises(SpeculativeDraftPlanError):
        engine.model_runner.run_speculative_discard(plan, seqs)
    assert not phases
    assert not engine.model_runner.draft_model.calls
    engine.scheduler.abort_speculative_step()


def test_runner_rejects_missing_target_capacity_before_draft_compute(monkeypatch):
    engine, seqs, phases, _ = _system(monkeypatch)
    plan = _plan(engine, seqs)
    engine.model_runner.kv_cache = engine.model_runner.kv_cache[:, :, :2]
    with pytest.raises(SpeculativeDraftPlanError):
        engine.model_runner.run_speculative_discard(plan, seqs)
    assert not phases and not engine.model_runner.draft_model.calls
    engine.scheduler.abort_speculative_step()


@pytest.mark.parametrize("block_size", [4, 256])
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("remaining", [1, 3, 8])
def test_block_boundary_request_and_position_sweep(monkeypatch, block_size, offset, batch, k, remaining):
    monkeypatch.setattr(Sequence, "block_size", block_size)
    length = block_size + offset
    model_limit = length + 2
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=8, max_num_batched_tokens=1024, max_model_len=model_limit,
        configured_k=k, eos=-1, kvcache_block_size=block_size,
        num_kvcache_blocks=32,
    ))
    workspace = plan_speculative_workspace(
        vocab_size=64, configured_k=k, max_num_seqs=8,
        max_num_batched_tokens=1024, max_model_len=model_limit,
        target_logits_dtype=torch.float32, draft_logits_dtype=torch.float32,
    )
    registry = build_draft_route_registry(workspace, enforce_eager=True)
    registry = registry.with_warmed_components(registry.desired_warm_components)
    seqs = []
    for index in range(batch):
        seq = Sequence([index + 1] * (length - 1), SamplingParams(max_tokens=remaining + 1))
        scheduler.block_manager.allocate(seq, 0)
        seq.append_token(5)
        seq.num_cached_tokens = seq.num_draft_cached_tokens = length - 1
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        scheduler.running.append(seq)
        seqs.append(seq)
    before = tuple(tuple(seq.block_table) for seq in seqs)
    selected, prefill = scheduler.schedule()
    assert not prefill and selected == seqs
    plan = scheduler.plan_speculative_step(
        selected, configured_workspace=workspace,
        route_admission=registry.resolve(batch_size=batch, catchup_tokens=0),
    )
    feasible = [candidate for candidate in range(1, k + 1)
                if candidate + 1 <= remaining and length + candidate - 1 < model_limit]
    assert plan.effective_k == max(feasible, default=0)
    if plan.uses_speculation:
        for row, seq in zip(plan.rows, seqs, strict=True):
            assert row.highest_target_write_position < model_limit
            assert len(seq.block_table) == (length + plan.effective_k + block_size - 1) // block_size
        scheduler.abort_speculative_step()
        assert tuple(tuple(seq.block_table) for seq in seqs) == before


def test_abort_failure_keeps_transaction_owned_and_retryable(monkeypatch):
    engine, seqs, _, _ = _system(monkeypatch)
    before = _snapshot(engine, seqs)
    plan = _plan(engine, seqs)
    with monkeypatch.context() as fault:
        def fail(*args):
            raise RuntimeError("injected release failure")
        fault.setattr(engine.scheduler.block_manager, "rollback_temporary_append", fail)
        with pytest.raises(RuntimeError, match="release failure"):
            engine.scheduler.abort_speculative_step()
        assert engine.scheduler._active_spec_transaction.plan is plan
        with pytest.raises(RuntimeError, match="transaction"):
            engine.scheduler.schedule()
    assert engine.scheduler.abort_speculative_step()
    assert _snapshot(engine, seqs) == before


@pytest.mark.parametrize("handoff", [False, True])
def test_targeted_cancel_aborts_batch_lease_without_cancelling_other_rows(monkeypatch, handoff):
    engine, seqs, _, _ = _system(monkeypatch, batch=2)
    plan = _plan(engine, seqs)
    if handoff:
        engine.scheduler.handoff_draft_discard(plan)
    assert engine.scheduler.cancel([seqs[0].seq_id]) == [seqs[0].seq_id]
    assert seqs[0].status is SequenceStatus.CANCELLED
    assert list(engine.scheduler.running) == [seqs[1]]
    assert seqs[1].num_cached_tokens == 4 and len(seqs[1].block_table) == 1
    assert engine.scheduler._active_spec_transaction is None
    assert len(engine._step_unlocked().events) == 1


def test_repeated_shadow_boundaries_release_all_capacity_on_completion(monkeypatch):
    engine, seqs, _, _ = _system(monkeypatch, batch=2, remaining=20)
    for _ in range(20):
        # This fixture emits call_index+1, so reset its diagnostics to keep its
        # synthetic tokens in-vocabulary during a long lifecycle test.
        engine.model_runner.sampler.calls.clear()
        output = engine._step_unlocked()
        assert len(output.events) == 2
        assert engine.scheduler._active_spec_transaction is None
        assert engine.scheduler._active_draft_discard is None
    assert all(seq.is_finished for seq in seqs)
    assert not engine.scheduler.block_manager.used_block_ids
    assert len(engine.scheduler.block_manager.free_block_ids) == 16


def test_mixed_prefill_decode_keeps_baseline_order_and_skips_speculation(monkeypatch):
    engine, seqs, phases, seen = _system(monkeypatch)
    waiting = Sequence([6] * 3, SamplingParams(max_tokens=4))
    engine.scheduler.add(waiting)
    output = engine._step_unlocked()
    assert [event.seq_id for event in output.events] == [seqs[0].seq_id, waiting.seq_id]
    assert output.num_decode_tokens == 1 and output.num_prefill_tokens == 3
    assert not phases and seen == ["run"]
