"""Independent commit oracle and fault injection for real speculative bursts."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.speculative_result import SpecExecutionResult, SpecRowResult
from nanovllm.engine.speculative_execution import target_probabilities
from nanovllm.engine.model_runner import DraftCycleRow
from nanovllm.layers.sampler import Sampler, ModifiedRejectionSampler
from nanovllm.utils.context import get_context
from test_speculative_v4_integration import _system, _plan, _snapshot


def result_for(plan, accepted, *, tokens=None):
    proposals = (11, 12, 13, 14)[:plan.effective_k]
    return SpecExecutionResult(
        plan.cycle_id, plan.workspace_fingerprint,
        tuple(SpecRowResult(row.seq_id, proposals,
                            tokens if tokens is not None else proposals[:accepted] + (20,),
                            accepted, accepted == plan.effective_k)
              for row in plan.rows),
        plan.target_query_tokens, plan.draft_catchup_tokens + plan.draft_query_tokens,
    )


def _mark_verifier_ready(runner, shapes):
    from nanovllm.engine.speculative_execution import (
        _verifier_readiness_fingerprint,
    )

    plan = runner.speculative_memory_plan
    batch = min(4, plan.batch_size)
    k = min(4, plan.max_effective_k)
    vocab = plan.vocab_size
    device = runner.kv_cache.device
    runner._spec_q_rows = torch.empty(
        batch * k, vocab, dtype=torch.float32, device=device
    )
    runner._spec_proposal_ids = torch.empty(
        batch * k, dtype=torch.int64, device=device
    )
    runner._spec_target_probability_rows = torch.empty(
        batch * (k + 1), vocab, dtype=torch.float32, device=device
    )
    runner._spec_bonus_noise = torch.empty(
        batch, vocab, dtype=torch.float32, device=device
    )
    runner._spec_result_rows = torch.empty(
        batch, k + 3, dtype=torch.int64, device=device
    )
    runner.speculative_verifier_shapes = frozenset(shapes)
    runner.speculative_verifier_fingerprint = (
        _verifier_readiness_fingerprint(runner)
    )
    runner.speculative_verifier_ready = True


@pytest.mark.parametrize("accepted", range(5))
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_commit_oracle(monkeypatch, accepted, batch):
    engine, seqs, _, _ = _system(monkeypatch, batch=batch)
    plan = _plan(engine, seqs)
    engine.scheduler.prepare_speculative_target_writes(plan)
    before = [s.token_ids[:] for s in seqs]
    result = result_for(plan, accepted)
    events = engine.scheduler.commit_speculative(plan, result)
    assert len(events) == batch * (accepted + 1)
    manager = engine.scheduler.block_manager
    for seq, prefix, row in zip(seqs, before, result.rows):
        assert seq.token_ids == prefix + list(row.committed_token_ids)
        assert seq.num_cached_tokens == len(seq) - 1
        assert seq.num_draft_cached_tokens == min(len(seq) - 1, len(prefix) + plan.effective_k - 1)
        assert len(seq.block_table) == (seq.num_cached_tokens + 3) // 4
        assert seq.token_times[-(accepted + 1):] == [seq.token_times[-1]] * (accepted + 1)
        assert seq.spec_metrics["spec_committed_tokens"] == accepted + 1
        # Every newly full target block is hashed, and only actual tokens enter it.
        previous = -1
        for i in range(seq.num_cached_tokens // 4):
            block = manager.blocks[seq.block_table[i]]
            expected = manager.compute_hash(seq.block(i), previous)
            # Fixture's original prefix block is not populated by real prefill.
            if i:
                assert block.token_ids == seq.block(i)
                assert block.hash == expected
            previous = block.hash
    assert engine.scheduler._active_spec_transaction is None
    assert not manager._active_temporary_reservations


@pytest.mark.parametrize("ignore", [False, True])
@pytest.mark.parametrize("eos_index", range(5))
def test_eos_burst_truncation(monkeypatch, ignore, eos_index):
    engine, seqs, _, _ = _system(monkeypatch)
    seq = seqs[0]
    seq.ignore_eos = ignore
    plan = _plan(engine, seqs)
    engine.scheduler.prepare_speculative_target_writes(plan)
    tokens = list((11, 12, 13, 14, 20))
    tokens[eos_index] = 63
    row = SpecRowResult(seq.seq_id, tuple(tokens[:4]), tuple(tokens), 4, True)
    result = replace(result_for(plan, 4), rows=(row,))
    events = engine.scheduler.commit_speculative(plan, result)
    assert len(events) == (5 if ignore else eos_index + 1)
    assert seq.is_finished == (not ignore)
    assert [event.finished for event in events] == [False] * (len(events) - 1) + [not ignore]
    if not ignore:
        assert not seq.block_table
        assert not engine.scheduler.running


@pytest.mark.parametrize("failure", ["append", "hash", "trim", "events"])
def test_commit_failure_restores_state_and_retry(monkeypatch, failure):
    engine, seqs, _, _ = _system(monkeypatch, batch=2)
    before = _snapshot(engine, seqs)
    plan = _plan(engine, seqs)
    engine.scheduler.prepare_speculative_target_writes(plan)
    manager = engine.scheduler.block_manager
    import nanovllm.engine.scheduler as module
    target, name = {"append": (seqs[1], "append_token"),
                    "hash": (manager, "hash_blocks"),
                    "trim": (manager, "finalize_temporary_append"),
                    "events": (module, "StreamOutput")}[failure]
    original = getattr(target, name)
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected commit failure")
    with monkeypatch.context() as patch:
        patch.setattr(target, name, fail)
        with pytest.raises(RuntimeError, match="injected"):
            engine.scheduler.commit_speculative(plan, result_for(plan, 4))
    engine.scheduler.abort_speculative_step()
    assert _snapshot(engine, seqs) == before
    plan = _plan(engine, seqs)
    engine.scheduler.prepare_speculative_target_writes(plan)
    assert len(engine.scheduler.commit_speculative(plan, result_for(plan, 4))) == 10


@pytest.mark.parametrize("field,value", [("cycle_id", -1), ("target_verification_positions", 1),
                                        ("workspace_fingerprint", "stale"), ("draft_positions", 0)])
def test_reject_stale_result_before_commit(monkeypatch, field, value):
    engine, seqs, _, _ = _system(monkeypatch)
    plan = _plan(engine, seqs)
    engine.scheduler.prepare_speculative_target_writes(plan)
    before = _snapshot(engine, seqs)
    with pytest.raises(ValueError):
        engine.scheduler.commit_speculative(plan, replace(result_for(plan, 4), **{field: value}))
    assert _snapshot(engine, seqs) == before
    engine.scheduler.abort_speculative_step()


def test_verifier_row_geometry_and_all_logits(monkeypatch):
    engine, _, _, _ = _system(monkeypatch)
    runner = engine.model_runner
    runner.sampler = Sampler()
    seen = {}
    class Model:
        def __call__(self, inputs, positions):
            context = get_context()
            seen.update(inputs=inputs.tolist(), positions=positions.tolist(),
                        q=context.cu_seqlens_q.tolist(), k=context.cu_seqlens_k.tolist(),
                        slots=context.slot_mapping.tolist(), maxq=context.max_seqlen_q,
                        maxk=context.max_seqlen_k)
            return torch.arange(inputs.numel() * 64, dtype=torch.float32).reshape(-1, 64)
        def compute_logits_all(self, hidden):
            return hidden
    runner.model = Model()
    rows = (DraftCycleRow(1, (1, 2, 3, 4, 5), 5, 4, 4, (2, 3), 1., -1, 1.),
            DraftCycleRow(2, (6, 7, 8), 3, 2, 2, (4, 5), 0., -1, 1.))
    p = target_probabilities(runner, rows, torch.tensor([[9, 10], [11, 12]]))
    assert p.shape == (2, 3, 64)
    assert seen == dict(inputs=[5, 9, 10, 8, 11, 12], positions=[4, 5, 6, 2, 3, 4],
                        q=[0, 3, 6], k=[0, 7, 12], slots=[12, 13, 14, 18, 19, 20], maxq=3, maxk=7)
    torch.testing.assert_close(p[0, 0], torch.arange(64, dtype=torch.float32).softmax(0))
    assert p[1, :, 63].tolist() == [1, 1, 1]
    assert not get_context().is_prefill


def test_engine_uses_verified_burst_and_consumes_rng(monkeypatch):
    engine, seqs, _, _ = _system(monkeypatch)
    runner = engine.model_runner
    _mark_verifier_ready(runner, ((1, k) for k in range(1, 5)))
    def call(method, plan, selected):
        assert method == "run_speculative"
        torch.rand(1)
        return result_for(plan, 4)
    runner.call = call
    before_rng = torch.get_rng_state().clone()
    step = engine._step()
    assert len(step.events) == 5
    assert step.num_decode_tokens == 1  # rows, not burst size
    assert not torch.equal(before_rng, torch.get_rng_state())


def test_engine_failure_restores_rng_and_schedule(monkeypatch):
    engine, seqs, _, _ = _system(monkeypatch)
    runner = engine.model_runner
    _mark_verifier_ready(runner, ((1, k) for k in range(1, 5)))
    def call(*args):
        torch.rand(10)
        raise RuntimeError("injected verify failure")
    runner.call = call
    before, rng = _snapshot(engine, seqs), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="injected verify"):
        engine._step()
    assert _snapshot(engine, seqs) == before
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("mode", ["plain", "temperature", "topk", "topp", "combined"])
def test_end_to_end_sampling_law(monkeypatch, mode):
    """Real scheduler->draft->target->accept->commit, independent categorical law.

    Compare complete fixed-length continuations, not accepted-only tokens (which
    would introduce selection bias). The toy model is context-independent.
    """
    torch.manual_seed(817)
    histogram = torch.zeros(64, dtype=torch.int64)
    logits = torch.full((64,), -30.)
    logits[:4] = torch.tensor([1.5, 1., .5, 0.])
    draft_logits = logits.flip(0).clone()
    draft_logits[:4] = torch.tensor([.4, 1.2, .9, .3])
    temperature = .7 if mode == "temperature" else 1.
    topk = 3 if mode in ("topk", "combined") else -1
    topp = .8 if mode in ("topp", "combined") else 1.
    expected_logits = logits.double() / temperature
    # Independent descending-prefix nucleus oracle, including crossing token.
    if topk > 0:
        expected_logits[expected_logits < expected_logits.topk(topk).values[-1]] = -torch.inf
    expected = expected_logits.softmax(0)
    if topp < 1:
        sorted_p, ids = expected.sort(descending=True)
        keep = sorted_p.cumsum(0) - sorted_p < topp
        expected[ids[~keep]] = 0
        expected /= expected.sum()
    class Model:
        def __init__(self, values):
            self.values = values
        def __call__(self, inputs, positions):
            return self.values.expand(inputs.numel(), -1).clone()
        def compute_logits(self, hidden):
            context = get_context()
            return hidden[context.cu_seqlens_q[1:].long() - 1] if context.is_prefill else hidden
        def compute_logits_all(self, hidden):
            return hidden
    for _ in range(90):
        engine, seqs, _, _ = _system(monkeypatch, batch=4, k=2, remaining=12)
        runner = engine.model_runner
        runner.model, runner.draft_model = Model(logits), Model(draft_logits)
        runner.sampler = Sampler()
        runner.speculative_rejection_sampler = ModifiedRejectionSampler()
        _mark_verifier_ready(
            runner, ((b, k) for b in range(1, 5) for k in (1, 2))
        )
        for seq in seqs:
            seq.temperature, seq.top_k, seq.top_p = temperature, topk, topp
        def call(method, *args):
            if method == "run_speculative":
                return runner.run_speculative(*args)
            selected, _ = args
            temps, buckets, nucleus = runner._prepare_draft_sample_metadata(tuple(selected))
            return runner.sampler.sample_exact_with_probabilities(
                logits.expand(len(selected), -1), temps, top_k_buckets=buckets, top_p_plan=nucleus,
            ).token_ids.tolist()
        runner.call = call
        while not engine.scheduler.is_finished():
            output = engine._step()
            histogram += torch.bincount(torch.tensor([event.token_id for event in output.events]), minlength=64)
        assert not engine.scheduler.block_manager.used_block_ids
    actual = histogram.double() / histogram.sum()
    assert histogram.sum() == 90 * 4 * 12
    assert .5 * (actual - expected).abs().sum() < .035


@pytest.mark.parametrize("accepted", range(4))
@pytest.mark.parametrize("ignore", [False, True])
def test_rejected_eos_does_not_finish(monkeypatch, accepted, ignore):
    engine, seqs, _, _ = _system(monkeypatch)
    seq = seqs[0]
    seq.ignore_eos = ignore
    plan = _plan(engine, seqs)
    engine.scheduler.prepare_speculative_target_writes(plan)
    result = result_for(plan, accepted)
    row = result.rows[0]
    proposed = list(row.proposed_token_ids)
    proposed[accepted] = 63
    result = replace(result, rows=(replace(row, proposed_token_ids=tuple(proposed)),))
    events = engine.scheduler.commit_speculative(plan, result)
    assert not seq.is_finished
    assert all(event.token_id != 63 and not event.finished for event in events)


@pytest.mark.parametrize("remaining", range(1, 7))
def test_remaining_budget_routes_bonus_or_baseline(monkeypatch, remaining):
    engine, seqs, _, _ = _system(monkeypatch, remaining=remaining)
    plan = _plan(engine, seqs)
    assert plan.effective_k == min(4, remaining - 1)
    if remaining == 1:
        assert not plan.uses_speculation
        events = engine.scheduler.postprocess(seqs, [20])
    else:
        engine.scheduler.prepare_speculative_target_writes(plan)
        events = engine.scheduler.commit_speculative(plan, result_for(plan, plan.effective_k))
    assert len(events) == min(5, remaining)
    assert seqs[0].is_finished == (remaining <= 5)


@pytest.mark.parametrize("abort", [False, True])
def test_overwritten_free_cache_hash_never_resurrected(monkeypatch, abort):
    engine, seqs, _, _ = _system(monkeypatch)
    manager = engine.scheduler.block_manager
    recycled = list(manager.free_block_ids)[1]  # first free block is ordinary append
    old_hash = 123456
    manager.blocks[recycled].update(old_hash, [51] * 4)
    manager.hash_to_block_id[old_hash] = recycled
    plan = _plan(engine, seqs)
    assert recycled in seqs[0].block_table
    engine.scheduler.prepare_speculative_target_writes(plan)
    if abort:
        engine.scheduler.abort_speculative_step()
    else:
        engine.scheduler.commit_speculative(plan, result_for(plan, 0))
    assert recycled in manager.free_block_ids
    assert old_hash not in manager.hash_to_block_id
    assert manager.blocks[recycled].hash == -1
    assert not manager.blocks[recycled].token_ids


def test_forced_empty_residual_counts_once_in_real_commit(monkeypatch):
    from nanovllm.layers.sampler import ModifiedRejectionResult
    engine, seqs, _, _ = _system(monkeypatch)
    runner = engine.model_runner
    _mark_verifier_ready(runner, ((1, k) for k in range(1, 5)))
    sampler = ModifiedRejectionSampler()
    # Force the analytically unreachable rejection seam explicitly; use the
    # REAL FP64 residual implementation and stochastic target fallback.
    def call(method, plan, selected):
        p = torch.zeros(1, 64)
        p[:, 9:11] = .5
        correction = sampler.sample_correction(p, p)
        assert correction.target_fallback.tolist() == [True]
        assert correction.token_ids.item() in (9, 10)
        row = SpecRowResult(selected[0].seq_id, (11, 12, 13, 14),
                            (correction.token_ids.item(),), 0, False, 1)
        return replace(result_for(plan, 0), rows=(row,))
    runner.call = call
    output = engine._step()
    assert len(output.events) == 1
    assert seqs[0].spec_metrics["spec_residual_numerical_fallbacks"] == 1
    assert seqs[0].num_cached_tokens == len(seqs[0]) - 1
    assert engine.scheduler._active_spec_transaction is None
