import gc
import weakref
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch

import nanovllm.engine.model_runner as runner_module
from nanovllm.engine.model_runner import (
    DraftDiscardResult,
    ModelRunner,
    SpeculativeDraftExecutionError,
    SpeculativeDraftPlanError,
)
from nanovllm.engine.scheduler import DraftDiscardPlan, DraftDiscardRow
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.speculative_memory import plan_speculative_workspace
from nanovllm.engine.speculative_routes import build_draft_route_registry
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, set_context


class _DraftModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.calls = []

    def __call__(self, input_ids, positions):
        context = get_context()
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "positions": positions.clone(),
                "is_prefill": context.is_prefill,
                "slot_mapping": (
                    None
                    if context.slot_mapping is None
                    else context.slot_mapping.clone()
                ),
                "context_lens": (
                    None
                    if context.context_lens is None
                    else context.context_lens.clone()
                ),
                "cu_q": (
                    None
                    if context.cu_seqlens_q is None
                    else context.cu_seqlens_q.clone()
                ),
                "cu_k": (
                    None
                    if context.cu_seqlens_k is None
                    else context.cu_seqlens_k.clone()
                ),
            }
        )
        rows = input_ids.numel()
        return torch.arange(
            rows * self.vocab_size, dtype=torch.float32
        ).reshape(rows, self.vocab_size)

    def compute_logits(self, hidden_states):
        return hidden_states


class _Sampler:
    def __init__(self, *, fail_call=None):
        self.calls = []
        self.fail_call = fail_call

    def sample_exact_with_probabilities(
        self,
        logits,
        temperatures,
        *,
        top_k_buckets=(),
        top_p_plan=None,
        probabilities_out=None,
    ):
        call_index = len(self.calls)
        self.calls.append(
            {
                "logits": logits.clone(),
                "temperatures": temperatures.clone(),
                "out": probabilities_out,
                "top_k": top_k_buckets,
                "top_p": top_p_plan,
            }
        )
        # This deliberate draw lets the production wrapper prove target-RNG
        # neutrality even if a future draft sampler consumes randomness.
        torch.rand(1)
        if self.fail_call == call_index:
            raise RuntimeError("injected proposal failure")
        probabilities_out.fill_(1.0 / logits.size(1))
        token_ids = torch.full(
            (logits.size(0),),
            call_index + 1,
            dtype=torch.int64,
            device=logits.device,
        )
        return SimpleNamespace(
            token_ids=token_ids,
            probabilities=probabilities_out,
        )


class _Graph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class _TargetPoison:
    def __getattribute__(self, name):
        raise AssertionError(f"target model must not be used: {name}")


def _sequence(
    token_ids,
    *,
    seq_id,
    draft_cached_tokens,
    block_table=(0, 1, 2),
    temperature=0.0,
):
    seq = Sequence(
        list(token_ids),
        SamplingParams(
            temperature=temperature,
            top_k=-1,
            top_p=1.0,
            max_tokens=32,
        ),
    )
    seq.seq_id = seq_id
    # Make all but the first token completion history so request-headroom
    # validation exercises the live value instead of a prompt-only shortcut.
    seq.num_prompt_tokens = 1
    seq.num_cached_tokens = len(seq) - 1
    seq.num_draft_cached_tokens = draft_cached_tokens
    seq.num_scheduled_tokens = 1
    seq.is_prefill = False
    seq.status = SequenceStatus.RUNNING
    seq.block_table[:] = block_table
    return seq


def _plan(runner, seqs, *, effective_k):
    rows = tuple(
        DraftDiscardRow(
            seq_id=seq.seq_id,
            committed_tokens=len(seq),
            target_cached_tokens=seq.num_cached_tokens,
            draft_cached_tokens=seq.num_draft_cached_tokens,
            remaining_completion_tokens=(
                seq.max_tokens - seq.num_completion_tokens
            ),
            model_position_headroom=(
                runner.config.max_model_len - len(seq)
            ),
            highest_proposal_input_position=(
                len(seq) + effective_k - 2
            ),
            block_table=tuple(seq.block_table),
        )
        for seq in seqs
    )
    batch_size = len(rows)
    catchup = sum(
        len(seq) - 1 - seq.num_draft_cached_tokens for seq in seqs
    )
    admission = runner.draft_route_registry.resolve(
        batch_size=batch_size,
        catchup_tokens=catchup,
    )
    route_key = admission.key_for(effective_k)
    assert route_key is not None
    return DraftDiscardPlan(
        cycle_id=7,
        rows=rows,
        configured_k=runner.config.configured_k,
        workspace_route_cap=runner.speculative_memory_plan.max_effective_k,
        effective_k=effective_k,
        draft_catchup_tokens=catchup,
        draft_step_token_counts=(batch_size,) * effective_k,
        target_query_tokens=batch_size,
        total_scheduled_tokens=catchup + batch_size * (effective_k + 1),
        fallback_reason=None,
        route_key=route_key,
    )


def _runner(monkeypatch, *, k=2, vocab_size=11, eager=True, sampler=None):
    runner = object.__new__(ModelRunner)
    runner.speculation_enabled = True
    runner.block_size = 4
    runner.enforce_eager = eager
    runner.config = SimpleNamespace(
        configured_k=k,
        max_model_len=32,
        max_num_batched_tokens=64,
        draft_hf_config=SimpleNamespace(vocab_size=vocab_size),
    )
    runner.speculative_memory_plan = plan_speculative_workspace(
        vocab_size=vocab_size,
        configured_k=k,
        max_num_seqs=8,
        max_num_batched_tokens=64,
        max_model_len=32,
        target_logits_dtype=torch.float32,
        draft_logits_dtype=torch.float32,
    )
    desired_registry = build_draft_route_registry(
        runner.speculative_memory_plan,
        enforce_eager=eager,
    )
    runner.draft_route_registry = desired_registry.with_warmed_components(
        desired_registry.desired_warm_components
    )
    runner.draft_kv_cache = torch.empty(2, 1, 16, 4, 1, 1)
    runner.draft_model = _DraftModel(vocab_size)
    runner.model = _TargetPoison()
    runner.sampler = sampler if sampler is not None else _Sampler()
    monkeypatch.setattr(
        runner_module,
        "speculative_route_fits_plan",
        lambda plan, *, batch_size, effective_k: (
            batch_size > 0 and 0 < effective_k <= plan.max_effective_k
        ),
    )
    return runner


def _install_cpu_rng_neutral_seam(runner, calls):
    def run_phase(self, phase_name, operation):
        calls.append(phase_name)
        state = torch.random.get_rng_state()
        try:
            return operation()
        finally:
            torch.random.set_rng_state(state)

    runner._run_draft_phase = MethodType(run_phase, runner)


def _assert_default_context():
    context = get_context()
    assert context.is_prefill is False
    assert context.cu_seqlens_q is None
    assert context.cu_seqlens_k is None
    assert context.slot_mapping is None
    assert context.context_lens is None
    assert context.block_tables is None


def test_catchup_excludes_committed_tail_and_decode_positions_are_sequential(
    monkeypatch,
):
    runner = _runner(monkeypatch, k=2)
    seq = _sequence(
        (2, 3, 4, 5, 6),
        seq_id=41,
        draft_cached_tokens=2,
    )
    plan = _plan(runner, [seq], effective_k=2)

    execution = runner._execute_draft_proposals(plan, [seq])

    assert execution.catchup_positions == 2
    catchup, step0, step1 = runner.draft_model.calls
    assert catchup["is_prefill"] is True
    assert catchup["input_ids"].tolist() == [4, 5]
    assert catchup["positions"].tolist() == [2, 3]
    assert catchup["cu_q"].tolist() == [0, 2]
    assert catchup["cu_k"].tolist() == [0, 4]
    assert step0["is_prefill"] is False
    assert step0["input_ids"].tolist() == [6]
    assert step0["positions"].tolist() == [4]
    assert step0["context_lens"].tolist() == [5]
    assert step1["input_ids"].tolist() == [1]
    assert step1["positions"].tolist() == [5]
    assert step1["context_lens"].tolist() == [6]
    _assert_default_context()


def test_k_major_probability_destinations_expose_zero_copy_bkv_view(monkeypatch):
    runner = _runner(monkeypatch, k=3, vocab_size=7)
    seqs = [
        _sequence((1, 2, 3), seq_id=10, draft_cached_tokens=2),
        _sequence((4, 5, 6), seq_id=11, draft_cached_tokens=2),
    ]
    plan = _plan(runner, seqs, effective_k=3)

    execution = runner._execute_draft_proposals(plan, seqs)

    assert execution.q_storage_kbv.shape == (3, 2, 7)
    assert execution.q_storage_kbv.is_contiguous()
    assert all(execution.q_storage_kbv[step].is_contiguous() for step in range(3))
    assert execution.q_probabilities.shape == (2, 3, 7)
    assert execution.q_probabilities.stride() == (7, 14, 1)
    assert (
        execution.q_probabilities.untyped_storage().data_ptr()
        == execution.q_storage_kbv.untyped_storage().data_ptr()
    )
    for step, call in enumerate(runner.sampler.calls):
        assert call["out"].data_ptr() == execution.q_storage_kbv[step].data_ptr()
        assert call["out"].is_contiguous()
    assert torch.equal(
        execution.proposal_token_ids,
        torch.tensor([[1, 2, 3], [1, 2, 3]]),
    )


def test_draft_decode_routes_to_captured_graph_or_eager(monkeypatch):
    graph_runner = _runner(monkeypatch, k=2, eager=False)
    seqs = [
        _sequence((1, 2, 3), seq_id=1, draft_cached_tokens=2),
        _sequence((4, 5, 6), seq_id=2, draft_cached_tokens=2),
    ]
    plan = _plan(graph_runner, seqs, effective_k=2)
    graph = _Graph()
    graph_runner.draft_graph_bs = [1, 2, 4]
    graph_runner.draft_graphs = {2: graph}
    graph_runner.draft_graph_vars = {
        "input_ids": torch.zeros(4, dtype=torch.int64),
        "positions": torch.zeros(4, dtype=torch.int64),
        "slot_mapping": torch.zeros(4, dtype=torch.int32),
        "context_lens": torch.zeros(4, dtype=torch.int32),
        "block_tables": torch.zeros(4, 8, dtype=torch.int32),
        "outputs": torch.zeros(4, 11),
    }

    graphed = graph_runner._execute_draft_proposals(plan, seqs)

    assert graphed.graph_decode_steps == 2
    assert graphed.eager_decode_steps == 0
    assert graph.replays == 2
    # No catch-up and graph replay mean the eager draft model was never called.
    assert graph_runner.draft_model.calls == []

    eager_runner = _runner(monkeypatch, k=2, eager=True)
    eager_seqs = [
        _sequence((1, 2, 3), seq_id=3, draft_cached_tokens=2),
        _sequence((4, 5, 6), seq_id=4, draft_cached_tokens=2),
    ]
    eager = eager_runner._execute_draft_proposals(
        _plan(eager_runner, eager_seqs, effective_k=2), eager_seqs
    )
    assert eager.graph_decode_steps == 0
    assert eager.eager_decode_steps == 2
    assert [call["positions"].tolist() for call in eager_runner.draft_model.calls] == [
        [2, 2],
        [3, 3],
    ]
    _assert_default_context()


def test_production_discard_is_host_only_rng_neutral_and_nonmutating(monkeypatch):
    runner = _runner(monkeypatch, k=2, sampler=_Sampler())
    phase_calls = []
    _install_cpu_rng_neutral_seam(runner, phase_calls)
    seq = _sequence((2, 3, 4, 5), seq_id=17, draft_cached_tokens=1)
    plan = _plan(runner, [seq], effective_k=2)
    before = (
        tuple(seq.token_ids),
        seq.num_tokens,
        seq.last_token,
        seq.num_cached_tokens,
        seq.num_draft_cached_tokens,
        tuple(seq.block_table),
    )
    torch.manual_seed(20260828)
    rng_before = torch.random.get_rng_state().clone()

    result = runner.run_speculative_discard(plan, [seq])

    assert isinstance(result, DraftDiscardResult)
    assert result.rows[0].seq_id == 17
    assert result.rows[0].coverage_after_commit == 4
    assert result.rows[0].proposed_token_ids == (1, 2)
    assert result.rows[0].proposal_count == 2
    assert result.draft_positions == 2
    assert result.q_shape == (1, 2, 11)
    assert result.q_stride == (11, 11, 1)
    assert result.q_dtype == "torch.float32"
    assert result.q_storage_contiguous and result.q_view_zero_copy
    assert phase_calls == ["V3 draft compute-then-discard"]
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert before == (
        tuple(seq.token_ids),
        seq.num_tokens,
        seq.last_token,
        seq.num_cached_tokens,
        seq.num_draft_cached_tokens,
        tuple(seq.block_table),
    )
    _assert_default_context()


def test_proposal_failure_restores_rng_context_and_public_state(monkeypatch):
    runner = _runner(monkeypatch, k=2, sampler=_Sampler(fail_call=1))
    _install_cpu_rng_neutral_seam(runner, [])
    seq = _sequence((2, 3, 4, 5), seq_id=19, draft_cached_tokens=0)
    plan = _plan(runner, [seq], effective_k=2)
    snapshot = (
        tuple(seq.token_ids),
        seq.num_tokens,
        seq.num_cached_tokens,
        seq.num_draft_cached_tokens,
        tuple(seq.block_table),
    )
    torch.manual_seed(99)
    rng_before = torch.random.get_rng_state().clone()
    set_context(False, slot_mapping=torch.tensor([123], dtype=torch.int32))

    with pytest.raises(
        SpeculativeDraftExecutionError,
        match="injected proposal failure",
    ):
        runner.run_speculative_discard(plan, [seq])

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert snapshot == (
        tuple(seq.token_ids),
        seq.num_tokens,
        seq.num_cached_tokens,
        seq.num_draft_cached_tokens,
        tuple(seq.block_table),
    )
    _assert_default_context()


@pytest.mark.parametrize(
    "corrupt",
    (
        "block_table",
        "draft_coverage",
        "target_coverage",
        "physical_block",
        "work_charge",
    ),
)
def test_stale_or_unsafe_plan_fails_before_model_and_sampler(
    monkeypatch, corrupt
):
    runner = _runner(monkeypatch, k=2)
    seq = _sequence((1, 2, 3, 4, 5), seq_id=23, draft_cached_tokens=2)
    plan = _plan(runner, [seq], effective_k=2)
    if corrupt == "block_table":
        seq.block_table[-1] = 4
    elif corrupt == "draft_coverage":
        seq.num_draft_cached_tokens += 1
    elif corrupt == "target_coverage":
        seq.num_cached_tokens -= 1
    elif corrupt == "physical_block":
        seq.block_table[-1] = runner.draft_kv_cache.size(2)
        plan = _plan(runner, [seq], effective_k=2)
    elif corrupt == "work_charge":
        plan = DraftDiscardPlan(
            cycle_id=plan.cycle_id,
            rows=plan.rows,
            configured_k=plan.configured_k,
            workspace_route_cap=plan.workspace_route_cap,
            effective_k=plan.effective_k,
            draft_catchup_tokens=plan.draft_catchup_tokens,
            draft_step_token_counts=plan.draft_step_token_counts,
            target_query_tokens=plan.target_query_tokens,
            total_scheduled_tokens=1,
            fallback_reason=None,
            route_key=plan.route_key,
        )
    torch.manual_seed(1234)
    rng_before = torch.random.get_rng_state().clone()
    set_context(False, slot_mapping=torch.tensor([77], dtype=torch.int32))

    with pytest.raises(SpeculativeDraftPlanError):
        runner.run_speculative_discard(plan, [seq])

    assert runner.draft_model.calls == []
    assert runner.sampler.calls == []
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    _assert_default_context()


def test_catchup_and_proposal_never_touch_target_model(monkeypatch):
    runner = _runner(monkeypatch, k=1)
    seq = _sequence((1, 2, 3, 4), seq_id=99, draft_cached_tokens=0)
    execution = runner._execute_draft_proposals(
        _plan(runner, [seq], effective_k=1), [seq]
    )
    assert execution.catchup_positions == 3
    assert execution.proposal_token_ids.tolist() == [[1]]
    _assert_default_context()


def test_unwarmed_route_is_a_host_only_admission_miss(monkeypatch):
    runner = _runner(monkeypatch, k=2, eager=True)
    seq = _sequence((1, 2, 3, 4), seq_id=101, draft_cached_tokens=3)
    desired = build_draft_route_registry(
        runner.speculative_memory_plan,
        enforce_eager=True,
    )
    runner.draft_route_registry = desired

    assert runner.resolve_draft_route_admission([seq]) is None
    assert runner.draft_model.calls == []
    assert runner.sampler.calls == []


def test_missing_registered_graph_is_a_pre_reservation_admission_miss(monkeypatch):
    runner = _runner(monkeypatch, k=2, eager=False)
    seq = _sequence((1, 2, 3, 4), seq_id=102, draft_cached_tokens=3)
    buckets = runner.draft_route_registry.graph_buckets
    batch_cap = buckets[-1]
    max_blocks = runner.config.max_model_len // runner.block_size
    runner.draft_graph_bs = list(buckets)
    runner.draft_graphs = {bucket: _Graph() for bucket in buckets}
    runner.draft_graph_vars = {
        "input_ids": torch.zeros(batch_cap, dtype=torch.int64),
        "positions": torch.zeros(batch_cap, dtype=torch.int64),
        "slot_mapping": torch.zeros(batch_cap, dtype=torch.int32),
        "context_lens": torch.zeros(batch_cap, dtype=torch.int32),
        "block_tables": torch.zeros(
            batch_cap, max_blocks, dtype=torch.int32
        ),
        "outputs": torch.zeros(batch_cap, 11),
    }
    assert runner.resolve_draft_route_admission([seq]) is not None

    runner.draft_graphs.pop(buckets[0])

    assert runner.resolve_draft_route_admission([seq]) is None
    assert runner.draft_model.calls == []
    assert runner.sampler.calls == []


def test_reordered_plan_rows_fail_before_draft_work(monkeypatch):
    runner = _runner(monkeypatch, k=2)
    first = _sequence((1, 2, 3, 4), seq_id=31, draft_cached_tokens=3)
    second = _sequence((5, 6, 7, 8), seq_id=32, draft_cached_tokens=3)
    plan = _plan(runner, [first, second], effective_k=2)
    reordered = replace(plan, rows=tuple(reversed(plan.rows)))

    with pytest.raises(SpeculativeDraftPlanError, match="row order"):
        runner.run_speculative_discard(reordered, [first, second])

    assert runner.draft_model.calls == []
    assert runner.sampler.calls == []


@pytest.mark.parametrize("mode", ("dtype", "negative", "vocab"))
def test_invalid_draft_sampler_token_ids_fail_closed(monkeypatch, mode):
    class InvalidTokenSampler(_Sampler):
        def sample_exact_with_probabilities(self, *args, **kwargs):
            sample = super().sample_exact_with_probabilities(*args, **kwargs)
            token_ids = sample.token_ids
            if mode == "dtype":
                token_ids = token_ids.to(torch.int32)
            elif mode == "negative":
                token_ids.fill_(-1)
            else:
                token_ids.fill_(11)
            return SimpleNamespace(
                token_ids=token_ids,
                probabilities=sample.probabilities,
            )

    runner = _runner(monkeypatch, k=1, sampler=InvalidTokenSampler())
    _install_cpu_rng_neutral_seam(runner, [])
    seq = _sequence((1, 2, 3), seq_id=41, draft_cached_tokens=2)

    with pytest.raises(
        SpeculativeDraftExecutionError,
        match="invalid token IDs",
    ):
        runner.run_speculative_discard(
            _plan(runner, [seq], effective_k=1),
            [seq],
        )

    _assert_default_context()


def test_retained_execution_error_does_not_retain_q_workspace(monkeypatch):
    class WeakFailSampler:
        q_ref = None

        def sample_exact_with_probabilities(
            self, logits, temperatures, *, probabilities_out, **kwargs
        ):
            self.q_ref = weakref.ref(probabilities_out)
            raise RuntimeError("tensor-bearing failure")

    sampler = WeakFailSampler()
    runner = _runner(monkeypatch, k=1, sampler=sampler)
    _install_cpu_rng_neutral_seam(runner, [])
    seq = _sequence((1, 2, 3), seq_id=51, draft_cached_tokens=2)

    with pytest.raises(SpeculativeDraftExecutionError) as exc_info:
        runner.run_speculative_discard(
            _plan(runner, [seq], effective_k=1),
            [seq],
        )
    retained_error = exc_info.value
    del exc_info
    gc.collect()

    assert retained_error.__context__ is None
    assert sampler.q_ref is not None
    assert sampler.q_ref() is None
    _assert_default_context()


def test_execution_error_note_copy_is_python_310_compatible(monkeypatch):
    class Python310StyleExecutionError(Exception):
        add_note = None

    class NotedFailSampler(_Sampler):
        def sample_exact_with_probabilities(self, *args, **kwargs):
            error = RuntimeError("noted draft failure")
            error.__notes__ = ["retained cleanup diagnostic"]
            raise error

    monkeypatch.setattr(
        runner_module,
        "SpeculativeDraftExecutionError",
        Python310StyleExecutionError,
    )
    runner = _runner(monkeypatch, k=1, sampler=NotedFailSampler())
    _install_cpu_rng_neutral_seam(runner, [])
    seq = _sequence((1, 2, 3), seq_id=52, draft_cached_tokens=2)

    with pytest.raises(
        Python310StyleExecutionError,
        match="noted draft failure",
    ):
        runner.run_speculative_discard(
            _plan(runner, [seq], effective_k=1),
            [seq],
        )

    _assert_default_context()


def test_authoritative_run_resets_context_after_model_failure():
    runner = object.__new__(ModelRunner)
    runner.rank = 0

    def prepare_decode(_seqs):
        set_context(False, slot_mapping=torch.tensor([7], dtype=torch.int32))
        return torch.tensor([1]), torch.tensor([0])

    runner.prepare_decode = prepare_decode
    runner.prepare_sample = lambda _seqs: (None, (), None, True)
    runner.run_model = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("injected target failure")
    )

    with pytest.raises(RuntimeError, match="injected target failure"):
        runner.run([SimpleNamespace()], False)

    _assert_default_context()
