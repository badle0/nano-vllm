from types import SimpleNamespace

import random, torch
from nanovllm import SamplingParams
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.context import get_context, set_context, reset_context

# Bitwise gate is valid only within one launch shape (F5 axis 1): the eager
# comparison must replicate the ROUTED tier's capture launch — cu/block_tables
# prefix-sliced to that tier's slot count (P13 two-tier captures).


def _assert_bitwise_vs_padded_eager(llm, ids, pos, ctx):
    mr = llm.model_runner
    with torch.inference_mode():
        graphed = mr.run_model(ids, pos, True).clone()     # fills buffers, replays
        v, t = mr.varlen_vars, ids.size(0)
        ns = ctx.cu_seqlens_q.numel() - 1
        key = mr._select_varlen_graph_key(t, ns)
        assert key is not None
        tp, sl = key
        set_context(True, v["cu_q"][:sl + 1], v["cu_k"][:sl + 1], tp, mr.config.max_model_len,
                    v["slot_mapping"][:tp], None, v["block_tables"][:sl])
        eager = mr.model.compute_logits(mr.model(v["input_ids"][:tp], v["positions"][:tp]))
    assert mr.varlen_miss == 0
    assert torch.equal(eager[:ns], graphed)                # padded rows are duplicates; real rows lead
    reset_context(); llm.scheduler.cancel_all()


def test_varlen_capture_boundaries_respect_model_length_and_slots():
    boundaries = ModelRunner._varlen_capture_boundaries(2048, 64, 512)
    assert boundaries is not None
    assert len(boundaries) == 65
    assert boundaries[0] == 0 and boundaries[-1] == 2048
    assert all(
        0 <= end - start <= 512
        for start, end in zip(boundaries, boundaries[1:])
    )
    assert ModelRunner._varlen_capture_boundaries(2048, 3, 512) is None


def test_varlen_graph_key_selection_handles_empty_and_sparse_captures():
    runner = ModelRunner.__new__(ModelRunner)
    runner.varlen_graphs = {}
    assert runner._select_varlen_graph_key(64, 1) is None

    runner.varlen_graphs = {
        (128, 513): object(),
        (256, 64): object(),
        (256, 513): object(),
    }
    assert runner._select_varlen_graph_key(100, 10) == (128, 513)
    assert runner._select_varlen_graph_key(200, 10) == (256, 64)
    assert runner._select_varlen_graph_key(100, 514) is None
    assert runner._select_varlen_graph_key(0, 1) is None


def test_varlen_graph_context_rejects_incompatible_persistent_shapes():
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(max_model_len=4)
    runner.varlen_vars = {"block_tables": torch.empty(4, 2)}
    context = SimpleNamespace(
        slot_mapping=torch.empty(3),
        cu_seqlens_k=torch.empty(3),
        max_seqlen_q=3,
        max_seqlen_k=4,
        block_tables=torch.empty(2, 2),
    )
    graph_key = (4, 4)
    assert runner._varlen_context_fits_graph(3, 2, context, graph_key)

    context.slot_mapping = torch.empty(2)
    assert not runner._varlen_context_fits_graph(3, 2, context, graph_key)
    context.slot_mapping = torch.empty(3)
    context.block_tables = torch.empty(1, 2)
    assert not runner._varlen_context_fits_graph(3, 2, context, graph_key)
    context.block_tables = torch.empty(2, 3)
    assert not runner._varlen_context_fits_graph(3, 2, context, graph_key)
    context.block_tables = torch.empty(2, 2)
    context.max_seqlen_k = 5
    assert not runner._varlen_context_fits_graph(3, 2, context, graph_key)


def test_varlen_graph_bitwise_per_bucket(llm, monkeypatch):
    random.seed(7)
    mr = llm.model_runner
    sp = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
    for target, budget in ((100, 128), (400, 512), (1500, 2048)):
        monkeypatch.setattr(llm.scheduler, "max_num_batched_tokens", budget)
        llm.add_request([random.randint(1000, 150000) for _ in range(target)], sp)
        seqs, is_prefill = llm.scheduler.schedule()
        assert is_prefill
        ids, pos = mr.prepare_ragged(seqs)
        _assert_bitwise_vs_padded_eager(llm, ids, pos, get_context())


def test_varlen_graph_bitwise_full_tier(llm, monkeypatch):
    # 65 x 8-token prompts in one step: ns=65 > lean tier (64) -> full-slot graph.
    # Sub-block prompts, so no prefix-cache coupling with other tests.
    random.seed(11)
    monkeypatch.setattr(llm.scheduler, "max_num_batched_tokens", 1024)
    mr = llm.model_runner
    sp = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
    for _ in range(65):
        llm.add_request([random.randint(1000, 150000) for _ in range(8)], sp)
    seqs, is_prefill = llm.scheduler.schedule()
    assert is_prefill and len(seqs) == 65
    ids, pos = mr.prepare_ragged(seqs)
    ctx = get_context()
    ns = ctx.cu_seqlens_q.numel() - 1
    key = mr._select_varlen_graph_key(ids.size(0), ns)
    assert key is not None and key[1] == mr.varlen_slots[-1]
    _assert_bitwise_vs_padded_eager(llm, ids, pos, ctx)
