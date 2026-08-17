import random, torch
from nanovllm import SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context

# Bitwise gate is valid only within one launch shape (F5 axis 1): the eager
# comparison must replicate the ROUTED tier's capture launch — cu/block_tables
# prefix-sliced to that tier's slot count (P13 two-tier captures).


def _assert_bitwise_vs_padded_eager(llm, ids, pos, ctx):
    mr = llm.model_runner
    with torch.inference_mode():
        graphed = mr.run_model(ids, pos, True).clone()     # fills buffers, replays
        v, t = mr.varlen_vars, ids.size(0)
        tp = next(x for x in mr.varlen_ts if x >= t)
        ns = ctx.cu_seqlens_q.numel() - 1
        sl = next(s for s in mr.varlen_slots if s >= ns)   # the tier run_model routed to
        set_context(True, v["cu_q"][:sl + 1], v["cu_k"][:sl + 1], tp, mr.config.max_model_len,
                    v["slot_mapping"][:tp], None, v["block_tables"][:sl])
        eager = mr.model.compute_logits(mr.model(v["input_ids"][:tp], v["positions"][:tp]))
    assert mr.varlen_miss == 0
    assert torch.equal(eager[:ns], graphed)                # padded rows are duplicates; real rows lead
    reset_context(); llm.scheduler.cancel_all()


def test_varlen_graph_bitwise_per_bucket(llm, monkeypatch):
    random.seed(7)
    mr = llm.model_runner
    sp = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
    for target, budget in ((100, 128), (400, 512), (1500, 2048)):
        monkeypatch.setattr(llm.scheduler, "_max_num_batched_tokens", budget)
        llm.add_request([random.randint(1000, 150000) for _ in range(target)], sp)
        seqs, is_prefill = llm.scheduler.schedule()
        assert is_prefill
        ids, pos = mr.prepare_ragged(seqs)
        _assert_bitwise_vs_padded_eager(llm, ids, pos, get_context())


def test_varlen_graph_bitwise_full_tier(llm, monkeypatch):
    # 65 x 8-token prompts in one step: ns=65 > lean tier (64) -> full-slot graph.
    # Sub-block prompts, so no prefix-cache coupling with other tests.
    random.seed(11)
    monkeypatch.setattr(llm.scheduler, "_max_num_batched_tokens", 1024)
    mr = llm.model_runner
    sp = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
    for _ in range(65):
        llm.add_request([random.randint(1000, 150000) for _ in range(8)], sp)
    seqs, is_prefill = llm.scheduler.schedule()
    assert is_prefill and len(seqs) == 65
    ids, pos = mr.prepare_ragged(seqs)
    ctx = get_context()
    ns = ctx.cu_seqlens_q.numel() - 1
    assert next(s for s in mr.varlen_slots if s >= ns) == mr.varlen_slots[-1]
    _assert_bitwise_vs_padded_eager(llm, ids, pos, ctx)
