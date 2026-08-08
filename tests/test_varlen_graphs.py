import random, torch
from nanovllm import SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context

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
        ctx = get_context()
        with torch.inference_mode():
            graphed = mr.run_model(ids, pos, True).clone()     # fills buffers, replays
            v, t = mr.varlen_vars, ids.size(0)
            tp = next(x for x in mr.varlen_ts if x >= t)
            ns = ctx.cu_seqlens_q.numel() - 1
            set_context(True, v["cu_q"], v["cu_k"], tp, mr.config.max_model_len,
                        v["slot_mapping"][:tp], None, v["block_tables"])
            eager = mr.model.compute_logits(mr.model(v["input_ids"][:tp], v["positions"][:tp]))
        assert mr.varlen_miss == 0
        assert torch.equal(eager[:ns], graphed)                # padded rows are duplicates; real rows lead
        reset_context(); llm.scheduler.cancel_all()