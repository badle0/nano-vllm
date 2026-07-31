# benchmarks/pr6/p5_kernel_config.py — is the per-bucket failure a kernel-config (M_q/M_k) effect?
import os, random, torch
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context

llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
random.seed(7)
for target, budget, t_pad in ((100, 128, 128), (400, 512, 512), (1500, 2048, 2048)):
    llm.scheduler.max_num_batched_tokens = budget
    llm.add_request([random.randint(1000, 150000) for _ in range(target)],
                    SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True))
    seqs, _ = llm.scheduler.schedule()
    ids, pos = llm.model_runner.prepare_ragged(seqs)
    ctx = get_context()
    with torch.inference_mode():
        eager = llm.model_runner.model.compute_logits(llm.model_runner.model(ids, pos)).clone()
        graphed = llm.model_runner.run_model(ids, pos, True).clone()
        d = (graphed - eager).float().abs()
        print(f"bucket {t_pad}: max {d.max().item():.4f} mean {d.mean().item():.2e} "
              f"exact-frac {(graphed == eager).float().mean().item():.3f}")
        # mechanism check: eager forced onto the graph's baked kernel config
        set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k, t_pad,
                    llm.model_runner.config.max_model_len, ctx.slot_mapping, None, ctx.block_tables)
        eager_baked = llm.model_runner.model.compute_logits(llm.model_runner.model(ids, pos))
        print(f"bucket {t_pad}: eager(baked M) == graphed bitwise:", torch.equal(eager_baked, graphed))
    reset_context(); llm.scheduler.cancel_all()

# P5b: does matching the FULL launch shape (padded batch, buffers, baked Ms) restore bitwise?
llm.scheduler.max_num_batched_tokens = 2048
llm.add_request([random.randint(1000, 150000) for _ in range(1500)],
                SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True))
seqs, _ = llm.scheduler.schedule()
ids, pos = llm.model_runner.prepare_ragged(seqs)
ctx = get_context()
mr = llm.model_runner
with torch.inference_mode():
    graphed = mr.run_model(ids, pos, True).clone()      # replay; also fills the persistent buffers
    v, t, ns = mr.varlen_vars, ids.size(0), ctx.cu_seqlens_q.numel() - 1
    tp = next(x for x in mr.varlen_ts if x >= t)
    set_context(True, v["cu_q"], v["cu_k"], tp, mr.config.max_model_len,
                v["slot_mapping"][:tp], None, v["block_tables"])       # capture-identical launch
    eager_padded = mr.model.compute_logits(mr.model(v["input_ids"][:tp], v["positions"][:tp]))
    eager_padded = mr.model.compute_logits(mr.model(v["input_ids"][:tp], v["positions"][:tp]))
    print("P5b eager(padded launch) == graphed bitwise:",
          torch.equal(eager_padded[: graphed.size(0)] if eager_padded.size(0) != graphed.size(0) else eager_padded, graphed))
reset_context(); llm.scheduler.cancel_all()