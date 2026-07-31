# benchmarks/pr6/p6_paged_tax.py — is large-T paged attention the regression mechanism?
import os, random, torch
from time import perf_counter
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context

llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
mr = llm.model_runner
random.seed(0)
sp = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)

def timed(fn, n=3):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((perf_counter() - t0) * 1e3)
    return min(ts)

# P6a — 16k-token eager step (bench.py's miss regime): paged vs fresh
for _ in range(8):
    llm.add_request([random.randint(1000, 150000) for _ in range(2000)], sp)
seqs, _ = llm.scheduler.schedule()
ids, pos = mr.prepare_ragged(seqs)
ctx = get_context()
with torch.inference_mode():
    paged = timed(lambda: mr.model(ids, pos))
    set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
                ctx.max_seqlen_k, ctx.slot_mapping, None, None)      # fresh branch
    fresh = timed(lambda: mr.model(ids, pos))
print(f"P6a 16k eager: paged {paged:.1f} ms | fresh {fresh:.1f} ms | tax {paged - fresh:+.1f} ms")
reset_context(); llm.scheduler.cancel_all()

# P6b — the long-prompt scenario's exact step: 2x2048, bucket 4096, graphed
for _ in range(2):
    llm.add_request([random.randint(1000, 150000) for _ in range(2048)], sp)
seqs, _ = llm.scheduler.schedule()
ids, pos = mr.prepare_ragged(seqs)
with torch.inference_mode():
    g = timed(lambda: mr.run_model(ids, pos, True))
print(f"P6b 4096-token graphed step: {g:.1f} ms   (fresh eager @3968 was 41.7)")
reset_context(); llm.scheduler.cancel_all()
print("P6c varlen_miss this process:", mr.varlen_miss)               # predict 0 (P6a bypassed run_model)