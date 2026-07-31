# benchmarks/pr6/p6_paged_tax.py — is large-T paged attention the regression mechanism?
# P6a only. P6b/P6c retired in the probe cleanup: the 2x2048 step is covered by p7, and
# with varlen_ts capped at 2048 the old "bucket 4096, graphed" label was stale — that
# shape takes run_model's miss-fallback (eager), which p7 now measures and labels honestly.
import random, torch
from probe_common import make_llm, ragged_step, timed
from nanovllm.utils.context import set_context, reset_context

llm = make_llm()
mr = llm.model_runner
random.seed(0)

# P6a — 16k-token eager step (bench.py's miss regime): paged vs fresh
ids, pos, ctx = ragged_step(llm, [2000] * 8)
with torch.inference_mode():
    paged = timed(lambda: mr.model(ids, pos))
    set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
                ctx.max_seqlen_k, ctx.slot_mapping, None, None)      # fresh branch
    fresh = timed(lambda: mr.model(ids, pos))
print(f"P6a 16k eager: paged {paged:.1f} ms | fresh {fresh:.1f} ms | tax {paged - fresh:+.1f} ms")
reset_context(); llm.scheduler.cancel_all()
