# p7: policy data at the exact long-prompt shape (2x2048) + interactive sanity (1024 bucket)
# NOTE: varlen_ts tops out at 2048 BY MEASUREMENT (P12): the 4096 replay floor is
# ~32 ms ~= paged-eager on host1 (past the E2 crossover), so the 4096-token step
# correctly takes the miss-fallback; the miss counter makes that measured, not assumed.
import random, torch
from probe_common import make_llm, ragged_step, timed
from nanovllm.utils.context import set_context, reset_context

llm = make_llm()
mr = llm.model_runner
random.seed(0)

ids, pos, ctx = ragged_step(llm, [2048, 2048], 16384)
with torch.inference_mode():
    t_paged = timed(lambda: mr.model(ids, pos))
    t_run = timed(lambda: mr.run_model(ids, pos, True))
    set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
                ctx.max_seqlen_k, ctx.slot_mapping, None, None)
    t_fresh = timed(lambda: mr.model(ids, pos))
print(f"P7 2x2048: fresh-eager {t_fresh:.1f} | paged-eager {t_paged:.1f} | "
      f"run_model {t_run:.1f} ms | miss {mr.varlen_miss}")
reset_context(); llm.scheduler.cancel_all()

ids, pos, ctx = ragged_step(llm, [64] * 16, 16384)            # the interactive-prefill shape
with torch.inference_mode():
    t_g = timed(lambda: mr.run_model(ids, pos, True))
    t_e = timed(lambda: mr.model(ids, pos))
print(f"P7 16x64:  paged-eager {t_e:.1f} | graphed(1024) {t_g:.1f} ms | miss {mr.varlen_miss}")
reset_context(); llm.scheduler.cancel_all()
