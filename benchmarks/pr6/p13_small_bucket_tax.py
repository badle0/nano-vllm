# benchmarks/pr6/p13_small_bucket_tax.py — decide the slot policy for the tau=512
# operating point. P12 measured the zero-length slot tax only at T=4096 (0.043 ms/slot);
# the tau=512 max_ITL residual (10.6 ms) is only worth slot surgery if (A) the rate is
# material at small T and (B) replay actually dominates the step. Both measured here.
import random, torch
from probe_common import make_llm, ragged_step, timed
from nanovllm.utils.context import set_context, reset_context

llm = make_llm()
mr = llm.model_runner
random.seed(0)

# A — slot-tax rate at small T: capture (T, slots) variants, replay a ~500-token load
ids, pos, ctx = ragged_step(llm, [500], 16384)
for T in (512, 1024):
    for slots in (18, 128, 513):
        cu = torch.zeros(slots + 1, dtype=torch.int32, device="cuda")
        cu[1:] = ids.size(0)
        buf_ids = torch.zeros(T, dtype=torch.int64, device="cuda"); buf_ids[:ids.size(0)] = ids
        buf_pos = torch.zeros(T, dtype=torch.int64, device="cuda"); buf_pos[:ids.size(0)] = pos
        slot_map = torch.full((T,), -1, dtype=torch.int32, device="cuda")
        bt = torch.full((slots, ctx.block_tables.size(1)), -1, dtype=torch.int32, device="cuda")
        bt[0] = ctx.block_tables[0]
        with torch.inference_mode():
            set_context(True, cu, cu.clone(), T, mr.config.max_model_len, slot_map, None, bt)
            g = torch.cuda.CUDAGraph()
            out = mr.model(buf_ids, buf_pos)
            with torch.cuda.graph(g, mr.graph_pool):
                out = mr.model(buf_ids, buf_pos)
            torch.cuda.synchronize()
            t = timed(lambda: g.replay())
        reset_context()
        print(f"P13a T={T} slots={slots:3d}: replay {t:.2f} ms")
reset_context(); llm.scheduler.cancel_all()

# B — decompose the production bucket-512 path (513-slot captures) on the same shape
ids, pos, ctx = ragged_step(llm, [500], 16384)
with torch.inference_mode():
    tp = mr._fill_varlen(ids, pos, ctx)
    t_replay = timed(lambda: mr.varlen_graphs[tp].replay())
    t_fill   = timed(lambda: mr._fill_varlen(ids, pos, ctx))
    t_logits = timed(lambda: mr.model.compute_logits(mr.varlen_vars["outputs"][:ids.size(0)]))
    t_run    = timed(lambda: mr.run_model(ids, pos, True))
    t_eager  = timed(lambda: mr.model(ids, pos))
print(f"P13b bucket {tp}: replay {t_replay:.2f} | fill {t_fill:.2f} | logits {t_logits:.2f} "
      f"| run_model {t_run:.2f} | paged-eager(model only) {t_eager:.2f} ms")
reset_context(); llm.scheduler.cancel_all()
