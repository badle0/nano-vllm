# benchmarks/pr6/p12_slot_tax.py — does the 4096-bucket replay tax scale with segment
# slots? (P1 proved zero-length segments CORRECT; their time cost at S+1 slots x large
# T_pad was never measured — E4's 6.3 ms replays were small shapes.) Captures T=4096
# graphs at varying (slot count, K-ceiling), replays the exact 2x2048 long-prompt load.
import random, torch
from probe_common import make_llm, ragged_step, timed
from nanovllm.utils.context import set_context, reset_context

llm = make_llm()
mr = llm.model_runner
random.seed(0)
ids, pos, ctx = ragged_step(llm, [2048, 2048], 16384)
T, H = 4096, mr.config.hf_config.hidden_size
real_ns = ctx.cu_seqlens_q.numel() - 1

for slots in (real_ns, 8, 64, 257):
    for k_ceil in (4096,) if slots != real_ns else (4096, 2048):
        cu = torch.zeros(slots + 1, dtype=torch.int32, device="cuda")
        cu[1:real_ns + 1] = ctx.cu_seqlens_q[1:]
        cu[real_ns + 1:] = ctx.cu_seqlens_q[-1]
        cu_k = cu.clone()
        buf_ids = torch.zeros(T, dtype=torch.int64, device="cuda"); buf_ids[:ids.size(0)] = ids
        buf_pos = torch.zeros(T, dtype=torch.int64, device="cuda"); buf_pos[:ids.size(0)] = pos
        slot_map = torch.full((T,), -1, dtype=torch.int32, device="cuda")
        bt = torch.full((slots, ctx.block_tables.size(1)), -1, dtype=torch.int32, device="cuda")
        bt[:real_ns] = ctx.block_tables
        with torch.inference_mode():
            set_context(True, cu, cu_k, T, k_ceil, slot_map, None, bt)
            g = torch.cuda.CUDAGraph()
            out = mr.model(buf_ids, buf_pos)                       # warmup
            with torch.cuda.graph(g, mr.graph_pool):
                out = mr.model(buf_ids, buf_pos)
            torch.cuda.synchronize()
            t = timed(lambda: g.replay())
        reset_context()
        print(f"P12 T=4096 slots={slots:3d} k_ceil={k_ceil}: replay {t:.1f} ms")
reset_context(); llm.scheduler.cancel_all()
