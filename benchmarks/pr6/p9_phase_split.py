# benchmarks/pr6/p9_phase_split.py — where do the ~750ms live: prefill steps, decode steps, or neither?
# usage: p9_phase_split.py {varlen|novarlen}
# Absorbs p8_pool_cohabitation (deleted in the probe cleanup): same workload, same A/B —
# its free-MiB and end-to-end tok/s lines are printed here (tok/s from the step loop,
# which excludes generate()'s final detokenize; the arm delta is what matters).
from probe_common import parse_arm, make_llm, bench_workload

ARM = parse_arm("varlen", "novarlen")
import time, torch

llm = make_llm()
prompts, sps = bench_workload(llm)                       # seed(0) load + warmup, as bench.py
print("free MiB pre-generate:", torch.cuda.mem_get_info()[0] // 2**20)
torch.cuda.reset_peak_memory_stats()
r0 = torch.cuda.memory_stats().get("num_alloc_retries", 0)
for p, sp in zip(prompts, sps):
    llm.add_request(p, sp)
np = nd = 0; tp = td = 0.0; wp = wd = 0.0
print(f"    reserved before first step: {torch.cuda.memory_reserved()//2**20} MiB")
while not llm.is_finished():
    t0 = time.time()
    _, ntok = llm.step()
    dt = time.time() - t0
    if ntok > 0 and np < 3:
        print(f"    prefill step {np+1}: {dt*1e3:7.1f} ms | reserved {torch.cuda.memory_reserved()//2**20} MiB")
    if ntok > 0: np += 1; tp += dt; wp = max(wp, dt)
    else:        nd += 1; td += dt; wd = max(wd, dt)
r1 = torch.cuda.memory_stats().get("num_alloc_retries", 0)
print(f"P9 [{ARM:8s}] prefill: {np:4d} steps {tp:6.2f}s (max {wp*1e3:6.1f}ms) | "
      f"decode: {nd:5d} steps {td:6.2f}s (max {wd*1e3:5.1f}ms) | "
      f"total {tp+td:6.2f}s | alloc_retries {r1-r0} | miss {getattr(llm.model_runner,'varlen_miss','n/a')}")
print(f"P8' [{ARM:8s}] {sum(sp.max_tokens for sp in sps)/(tp+td):.2f} tok/s   ({tp+td:.2f}s)")
