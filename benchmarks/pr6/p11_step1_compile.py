# benchmarks/pr6/p11_step1_compile.py
# P11 — bisect step-1 compile cost across construction arms; count Dynamo work via
# counters (a measurement) rather than log strings (a recollection).
# usage: p11_step1_compile.py {novarlen|pretouch|varlen}
#   novarlen  bucket captures AND pre-touch stubbed    (= dev-equivalent control)
#   pretouch  ONLY the post-restore pre-touch forward  (buckets stubbed)
#   varlen    unmodified                               (= P10 treatment)
# Step 1 (~16384 tok) > varlen_ts[-1] (2048): run_model takes the eager branch in
# EVERY arm. The step-1 forward is identical; only __init__'s compile state differs.
# History: P11 originally bisected __init__'s *shape* history and falsified it —
# the guard is GLOBAL_STATE default_dtype (see _pretouch_eager_prefill's comment);
# the pre-touch now runs post-restore in production, and this probe verifies it.
from probe_common import parse_arm, make_llm, bench_workload, clean_exit

ARM = parse_arm("novarlen", "pretouch", "varlen")
import copy, sys, time, torch
from nanovllm.engine.model_runner import ModelRunner

if ARM == "pretouch":
    ModelRunner.capture_varlen_graphs = lambda self: None   # production pre-touch kept

import torch._dynamo.utils as du

snap = lambda: copy.deepcopy(du.counters)
def diff(a, b):
    out = {}
    for k, cnt in b.items():
        d = {kk: vv - a[k][kk] for kk, vv in cnt.items() if vv - a[k][kk]}
        if d: out[k] = d
    return out

t0 = time.perf_counter()
llm = make_llm()
t_init = time.perf_counter() - t0
c_init = snap()

prompts, sps = bench_workload(llm)
for p, sp in zip(prompts, sps):
    llm.add_request(p, sp)
c_pre = snap()

print("STEP1-BEGIN", file=sys.stderr, flush=True)   # stderr: same stream as TORCH_LOGS
torch.cuda.synchronize(); t0 = time.perf_counter()
llm.step()
torch.cuda.synchronize(); t_step1 = time.perf_counter() - t0
print("STEP1-END", file=sys.stderr, flush=True)
c_post = snap()

mr = getattr(llm, "model_runner", None)
print(f"=== P11 arm={ARM} ===")
print(f"init            {t_init*1e3:8.1f} ms")
print(f"step1           {t_step1*1e3:8.1f} ms")
print(f"varlen_miss     {getattr(mr, 'varlen_miss', 'n/a')}")
print(f"counters@init   {dict(c_init)}")
print(f"counters@step1  {diff(c_pre, c_post)}")
try:
    print("compile_times:"); print(du.compile_times(repr="csv", aggregate=True))
except Exception as e:
    print("compile_times unavailable:", type(e).__name__, e)
clean_exit(llm)
