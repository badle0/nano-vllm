# benchmarks/pr6/p11_step1_compile.py
# P11 — bisect step-1 compile cost across construction arms; count Dynamo work via
# counters (a measurement) rather than log strings (a recollection).
# usage: p11_step1_compile.py {novarlen|pretouch|varlen}
#   novarlen  capture_varlen_graphs stubbed entirely   (= P10 control)
#   pretouch  ONLY the budget-sized pre-touch forward  (buckets skipped)
#   varlen    unmodified                               (= P10 treatment)
# Step 1 (~16384 tok) > varlen_ts[-1] (2048): run_model takes the eager branch in
# EVERY arm. The step-1 forward is identical; only __init__'s shape history differs.
from probe_common import parse_arm, make_llm, bench_workload, clean_exit

ARM = parse_arm("novarlen", "pretouch", "varlen")
import copy, sys, time, torch
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.context import set_context, reset_context


@torch.inference_mode()
def _pretouch_only(self):
    # verbatim tail of capture_varlen_graphs. Sets no varlen_* attrs, so run_model's
    # hasattr gate and exit()'s guard both behave exactly as in the novarlen arm.
    T = self.config.max_num_batched_tokens
    L = min(self.config.max_model_len, T)
    ns = (T + L - 1) // L
    cu = torch.arange(0, ns + 1, dtype=torch.int32) * L
    cu[-1] = T
    set_context(True, cu, cu.clone(), L, L,
                torch.full((T,), -1, dtype=torch.int32), None, None)
    self.model(torch.zeros(T, dtype=torch.int64), torch.arange(T, dtype=torch.int64) % L)
    torch.cuda.synchronize()
    reset_context()

if ARM == "pretouch":
    ModelRunner.capture_varlen_graphs = _pretouch_only

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
