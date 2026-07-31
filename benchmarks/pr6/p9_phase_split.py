# benchmarks/pr6/p9_phase_split.py — where do the ~750ms live: prefill steps, decode steps, or neither?
# usage: p9_phase_split.py {varlen|novarlen}
import os, sys, time, torch
from random import randint, seed
if len(sys.argv) != 2 or sys.argv[1] not in ("varlen", "novarlen"):
    sys.exit("usage: p9_phase_split.py {varlen|novarlen}")
if sys.argv[1] == "novarlen":
    from nanovllm.engine.model_runner import ModelRunner
    ModelRunner.capture_varlen_graphs = lambda self: None
from nanovllm import LLM, SamplingParams

seed(0)
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
prompts = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(256)]
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024)) for _ in range(256)]
llm.generate(["Benchmark: "], SamplingParams())          # warmup, same as bench.py
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
print(f"P9 [{sys.argv[1]:8s}] prefill: {np:4d} steps {tp:6.2f}s (max {wp*1e3:6.1f}ms) | "
      f"decode: {nd:5d} steps {td:6.2f}s (max {wd*1e3:5.1f}ms) | "
      f"total {tp+td:6.2f}s | alloc_retries {r1-r0} | miss {getattr(llm.model_runner,'varlen_miss','n/a')}")