# benchmarks/pr6/p10_profile_step1.py — profile the first big prefill; let the delta name its op
# usage: p10_profile_step1.py {varlen|novarlen}
import os, sys
from random import randint, seed
if len(sys.argv) != 2 or sys.argv[1] not in ("varlen", "novarlen"):
    sys.exit("usage: p10_profile_step1.py {varlen|novarlen}")
if sys.argv[1] == "novarlen":
    from nanovllm.engine.model_runner import ModelRunner
    ModelRunner.capture_varlen_graphs = lambda self: None
import torch
from torch.profiler import profile, ProfilerActivity
from nanovllm import LLM, SamplingParams

seed(0)
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
prompts = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(256)]
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024)) for _ in range(256)]
llm.generate(["Benchmark: "], SamplingParams())
for p, sp in zip(prompts, sps):
    llm.add_request(p, sp)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    llm.step()                                          # the 300ms-vs-970ms step
print(f"=== {sys.argv[1]}: step-1 by CUDA time ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=8))
print(f"=== {sys.argv[1]}: step-1 by CPU time ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))
while not llm.is_finished(): llm.step()                 # drain for a clean exit