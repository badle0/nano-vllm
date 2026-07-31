# benchmarks/pr6/p10_profile_step1.py — profile the first big prefill; let the delta name its op
# usage: p10_profile_step1.py {varlen|novarlen}
from probe_common import parse_arm, make_llm, bench_workload, clean_exit

ARM = parse_arm("varlen", "novarlen")
from torch.profiler import profile, ProfilerActivity

llm = make_llm()
prompts, sps = bench_workload(llm)
for p, sp in zip(prompts, sps):
    llm.add_request(p, sp)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    llm.step()                                          # the 300ms-vs-970ms step
print(f"=== {ARM}: step-1 by CUDA time ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=8))
print(f"=== {ARM}: step-1 by CPU time ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))
clean_exit(llm)     # replaces the full drain loop; exit-with-queued-work proven by the P11 run
