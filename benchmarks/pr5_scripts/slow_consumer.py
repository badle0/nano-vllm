"""Sync-design coupling: consumer time lands on the critical path. Run from repo root."""
import os, statistics, torch
from time import perf_counter, sleep
from nanovllm import LLM, SamplingParams

PROMPTS = ["The capital of France is"] * 8
SP = SamplingParams(temperature=0.6, max_tokens=64, ignore_eos=True)
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)

def run(sleep_s):
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    gaps, prev, seen = [], None, set()
    for ev in llm.stream(PROMPTS, SP):
        if ev.seq_id in seen:                     # first event of a new step
            now = perf_counter()
            if prev is not None: gaps.append(now - prev)
            prev = now
            seen.clear()
        seen.add(ev.seq_id)
        if sleep_s: sleep(sleep_s)
    return statistics.median(gaps) * 1000

for s in (0.0, 0.0, 0.002, 0.010):          # first is warmup
    print(f"| consumer sleep {s*1000:5.1f} ms | median inter-event {run(s):7.3f} ms |")