"""Throughput cost of the streaming path with a do-nothing consumer. Run from repo root."""
import os, statistics, torch
from time import perf_counter
from nanovllm import LLM, SamplingParams

PROMPTS = ["The capital of France is"] * 32
SP = SamplingParams(temperature=0.6, max_tokens=256, ignore_eos=True)
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
NTOK = len(PROMPTS) * SP.max_tokens

def run_batch():
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    t = perf_counter(); llm.generate(PROMPTS, SP, use_tqdm=False); return NTOK / (perf_counter() - t)

def run_stream():
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    t = perf_counter(); n = 0
    for _ in llm.stream(PROMPTS, SP): n += 1
    assert n == NTOK, f"event count {n} != {NTOK}"     # free equivalence check
    return NTOK / (perf_counter() - t)

b, s = [], []
for i in range(4):
    bb, ss = run_batch(), run_stream()
    if i: b.append(bb); s.append(ss)
mb, ms = statistics.median(b), statistics.median(s)
print(f"| generate()      | {mb:8.1f} tok/s |")
print(f"| stream() null   | {ms:8.1f} tok/s |")
print(f"| delta           | {(ms-mb)/mb*100:+7.2f}% |")