"""Caller-visible time-to-first-content: batch vs stream. Run from repo root."""
import os, statistics, torch
from time import perf_counter
from nanovllm import LLM, SamplingParams

PROMPTS = ["The capital of France is", "def fibonacci(n):", "In 1969, humans first"]
SP = SamplingParams(temperature=0.6, max_tokens=256, ignore_eos=True)
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)

def batch_first_content():
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    t = perf_counter(); llm.generate(PROMPTS, SP, use_tqdm=False); return perf_counter() - t

def stream_first_content():
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    t = perf_counter()
    it = llm.stream(PROMPTS, SP)
    first = next(it); dt = perf_counter() - t
    for _ in it: pass                      # drain: leave the engine clean for the next trial
    return dt

b, s = [], []
for i in range(4):                          # 1 warmup + 3 measured, interleaved
    bb, ss = batch_first_content(), stream_first_content()
    if i: b.append(bb); s.append(ss)
mb, ms = statistics.median(b), statistics.median(s)
print(f"| batch (generate returns) | {mb*1000:8.1f} ms |")
print(f"| stream (first event)     | {ms*1000:8.1f} ms |")
print(f"| collapse factor          | {mb/ms:8.1f}x |")
print(f"raw batch={[round(x*1000,1) for x in b]} stream={[round(x*1000,1) for x in s]}")