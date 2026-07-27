"""Cumulative-decode cost vs sequence length. Run from repo root. No GPU needed."""
import os, statistics
from time import perf_counter
from transformers import AutoTokenizer
from nanovllm import StreamingDetokenizer

tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/huggingface/Qwen3-0.6B"))
ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 400)
for n in (64, 256, 1024, 2048):
    seq = ids[:n]
    per = []
    for _ in range(3):
        d = StreamingDetokenizer(tok)
        t = perf_counter()
        for i, t_id in enumerate(seq): d.feed(0, t_id)
        per.append((perf_counter() - t) / n * 1e6)
    print(f"| n={n:5d} | {statistics.median(per):8.1f} us/token |")