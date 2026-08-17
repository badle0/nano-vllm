"""Bounded-window plus exact-flush decode cost. Run from repo root. No GPU needed."""
import statistics
from time import perf_counter
from transformers import AutoTokenizer
from nanovllm import StreamingDetokenizer

tok = AutoTokenizer.from_pretrained("/workspace/models/Qwen3-0.6B")
ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 400)
for n in (64, 256, 1024, 2048):
    seq = ids[:n]
    per = []
    for _ in range(3):
        d = StreamingDetokenizer(tok)
        t = perf_counter()
        for token_id in seq:
            d.feed(0, token_id)
        d.flush(0)
        per.append((perf_counter() - t) / n * 1e6)
    print(f"| n={n:5d} | {statistics.median(per):8.1f} us/token |")
