import os, statistics
from random import randint, seed
from nanovllm import LLM, SamplingParams

seed(0)
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
llm.generate([[randint(0, 10000) for _ in range(64)] for _ in range(16)],
             SamplingParams(ignore_eos=True, max_tokens=4), use_tqdm=False)

short = [[randint(0, 10000) for _ in range(64)] for _ in range(16)]
long_ = [[randint(0, 10000) for _ in range(2048)] for _ in range(2)]
sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=256)

collected = {}
def drain(out):
    for seq_id, token_ids, metrics in out:
        collected[seq_id] = metrics

for p in short:
    llm.add_request(p, sp)
for _ in range(40):            # get the interactive group deep into decode
    out, _ = llm.step()
    drain(out)
for p in long_:
    llm.add_request(p, sp)
while not llm.is_finished():
    out, _ = llm.step()
    drain(out)

groups = {"interactive": [], "long": []}
for sid, m in collected.items():
    groups["interactive" if m["num_prompt_tokens"] == 64 else "long"].append(m)
for name, ms in groups.items():
    ttfts = sorted(m["ttft"] * 1e3 for m in ms)
    mitl  = [m["mean_itl"] * 1e3 for m in ms]
    xitl  = [m["max_itl"] * 1e3 for m in ms]
    print(f"{name:12s} n={len(ms):2d}  TTFT p50={ttfts[len(ttfts)//2]:8.1f}ms p99={ttfts[-1]:8.1f}ms" # p99 == max at these n; labels kept for the chunked-prefill rerun
          f"  mean_ITL={statistics.median(mitl):6.2f}ms  max_ITL p50={statistics.median(xitl):7.1f}ms")