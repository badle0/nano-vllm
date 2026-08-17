import os, random, statistics
from time import perf_counter
from nanovllm import LLM, SamplingParams
from probe_common import clean_exit

sp = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
T = 3968                                   # +1 completion stays under max_model_len
random.seed(0)
mk = lambda: [random.randint(1000, 150000) for _ in range(T)]   # unique prompts: no prefix-cache hits

print("| C | steps | median step ms | total prefill ms | us/token |")
for C in (64, 128, 256, 512, 1024, 2048, 3968):
    llm = LLM(
        os.path.expanduser("~/huggingface/Qwen3-0.6B"),
        enforce_eager=False,
        max_model_len=4096,
        max_num_batched_tokens=C,
        max_num_seqs=min(512, C),
    )
    try:
        llm.generate([mk()[:256]], sp, use_tqdm=False)          # per-config warmup
        med, tot = [], []
        for _ in range(3):
            llm.add_request(mk(), sp)
            times = []
            while not llm.is_finished():
                t0 = perf_counter(); llm.step(); times.append((perf_counter()-t0)*1e3)
            med.append(statistics.median(times)); tot.append(sum(times))
        m, s = statistics.median(med), statistics.median(tot)
        print(f"| {C:5d} | {-(-T//C):3d} | {m:8.2f} | {s:8.1f} | {s/T*1e3:6.1f} |")
    finally:
        clean_exit(llm)
