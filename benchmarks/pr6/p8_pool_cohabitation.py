# benchmarks/pr6/p8_pool_cohabitation.py — is the bench regression caused by varlen graphs sharing the pool?
# usage: p8_pool_cohabitation.py {varlen|novarlen}
import os, sys, time
from random import randint, seed
if len(sys.argv) != 2 or sys.argv[1] not in ("varlen", "novarlen"):
    sys.exit("usage: p8_pool_cohabitation.py {varlen|novarlen}")
if sys.argv[1] == "novarlen":
    from nanovllm.engine.model_runner import ModelRunner
    ModelRunner.capture_varlen_graphs = lambda self: None   # State C code, minus the captures
from nanovllm import LLM, SamplingParams

seed(0)
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
prompts = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(256)]
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024)) for _ in range(256)]
llm.generate(["Benchmark: "], SamplingParams())
import torch; print("free MiB pre-generate:", torch.cuda.mem_get_info()[0] // 2**20)
t = time.time()
llm.generate(prompts, sps, use_tqdm=False)
t = time.time() - t
print(f"P8 [{sys.argv[1]:8s}] {sum(sp.max_tokens for sp in sps)/t:.2f} tok/s   ({t:.2f}s)")