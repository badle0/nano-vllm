# benchmarks/pr6/gate_generate.py — fixed-seed generate, dumps token_ids; the byte-gate instrument
import json, sys, os, torch
from nanovllm import LLM, SamplingParams
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
out = llm.generate(["The capital of France is", "def fibonacci(n):", "In 1969, humans first"],
                   SamplingParams(temperature=0.6, max_tokens=32, ignore_eos=True), use_tqdm=False)
json.dump([o["token_ids"] for o in out], open(sys.argv[1], "w"))
print("wrote", sys.argv[1])