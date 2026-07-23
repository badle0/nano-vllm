import os
from random import randint, seed
from nanovllm import LLM, SamplingParams
seed(0)
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
prompts = [[randint(0, 10000) for _ in range(512)] for _ in range(256)]
llm.generate(prompts, SamplingParams(temperature=0.6, top_k=50, ignore_eos=True, max_tokens=64), use_tqdm=False)
