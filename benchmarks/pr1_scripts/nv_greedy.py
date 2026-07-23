import os, json
from nanovllm import LLM, SamplingParams
prompts = ["The capital of France is", "def fibonacci(n):", "In 1969, humans first"]
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"),
          enforce_eager=True, max_model_len=4096)
out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True))
json.dump([o["token_ids"] for o in out], open("nv.json", "w"))
print("wrote nv.json")
