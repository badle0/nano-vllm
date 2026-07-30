# greedy token-gate: natural prompts, dumps token_ids to argv[1]
import json, sys, os
from nanovllm import LLM, SamplingParams
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
prompts = ["The history of the Roman Empire begins with",
           "In machine learning, gradient descent works by",
           "The recipe calls for two cups of flour and",
           "Photosynthesis converts sunlight into"]
out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True), use_tqdm=False)
json.dump([o["token_ids"] for o in out], open(sys.argv[1], "w"))
print("wrote", sys.argv[1])