# greedy token-gate under a forced budget: same prompts/params as gate_generate.py,
# with the scheduler budget shrunk at runtime (the tests' established knob) so the
# C3 scheduler must chunk the prompts and mix decodes into ragged steps.
# usage: gate_generate_tau.py TAU OUTPUT.json
import json, sys, os
if len(sys.argv) != 3 or not sys.argv[1].isdigit():
    sys.exit("usage: gate_generate_tau.py TAU OUTPUT.json")
tau = int(sys.argv[1])
from nanovllm import LLM, SamplingParams
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
llm.scheduler.max_num_batched_tokens = tau
prompts = ["The history of the Roman Empire begins with",
           "In machine learning, gradient descent works by",
           "The recipe calls for two cups of flour and",
           "Photosynthesis converts sunlight into"]
out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True), use_tqdm=False)
json.dump([o["token_ids"] for o in out], open(sys.argv[2], "w"))
print("wrote", sys.argv[2], "tau", tau)
