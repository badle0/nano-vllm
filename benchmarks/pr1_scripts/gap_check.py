import os, sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
i, div = int(sys.argv[1]), int(sys.argv[2])          # prompt index, divergence index
path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
prompts = ["The capital of France is", "def fibonacci(n):", "In 1969, humans first"]
tok = AutoTokenizer.from_pretrained(path)
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
nv = json.load(open("nv.json"))
ids = tok(prompts[i], return_tensors="pt").input_ids[0].tolist() + nv[i][:div]
with torch.no_grad():
    logits = m(torch.tensor([ids]).cuda()).logits[0, -1].float()
top2 = logits.topk(2)
print("top-2 tokens:", top2.indices.tolist(), "logit gap:", (top2.values[0]-top2.values[1]).item())
