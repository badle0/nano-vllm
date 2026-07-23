import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
prompts = ["The capital of France is", "def fibonacci(n):", "In 1969, humans first"]
tok = AutoTokenizer.from_pretrained(path)
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
res = []
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        g = m.generate(ids, do_sample=False, max_new_tokens=32, use_cache=True)
    res.append(g[0, ids.shape[1]:].tolist())
json.dump(res, open("hf.json", "w"))
print("wrote hf.json")
