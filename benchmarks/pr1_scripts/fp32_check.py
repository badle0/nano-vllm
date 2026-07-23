import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
tok = AutoTokenizer.from_pretrained(path)
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).cuda().eval()
nv = json.load(open("nv.json"))
ids = tok("In 1969, humans first", return_tensors="pt").input_ids[0].tolist() + nv[2][:31]
with torch.no_grad():
    lg = m(torch.tensor([ids]).cuda()).logits[0, -1]
t2 = lg.topk(2)
print("fp32 top-2:", t2.indices.tolist(), "gap:", (t2.values[0]-t2.values[1]).item())
print("decoded:", repr(tok.decode([1096])), "vs", repr(tok.decode([576])))
