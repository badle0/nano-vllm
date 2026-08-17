# /tmp/graph_feasibility.py — can we capture a C=512 prefill chunk, and what does replay cost?
import os, random, torch
from time import perf_counter
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import reset_context

llm = LLM(
    os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False,
    max_model_len=4096, max_num_batched_tokens=512,
)
random.seed(0)
llm.add_request([random.randint(1000, 150000) for _ in range(3968)],
                SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True))
seqs, _ = llm.scheduler.schedule()
ids, pos = llm.model_runner.prepare_prefill(seqs)     # first chunk: fresh-K/V varlen path
model = llm.model_runner.model
with torch.inference_mode():
    model(ids, pos); torch.cuda.synchronize()          # warm this exact shape
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(ids, pos)
    torch.cuda.synchronize()
    print("capture: OK")
    for rep in range(3):
        t0 = perf_counter(); g.replay(); torch.cuda.synchronize()
        print(f"replay {rep}: {(perf_counter()-t0)*1e3:6.2f} ms")
reset_context(); llm.scheduler.cancel_all()
