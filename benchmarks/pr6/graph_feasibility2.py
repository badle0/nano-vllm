# /tmp/graph_feasibility2.py — the paged-varlen path: chunk 2, block_table gather under capture
import os, random, torch
from time import perf_counter
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, reset_context

llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
random.seed(0)
llm.scheduler.max_num_batched_tokens = 512
llm.add_request([random.randint(1000, 150000) for _ in range(3968)],
                SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True))
llm.step()                                             # chunk 1 through the real engine
seqs, is_prefill = llm.scheduler.schedule()            # chunk 2: resumption path
assert is_prefill and seqs[0].num_cached_tokens == 512
ids, pos = llm.model_runner.prepare_prefill(seqs)
assert get_context().block_tables is not None, "not on the paged path"
model = llm.model_runner.model
with torch.inference_mode():
    model(ids, pos); torch.cuda.synchronize()          # warm this shape
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        model(ids, pos)
    torch.cuda.synchronize(); print("capture paged-varlen: OK")
    for rep in range(3):
        t0 = perf_counter(); g.replay(); torch.cuda.synchronize()
        print(f"replay {rep}: {(perf_counter()-t0)*1e3:6.2f} ms")
reset_context(); llm.scheduler.cancel_all()