import os, random, torch
from time import perf_counter
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context

llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
random.seed(0)
llm.scheduler.max_num_batched_tokens = 512
for _ in range(8):
    llm.add_request([random.randint(1000, 150000) for _ in range(64)],
                    SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True))
seqs, is_prefill = llm.scheduler.schedule()
assert is_prefill and len(seqs) == 8                   # 8 segments x 64 tokens, T=512
ids, pos = llm.model_runner.prepare_prefill(seqs)
ctx = get_context()
model = llm.model_runner.model

def capture(m):
    set_context(True, cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=m, max_seqlen_k=ctx.max_seqlen_k,
                slot_mapping=ctx.slot_mapping, block_tables=ctx.block_tables)
    with torch.inference_mode():
        model(ids, pos); torch.cuda.synchronize()      # warm this grid config
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            model(ids, pos)
        torch.cuda.synchronize()
    return g

for label, m in (("tight M=64 ", 64), ("loose M=512", 512)):
    g = capture(m)
    for rep in range(4):
        torch.cuda.synchronize(); t0 = perf_counter()
        g.replay(); torch.cuda.synchronize()
        if rep: print(f"P2 {label}: replay {(perf_counter()-t0)*1e3:6.2f} ms")
reset_context(); llm.scheduler.cancel_all()