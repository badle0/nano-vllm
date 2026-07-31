# p7: policy data at the exact long-prompt shape (2x2048) + interactive sanity (1024 bucket)
import os, random, torch
from time import perf_counter
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context

llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
mr = llm.model_runner
random.seed(0)
sp = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)

def timed(fn, n=3):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((perf_counter() - t0) * 1e3)
    return min(ts)

def setup(lens, budget):
    llm.scheduler.max_num_batched_tokens = budget
    for L in lens:
        llm.add_request([random.randint(1000, 150000) for _ in range(L)], sp)
    seqs, _ = llm.scheduler.schedule()
    return mr.prepare_ragged(seqs), get_context()

(ids, pos), ctx = setup([2048, 2048], 16384)
with torch.inference_mode():
    t_paged = timed(lambda: mr.model(ids, pos))
    t_graph = timed(lambda: mr.run_model(ids, pos, True))
    set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
                ctx.max_seqlen_k, ctx.slot_mapping, None, None)
    t_fresh = timed(lambda: mr.model(ids, pos))
print(f"P7 2x2048: fresh-eager {t_fresh:.1f} | paged-eager {t_paged:.1f} | graphed(4096) {t_graph:.1f} ms")
reset_context(); llm.scheduler.cancel_all()

(ids, pos), ctx = setup([64] * 16, 16384)            # the interactive-prefill shape
with torch.inference_mode():
    t_g = timed(lambda: mr.run_model(ids, pos, True))
    t_e = timed(lambda: mr.model(ids, pos))
print(f"P7 16x64:  paged-eager {t_e:.1f} | graphed(1024) {t_g:.1f} ms")
reset_context(); llm.scheduler.cancel_all()