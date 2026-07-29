# /tmp/dispatch_probe.py — locate the flat ~47ms: CPU dispatch vs GPU wall vs plumbing
import os, random, torch
from time import perf_counter
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import reset_context

llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
sp = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
random.seed(0)

for C in (64, 512, 3968):
    llm.scheduler.max_num_batched_tokens = C
    llm.add_request([random.randint(1000, 150000) for _ in range(3968)], sp)
    seqs, _ = llm.scheduler.schedule()
    ids, pos = llm.model_runner.prepare_prefill(seqs)
    with torch.inference_mode():
        llm.model_runner.model(ids, pos)          # warmup this shape
        torch.cuda.synchronize()
        for rep in range(2):
            t0 = perf_counter()
            llm.model_runner.model(ids, pos)
            t_cpu = (perf_counter() - t0) * 1e3   # CPU done issuing
            torch.cuda.synchronize()
            t_wall = (perf_counter() - t0) * 1e3  # GPU done executing
            print(f"C={C:4d} rep{rep}: dispatch {t_cpu:6.1f} ms | wall {t_wall:6.1f} ms")
    reset_context(); llm.scheduler.cancel_all()