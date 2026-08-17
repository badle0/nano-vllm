# benchmarks/pr6/probe_common.py — shared plumbing for probes p5+. No GPU work at import.
# House rule preserved: parse_arm() runs the argument check before any torch/nanovllm import.
import atexit, os, sys
from time import perf_counter

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")


def parse_arm(*arms):
    """Usage-gate first, GPU work second. 'novarlen' stubs the whole varlen init
    (bucket captures AND the post-restore pre-touch) so the arm matches dev.
    Class-level patch, applied before any LLM is constructed. Returns the arm."""
    if len(sys.argv) != 2 or sys.argv[1] not in arms:
        sys.exit(f"usage: {os.path.basename(sys.argv[0])} {{{'|'.join(arms)}}}")
    if sys.argv[1] == "novarlen":
        from nanovllm.engine.model_runner import ModelRunner
        ModelRunner.capture_varlen_graphs = lambda self: None
        ModelRunner._pretouch_eager_prefill = lambda self: None
    return sys.argv[1]


def make_llm(max_num_batched_tokens=16384):
    from nanovllm import LLM
    return LLM(
        MODEL,
        enforce_eager=False,
        max_model_len=4096,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=min(512, max_num_batched_tokens),
    )


def timed(fn, n=3):
    """Best-of-n wall time in ms, synchronized; first (warmup) call discarded."""
    import torch
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((perf_counter() - t0) * 1e3)
    return min(ts)


def bench_workload(llm, warmup=True):
    """The bench.py-shaped load shared by p9-p11: seed(0), 256 prompts of 100-1024
    random tokens, matching SamplingParams, then the standard 1-prompt warmup.
    randint stream is identical to the pre-refactor scripts (nanovllm consumes no
    Python random, so seeding here vs before the LLM ctor is equivalent)."""
    from random import randint, seed
    from nanovllm import SamplingParams
    seed(0)
    prompts = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(256)]
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024)) for _ in range(256)]
    if warmup:
        llm.generate(["Benchmark: "], SamplingParams())
    return prompts, sps


def ragged_step(llm, lens, budget=None):
    """Queue one prompt per length (caller seeds `random`), schedule a single step,
    prepare it. Returns (input_ids, positions, ctx) with the ragged context LIVE —
    caller must reset_context() and scheduler.cancel_all() when done."""
    import random
    from nanovllm import SamplingParams
    from nanovllm.utils.context import get_context
    if budget is not None and llm.scheduler.max_num_batched_tokens != budget:
        raise ValueError(
            "ragged_step budget must be configured by make_llm before graph capture: "
            f"configured={llm.scheduler.max_num_batched_tokens}, requested={budget}"
        )
    sp = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
    for L in lens:
        llm.add_request([random.randint(1000, 150000) for _ in range(L)], sp)
    seqs, _ = llm.scheduler.schedule()
    ids, pos = llm.model_runner.prepare_ragged(seqs)
    return ids, pos, get_context()


def clean_exit(llm):
    """Explicit teardown + deregister the atexit hook, so exit() runs exactly once
    (the double-call raised a benign AttributeError; safe with work still queued —
    verified by the P11 varlen run)."""
    try:
        llm.exit()
        atexit.unregister(llm.exit)
    except Exception:
        pass
