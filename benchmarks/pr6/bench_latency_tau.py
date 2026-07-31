# benchmarks/pr6/bench_latency_tau.py — run the PINNED bench_latency.py (fetched via
# git show 5b6f013, bytes unmodified) with the scheduler budget shrunk post-construction
# (the tests' established runtime knob). This is the C3 tau sweep instrument: capture
# sizes stay default; only the scheduling budget changes. NOT the pinned protocol run —
# label results as tau-wrapped.
# usage: bench_latency_tau.py TAU
import subprocess, sys
if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    sys.exit("usage: bench_latency_tau.py TAU")
tau = int(sys.argv[1])

import nanovllm
_LLM = nanovllm.LLM
class _TauLLM(_LLM):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.scheduler.max_num_batched_tokens = tau
nanovllm.LLM = _TauLLM

src = subprocess.check_output(
    ["git", "show", "5b6f013:benchmarks/bench_latency.py"], text=True)
print(f"=== tau={tau} (pinned 5b6f013, tau-wrapped) ===")
exec(compile(src, "bench_latency.py@5b6f013", "exec"), {"__name__": "__main__"})
