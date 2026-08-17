# benchmarks/pr6/bench_latency_tau.py — run the PINNED bench_latency.py (fetched via
# git show 5b6f013, bytes unmodified) with the scheduler budget supplied to the
# constructor. This is the C3 tau sweep instrument: graph capture and scheduling
# are configured coherently for the requested budget. NOT the pinned protocol run —
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
        requested = k.get("max_num_batched_tokens", tau)
        if requested != tau:
            raise ValueError(
                f"pinned benchmark requested tau={requested}, wrapper requires {tau}"
            )
        k["max_num_batched_tokens"] = tau
        k["max_num_seqs"] = min(k.get("max_num_seqs", 512), tau)
        super().__init__(*a, **k)
nanovllm.LLM = _TauLLM

src = subprocess.check_output(
    ["git", "show", "5b6f013:benchmarks/bench_latency.py"], text=True)
print(f"=== tau={tau} (pinned 5b6f013, tau-wrapped) ===")
exec(compile(src, "bench_latency.py@5b6f013", "exec"), {"__name__": "__main__"})
