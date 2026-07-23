# PR 2 (top-k sampling) — validation record

Branch: feat/topk-sampling (single commit over feat/greedy-sampling e4a8c91 over upstream bb823b3)
Hardware: Vast.ai A100, env in pr2_env.txt. Model: Qwen3-0.6B bf16.
Protocol: all comparisons interleaved within one session; first runs discarded (compile warmup);
medians reported. Cross-session absolute throughput varies ~3% on this host (8670 vs 8950 tok/s
for identical code across two rentals) — only within-session deltas are quoted.

## Decode step-time (torch.cuda.Event around run_model / sampler+tolist, B=256, 512-token
## prefill, 64 decode steps, median of 61 after dropping 2 warmup lines)

| configuration                     | model ms | sampler ms | sampler share |
|-----------------------------------|----------|------------|---------------|
| PR1 greedy baseline               | 14.651   | 0.821      | 5.3%          |
| top-k, unconditional sort (naive) | 14.663   | 8.578      | 36.9%         |
| top-k final, disabled (fast path) | 14.655   | 0.820      | 5.3%          |
| top-k final, enabled (top_k=50)   | 14.646   | 8.577      | 36.9%         |

The "sampler" column includes .tolist() (the step's GPU->CPU sync), i.e. it is the quantity
that lands on the decode critical path.

## The design story (chronological, predictions written before measurement)

1. Roofline estimate predicted the full-vocab sort at ~10x the elementwise sampler cost.
2. First implementation ran the sort unconditionally. Measured: 8.578 ms vs 0.821 ms
   baseline = +7.76 ms/step, a 9-10.5x multiple — estimate validated. At B=256 this taxed
   the *disabled* default path ~33% of decode step time.
3. Fix: prepare_sample passes top_ks=None when every sequence has top_k=-1; the sampler's
   sort block is guarded by `if top_ks is not None`. torch.compile specializes on the
   None-vs-tensor distinction, producing two graphs; the disabled graph is op-for-op
   identical to the PR1 sampler.
4. Re-measured: disabled path restored to baseline (0.820 ms); enabled path pays 8.577 ms.
   Enabled cost is the honest price of top-k at this vocab (151,936) and batch; it is paid
   only by batches containing at least one top_k-enabled sequence (the sort is batch-wide).

Note: two-specialization design implies one extra compile the first time each path is hit
per process (~300 ms, observed on the first enabled step). One-time, bounded.

Note: Inductor emits "Online softmax is disabled ... split reduction" for the vocab-dim
softmax in the compiled sampler; informational, correctness-unaffected.

## End-to-end no-regression (bench.py, temperature=0.6, top_k disabled, 3 runs/branch,
## interleaved; first run of each discarded)

| branch               | throughput tok/s (kept runs) | median |
|----------------------|------------------------------|--------|
| feat/greedy-sampling | 8949.88, 8948.51             | 8948.5 |
| feat/topk-sampling   | 8920.94, 8934.06             | 8927.5 |

Delta -0.2%, within the 0.5% within-branch spread: the disabled default path shows no
regression. (An earlier benchmark pair in this session was voided: a dirty working tree
meant both "branches" ran the naive top-k code; it measured 7050 tok/s on both — which is
itself the end-to-end signature of the unconditional sort, consistent with the step-time
arithmetic: +7.76 ms on a ~15 ms step.)

## Seed-equivalence (unit test, compiled-vs-compiled)

With top_ks=None, the sampler is bit-identical to the compiled PR1 sampler under a fixed
seed, RNG stream included (the sort/mask/scatter draw no randomness; with the fast path
the disabled graph is literally the PR1 graph). Note: eager and compiled RNG streams
differ under the same seed (Inductor lowers exponential_ through its own RNG), so the
test compiles its reference — bit-equality claims must compare like with like.

## Kept-set semantics vs HF TopKLogitsWarper

Value-threshold semantics: keep all tokens with logit >= k-th largest value (ties at the
boundary all survive; deterministic, sort-stability-independent). bf16 ties are common in
practice (~256 representable values per binade).
xval_topk.py, 200 trials x {1,5,50} x 4 rows incl. forced ties: [TODO: fill from xval run]

## Unit tests

10/10 (pytest tests/ -v on the box, torch.compile enabled): greedy/argmax equality, mixed-
batch routing, stochastic non-constancy, params validation (temperature and top_k bounds),
seed-equivalence vs compiled PR1 reference, top-k support containment, boundary-tie
inclusion, top_k=1 == greedy.
