# V7: bounded experimental speculative decoding qualification

The executable path is no longer a shadow/draft-discard milestone. It performs
draft catch-up and proposals, target verification, modified rejection, correction
or bonus sampling, atomic KV/token commit, and ordinary streaming delivery.
Speculation remains **opt-in and experimental**. This document distinguishes
correctness qualification from acceleration; a slower result is not a correctness
failure and is not advertised as a speedup.

## Revisions and reproducibility

- Runtime: `a165b654660e60ca60cecd47b89c95de1d65735b` and its unchanged `nanovllm`
  subtree in later V7 commits.
- Heterogeneous cold correctness producer: `a715a199d413a67ba563271f1b4fa8fe87f00eaa`.
- Paired timing producer/harness: `89829e6052c17e0ef4fcd65e294d0f1e78139184`.
- Pre-V5 speculation-off control: `2678d764ad0341bbfbdd2a93ac0e5528959058a4`.
- Target Qwen3-4B: HF revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Draft Qwen3-0.6B: the same weight digest used by the V2–V6 retained archives.

Every timing process checks its imported runtime against the declared Git blobs.
The older control is an isolated `git archive` export, not a checkout/rewrite of
the working branch. The phase profiler records its own committed source/harness
identity. Later evidence/document commits do not change the measured runtime.
The archive records full weight/configuration hashes; it does not contain models.

The 4B checkpoint was downloaded into a temporary RAM-backed directory because
workspace disk space was insufficient. It will not survive reboot. No original
repository, historical worktree, or user file was deleted. An initial attempt to
put compiler caches there failed because `/dev/shm` is non-executable; the retained
cold run uses fresh disk-backed compiler caches instead.

## Correctness qualification

The heterogeneous graph-on run executes **271 speculative cycles** and all
**80** registered `(B=1..4, K=1..4, sampling-family)` cells. Matching graph-off
controls are token-identical for ragged greedy batches at context boundaries
16/255/256/257. Both target lanes execute, with no compiler/capture change inside
any guarded speculative interval. Natural residual fallback count is zero; the
explicitly forced empty-residual recovery is counted exactly once.

The same run covers cache reruns, streaming reconstruction, actual pending-burst
close and GC, post-verification failure with RNG/state rollback and retry, and
non-vacuous causal verification. The normal live-allocation peak is 170,415,104
bytes, below its registered workspace model. The lifecycle fixture now waits for
an actual accepted burst: two heterogeneous models need not accept their first
proposal. This is a harness correction, not a weakened lifecycle assertion.

The preceding [V5/V6 archive](10_v6_retained_validation.md) independently retains
335 cycles across eager/graph, on/off, and automatic-KV-sizing controls. Its source
subtree is identical to the V7 runtime. CPU property/oracle and fault-injection
tests cover cases that finite natural-model samples cannot force, including
rejection and EOS at every position, every commit failure phase, and numerical
residual corners. Neither archive claims to prove the absence of all bugs.

"Exact" refers to modified rejection for the represented canonical FP32 p/q
laws, evaluated robustly in FP64. It does not promise bit-identical probabilities
from differently batched BF16 model kernels. Matching-mode greedy compatibility
is a separate gate, not a tolerance silently applied to sampled output IDs.
Rollback tests inject recoverable failures; they do not promise reuse after a
device-fatal CUDA error or a poisoned CUDA context.

## Timing protocol and interpretation

The [preregistered protocol](11_v7_measurement_policy.md) fixes five fresh-process
pairs in AB/BA/AB/BA/AB order, two complete warmups per cell, and seeds 17/23/41.
Primary cells: B=1/4/8, contexts 32/256, 64 output tokens, greedy/plain/top-k/top-p/
combined sampling. Both sides use graph-enabled engines, 64 KV blocks, model and
input-token limits 4096, and memory utilization .8. EOS is ignored to keep work
length fixed. Different sampled output IDs are expected; greedy controls must
match. CPU/GPU contention, temperature, and timed compilation exclusions were
declared before the runs, not chosen after seeing latency.

Prompts repeat a fixed prose/code phrase to the requested token length; this is
a controlled synthetic workload, not a production request trace. Warmups and
samples reuse those prompts, so full target-prefix blocks may be cached. Draft
catch-up still obeys its independent coverage contract. Reported TTFT is therefore
for this warmed workload, **not cold-request TTFT**. Cold compiler/first-eligible
route checks are a separate correctness experiment, not a cold-serving latency
benchmark. No quality, production-tail-latency, or arrival-process claim is made.

`speed ratio = speculation-off wall seconds / speculation-on wall seconds`.
Values below one mean speculation is slower. Each process-pair contributes the
median of its three same-seed ratios. The reported median and exact bootstrap
interval resample **five process pairs**, not 15 supposedly independent seeds.
Intervals are descriptive on this host, not universal confidence guarantees.

The separate pre-V5/current speculation-off comparison uses the same protocol
with three B=1/4/8 plain-sampling cells and a preregistered ±5% noise band.
Supplemental cells cover B=2/16/32/64/128, contexts 1024/2048, output lengths
1/2/4/5/256, ordinary and 2-ms/token slow streaming consumers, and configured
K=5/6 (capped to effective K<=4). Supplemental cells have three seeds but only
one process per side; they do not inherit the five-pair headline confidence.

## Roofline calculation

All units below use bytes and FLOP/s; GB and TB are decimal unless labeled GiB.
The report calibrates this device using 256-MiB device copies, counting both read
and write traffic, and a 4096-square BF16 GEMM. Each has ten warmups and fifty
CUDA-event-timed repetitions. These are *attainable microbenchmark ceilings*, not
a promise that decode, sorting, rejection, or small GEMMs attain either ceiling.

The target has L=36 layers, hidden width H=2560, intermediate width I=9728,
32 query heads, 8 KV heads, head dimension D=128, vocabulary V=151936, and BF16
weights/KV. Its unique weight storage is 8,044,936,192 bytes. A naive sum over
parameter objects gives 8,822,848,512 bytes because the embedding and LM-head
objects share 777,912,320 bytes. The diagnostic storage ledger deduplicates
storage pointers and verifies this alias. The timing JSON's historical
`target_weight_bytes`/`draft_weight_bytes` fields are **parameter-object sums**,
not physical resident-byte measurements; analysis explicitly corrects them using
the retained storage ledger.

For one query token, dense linear-weight elements are

```text
L * [2*H*(query_heads*D) + 2*H*(KV_heads*D) + 3*H*I] + V*H
= 36 * [20,971,520 + 5,242,880 + 74,711,040] + 388,956,160
= 4,022,272,000
```

Thus linear work is about **8.044544 GFLOP/query**, counting multiply-add as two
FLOPs. Embedding lookup is not charged as a dense matrix multiplication; norms,
rotary, activation, sampling and attention are additional work. Under ideal
weight reuse, ordinary batch-B decode performs this linear work B times while
streaming roughly one target-weight set per step. Cache reuse, extra reads, and
kernel launch costs make this an optimistic model, not an exact trace.

Target KV storage per cached token is

```text
2(K,V) * 36 layers * 8 KV heads * 128 head dimension * 2 BF16 bytes
= 147,456 bytes = 144 KiB.
```

At context C, an ideal one-step KV-read floor is approximately `B*C*147456` bytes,
plus KV writes and metadata. Attention work is approximately
`4*L*B*C*query_heads*D` FLOPs. GQA reuses KV across query-head groups; charging
32 independent KV heads would overcount the ideal floor. The 64 physical blocks,
each of 256 tokens, occupy 2,415,919,104 target-KV bytes regardless of occupancy.
The draft has 28 layers and the same KV-head count/dimension: 114,688 bytes per
cached token, or 1,879,048,192 bytes for its 64-block pool. The two physical pools
therefore occupy exactly 4 GiB together. Logical block IDs are shared; physical
target and draft KV tensors are not. Driver memory occupancy also includes CUDA
context, reserved allocator blocks, graphs and buffers, not just weights and KV.

With measured bandwidth `BW` and compute roof `F`, a useful optimistic step model
is `max((target_weights + KV_reads)/BW, (linear_FLOPs + attention_FLOPs)/F)`.
Different kernels execute serially and cannot necessarily overlap their ceilings;
the max is a lower-bound model, not a wall-time prediction. Weight reuse, HBM
traffic beyond the floor, launch overhead, and CPU preparation must be checked
against the phase evidence rather than fitted away.

At B=4, K=4, retaining canonical FP32 proposal and target laws alone needs
`4 * (K + K + 1) * V * 4 = 21,878,784` bytes. This is **not** the workspace peak:
logits, filters, sort scratch, FP64 acceptance/residual temporaries, and allocator
margin are priced separately. The retained 20-row top-p probe bounds private
scratch by `20 * V * 52` bytes. See [V5's lifetime discussion](09_v5_verified_execution.md).

## Why a valid speculative implementation can be slower

For a steady-state row, the break-even condition is

```text
emitted_tokens_per_cycle * ordinary_decode_step_time
  > draft + verify + accept + bonus + commit + catch-up + host overhead.
```

Pure stochastic batches use one paged all-query target pass, but draft decoding
still makes K sequential calls. Exact full-vocabulary FP32 probability retention,
FP64 robust rejection, top-k/top-p transforms, and host synchronization have real
costs. Good prefix acceptance alone does not imply a win. Initial catch-up and
short-output tails further weaken end-to-end amortization.

Greedy/mixed batches deliberately use K+1 ordinary target calls to preserve
matching-mode BF16 behavior. Even full acceptance does not remove target calls;
draft and rejection work are added. The familiar single-target-pass speculative
speedup formula is therefore inapplicable to this compatibility lane.

The diagnostic profiler synchronizes at phase boundaries and retains each cycle's
acceptance length and phase times for graph/eager, B=1/4, and prose/code/repetitive/
adversarial inputs. Its overhead makes those timings unsuitable as headlines.
The accepted-prefix fraction is not an unbiased per-position acceptance law;
condition on reaching a position when analyzing the length histogram.

## Operational limits

- Speculation is disabled by default. Opt in with compatible local target/draft
  model paths and `num_speculative_tokens=4`; see [the API example](09_v5_verified_execution.md).
- Only TP=1 and the exact sampler backend are supported. FlashInfer speculation
  and TP>1 fail early; neither is silently routed through an unqualified backend.
- Live B>4 bypasses speculation as a whole. K>4 is capped. Budget, model-length,
  completion, warmed-route and free-block limits can reduce K or cause bypass.
- Initial draft catch-up is charged to the aggregate input budget. With budget
  4096, B=4/C=1024 or 2048 cannot fit catch-up plus verification, so those cells
  can remain ordinary decoding; this is not a long-context speculative speedup.
- Draft weights and KV stay resident during bypass. Equal-block-capacity results
  are not an automatic-sizing throughput/capacity claim.
- Public tokens are committed tokens only; bursts preserve per-request order.
  Compute ITL can be zero within a burst, while consumer delivery still takes time.
- There is no claimed optimal crossover, adaptive K, greedy acceleration,
  graph-captured parallel target verifier, broad model-fleet certification, or
  production speedup. Those are separate optimization/qualification tasks.

## Result status

The five-pair timing, off-regression, supplemental and phase runs are in progress.
Final numeric tables, archive digest and test totals will replace this paragraph
only after all raw artifacts pass the model-free validator.
