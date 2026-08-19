# Roofline and benchmark experiments for the major fork contributions

Status: retained contribution evidence assembled on 2026-08-19 from
`fork-main` at `4b8053fbd423f69db355635112ef3dfce658592f`
(`v0.3.0-rc.1`).

This document records the performance models, benchmark protocols, acceptance
gates, measurements, and unresolved limits for three contribution groups:

1. greedy, top-k, and top-p sampling;
2. token streaming; and
3. chunked prefill.

Request metrics are intentionally excluded. Their cost is covered by a separate
paired certification archive and does not need another roofline in this report.

## 1. How to read this report

### 1.1 Evidence is tied to code, not just to a branch name

The raw artifacts were produced at specific clean commits. Those commits are
ancestors of the release-candidate stack, but a later merge can still change a
shared file. Accordingly, this report uses the following labels:

- **certified**: the exact pinned artifact passed its predeclared release gates;
- **measured pass**: the experiment passed its declared functional or
  performance gate but is not a full release certificate;
- **diagnostic**: useful attribution or component data that is deliberately
  ineligible to certify the public workload;
- **historical**: genuine evidence from an earlier contribution commit whose
  provenance or source identity is weaker than the current certificate;
- **rejected**: a prototype that was measured and explicitly not integrated;
- **unverified**: a required experiment could not be run on the available
  hardware.

The distinction matters most for token streaming and chunked-prefill latency.
Their retained certificates target contribution commits rather than the final
RC merge. The current full test suite is regression evidence, but a fresh A100
run at the RC commit would be required to call the RC byte-for-byte performance
certified.

### 1.2 “Roofline” is feature-specific

The classical kernel roofline is:

```text
attainable work rate <= min(peak compute, memory bandwidth * arithmetic intensity)
```

That is useful for full-vocabulary sampling kernels, but it is the wrong sole
model for an API-delivery feature or a scheduler. This report therefore uses
four related kinds of roof:

1. **device traffic roof** — minimum time implied by bytes moved through HBM;
2. **algorithmic work roof** — how cost grows with batch, vocabulary, queue,
   token budget, or sequence length;
3. **service-level budget** — maximum latency or throughput loss allowed by the
   public acceptance contract; and
4. **structural roof** — an invariant such as bounded pending events, bounded
   detokenizer input, bounded admitted requests, or bounded transport frames.

For a baseline elapsed time `T0`, enabled time `T1`, baseline throughput
`R0`, and enabled throughput `R1`:

```text
throughput change (%) = 100 * (R1 / R0 - 1)
time overhead (%)      = 100 * (T1 / T0 - 1)
```

If the maximum permitted throughput loss is `L`, the corresponding time budget
is not `L`; it is:

```text
T1 <= T0 / (1 - L)
Delta T <= T0 * (1 / (1 - L) - 1)
```

Thus a 10% throughput-loss ceiling permits at most 11.111% elapsed-time
overhead. Paired experiments use within-pair ratios first and aggregate those
ratios; they do not infer a result by dividing two unrelated global medians.

### 1.3 Common measured platform

Unless a subsection says otherwise, GPU evidence used this environment:

| Item | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, compute capability 8.0 |
| Reported GPU memory | 42,406,903,808 bytes |
| Nominal HBM2 bandwidth | 1,555 GB/s |
| Driver | 570.133.20 |
| Python | 3.12.13 |
| PyTorch / CUDA wheel | 2.10.0+cu128 / CUDA 12.8 |
| Triton | 3.6.0 |
| Transformers | 5.14.1 |
| FlashAttention | 2.8.1 |
| Model | Qwen3-0.6B, BF16 |
| Vocabulary | 151,936 tokens |
| Tensor parallelism | TP1 unless explicitly stated |

The 1,555 GB/s figure is the nominal vendor bandwidth, not a measured STREAM
result for this host; see the
[NVIDIA A100 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf).
The scheduler backlog experiment is CPU-only and records an AMD EPYC 7713 host.

### 1.4 Executive verdict

| Contribution | Correctness | Performance verdict | Remaining boundary |
|---|---|---|---|
| Homogeneous greedy | Pass | Certified sampler fast path: 0.177 ms median | Mixed batches still use the stochastic route |
| Top-k | Pass, including cutoff ties and inactive-row RNG layout | Certified at B256/top-k 50: -6.15% paired E2E throughput, inside 10% budget | Other `k`, batch, model, and GPU values need their own runs |
| Exact top-p | Pass for audited Transformers support and fixed-seed layout | All-active exact path remains expensive: 11.649 ms complete sampling | This is the cost of the exact sort/tie/RNG contract; the exact backend is not performance-resolved |
| Opt-in FlashInfer top-p | Pass for the documented statistical contract | 1.017 ms median; +59.56% median paired E2E versus exact | Different boundary ties, token identities, and Philox consumption by design |
| Token streaming | Pass for the synchronous, single-owner session contract | Accepted certificate: 90% CI [-0.204%, +0.245%] inside the +/-2% overhead band | Certificate pins ancestor `14002ae`; async/multi-owner serving is outside scope |
| Chunk scheduler/admission | Pass | 0.67 to 0.76 us from 0 to 500k injected waiters | Injected backlog is a complexity probe, not a reachable public state |
| Ragged graphs/chunk scheduling | Pass on retained graph and token gates | Stock throughput -0.20%; small-tau mechanism improves TTFT/ITL tradeoff | Cross-template BF16 token ties are not a bitwise-logit contract |
| TP transport | Spawn/shared-memory transport pass | 93.14% payload reduction in the realistic envelope | Real TP2 NCCL/inference remains unverified on the one-GPU host |
| Chunked-prefill request latency | Functional pass | **Not latency-certified** at the current self-pinned evidence commit: tau256 passed 3/5 strict runs | Rare decode-only/synchronization stalls remain under investigation; tau512 is policy-ineligible for latency certification |

## 2. Sampling: greedy, top-k, and top-p

### 2.1 Branch and evidence identity

| Contribution | Original feature tip | Repair/evidence identity |
|---|---|---|
| Greedy | `feat/greedy-sampling` `e4a8c914` | `fix/greedy-sampling` `ec988708` |
| Top-k | `feat/topk-sampling` `93a46649` | API repair `44417f7`; active-row repair `8759c877` |
| Exact top-p | `feat/topp-sampling` `b9b05cf2` | Transformers parity `716274f`; oracle coverage `1afec9d`; retained decision tip `42affe47` |
| Greedy/top-k evidence | — | `fix/sampling-evidence` `178e6142` |
| Optional fast top-p | no upstream feature branch | implementation `0e3d9ee9`; micro evidence `04f095f0`; E2E evidence `b68f1913`; archive `b1d98024` |

The primary retained manifests are
[sampling evidence](../benchmarks/sampling_evidence/README.md) and
[fast top-p evidence](../benchmarks/topp_performance/evidence/README.md).
The fast-top-p manifest states that production source is unchanged after
`0e3d9ee9`; the later commits harden and archive evidence.

### 2.2 Common B256 x V151936 tensor roof

For `B=256`, `V=151,936`:

```text
N = B * V = 38,895,616 logits
BF16 input       = 2N =  77,791,232 bytes =  74.1875 MiB
FP32 values      = 4N = 155,582,464 bytes = 148.3750 MiB
int64 indices    = 8N = 311,164,928 bytes = 296.7500 MiB
one-byte mask    =  N =  38,895,616 bytes =  37.0938 MiB
```

At the nominal 1,555 GB/s HBM roof, a single sequential pass over the BF16
input cannot be faster than:

```text
77,791,232 / 1.555e12 = 0.000050027 s = 0.05003 ms
```

The FP32 and int64 one-pass floors are 0.10005 and 0.20011 ms. These values are
deliberately optimistic: argmax needs a reduction; top-k needs selection;
top-p needs scaling, ordering or selection, softmax, a cumulative decision,
masking, and sampling. Allocated-memory peaks are capacity measurements, not
bytes actually transferred.

### 2.3 Greedy roofline and experiment

#### Original defect

The original greedy feature made `temperature=0` legal and returned
`argmax(logits)`, but it still executed the full stochastic graph:

1. cast and temperature-scale all logits in FP32;
2. materialize softmax probabilities;
3. create a full-vocabulary exponential random tensor;
4. compute the stochastic sample; and
5. discard it with `torch.where` for greedy rows.

The output rule was correct, but a homogeneous greedy batch paid stochastic
time, allocation, and RNG advancement.

#### Repaired route and correctness contract

`ec988708` adds a compiled homogeneous method:

```python
@torch.compile
def greedy(self, logits):
    return logits.argmax(dim=-1)
```

Host sequence metadata selects this route without a device `.item()`
synchronization. Constructor warmup compiles B1 and B2 shapes before requests.
Tests pin exact `torch.argmax` equality, first-index tie behavior, and unchanged
CPU/CUDA RNG state. Positive-temperature sampling remains operation-for-operation
equivalent to the original common sampler.

Mixed greedy/stochastic batches intentionally retain the common sampler so the
full-batch random-draw layout does not change. That is a remaining optimization
opportunity, not a correctness failure.

#### Protocol

The retained microbenchmark used three fresh Python processes at the exact
`ec988708` commit. Each had a unique empty Inductor cache and recorded:

- one true first call, including compilation and incremental allocation;
- five warmups; and
- 25 CUDA-event samples, summarized per process before cross-process
  aggregation.

The aggregate uses the median of process cold/median/p95 statistics and the
maximum process allocation peak.

#### Result and roofline interpretation

| Cold wall | Steady median | Steady p95 | Peak incremental allocation |
|---:|---:|---:|---:|
| 1,171.37 ms | 0.17715 ms | 0.18534 ms | 0.00195 MiB |

The cold number is compilation cost, not steady request latency. The steady
kernel is close to the 0.173 ms direct-argmax audit reference.

Using only the compulsory 74.1875 MiB read:

```text
effective input rate = 77,791,232 / 0.000177152 = 439.1 GB/s
fraction of nominal  = 439.1 / 1,555 = 28.2%
observed / one-read floor = 0.17715 / 0.05003 = 3.54x
```

That gap is reasonable for a reduction with launch and final-write overhead; it
is not evidence that another 3.54x is practically available.

**Verdict:** the avoidable homogeneous-greedy stochastic work and RNG advance
are resolved. No newly archived repaired-greedy E2E A/B exists, so the release
claim is sampler-level. Earlier feature-tip E2E deltas are historical and do
not measure this fast path.

### 2.4 Top-k roofline and experiment

#### Original defect

The original implementation represented disabled rows as `k=V` and sorted the
full vocabulary for every row. One active row therefore cost approximately the
same as all 256 active rows, with roughly 1.41 GiB peak allocation.

#### Repaired algorithm

The repaired path computes active rows and effective `k` values on the host,
groups rows by equal `k`, and invokes:

```python
values = torch.topk(active_logits, k, sorted=False).values
threshold = values.amin(dim=-1, keepdim=True)
active_logits.masked_fill_(active_logits < threshold, -inf)
```

Using `< threshold` rather than `<=` retains every token tied at the kth
boundary. Only active rows are copied into a compact bucket. The unchanged
full-batch stochastic sampler runs once afterward, preserving inactive-row
fixed-seed tokens and RNG position.

For `m_k` active rows at a particular `k`, the optional work model is:

```text
L_total = L_common_sampler + sum_k L_topk(m_k, V, k)
```

It must scale with active rows/buckets, not with the full batch whenever any one
row enables top-k.

#### Micro protocol and results

The three-fresh-process protocol matches greedy: one cold call, five warmups,
25 CUDA-event samples, `B=256`, `V=151,936`, BF16, and exact repair commit
`8759c877`.

| Route | Cold wall | Steady median | p95 | Peak incremental allocation |
|---|---:|---:|---:|---:|
| Top-k disabled | 1,893.14 ms | 1.01069 ms | 1.02605 ms | 148.377 MiB |
| One active row, `k=50` | 2,003.54 ms | 1.06906 ms | 1.11206 ms | 148.377 MiB |
| All 256 rows, `k=50` | 1,943.67 ms | 2.03776 ms | 2.04288 ms | 148.377 MiB |

```text
one-active optional increment = 1.069056 - 1.010688 = 0.058368 ms
all-active optional increment = 2.037760 - 1.010688 = 1.027072 ms
all-active / one-active total = 1.906x
all-active / one-active optional increment = 17.60x
```

The non-linear optional ratio is expected: one row is launch/indexing dominated,
whereas a 256-row `topk` exposes much more parallel work. The important result
is that one row no longer pays the all-row cost. The common FP32 stochastic
sampler dominates the identical 148.377 MiB peak.

#### E2E protocol and service budget

Four fresh `8759c877` processes generated 32 tokens for 256 distinct 128-token
integer prompts at temperature 0.6. CUDA graphs were enabled with
`max_model_len=1024`, `max_num_seqs=256`, and 80% GPU-memory utilization.
Each process alternated disabled and all-row `k=50` four times. Observation
zero is retained as shape-cold evidence and excluded; the first scenario is
balanced disabled/enabled/disabled/enabled across seeds.

| Seed | First scenario | Disabled tok/s | `k=50` tok/s | Paired change |
|---:|---|---:|---:|---:|
| 20260817 | disabled | 17,434.12 | 16,334.23 | -6.309% |
| 20260818 | enabled | 17,397.21 | 16,356.62 | -5.981% |
| 20260819 | disabled | 17,404.29 | 16,275.89 | -6.483% |
| 20260820 | enabled | 17,396.03 | 16,364.24 | -5.931% |
| Median paired result | balanced | 17,400.75 | 16,345.42 | **-6.145%** |

The predeclared budget was no more than 10% throughput loss. All four runs pass.
A 6.145% throughput loss corresponds to approximately 6.55% elapsed-time
overhead:

```text
1 / (1 - 0.061451) - 1 = 0.06548
```

**Verdict:** API compatibility, cutoff ties, heterogeneous-row scaling, memory,
RNG layout, and the declared B256 E2E budget are resolved. This is not a
universal result for all `k`, models, batches, or GPUs.

### 2.5 Exact top-p roofline and experiment

#### Exact semantic contract

The default `top_p_backend="exact"` follows the audited Transformers 5.14.1
processor order:

1. divide active BF16 logits by temperature in FP32;
2. sort ascending with the default `torch.sort` tie behavior;
3. compute softmax then cumulative probability;
4. remove tokens while cumulative probability is `<= 1 - top_p`;
5. force the final sorted token to remain; and
6. scatter the removal mask to original token IDs and mask the raw logits.

The host computes `1.0 - top_p` before FP32 transfer. Recomputing the complement
from an already rounded FP32 `top_p` changes equality-adjacent support. Top-k
runs before top-p. Only stochastic active rows enter the filter, and all-active
top-p is processed in chunks of 64 rows to cap workspace.

The final exponential sampler still runs once at the original batch shape.
Therefore optional filtering does not silently alter inactive-row random-draw
count or order.

Correctness tests include FP32/BF16 random inputs, forced ties, the
`[0.5, 0.5], p=0.5` boundary, crossing-token inclusion, tiny positive `p`,
host-cutoff rounding, top-k followed by top-p, inactive rows, CPU/CUDA, and
64/65-row chunk boundaries.

#### Repaired exact gates

| Scenario | Median ceiling | Scratch ceiling | Measured median |
|---|---:|---:|---:|
| Top-p disabled | 1.30 ms | 1 MiB filter overhead | 1.15 ms |
| One active row, `p=0.9` | 1.75 ms | 16 MiB | 1.41 ms |
| All 256 rows, `p=0.9` | 12.0 ms | 650 MiB | 10.18 ms |

Active-row isolation fixed the one-row scaling defect, and row chunking reduced
the original roughly 1.41 GiB transient. A later clean complete-route run
(filter plus exponential sampler) measured:

| Route | Median | p95 | Peak transient |
|---|---:|---:|---:|
| Exact complete top-p | 11.649 ms | 11.816 ms | 616.96 MiB |

Relative to the 0.05003 ms one-read floor, 11.649 ms is 233x. Even one
capacity-equivalent transfer of 616.96 MiB would take only about 0.416 ms at
nominal bandwidth. The excess is caused by multiple passes and the ordering,
index, softmax, cumulative-sum, scatter, mask, and RNG work; this is not a
single streaming-copy kernel.

#### E2E time budget

The retained rejected-candidate artifact has weaker provenance, but it contains
a useful same-process disabled baseline: 0.476453 seconds for B256 x 32 output
tokens. It implies:

```text
10% throughput-loss budget:
Delta T_total <= 0.476453 * (1/0.90 - 1) = 52.939 ms
Delta t_decode <= 52.939 / 32 = 1.654 ms per decode step

15% throughput-loss budget:
Delta T_total <= 0.476453 * (1/0.85 - 1) = 84.080 ms
Delta t_decode <= 84.080 / 32 = 2.627 ms per decode step
```

The exact filter alone is several times those budgets. Its all-active E2E loss
was approximately 38% in the original audit. Under the current exact
Transformers/tie/RNG contract, this remains a material enabled-path cost.

This distinction is important:

- the previous one-active and 1.41 GiB behavior was an implementation defect
  and is fixed;
- the remaining all-active exact cost is dominated by the selected semantic
  contract and current full-vocabulary primitives; and
- calling it “resolved” would require either a much faster oracle-equivalent
  kernel or an explicit acceptance of that cost.

#### Exact/approximate development search

These experiments are development decisions, not interchangeable release
results:

| Candidate | Timing / memory result | Correctness result | Decision |
|---|---|---|---|
| Stable BF16 sort before FP32 scale | filter 9.235 -> 7.981 ms; 617.9 -> 531.7 MiB | tie identity not portable; non-unit scaling can merge/overflow values | rejected |
| Checked BF16 boundary plus FP32 fallback | filter 9.160 -> 8.508 ms; E2E 17,193.7 -> 10,905.8 tok/s (-36.57%) | tested oracle behavior retained | rejected: synchronization erased the gain |
| Stable descending minimal nucleus | 10.730 -> 9.455 ms | differed in 252/256 rows and 13,260 support positions | rejected |
| Exact BF16 histogram reconstruction | 2.57 ms through reconstruction/divide; 3.99 ms through softmax+cumsum | cumulative tensor exact, token-ID boundary recovery still required | rejected by complete-route budget |
| CUB segmented BF16 keys-only sort | 1.738 ms | exact sorted values | stopped: sort alone failed 0.75 ms primitive gate and exceeded 1.6 ms complete target |
| Grouped histogram cutoff | 2.894 ms | 255/256 production-shaped Gaussian rows; forced ties worse | rejected: no exact certificate |
| Qrita sorting-free mask | 5.955 ms p50, 5.967 p95, 62.60 MiB persistent versus exact filter 9.155/9.344 ms and 616.96 MiB | 14,720 support-bit differences on production Gaussian; uniform/tie token differences | rejected |
| Adaptive top-M, Gaussian `M=4096` | 12.764/14.868 ms versus exact 10.869/10.976; 860.54 versus 691.15 MiB | 256/256 rows fell back; exact output | rejected |
| Adaptive top-M, Gaussian `M=8192` | 13.117/13.155 versus exact 9.324/9.509; 880.63 versus 691.15 MiB | 256/256 rows fell back | rejected |
| Adaptive top-M, peaked `M=4096` | 10.510/10.686 versus exact 9.312/9.404; 824.03 versus 691.15 MiB | 193/256 fallbacks; exact output | rejected |
| Adaptive top-M, peaked `M=8192` | 10.899/11.012 versus exact 9.304/9.558; 844.12 versus 691.15 MiB | 193/256 fallbacks; exact output | rejected |

The adaptive route is “exact” only because uncertain rows take the original
full-sort path. Its equality results validate fallback safety, not a useful
optimization.

### 2.6 Optional FlashInfer top-p experiment

The accepted performance path is a separate, opt-in semantic contract:

```text
top_p_backend="exact"       # default, audited support and fixed-seed stream
top_p_backend="flashinfer"  # opt-in statistical top-p
```

The production wrapper performs a greedy argmax for temperature-zero rows,
FlashInfer FP32 softmax with per-row temperature, sorting-free deterministic
top-p sampling, dtype normalization, and a final greedy/stochastic
`torch.where`. FlashInfer 0.6.17 identifies source commit
`a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`.

#### Complete-sampler micro protocol

The clean B256 x V151936 BF16 run used per-row FP32 temperatures at 0.6,
`p=0.9`, seed 20260826, five warmups, and 25 CUDA-event iterations. Inputs are
restored outside the event interval because production receives fresh model
logits. The routes were measured sequentially exact, direct primitive, then
production wrapper.

| Complete route | Median | p95 | Maximum | Peak transient |
|---|---:|---:|---:|---:|
| Exact filter + sampler | 11.649 ms | 11.816 ms | — | 616.96 MiB |
| FlashInfer direct primitive | 0.912 ms | 0.971 ms | — | 296.75 MiB |
| FlashInfer production wrapper | 1.017 ms | 1.067 ms | 1.295 ms | 296.753 MiB |

The production wrapper passes the 1.6 ms median and maximum-observation gate.
The descriptive exact/wrapper ratio is 11.456x and transient allocation falls
by about 51.9%. Because exact samples shifted late in the sequential run, the
absolute wrapper gate is stronger evidence than the speedup ratio: every one of
the 25 wrapper samples was below 1.6 ms.

#### Eight-pair E2E protocol and result

Each backend ran in its own fresh process. The workload was B256, 128 prompt
tokens, 32 output tokens, Qwen3-0.6B, temperature 0.6, `p=0.9`, CUDA graphs,
and six observations. Observation zero is shape-cold after engine/sampler JIT
warmup and excluded; the median of the other five is one backend observation.
Each exact/FlashInfer process pair is one timing replicate. Four prompt seeds
are mirrored in the opposite execution order.

| Pair | Order | Exact tok/s | FlashInfer tok/s | Ratio |
|---:|---|---:|---:|---:|
| 1 | exact -> fast | 10,742.5 | 17,097.0 | 1.5915x |
| 2 | fast -> exact | 10,587.8 | 17,109.5 | 1.6160x |
| 3 | fast -> exact | 10,702.6 | 16,889.1 | 1.5780x |
| 4 | exact -> fast | 10,731.8 | 17,184.6 | 1.6013x |
| 5 | exact -> fast | 10,745.8 | 17,105.1 | 1.5918x |
| 6 | fast -> exact | 10,732.0 | 17,214.2 | 1.6040x |
| 7 | fast -> exact | 10,735.9 | 17,156.6 | 1.5981x |
| 8 | exact -> fast | 10,717.5 | 17,075.1 | 1.5932x |

```text
median pair ratio = 1.595626x = +59.5626%
mean pair ratio   = 1.596731x
sample SD         = 0.011076
95% t interval    = [1.58747x, 1.60599x], df=7
observed range    = [1.57804x, 1.61597x]
```

All eight pairs improved. The interval describes this pinned workload; four
prompt seeds are mirrored, so it is not an eight-corpus or universal hardware
generalization.

#### Semantic boundary

Same-seed repeatability passes within each backend, but equivalence between
backends is intentionally false:

- 256/256 fixed-seed production rows selected different tokens;
- both began at the same CUDA RNG state;
- exact ended at Philox offset 4 and FlashInfer at offset 8,192;
- uniform eight-way `p=0.6`: exact retained token IDs `[0,1,2,3,5]`, while
  FlashInfer sampled all eight symmetrically; and
- three tied maxima at `p=0.5`: exact retained `[0,1]`, while FlashInfer
  sampled `[0,1,2]`.

The statistical gate used 131,072 draws from a known unique-logit distribution.
It produced no draw outside the mathematical three-token nucleus and a maximum
absolute standardized residual of 1.79, below its six-sigma gate.

**Verdict:** FlashInfer resolves all-active performance for callers who
explicitly accept distributional top-p rather than exact fixed-seed/tie
identity. It does not make the default exact route faster and must not be
described as exact-backend equivalence.

### 2.7 Sampling evidence integrity and reproduction

The current CPU-only validators pass:

```text
validated 16 release raw files, 2 harnesses, 1 rejected artifact
validated 1 micro artifact, 16 E2E raw files, 8 fresh-process pairs, 2 harnesses
```

Run them from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  /venv/main/bin/python benchmarks/sampling_evidence/validate_provenance.py

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  /venv/main/bin/python \
  benchmarks/topp_performance/evidence/validate_provenance.py
```

Add `--check-model` when the pinned Qwen3-0.6B snapshot is available. The
sampling manifest pins config, tokenizer, weights, commands, process order,
seeds, commits, environments, and all raw hashes. The fast-top-p manifest pins
one micro artifact, 16 E2E process artifacts, two harnesses, and five model
files.

The greedy/top-k micro harness hard-pins the exact repair commits. Reproducing
those release numbers therefore requires an exact detached checkout of
`ec988708` or `8759c877`, a new empty Inductor cache per process, and new
immutable output paths. The development commands and contract caveats for
exact and FlashInfer top-p are in
[benchmarks/topp_performance/README.md](../benchmarks/topp_performance/README.md).

## 3. Token streaming

### 3.1 What the feature changes

Token streaming does not add model FLOPs. `generate()` and `stream()` consume
the same private engine-step loop; the difference is when completion token IDs
cross the public caller boundary.

The repaired API has a deliberately narrow contract:

- one eager `StreamSession` owns the synchronous engine;
- another stream, generate, manual queue, or step operation is rejected while
  that session is active;
- admission is validated and transactional;
- cancellation names only the owning request IDs;
- early exit requires a context manager or explicit `close()`;
- events carry `(seq_id, token_id, finished)`; and
- consumer-side `TextUpdate` objects can replace a previously rendered suffix.

This makes a generic compute roofline inappropriate. The meaningful roofs are:

1. throughput equivalence to matched `generate()`;
2. engine-to-caller delivery latency;
3. caller exposure relative to batch return;
4. paired peak allocation;
5. pending-event storage;
6. synchronous slow-consumer coupling; and
7. detokenizer work and state scaling.

### 3.2 Accepted evidence identity

The accepted raw artifact is
[repaired_streaming_cert_a100_2026-08-18_14002ae.json](../benchmarks/pr5_results/repaired_streaming_cert_a100_2026-08-18_14002ae.json):

```text
size       26,776,996 bytes
SHA-256    df662215db769d0c93129b8d29fc9fbcda4998e41cc412d312864c437da7f8c7
commit     14002ae04102eef58aea09fa8a2a78eca0103b5f
tree       383ba5daa010907b7d12c340cdbd0e7b510da9ef
branch     fix/token-streaming-cert
```

The adjacent
[manifest](../benchmarks/pr5_results/repaired_streaming_cert_a100_2026-08-18_14002ae.manifest.json)
records the source/model fingerprints, environment, all recomputed gates, and
the explicitly rejected older certificate.

Commit `14002ae` is an ancestor of `fork-main`, but later chunked-prefill
integration changed shared engine/scheduler/sequence code. Therefore the
artifact certifies the contribution commit, not byte-for-byte RC1 performance.
The archive validator still passes against its retained bytes.

### 3.3 Core matched-work protocol

The statistical unit is one fresh worker process, not one in-process round:

- eight fresh workers;
- one engine, model, and CUDA context per worker;
- Qwen3-0.6B, graph mode, TP1;
- `B=16`, 128 output tokens, temperature 0.6, EOS ignored;
- 2,048 output tokens/events per route;
- distinct prompt set and seed for each of 32 timed rounds;
- short and full-128-token warmup on both routes;
- four timed generate/stream pairs per worker;
- two `generate -> stream` and two `stream -> generate` orders per worker;
- prefix-cache metadata reset only while the engine is idle and owns no blocks;
  and
- prompt plus completion asserted below one 256-token KV block.

Timed work is matched. `generate()` performs one final full tokenizer decode per
request; drained `stream()` explicitly performs the same final decode before
its route timer stops. The comparison is therefore not an IDs-only stream
against a decoded batch result.

For round `r` in worker `i`:

```text
d_i,r = 100 * (R_stream / R_generate - 1)
D_i   = median over the four d_i,r values
```

Only the eight `D_i` values enter the confidence interval:

```text
mean = sum(D_i) / 8
SE   = sample_stddev(D_i) / sqrt(8)
central 90% interval = mean +/- t(df=7, 0.95) * SE
                     = mean +/- 1.894579 * SE
```

The predeclared throughput-equivalence gate requires the complete interval to
lie in `[-2%, +2%]`.

### 3.4 Throughput-equivalence result

The eight worker units were:

```text
[+0.277102, +0.079534, -0.409943, -0.566836,
 +0.197425, +0.019661, +0.192369, +0.375652] %
```

| Statistic | Result |
|---|---:|
| Mean paired delta | +0.02062% |
| Central 90% t interval | [-0.20391%, +0.24515%] |
| Descriptive generate median | 4,687.78 output tok/s |
| Descriptive stream median | 4,688.55 output tok/s |
| Gate | **Pass: entire interval inside +/-2%** |

This supports throughput equivalence/no material overhead, not a streaming
speedup claim.

All 32 raw paired rounds remain in the artifact. Their median was -0.06968% and
range was -16.46728% to +4.05791%. The -16.47% outlier occurred in worker 3,
round 2 with stream first; the predeclared four-round worker median was
-0.56684%. Robust aggregation prevents one round from dominating while retaining
the variability for audit.

### 3.5 Caller delivery and exposure roofs

Definitions:

- first-event time is caller `perf_counter` immediately after the first yield
  minus route start; it is not model-only TTFT;
- engine-to-caller delay is caller receipt minus the scheduler token timestamp;
  both clocks are in one process; and
- caller exposure is `generate_return_seconds / stream_first_event_seconds`.

Acceptance roofs:

```text
p95(engine-to-caller) <= 1.0 ms
minimum worker caller exposure >= 10x
```

Results:

| Metric | Median | p95 | Maximum / minimum |
|---|---:|---:|---:|
| Generate API return | 436.884 ms | — | range 435.633-448.999 ms |
| Stream first event | 29.073 ms | — | range 28.610-32.742 ms |
| Caller exposure | 14.9709x | 15.2013x | minimum 13.7730x |
| Engine-to-caller, 65,536 events | 0.02949 ms | 0.03972 ms | maximum 0.51713 ms |
| First-token to first-delivery, 512 requests | 0.04025 ms | 0.04842 ms | maximum 0.05696 ms |
| Engine-finish to final-delivery, 512 requests | 0.14290 ms | 0.24371 ms | maximum 0.37996 ms |

Both gates pass. The delivery p95 is about 25x below its 1 ms ceiling. The
exposure number means the caller sees content before full-batch return; it must
not be advertised as a 15x reduction in model TTFT.

### 3.6 Memory and pending-storage roofs

Every timed pair applies a relative peak-allocation gate:

```text
M_stream <= 1.01 * M_generate
```

If the generate peak is zero, only a zero stream peak passes. In this workload,
both routes reached exactly 37,092,210,176 allocated bytes in every pair and all
32 stream-minus-generate differences were zero bytes.

This is absolute PyTorch allocator peak, not incremental route allocation or
total board use. Reserved memory and RSS are retained but not acceptance-gated.

The session drains all events from one engine step before starting the next.
For a slow-consumer batch `B_s=8`, the pending deque roof is:

```text
Q_max <= B_s - 1 = 7 events
```

The observed maximum was exactly 7 for every worker and sleep setting. This
proves bounded in-process step storage; it does not describe a decoupled async
producer, because none exists in this contract.

### 3.7 Synchronous backpressure roofline

A decode step emits `B_s` events. If the caller sleeps `delta` after every
event, sleep cost is paid `B_s` times before the engine can advance:

```text
G(B_s, delta) = G0 + B_s * delta
```

Here `B_s=8`. `G0` is the across-worker median at zero sleep. For each
nonzero delay:

```text
residual  e = G_observed - G_predicted
tolerance = max(2 ms, 0.10 * B_s * delta_ms)
gate      = abs(e) <= tolerance
```

| Sleep per event | Expected gap | Observed gap | Residual | Tolerance | Result |
|---:|---:|---:|---:|---:|---|
| 0 ms | 3.19529 ms | 3.19529 ms | 0.00000 ms | 2.0 ms | pass |
| 1 ms | 11.19529 ms | 11.51848 ms | +0.32319 ms | 2.0 ms | pass |
| 4 ms | 35.19529 ms | 36.10309 ms | +0.90780 ms | 3.2 ms | pass |

This is the measured price of the synchronous design, not an accidental
regression: a slow caller directly stalls stepping. Removing that coupling
requires an async producer/queue architecture with a new memory and cancellation
contract.

### 3.8 Detokenizer structural and scaling rooflines

Tokenizer decode is not necessarily prefix-monotone. A later token can replace
a preceding space, normalization fragment, or incomplete UTF-8 character.
`StreamingDetokenizer.feed()` therefore returns:

```text
TextUpdate(seq_id, replace_from, delete_count, insert, final)
```

Offsets are Python Unicode code points. The default unstable-window parameters
are `W=32`, overlap `O=8`, and hard incremental input roof:

```text
H = W + 2O = 48 tokens
```

Each feed decodes at most `H` tokens. Frontier advancement tests at most
`O+1` candidate splits, each using bounded token slices. For fixed `W,O`,
tokenizer work is O(N), followed by exactly one O(N) full decode at `flush()`.
The implementation retains all token IDs so final flush can be exact.

Normal Qwen tokenizer measurements include every feed, immutable
`TextUpdate.apply`, final flush, and application of the final correction:

| Input tokens | Median us/input token | Maximum incremental decode | Full-length flushes |
|---:|---:|---:|---:|
| 64 | 27.55 | 40 tokens | 1 |
| 256 | 21.97 | 40 tokens | 1 |
| 1,024 | 21.83 | 40 tokens | 1 |
| 2,048 | 21.94 | 40 tokens | 1 |

The 40-token maximum is below the 48-token roof, and steady per-token cost is
flat rather than growing linearly with sequence length.

The correction-heavy CPU experiment alternates spaces and punctuation so 50% of
feeds replace a prior suffix. It compares `N1=8,000` and `N2=32,000`, a 4x
input increase, over three process-time repetitions.

Predeclared roofs:

```text
process-time ratio <= 1.5 * 4 = 6.0x
retained-state ratio <= 1.125 * 4 = 4.5x
```

Observed:

| Quantity | 8k | 32k | Ratio | Roof | Result |
|---|---:|---:|---:|---:|---|
| CPU process time | 0.04246 s | 0.19348 s | 4.5567x | 6.0x | pass |
| Selected shallow state | 67,577 B | 277,689 B | 4.1092x | 4.5x | pass |

The probe also requires exact final text, one full flush, state release, 50%
corrections, and max feed decode <=48; all pass.

Limitations are explicit: `TextUpdate.apply` copies immutable Python strings;
the selected state-byte count is shallow rather than total RSS; intermediate
exactness assumes rewrites remain within the overlap; and a non-splittable
tokenizer raises before admitting the next token once the hard limit is reached.
Final flush remains exact.

### 3.9 Correctness, ownership, and cleanup gates

Tests and the certificate jointly pin:

- same-seed stream reassembly equals `generate()` token IDs and decoded text;
- one event per actual completion token, with exactly one final event/request;
- intermediate prefill chunks emit nothing; a completing prefill emits the first
  completion token;
- prefix-cache stream/generate equivalence;
- an atomic single-owner lease, including a simultaneous-start one-winner test;
- fail-before-admit length/capacity checks;
- transactional rollback of only IDs admitted by the failed operation;
- manual queued work is rejected without global cancellation;
- context-manager/explicit-close cleanup restores scheduler/KV state;
- public legacy `step()` pairs and opt-in metrics remain separate;
- Unicode, CJK, emoji/ZWJ, combining marks, whitespace/normalization rewrite,
  incomplete UTF-8 holdback, cross-window rewrite, interleaved sequences,
  overlap exhaustion, exact flush, and state release.

The ownership lock does not make arbitrary engine methods generally
thread-safe. Callers must serialize all engine access. Breaking a retained
iterator does not itself guarantee cleanup; use `with` or `close()`.

### 3.10 Superseded evidence and remaining scope

The earlier schema-1 artifact used four workers, reused one seed, and omitted
the final detokenization on the null stream. It remains historical, not an
equivalence certificate.

The schema-2 `cf6da50` run is explicitly rejected: it compared memory against
1% of total board capacity instead of 1% of the paired generate peak, lacked a
full-shape warmup design, and failed its own throughput interval
`[-5.20386%, +6.01175%]`. The validator rejects its hash and commit.

The accepted result is limited to one A100, Qwen3-0.6B, TP1, B16, 128 outputs,
temperature 0.6, in-process delivery, and prompts below one cache block. It does
not certify HTTP/network serialization, async serving, TP2, other hardware,
longer outputs, or other tokenizers.

Validate the accepted archive without CUDA:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  /venv/main/bin/python \
  benchmarks/pr5_scripts/validate_repaired_stream_certificate.py
```

To certify a current commit rather than the retained ancestor, run the hardened
eight-worker harness from a clean checkout with a new output path:

```bash
CERT_COMMIT="$(git rev-parse HEAD)"
PYTHONPATH=. /venv/main/bin/python \
  benchmarks/pr5_scripts/repaired_stream_benchmark.py \
  --model /workspace/models/Qwen3-0.6B \
  --runs 8 --seed 20260818 \
  --output "/workspace/.feat_bench/results/repaired_streaming_cert_${CERT_COMMIT}.json"
```

## 4. Chunked prefill

### 4.1 What the contribution changes

Chunked prefill lets decode work and a bounded slice of one or more prompt
prefills share an engine step. Its purpose is not to make a given prompt token
free. It trades a small amount of scheduling and ragged-execution complexity
for bounded decode interference, earlier service for waiting work, and better
control over the TTFT/ITL/throughput trade-off.

Let:

- $\tau$ be `max_num_batched_tokens`, the total token budget for a step;
- $D$ be the number of live one-token decode rows selected first;
- $P$ be the number of prefill tokens scheduled in the remainder; and
- $L$ be the remaining length of a waiting prompt.

The repaired scheduler enforces

$$
P_{\max} = \max(0,\tau-D), \qquad D+P \le \tau.
$$

Decode rows retain priority. Waiting sequences fill the remaining positive
budget in FIFO order, and at most one partially consumed request is tracked as
the mid-chunk owner. In the simple case of one long prompt and constant decode
load, the number of mixed prefill steps is

$$
n_{\text{mixed}} =
  \left\lceil \frac{L}{\tau-D} \right\rceil ,
$$

provided $\tau>D$. For example, the phase workload with 63 decoders and a
2,048-token prompt predicts:

| $\tau$ | Prefill budget $\tau-D$ | Predicted mixed steps | Chunk decomposition |
| ---: | ---: | ---: | --- |
| 256 | 193 | 11 | $10\times193+118$ |
| 512 | 449 | 5 | $4\times449+252$ |

The retained diagnostic observed exactly 11 and 5 mixed steps respectively.
That is a mechanism check, not by itself a latency certificate.

### 4.2 Original issues and repair map

The feature audit separated six concerns rather than treating “chunked
prefill” as one pass/fail item:

| Concern | Original failure mode | Repaired mechanism | Present verdict |
| --- | --- | --- | --- |
| Admission/fairness | Unbounded accepted waiting work weakened the fairness argument and allowed a large batch to queue transactionally inconsistent work. | Public admission is capacity-bounded by `max_num_seqs`; a whole batch is preflighted before mutation and a retryable capacity failure leaves no partial batch. | **Resolved** for the documented hard-backpressure contract. |
| Scheduler complexity | The hot path scanned a potentially enormous waiting deque even after the token budget was consumed. | Stop once the budget is nonpositive; retain one mid-chunk owner rather than rescanning skipped work. | **Resolved** by the five-process O(1)-with-backlog roofline. |
| Budget coherence | The token budget could be mutated after construction and become inconsistent with graph capture and other derived state. | Validate at construction, require $\tau\ge\text{max_num_seqs}$, expose the derived budget as read-only, and guard nonpositive residual budgets. | **Resolved** and covered by CPU tests. |
| Ragged CUDA graphs | Empty/sparse bucket sets, low $\tau$, short `max_model_len`, unsafe dummy layouts, or live shapes beyond a captured key could select an illegal graph or miss silently. | Safe eager fallback, sparse graph selection, live q/k guards, legal dummy capture layouts, and variable-length graph contracts. | **Resolved** for the retained matrix and boundary cases. |
| TP transport | Pickling full `Sequence` objects could overflow the fixed 1 MiB shared-memory frame; ordinary pickle state was also modified globally. | Rank-0-only compact `ScheduledSequence` DTO, validated framed transport, config-derived page-rounded shared memory, and unchanged ordinary `Sequence` pickle behavior. | **Transport repaired; TP2 inference remains unverified** because this host has one GPU. |
| Request-level tail latency | Medians and bounded phase tests could hide isolated whole-step stalls. | Self-pinned five-process full-completion certificate with a strict per-run maximum-ITL gate, plus a separate intrusive jitter diagnostic. | **Not certified**: current $\tau=256$ evidence passes only 3/5 runs. |

The production repairs and their tests are ancestors of the fork release
candidate. Some retained artifacts intentionally pin earlier clean commits.
The evidence hashes prove what was measured; ancestry plus the current CPU
suite provide regression evidence, but they do not turn an ancestor benchmark
into byte-for-byte performance certification of the release candidate.

### 4.3 Analytical A100/Qwen roofline

This section uses a deliberately simple lower-bound model. It is useful for
classifying results, not for predicting every kernel. The vendor nominal A100
SXM4 figures used here are 312 BF16 Tensor TFLOP/s and 1,555 GB/s HBM2
bandwidth, giving the ridge point

$$
I_{\text{ridge}} =
  \frac{312\times10^{12}}{1{,}555\times10^9}
  = 200.64\ \text{FLOP/byte}.
$$

For the measured Qwen3-0.6B configuration:

- hidden size $h=1024$;
- intermediate size $i=3072$;
- 28 layers;
- 16 query heads, 8 KV heads, head dimension 128;
- vocabulary $V=151{,}936$.

Counting the dense attention projections and MLP matrices gives

$$
W_{\text{layer}}=15{,}728{,}640,\qquad
W_{\text{core}}=28W_{\text{layer}}=440{,}401{,}920.
$$

The vocabulary projection contributes

$$
W_{\text{vocab}}=Vh=155{,}582{,}464.
$$

Ignoring norms, elementwise operations, activation traffic, KV traffic, and
attention for the moment, the dense-linear work for $T$ model input tokens
and $N$ rows that require vocabulary logits is

$$
F_{\text{linear}}
  = 2W_{\text{core}}T + 2W_{\text{vocab}}N.
$$

One BF16 read of core plus vocabulary weights is at least

$$
B_W=2(W_{\text{core}}+W_{\text{vocab}})
   =1{,}191{,}968{,}768\ \text{bytes}.
$$

The resulting idealized roofs are:

| Route shape | $T$ | $N$ | Dense work | Weight-only intensity | Ideal lower bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64-row decode | 64 | 64 | 76.286 GFLOP | 64.00 FLOP/B | 0.766 ms bandwidth roof |
| $\tau=256$ mixed | 256 | 64 | 245.400 GFLOP | 205.88 FLOP/B | 0.787 ms compute roof |
| $\tau=512$ mixed | 512 | 64 | 470.886 GFLOP | 395.05 FLOP/B | 1.509 ms compute roof |

These values are unattainable service-time floors: they omit kernel launch,
attention, KV-cache reads/writes, ragged metadata, sampling, scheduler work,
CPU/CUDA synchronization, and imperfect Tensor Core utilization. They are
still informative. Pure decode sits below the ridge and is weight-bandwidth
sensitive; larger mixed steps cross into the compute-sensitive region.

For a long prefill slice of $q$ queries with cached prefix length $c$, the
number of causal query/key pairs is

$$
A(q,c)=qc+\frac{q(q+1)}{2}.
$$

A useful approximate attention-work term across this model is

$$
F_{\text{attn}}\approx
  4(28)(16)(128)A(q,c)
  =229{,}376A(q,c).
$$

The retained phase diagnostic illustrates why a fixed token count does not
imply a fixed mixed-step cost:

| Profile | Long-slice state | Approx. attention work | Linear + attention | Observed model CUDA span |
| --- | --- | ---: | ---: | ---: |
| $\tau=256$ | $q=193,c=0$ | 4.294 GFLOP | 249.694 GFLOP | 4.803 ms |
| $\tau=256$ | $q=193,c=1737$ | 81.190 GFLOP | 326.591 GFLOP | 6.598 ms |
| $\tau=512$ | $q=449,c=0$ | 23.173 GFLOP | 494.059 GFLOP | 6.686 ms |
| $\tau=512$ | $q=449,c=1347$ | 161.900 GFLOP | 632.786 GFLOP | 8.418 ms |

The arithmetic model is explanatory rather than an oracle: FlashAttention has
tiling and memory effects that the simple FLOP count omits. The important
result is structural. A later chunk against a longer prefix is more expensive
than an early chunk of the same $q$, and $\tau=512$ legitimately has a
higher deterministic latency floor than $\tau=256$.

### 4.4 Scheduler backlog roofline

#### Protocol

The certified CPU harness times only `Scheduler.schedule()` after constructing
the backlog. It uses five fresh Python processes, distinct seeds, 2,000 warmup
calls and 20,000 retained nanosecond samples at each injected backlog
$W\in\{0,100{,}000,500{,}000\}$. Two decode rows exercise the normal
scheduling path. Queue construction is outside every timed interval, GC state
is checked, and backlog order is randomized by process.

The injected queues intentionally exceed the public admission cap. This is an
algorithmic complexity probe: the state is unreachable through the repaired
public API but exposes whether hot-path work is $O(W)$.

The predeclared aggregate gate is

$$
M_{500k}\le\max(2M_0,5\ \mu s),
$$

and every individual run must also satisfy its embedded gate.

#### Results

| Injected waiters | Per-process median ($\mu s$) | Median of medians ($\mu s$) | Per-process p95 ($\mu s$) |
| ---: | --- | ---: | --- |
| 0 | 0.67, 0.67, 0.79, 0.80, 0.66 | 0.67 | 0.71, 0.83, 1.07, 0.84, 0.70 |
| 100,000 | 0.76, 0.74, 0.88, 0.88, 0.74 | 0.76 | 0.80, 0.90, 1.04, 0.92, 0.78 |
| 500,000 | 0.75, 0.88, 0.76, 0.89, 0.76 | 0.76 | 0.79, 0.93, 1.02, 0.94, 0.80 |

$$
\frac{M_{500k}}{M_0}=1.1343.
$$

This passes both the relative and 5-$\mu s$ absolute roofs. For context, the
audited pre-repair scan grew from roughly 0.0016 ms at no waiters to 4.09 ms at
100k and 19.73 ms at 500k. The repaired result is therefore not “the queue is
usually short”; it is direct evidence that timed scheduler work no longer
scales with the hidden tail of the deque.

The retained
[scheduler manifest](../benchmarks/chunked_prefill_tail/evidence/2026-08-18-scheduler-da93670/scheduler_da93670_manifest.json)
has SHA-256
`e1fe1cb0cb8b781771c113f516a542f139ce7b89d4283b5192f5d9a7594efeba`.

### 4.5 Ragged CUDA-graph correctness rooflines

The graph mechanism captures total-token buckets
$\{128,256,512,1024,2048\}$ no larger than $\tau$, with segment-slot tiers
that include the live-shape requirement. Selection must satisfy both total
token and segment-count bounds; otherwise the route must use eager execution.

The low-budget/short-context contract matrix exercises six cells:

| $\tau$ | `max_model_len` | Real unpadded shape | Expected route | Observed result |
| ---: | ---: | --- | --- | --- |
| 64 | 512, 1024, 4096 | $4\times15=60$ tokens | No legal captured bucket; eager, one routed miss | All three token vectors match eager |
| 128 | 512, 1024, 4096 | $4\times31=124$ tokens | Graph key `(128, 9)`, zero misses | All three token vectors match eager |

The separate boundary test uses four 511-token segments with
`max_model_len=512`: 2,044 real tokens route to `(2048, 5)`, with zero graph
misses, exact eager/graph/end-to-end greedy-token equality, and maximum logit
difference zero.

Randomized valid-unpadded tests also exercise:

- bucket 128 with segment lengths `(7, 19, 43)`;
- bucket 256 with `(31, 57, 81)`; and
- bucket 512 with `(33, 91, 137, 141)`.

The comparison is intentionally on live rows and selected greedy token IDs.
It does not claim bitwise equality of padded and unpadded logits across
different graph templates.

The graph/phase archive pins commit `e50e732568727ae9127eac22d23c7dd7054e6b71`,
source SHA-256
`ba43d751ae343d47f5efd642bf070bacf64f86931e10c205474dbfa2d8e74db8`,
and model SHA-256
`0c659d1dba2804b0943c24bbece1858e93273147f7a11693fa99062df8c5997b`.
The retained 4x511 artifact SHA-256 is
`56f1ad9f901bb5c719615958dcda6fb90b3b95d57269c334ddadf029ed90ae6d`;
the six-cell matrix is
`2ecb58ff0e7695166f381a83056d378e1ce089b7c7d312a13903aaa8cff002a9`.

### 4.6 Mixed-load mechanism and profile sweep

The repaired contribution retained several complementary experiments:

1. a feature A/B sweep showing the intended TTFT/ITL trade-off;
2. a stock-workload regression guard;
3. graph-contract tests that answer “is the route legal and equivalent?”;
4. bounded phase diagnostics that answer “where is time spent?”; and
5. full-completion certification that answers “does every request-level tail
   observation meet the release SLO?”

Those questions must not be collapsed into one metric.

#### Repaired constructor sweep

For a mixed workload on the repaired branch, the retained sweep reports:

| $\tau$ | Mixed throughput (tok/s) | Interactive TTFT (ms) | Maximum interactive ITL (ms) | Long-request TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 3,098.8 | 20.176 | 7.001 | 133.277 |
| 256 | 3,185.9 | 14.848 | 7.099 | 72.706 |
| 512 | 3,165.8 | 10.566 | 9.058 | 58.212 |
| 1,024 | 2,900.3 | 11.531 | 15.456 | 53.860 |
| 2,048 | 2,953.7 | 11.572 | 19.517 | 41.791 |
| 16,384 | 3,275.0 | 11.101 | 36.741 | 36.706 |

This is the central product trade-off. Increasing $\tau$ gets long requests
to service in fewer chunks and generally lowers their TTFT, but makes each
mixed step more expensive and eventually violates the interactive ITL target.
The table supports $\tau=256$ as the conservative latency profile and
$\tau=512$ as a lower-long-TTFT profile; it does not establish a universal
best $\tau$.

Repaired A/B medians at three anchor points were:

| $\tau$ | Throughput delta | Interactive TTFT delta | Maximum ITL delta | Long TTFT delta |
| ---: | ---: | ---: | ---: | ---: |
| 128 | +0.62% | +0.30% | -14.17% | -8.88% |
| 1,024 | -0.07% | +0.21% | +0.64% | +1.60% |
| 16,384 | +0.51% | +3.95% | +1.45% | +1.45% |

The historical A/B development data are useful mechanism evidence, but the
original PR6 scripts and status text predate several repairs. The preserved
raw bundle on the repaired `fix/chunked-prefill` branch is the authoritative
source for those numbers; the self-pinned tail archives below are authoritative
for the later release classification.

#### Stock-workload guard

Fresh stock medians were 8,779.672 tok/s for the development parent and
8,761.927 tok/s for chunked prefill, a descriptive $-0.20\%$ ratio; the
paired median delta was $-0.05\%$. This passes the predeclared 1% stock
regression ceiling. It shows that merely enabling the repaired machinery does
not materially slow the non-mixed benchmark on this system.

#### Bounded phase diagnostic

The phase probe conditions 63 interactive decoders, admits one 2,048-token
long request, then records 96 steps at temperature zero with engine-owned
cyclic GC disabled:

| $\tau$ | Route counts | Selected keys | Wall median / p95 / max (ms) | Mixed-step median / max (ms) | Graph misses |
| ---: | --- | --- | ---: | ---: | ---: |
| 256 | 85 decode, 11 mixed | `(64)`, `(256,64)` | 5.254 / 6.514 / 7.209 | 6.373 / 7.209 | 0 |
| 512 | 91 decode, 5 mixed | `(64)`, `(512,64)` | 5.252 / 5.441 / 9.085 | 8.556 / 9.085 | 0 |

The summed mixed-step wall time was 69.871 ms at $\tau=256$ and 41.782 ms at
$\tau=512$, illustrating why the larger profile improves long-request TTFT:
it pays more per mixed step but needs fewer steps. Both artifacts record
`current_run_latency_certified=false`. They are intrusive, bounded diagnostics,
not full-request SLO evidence.

### 4.7 Full-completion latency certification

#### Why this protocol is stricter

The release SLO is not “median ITL below 10 ms.” It is:

> In each of five fresh processes, every recorded interactive inter-token
> latency must be strictly below 10 ms.

The eligible harness runs:

- one fresh process per seed;
- Qwen3-0.6B, TP1, graphs enabled, `max_model_len=4096`;
- 16 interactive requests with 64-token prompts;
- 40 engine steps before two 2,048-token requests are admitted;
- 256 completion tokens/request, temperature 0.6;
- `max_num_seqs=min(512,tau)` and GPU-memory utilization 0.8;
- engine-owned cyclic-GC suppression during warmup/measurement, with the prior
  process state restored at exit; and
- complete raw request metrics, all 255 ITLs/request, token hashes, graph-miss
  counters, memory, KV-block state, clean Git/tree/source pins, model-file
  hashes, argv, environment, and immutable output.

A single run is never certified. The aggregate accepts exactly five distinct
seeds and recomputes all summaries from raw records.

#### $\tau=256$: predeclared gate fails

| Seed | Maximum interactive ITL (ms) | Long-request TTFT (ms) | Completion throughput (tok/s) | Run verdict |
| ---: | ---: | ---: | ---: | --- |
| 20260826 | 6.982 | 106.497 | 3,186.67 | pass |
| 20260827 | 7.142 | 107.532 | 3,195.52 | pass |
| 20260828 | 9.177 | 108.366 | 3,171.23 | pass |
| 20260829 | 13.838 | 107.370 | 3,170.51 | **fail** |
| 20260830 | 12.854 | 109.919 | 3,162.06 | **fail** |

The medians are 9.177 ms for the per-process maximum ITL, 107.532 ms for long
TTFT, and 3,171.23 tok/s. Only 3/5 processes meet the strict SLO, so the
retained verdict is `latency_not_certified`.

#### $\tau=512$: useful comparison, policy-ineligible

| Seed | Maximum interactive ITL (ms) | Long-request TTFT (ms) | Completion throughput (tok/s) | Observation |
| ---: | ---: | ---: | ---: | --- |
| 20260831 | 9.380 | 76.962 | 3,170.57 | below 10 ms |
| 20260832 | 7.762 | 64.265 | 3,211.99 | below 10 ms |
| 20260833 | 13.705 | 70.428 | 3,151.22 | **breach** |
| 20260834 | 7.762 | 64.381 | 3,209.02 | below 10 ms |
| 20260835 | 9.312 | 76.857 | 3,156.14 | below 10 ms |

The medians are 9.312 ms, 70.428 ms, and 3,170.57 tok/s. Compared with the
$\tau=256$ medians, throughput is effectively unchanged
($-0.0209\%$) while long TTFT is 34.51% lower. Nevertheless, $\tau=512$
is explicitly a throughput/TTFT profile and can never receive the sub-10-ms
latency label under this policy. It also observed one breach.

Earlier manually GC-disabled evidence at an older commit produced 5/5 passing
$\tau=256$ runs. Those files lacked the later model-content and self-pinned
source manifests. They remain historical reference and cannot override the
stronger current negative certificate.

The full archive pins commit `ba1bde44ce0735923ee41d7547121809dd1cfb91`,
tree `506d7305ee818dcbdf84354790d3bec711380554`, source SHA-256
`a0c6c56c73ad4784f3e2c8c5f133d0646bdde604ceb3216f3c716fdf3ebfd210`,
and model SHA-256
`0c659d1dba2804b0943c24bbece1858e93273147f7a11693fa99062df8c5997b`.
The $\tau=256$ aggregate and manifest hashes are
`940abdbd92baa91dead963bff71b0c85ef85d448923340b3b77e1d78d839cadd`
and
`f2ed8f1d111f816fa1fde4105961eb8e7678829caa44903f060a40f2284e5e11`.
The $\tau=512$ counterparts are
`96cbd5c79b004fe86592877bb4e10cb70e510bb7a9c6591acfb762ed74f16df1`
and
`35da99b9695cdd693f6e30af576ab25338efab106dba3392598a98e4d0d82383`.

### 4.8 Decode-jitter attribution

The two $\tau=256$ certificate breaches occurred in pure-decode steps rather
than mixed/prefill steps:

- seed 20260829: an 18-row decode, graph bucket 32;
- seed 20260830: a 16-row decode, graph bucket 16.

The $\tau=512$ breach likewise occurred in a 16-row decode. Across all five
full-completion $\tau=256$ runs, mixed-step maximum latency was 7.284 ms and
had zero threshold breaches. This is strong evidence against changing chunk
size or mixed scheduling merely to chase these isolated pauses.

A separate, intentionally intrusive diagnostic on branch
`fix/chunked-prefill-tail` ran 1,000 pure-decode steps at B16/bucket16 and
1,000 at B18/bucket32. It recorded scheduler, prepare, model, sampler,
postprocess and API wall/thread-CPU spans; CUDA events; thread context-switch
deltas; graph key; memory; and the residual across the existing
`tokens.tolist()` synchronization.

Exactly 8/2,000 API steps exceeded 10 ms:

- seven had runner CUDA time below the profile p99 but more than 10 ms of
  residual after the runner-end enqueue/across synchronization glue;
- one B16 step had 10.027 ms model CUDA, 10.555 ms runner CUDA, and only
  0.417 ms residual; and
- all eight recorded zero voluntary/involuntary context-switch deltas, while
  thread CPU time closely matched wall time.

The evidence is consistent with seven busy driver/runtime synchronization
pauses and one real GPU/model spike. It does not prove a single root cause:
instrumentation changes timing, CUDA events cover queued stream work rather
than every driver action, and there is no kernel-level profiler in this
container. It is therefore explicitly non-certifying and does not justify a
production “fix” on its own.

The 5,571,547-byte raw artifact has SHA-256
`6e8236dfb68c4ab42047c39ade251bad27d0697d2767d8bac215abd57092de79`.
It is retained on the
[diagnostic evidence branch](https://github.com/badle0/nano-vllm/tree/37e4021d8b52d3d462278b493cc87993aa5b0d80/benchmarks/chunked_prefill_tail/evidence/2026-08-18-a100-decode-jitter-d9639fc).

### 4.9 Tensor-parallel transport roofline

The original TP path serialized complete `Sequence` objects into a fixed 1 MiB
shared-memory slot. A realistic 64-sequence, length-4,096 case in the original
feature evidence produced a 1,089,291-byte payload, exceeding that slot by
40,715 bytes. This is a correctness failure, not merely a performance cost.

The repair sends a module-level, spawn-pickleable `ScheduledSequence` DTO
containing only the scheduled slice, mode/counts, last token and block table.
Ordinary local execution keeps the original `Sequence` object, so global
pickle behavior is no longer altered.

For token budget $\tau$, maximum sequences $S$, maximum model length $L$,
and KV block size $b$, the derived shared-memory allocation is:

$$
B_{\text{shm}} =
\operatorname{roundup}_{4096}
\left(
  8 + 65{,}536 +
  16\left[\tau+S\left\lceil\frac{L}{b}\right\rceil\right]
  +256S
\right),
$$

with a 64 MiB cap and an 8-byte validated frame header. The realistic
$\tau=16{,}384,S=64,L=4096,b=256$ case derives 364,544 bytes total and
364,536 bytes payload capacity.

In the current-source CPU serialization probe, the ordinary object was
782,509 bytes and the compact DTO was 53,710 bytes:

$$
1-\frac{53{,}710}{782{,}509}=93.136\%\ \text{reduction}.
$$

Tests cover actual multiprocessing `spawn`, OS `SharedMemory`, reusable events,
sequential compact prefill/decode/exit frames, header validation, checked
writes/reads, and overflow-before-publication. These provide strong transport
confidence.

They do not certify real TP2 model execution. This host exposes exactly one
A100, so no two-rank NCCL initialization, chunked-prefill/decode inference,
TP1/TP2 token comparison, hang check, or live shared-memory envelope was run.
TP2 remains an explicit release gate for a multi-GPU machine.

### 4.10 Evidence validation and fresh-run commands

Validate the scheduler samples with their scheduler-specific recomputing
aggregator. The `--output` path must not already exist:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /venv/main/bin/python \
  benchmarks/chunked_prefill_tail/validate_scheduler_roofline.py \
  --output /tmp/chunk_scheduler_validation.json \
  benchmarks/chunked_prefill_tail/evidence/2026-08-18-scheduler-da93670/scheduler_da93670_seed20260818.json \
  benchmarks/chunked_prefill_tail/evidence/2026-08-18-scheduler-da93670/scheduler_da93670_seed20260819.json \
  benchmarks/chunked_prefill_tail/evidence/2026-08-18-scheduler-da93670/scheduler_da93670_seed20260820.json \
  benchmarks/chunked_prefill_tail/evidence/2026-08-18-scheduler-da93670/scheduler_da93670_seed20260821.json \
  benchmarks/chunked_prefill_tail/evidence/2026-08-18-scheduler-da93670/scheduler_da93670_seed20260822.json
```

The graph/phase and full-completion archives use the general retained-evidence
validator:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /venv/main/bin/python -B \
  benchmarks/chunked_prefill_tail/validate_retained_evidence.py \
  benchmarks/chunked_prefill_tail/evidence/2026-08-18-a100-contract-phase-e50e732 \
  benchmarks/chunked_prefill_tail/evidence/2026-08-18-a100-full-completion-ba1bde4
```

Run the focused CPU contracts:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  /venv/main/bin/pytest -q -p no:cacheprovider \
  tests/test_chunk_scheduler_roofline.py \
  tests/test_chunked_prefill_retained_evidence.py \
  tests/test_chunk_tail_certification.py \
  tests/test_tp_transport.py \
  tests/test_varlen_graphs.py
```

For a new performance claim, use a clean committed checkout, compute the
harness source pin printed by `full_completion_cert.py`, run five new
$\tau=256$ seeds in five fresh processes, and aggregate them into a new
immutable archive:

```bash
PIN_COMMIT="$(git rev-parse HEAD)"
PIN_SOURCE="$(PYTHONPATH=. /venv/main/bin/python \
  benchmarks/chunked_prefill_tail/full_completion_cert.py \
  --print-source-sha256)"

for SEED in 20260840 20260841 20260842 20260843 20260844; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
    TORCHINDUCTOR_CACHE_DIR="/tmp/nv_chunk_full_${SEED}" \
    /venv/main/bin/python \
      benchmarks/chunked_prefill_tail/full_completion_cert.py \
      --model /workspace/models/Qwen3-0.6B \
      --tau 256 --seed "${SEED}" \
      --expected-commit "${PIN_COMMIT}" \
      --expected-source-sha256 "${PIN_SOURCE}" \
      --output "/workspace/.feat_bench/chunk-tail/full_tau256_seed${SEED}.json"
done
```

Do not overwrite, edit, pool with, or relabel the retained negative archive.
A code change requires a new commit pin and five entirely new processes.

## 5. Which costs are feature prices and which are defects?

The rooflines make this distinction more precise than “the feature is slower.”

### 5.1 Sampling

- **Greedy stochastic work was a defect.** Argmax semantics do not require
  softmax, exponential noise, or RNG advancement. The direct 0.177 ms route
  proves that nearly all of the old homogeneous-greedy cost was avoidable.
- **Active top-k selection is a feature cost.** Some work must identify the
  allowed set. The defect was applying it to inactive rows and perturbing the
  existing full-batch sampling graph. The repaired one-active/all-active
  scaling and the 6.15% E2E loss show the paid work is bounded for the tested
  configuration.
- **Exact top-p sorting is both a semantic price and an unresolved engineering
  target.** Under the audited Transformers support, tie identity, and fixed
  exponential RNG contract, a global ordered cumulative decision is required.
  The current 11.649 ms implementation is not proven optimal, however. The
  1.017 ms FlashInfer result proves that top-p as a statistical feature need
  not be slow; it reaches that speed by changing boundary-tie and RNG semantics.
  Therefore “exact top-p remains slow” is accurate, while “all top-p must cost
  11.6 ms” is not.

### 5.2 Token streaming

- **Sub-millisecond delivery bookkeeping is the implementation cost.** Its
  p95 is 0.0397 ms and the matched-work throughput interval is well inside
  $\pm2\%$, so there is no measured streaming overhead defect in the
  certified workload.
- **The $B\delta$ slow-consumer term is an API-contract price.** The public
  iterator is synchronous and intentionally does not run an unbounded
  producer. A caller sleeping after every event stalls the next engine step.
  Removing that term would require a separately designed bounded asynchronous
  API, not a local optimization.
- **Bounded incremental decoding is necessary work.** The prior cumulative
  decoder's quadratic behavior and inability to retract text were defects.
  The repaired 48-token input roof plus one exact full flush is the deliberate
  correctness/performance contract.

### 5.3 Chunked prefill

- **More attention against a longer prefix is a feature price.** The
  $qc+q(q+1)/2$ term explains deterministic growth within a long prefill.
  Larger $\tau$ legitimately buys fewer mixed steps and lower long TTFT at
  the cost of higher interactive ITL.
- **Backlog scanning, incoherent mutable budgets, illegal graph selection, and
  fixed-frame TP overflow were defects.** Their repaired complexity,
  correctness, and transport roofs pass.
- **The isolated over-10-ms decode stalls are unresolved system behavior, not
  demonstrated chunk-work cost.** They occurred in decode-only graph steps,
  while mixed steps remained below the threshold. The current evidence does
  not support changing chunk scheduling to fix them.

## 6. Release-candidate judgment and required reruns

| Surface | Contribution-level evidence | Safe current claim | Next release-grade experiment |
| --- | --- | --- | --- |
| Homogeneous greedy | Direct compiled argmax passes time, memory, output and RNG gates | Resolved for all-greedy batches | Fresh current-commit B1/B64/B256 sampler plus E2E A/B if byte-for-byte RC certification is desired |
| Top-k | Active-row semantics pass; four B256 E2E pairs all stay within the 10% loss budget | Resolved for the tested $k=50$, B256, A100 workload | Repeat across representative $k$, B, vocab/model and at least two prompt/seed families |
| Exact top-p | Correctness passes; all-active performance misses the 10% E2E time budget | Correct default; performance unresolved | Either retain exact contract and prototype a source-pinned exact kernel, or explicitly adopt the statistical backend contract |
| FlashInfer top-p | Complete wrapper below 1.6 ms; eight fresh E2E pairs improve throughput | Fast opt-in backend for its documented distributional/tie/RNG semantics | Re-run wrapper and eight balanced pairs at the release commit and supported dependency lane |
| Token streaming | Eight-worker contribution certificate passes all gates | Resolved for synchronous single-owner in-process streaming at the pinned commit | Re-run schema-3 harness at the final RC commit; add separate async/server benchmarks only if such APIs are introduced |
| Chunk scheduler and graphs | Backlog, admission, low-$\tau$, short-context, 4x511 and unpadded-token gates pass | Mechanisms resolved for TP1 | Keep these CPU/GPU contracts in CI and rerun graph cases on each supported GPU/software lane |
| Chunk request latency | Five-run current self-pinned archive fails 3/5 strict $\tau=256$ criterion | **Do not advertise sub-10-ms certification** | Run five fresh current-commit $\tau=256$ processes after any justified latency change; all five maxima must be <10 ms |
| TP compact transport | Spawn/shared-memory tests and payload bound pass | Transport repair is credible | On a two-GPU host, run TP2 init/exit, mixed prefill+decode, realistic 64x4096 envelope, TP1 token comparison and hang/overflow checks |

No result in this document changes the separately maintained request-metrics
verdict. That contribution was intentionally excluded from this roofline scope.

## 7. Interpretation limits

- The principal GPU measurements use one A100-SXM4-40GB and one small model.
  Roofline ratios based on nominal peak bandwidth/compute are classification
  aids, not achieved-kernel efficiency claims.
- CUDA-event time, synchronized host wall time, caller timestamps, CPU process
  time, and allocator peaks answer different questions. This report preserves
  those domains rather than combining them into one “latency.”
- A fresh process is the statistical unit only when the protocol says so.
  Repetitions within one process primarily estimate steady-state variation and
  must not be counted as independent machines.
- A median cannot certify a maximum-tail SLO. This is why the chunked-prefill
  $\tau=256$ profile remains uncertified despite a median per-run maximum of
  9.177 ms.
- Exact token equality, statistical distributional equivalence, and matching
  support are distinct contracts. The exact and FlashInfer top-p routes are
  intentionally not interchangeable.
- Retained raw evidence should remain immutable. New source, dependencies,
  model files, or benchmark logic require a new commit/source/model pin and a
  new output artifact rather than editing an old JSON.
