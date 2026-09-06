# V4 retained GPU validation — 2026-09-06

## Result and claim boundary

**PASS for the registered fixed-capacity V4 shadow workload in eager and CUDA
graph modes.** This is not a certificate for all possible configurations, for
automatic KV sizing, or for completed speculative decoding.

Runtime commit: `22b63e8e24db3c7bc9c24b61aedd93b25d76d289`.
Runtime `nanovllm` tree: `922d81417cf72ec912da13267fbb024c145a6a15`.
Clean GPU producer: `6ca56b43541237a9c0e588fa8101b7fa08c9409d`.

Archive: [2026-09-06-a100-v4-6ca56b4](../../benchmarks/speculative_v4/evidence/2026-09-06-a100-v4-6ca56b4/manifest.json).
The archive contains six raw JSON records, six unmodified stdout/stderr logs,
and a manifest. The standalone checker pins the manifest SHA-256:

```text
582e1e112213b2b5bdd796febce683af5d1530ee6cc45678dd374859b7e6bc1e
```

The runtime still **does not execute a speculative target verifier, accept or
reject draft tokens, or emit a burst/bonus token**. Draft proposals are discarded;
the ordinary target produces each public token. All plan certificates retain
`gpu_certified=False`. No throughput, speedup, acceptance-rate or maximum
workspace-peak claim is made.

## Registered environment

| Item | Value |
| --- | --- |
| GPU | One NVIDIA A100-SXM4-40GB, compute capability 8.0 |
| Visible GPU memory | 42,406,903,808 bytes |
| GPU UUID | `61d56efc-2291-5bb3-c9b2-25d16c9179d9` |
| Driver | 570.133.20 |
| Python | 3.12.13 |
| PyTorch / wheel CUDA | 2.10.0+cu128 / 12.8 |
| Target and draft | Same local Qwen3-0.6B checkpoint |
| Vocabulary | 151,936 |
| Configured K / maximum live B | 2 / 4 |
| Model length / work-token budget | 512 / 1,024 |
| KV pool | **64 blocks explicitly configured**, block size 256 |
| GPU memory utilization setting | 0.5 |
| Sampling backend / TP | Exact / TP=1 |
| Seed | 20260906 |

The model safetensors file is 1,503,300,328 bytes, SHA-256
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
Full model/tokenizer metadata and weight hashes are recorded before and after
each producer. Single visible GPU identity is recorded; continuous isolation
from other host processes is not certified.

## Experiments and results

Each cell ran in a fresh process with independent, initially empty Inductor and
Triton directories. Remote/autograd/autotune/PGO cache flags were disabled for
the proof. Dynamo was enabled, errors were not suppressed, and constructor
pretouch had to produce nonempty compiler artifacts and compiled graphs.

| Mode | Side | Guarded draft intervals | Draft graph-replay steps | Result |
| --- | --- | ---: | ---: | --- |
| Eager | Off | 0 | 0 | PASS |
| Eager | Cold draft KV filled with zero | 46 | 0 | PASS |
| Eager | Cold draft KV filled with NaN | 46 | 0 | PASS |
| Graph | Off | 0 | 0 | PASS |
| Graph | Cold draft KV filled with zero | 46 | 75 | PASS |
| Graph | Cold draft KV filled with NaN | 46 | 75 | PASS |

That is **184 guarded draft intervals**, including **128 route-sweep intervals**
across the four enabled producers. Each graph-enabled producer recorded 20
graph-object constructions and 20 capture contexts during initialization, with
the same counts after the workload. These counters include the constructor's
profiling/capture work, not 20 distinct runtime serving routes.

### 1. Public output and RNG control

Two prompts run to six completion tokens each, using temperature 0.8, top-k 8,
top-p 0.9, and `ignore_eos=True`. Within each mode, off/zero/NaN producers have
identical request IDs, all 12 public token events including finish flags, and
all seven CPU/CUDA RNG checkpoints. Completion-tail admission naturally reduces
K and then falls back to an ordinary final target step. There are four positive
shadow intervals in each enabled control.

This is same-mode shadow/baseline identity, not a general promise of sampled
eager-versus-graph token identity or future speculative same-seed identity.

### 2. Complete registered draft-route sweep

For every live B in 1–4 and K in 1–2, execute two repetitions of a cold catch-up
cycle followed by a warm cycle. The test truncates an already-ready route
admission to the requested K; it does not add routes or bypass the planner's
budget, workspace, state or physical-capacity checks.

The real `LLMEngine._step()` path must call `Scheduler.plan_speculative_step()`
and pass a positive `SpecStepPlan` to the production runner. There are 32 route
intervals per enabled producer. All four eager registry keys and all 12 graph
registry keys are visited. Eager keys describe dynamic batches, so separately
enumerating live B=1–4 matters; graph B=3 exercises the padded bucket 4.

The tested row mix is combined top-k/top-p, greedy, unfiltered sampling and
top-k sampling (truncated by live B). This is not an exhaustive sweep of sampler
parameters; in particular, standalone full-vocabulary top-p-only sampling is
not separately certified by this GPU workload.

Every guarded interval uses `fail_on_recompile` and compares full compiler
counters, guard failures, graph-break reasons, compiler-cache file manifests,
and capture counters before/after. CPU/CUDA RNG hashes must match and attention
context must be reset. Run-bound begin/end markers in the raw log must match
the JSON interval ledger exactly, with no recompile/graph-break/warning text
inside a guarded interval.

**Scope caveat:** whole-process compilation is not unchanged. The enabled
zero-fill logs contain 19 eager and 27 graph recompilation announcements outside
the guarded intervals. Initialization and ordinary target work are outside this
no-recompile claim. This does not certify the unimplemented `[B*(K+1), V]`
target-verifier path or its warmup requirements.

### 3. Planned geometry and modeled workspace

The runner independently revalidates the full V4 plan before draft compute.
The archive checker additionally verifies row state/headroom, highest target
and draft write positions, reserved block-table capacity, integer K, count
identities, a q+p memory floor, and `gpu_certified=False`.

For the largest registered cold route, B=4, K=2, C=16:

```text
draft work              = B*K       = 8
planned verifier work   = B*(K+1)   = 12
planned full cycle      = C+8+12    = 36 <= 1024
actual V4 shadow work   = C+8+B     = 28

q bytes = 4*4*2*151936 =  4,861,952
p bytes = 4*4*3*151936 =  7,292,928
q+p floor             = 12,154,880

modeled live peak      = 85,085,308 bytes
workspace reservation  = 152,194,172 bytes
```

The peak/reservation include the existing conservative transform/filter/race/
rejection phase model and allocator margin. They are **not measured peaks**;
V4 does not materialize the future p/verifier owner. The standalone checker
does not reimplement every memory-model phase or claim a second numerical
memory oracle; runtime certificate recomputation and CPU planning tests cover
that layer. Historical source and raw bytes are pinned independently.

### 4. Physical boundaries, failure undo and retry

For prompts of length 254/255/256/257, execute a real draft interval, inject an
exception after handoff but before the ordinary target call, and compare exact
allocator and sequence snapshots before/after failure. Snapshot hashes include
free-list order, used IDs, refcounts, hashes/token metadata, token histories,
cache coverages, scheduled counts and block tables. Then retry successfully.

| Prompt length | Committed L after prefill | Highest future target write, K=2 | Additional speculative blocks |
| ---: | ---: | ---: | ---: |
| 254 | 255 | 256 | 1 |
| 255 | 256 | 257 | 1 |
| 256 | 257 | 258 | 0 |
| 257 | 258 | 259 | 0 |

The length-256 case also exercises undo of the ordinary decode-boundary append.
Every failure restores its snapshot and releases transaction ownership; every
retry succeeds. Successful target commit retains only the baseline block table.
Cancellation after each case leaves all 64 blocks free and no live leases.
This does not test rollback after a partial multi-row target postprocess commit.

### 5. Zero/NaN fill and prefix-cache numerics

Only **cold draft** rows are poisoned; valid warm draft coverage and target KV
are never overwritten by the harness. For every one of the 46 enabled intervals,
zero and NaN producers agree on the full plan, logits/probability SHA-256 hashes,
probability row sums, proposal token IDs and RNG observations. Probabilities
must be finite, nonnegative, and normalized within absolute tolerance 2e-6.

A 257-token prompt is also run cold and then with a target-prefix cache hit:
prefill processes 257 tokens then one token. Draft logits/probability hashes,
proposal tokens and the next target token match exactly. Full arrays are not
retained: this is producer-observed hash equality, not an independently
reconstructed HF/model numerical oracle. The V1 sampling-law proof remains a
separate certificate.

## Separate failed experiment: automatic KV sizing

Before fixing the control pool, producer
`87c3b91dd1aba0a043e55dda411a2e2c408fc9f1` ran the same configuration with the
default automatic block count. Its first eager enabled run passed; a later
fresh eager enabled process failed during construction, before any draft
interval, with `SpeculativeKVCacheCapacityError`:

```text
budget_headroom     = -177,405,952 bytes
runtime_transient   =   67,126,784 bytes
workspace_reserved  =  152,194,172 bytes
shortfall           =  396,726,908 bytes
```

The headroom guard failed closed. Its root cause/reproducibility has not been
fully isolated, and **automatic sizing has not been fixed or certified here**.
An explicit 64-block pool makes paired workload capacity deterministic and
leaves ample room; it does not establish that the auto-sized configuration is
safe. This is a follow-up lifecycle/memory-sizing investigation, not a reason
to suppress the guard or claim all known issues are resolved.

Raw diagnostic: [auto-KV startup failure](../../benchmarks/speculative_v4/diagnostics/2026-09-06-auto-kv-init-failure.log),
12,951 bytes, SHA-256
`b864cee5b89c0d27089afaa3f12b0b010aa914584f940e42142fc34a13be7c4f`.
It is deliberately separate from the passing six-cell archive.

## Validate or reproduce

Final local regression: **1,073 passed, 31 skipped, 14 deprecation warnings in
97.57s**, with CUDA hidden. The retained-evidence suite contributes 32 passing
tests, including altered manifest/payload bytes, forged provenance, missing
route/log coverage, changed public tokens and empty normalization observations.
The standalone validator also passed under `python -S` (site packages disabled),
and the frozen V3 archive still validates. GPU-dependent skipped pytest cases
are not certified by this CPU result. CI now includes the V4 archive checker
and tests; remote CI has not been run in this work session.

Validate the retained bytes, historical sources, interval logs, and paired
oracles without Torch/CUDA or the model:

```bash
python benchmarks/speculative_v4/validate_retained_evidence.py \
  benchmarks/speculative_v4/evidence/2026-09-06-a100-v4-6ca56b4
```

Full Git history must include the producer and runtime commits. The manifest
is a pinned trust root, not an editable index that can bless altered payloads.
The validator rejects altered bytes, duplicate/nonfinite JSON, symlinks/hard
links, missing/extra archive members, mismatched producer/source/model identity,
overlapping compiler caches, invalid plans and incomplete route/log coverage.
It does not execute code supplied by an artifact. The historical files are
checked with Git, not imported. As with the V3 validator, the filesystem is
assumed quiescent during validation/sealing.

For a new GPU reproduction, use a separate clean checkout at the producer SHA
(not today's evidence/docs commit), the same local model content, and the
registered software/hardware. For each `mode` in `eager graph` and `side` in
`off zero nan`, run `tests/run_speculative_v4_gpu.py` with:

```text
--mode <mode> --side <side>
--model /workspace/models/Qwen3-0.6B
--expected-commit 6ca56b43541237a9c0e588fa8101b7fa08c9409d
--output <new-outside-repo-path>/<mode>-<side>.json
```

Capture stdout/stderr to the adjacent `<mode>-<side>.log`. Set
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=.`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and `TORCH_LOGS=recompiles,graph_breaks`. Each invocation
needs fresh, distinct `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` outside
the source/model/output paths. Set each of the following to `0`:

```text
TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE
TORCHINDUCTOR_AUTOGRAD_CACHE
TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE
TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE
TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE
TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO
TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO
```

The exact invocations, source snapshots, paths, software versions and model
hashes used here are also embedded in every raw JSON artifact. New runs need a
new reviewed archive/pin; they must not overwrite or relabel this archive.

## What remains

- Review the joint queue/pool/route-cap hard-gate coverage before declaring all
  of V4 complete. This is a narrow B<=4/K<=2/fixed-pool GPU certificate.
- Investigate the failed auto-KV constructor experiment separately.
- V5 target verification, exact rejection/bonus handling and atomic physical/
  logical/cache-hash commit are still unimplemented. So are acceptance-driven
  streaming/metrics and V7 speedup/peak-memory certification.
- Heterogeneous draft/target models, TP>1 and FlashInfer remain outside scope.
