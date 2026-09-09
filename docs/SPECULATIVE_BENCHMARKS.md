# Speculative decoding: benchmark and integration record

For the later invariant/speculative repair bundle, see the
[2026-09-09 replacement A100 qualification](REPLACEMENT_A100_QUALIFICATION.md),
including measured speedups and the unresolved automatic-KV sizing regression.
The record below describes the original integration runtime.

## Provenance and scope

This branch is a curated integration, not a rerun or reinterpretation of the
historical qualification. The runtime is copied unchanged from experiment tip
`fae471785e72c20f9e17b4329d1ff10dbf429b31`; its `nanovllm` Git tree is
`c2c9fbce937217ad2227955fbb4451441a721dd3`. Integration starts at fork-main
`663753b99131945c297c1fbe02341108f422dce7` without importing the experiment ancestry.

The frozen experiment branch is `feat/spec-v2-performance`. Its
[full report](https://github.com/badle0/nano-vllm/blob/fae471785e72c20f9e17b4329d1ff10dbf429b31/docs/pr8/12_v7_experimental_qualification.md)
and [raw V7 archive](https://github.com/badle0/nano-vllm/tree/fae471785e72c20f9e17b4329d1ff10dbf429b31/benchmarks/speculative_v7/evidence/2026-09-06-a100-v7-89829e6)
include per-cell results, exclusions, failed attempts, weight hashes and raw
compiler/lifecycle evidence. **These links require publishing the experiment
branch to the fork; preparing or pushing the slim branch alone does not do so.**
The full data remains available in the original local experiment worktree.

Historical producers:

| Evidence | Commit / digest |
|---|---|
| Final runtime first committed | `a165b654660e60ca60cecd47b89c95de1d65735b` |
| Heterogeneous cold correctness | `a715a199d413a67ba563271f1b4fa8fe87f00eaa` |
| Timing harness/runtime | `89829e6052c17e0ef4fcd65e294d0f1e78139184` |
| Pre-V5 speculation-off control | `2678d764ad0341bbfbdd2a93ac0e5528959058a4` |
| V7 manifest SHA256 | `c7fe59acd0db36882ba7bf8fb8f4c9c60d607fe602a283511d96bdccc9bd44ee` |

## Results: correct within the tested envelope, not faster

Hardware/software: one A100-SXM4-40GB, driver 570.133.20, Torch 2.10.0+cu128,
Python 3.12.13. Target Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c`; draft Qwen3-0.6B.

- V5/V6: 335 guarded speculative intervals across eager/graph, enabled/disabled
  and automatic KV sizing, with the final runtime tree.
- Heterogeneous graph correctness: 271 intervals, all 80 registered B=1..4,
  K=1..4 and sampling-family cells, matching-mode greedy/cache/stream controls,
  recoverable fault and RNG rollback, actual pending-burst close/GC and causal
  checks. No compiler/capture changes in guarded intervals. Normal measured live
  workspace peak was 170,415,104 bytes, below its registered model.
- V7: 32 successful fresh-process runs, 1,308 raw timing samples and 238
  intrusive diagnostic cycles. Twelve timing samples were excluded for observed
  GPU contention; preregistered repair pairs restored five valid pairs per
  primary/off-regression cell. Two failed setup/fixture attempts are retained.

| Comparison | Observed result |
|---|---|
| Active B=1/4, speculation off/on wall-time ratio | 0.276–0.521: **1.92–3.62x slower** with speculation |
| B=8 whole-batch bypass, off/on ratio | 0.963–0.974; draft resources remain resident |
| Speculation-off old/current latency change, B=1/4/8 | -0.17% / +0.04% / +0.40%, within the preregistered +/-5% band |

No primary cell showed acceleration. Neither universal correctness nor production
tail-latency, optimal K or an optimal routing crossover is claimed. TP>1 and
FlashInfer speculation are unsupported. Greedy/mixed batches deliberately use
K+1 ordinary target calls for BF16 compatibility, not a parallel speedup lane.

## Timing protocol

Primary cells used B=1/4/8, context 32/256, greedy/plain/top-k/top-p/combined,
64 output tokens and `ignore_eos=True`. Both sides used graph-enabled engines,
64 KV blocks, model and token budgets 4096, utilization 0.8 and configured K=4.
Five process pairs followed AB/BA/AB/BA/AB, with two full warmups and seeds
17/23/41 per cell. The first five complete valid pairs were selected under the
preregistered exclusion/repair policy; failed or incomplete pairs were not
silently averaged in. Each pair contributes the median of three seed ratios;
descriptive bootstrap intervals resample process pairs, not independent seeds.

Timing wraps the public API with CUDA synchronization and no internal phase
hooks. Warmups reuse synthetic prompts, so target prefix cache hits are possible:
TTFT is for this warmed workload, not cold-request serving. Cold compiler checks
are separate correctness runs. Boundary process snapshots do not continuously
prove exclusive GPU occupancy. Supplemental batch/context/tail/stream cells had
only one process per side and do not inherit five-pair confidence.

## Roofline and phase interpretation

The measured attainable roofs were approximately **1.372 TB/s** device copy
bandwidth and **253.787 TFLOP/s** dense BF16 GEMM, giving a ridge near 185 FLOP/byte.
Calibration used a 256-MiB copy (read + write bytes), a 4096-square GEMM, ten
warmups and fifty CUDA-event-timed repetitions. These are microbenchmark roofs,
not promised sorting, small-GEMM or decode throughput. Units are decimal except GiB.

Target unique weight storage is 8,044,936,192 bytes; draft is 1,192,099,840 bytes.
Tied embedding/LM-head storage is counted once. Raw timing fields named
`target_weight_bytes`/`draft_weight_bytes` sum parameter objects and double-count
ties; use the phase profiler's unique-storage ledger for physical bytes.

For target L=36, H=2560, I=9728, query heads=32, KV heads=8, D=128, V=151936:

```text
linear weight elements/query
  = L * [2*H*(32*D) + 2*H*(8*D) + 3*H*I] + V*H
  = 4,022,272,000
linear FLOPs/query = 2 * elements = 8.044544 GFLOP
target KV bytes/cached token = 2 * 36 * 8 * 128 * 2 = 147,456
draft KV bytes/cached token  = 2 * 28 * 8 * 128 * 2 = 114,688
```

An optimistic ordinary batch-B step streams one weight set plus roughly
`B*C*147456` target KV bytes at context C. Attention additionally costs about
`4*L*B*C*32*D` FLOPs; writes, norms, rotary, activation and sampling add work.
Ideal weight-stream floors are approximately 5.864 ms target and 0.869 ms draft.
Two 64-block pools at 256 tokens/block occupy exactly 4 GiB combined; this is
reserved capacity, not logical occupancy or total process memory.

At B=4, K=4, retained FP32 p/q alone cost
`B * (K + K + 1) * V * 4 = 21,878,784` bytes. This is not total workspace:
logits, filtering, two top-p sort scratch payloads, FP64 residual/rejection,
metadata and allocator margin are separate live owners. The first GPU sweep
found a sort-scratch underestimate; the qualified runtime includes its correction.

Break-even requires:

```text
emitted tokens/cycle * ordinary decode step time
  > draft catch-up/proposals + target verify + accept/bonus + commit + overhead
```

High acceptance alone is insufficient. For B=1 code, 32 output tokens, intrusive
graph-mode means were 89.466 ms/cycle for greedy (42.546 draft, 44.495 verify;
5 emitted tokens/cycle) and 69.078 ms for plain sampling (30.122 draft, 35.994
verify; 3.875 emitted tokens/cycle). These synchronized diagnostics are not
headline timings. Draft catch-up is eager even for one token, parallel target
verification is eager, and full-vocabulary exact sampling remains costly.
These are profiling/optimization candidates, not already implemented speedups.

## Reproduce on the slim branch

Use the GPU environment with compatible dependencies and local checkpoints.
Run from the repository root. Output paths must be fresh; keep generated results
outside the source tree. Each invocation constructs and exits its own engine.

```bash
# Full available test suite; CUDA tests need the real GPU dependencies.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m pytest -q -p no:cacheprovider

# Fresh-process integration checks (repeat with --mode eager).
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python tests/run_speculative_v5_gpu.py \
  --model /path/to/Qwen3-4B --draft-model /path/to/Qwen3-0.6B \
  --mode graph --enabled --sweep --output /tmp/spec-graph-check.json

# A paired smoke experiment, not the complete five-pair historical protocol.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python tests/run_speculative_v7_benchmark.py \
  --model /path/to/Qwen3-4B --draft-model /path/to/Qwen3-0.6B \
  --revision HEAD --mode graph --suite smoke --output /tmp/spec-off.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python tests/run_speculative_v7_benchmark.py \
  --model /path/to/Qwen3-4B --draft-model /path/to/Qwen3-0.6B \
  --revision HEAD --mode graph --suite smoke --enabled --output /tmp/spec-on.json
```

Check output token counts and `spec_cycles` to distinguish actual execution from
bypass. Inspect each sample's `accepted` flag and rejection reasons; program exit
alone does not make all timing samples acceptable. Compare greedy outputs with
matching mode controls; stochastic output IDs need not match at the same seed.
The timing tool requires committed runtime files matching `--revision`.

The V5 tool's stricter `--retained --expected-commit SHA` mode additionally
requires a clean committed checkout, isolated empty executable compiler caches
and remote caches disabled. Follow the frozen report for exact environment and
archive-validation commands. Do not use its old pinned manifests to certify a
new run or edit their hashes to make an unrelated artifact pass.

Raw evidence and historical validators remain on the full experiment branch.
The slim branch carries runtime regression tests and usable GPU tools, not the
historical archive-specific test count. Its own validation is recorded separately
below; historical measurements above are not newly collected integration results.

## Slim integration validation (2026-09-06)

Implementation/test/CI checkpoint:
`7e2c6b560334a6617380915e3b62bc358b6fe7b2`. The subsequent validation-record
commit changes documentation only. The complete `nanovllm` tree matches the
qualified experiment exactly (`c2c9fbce937217ad2227955fbb4451441a721dd3`).

| Check | Result |
|---|---|
| CPU run with Qwen shim, CUDA hidden | 991 passed, 31 skipped, 14 warnings |
| Full suite with real CUDA dependencies | 1,021 passed, 1 skipped, 14 warnings |
| Exact CPU-CI selection in independent depth-one clone | 991 passed, 26 skipped, 14 warnings |
| Syntax and all six existing evidence-validation commands in depth-one clone | Passed |
| Fresh Qwen3-4B / Qwen3-0.6B graph-enabled integration sweep in depth-one clone | Passed: 255 cycles, 80 sweep cells |
| GPU integration, timing and phase tool CLI imports in depth-one clone | Passed |
| Whitespace, runtime identity, unchanged existing benchmarks/package/license | Passed |

The shallow clone contained only this checkpoint and could not resolve the
experiment tip. Thus these checks did not depend on hidden experimental Git
history or speculative raw archives. The smaller test count is intentional:
272 historical artifact/harness tests remain on the experiment branch, rather
than being marked skipped here. The additional five CPU-CI exclusions are the
Triton-importing histogram module; those tests ran in the CUDA suite.
Warnings were the existing TorchScript deprecation warnings.

The fresh graph sweep used the command above with `--enabled --sweep`, default
B<=4/K=4, model length 512, input budget 1024 and fixed 64-block KV pools. It
observed four pending tokens before both close and abandonment, passed causal
verification and recoverable fault/RNG rollback with retry, and produced four
13-token outputs in each stochastic/mixed case. This was a tool/integration
check, **not** a new five-pair performance qualification, matching off-control
pair, or isolated-cold-cache `--retained` certificate.

Its temporary raw result is outside Git at
`/tmp/nano-vllm-slim-graph-7e2c6b5.json`, SHA256
`49d87b7c400343c813b539090debafb2b4b83e40dfb6b7e4b69b61d66bbcf110`.
This temporary file is not a durable archive; the complete historical evidence
remains on the pinned experiment branch.

Local tests used Python 3.12.13 and Torch 2.10.0+cu128. The GitHub Actions
Python 3.10/3.12, Torch 2.4.1 CPU dependency matrix has **not** run remotely yet;
passing the local selection does not establish that matrix's result. Require
the fork-local PR checks before merging. Neither branch has been pushed or
merged as part of this preparation.

### First remote CI follow-up

[PR #7's initial run](https://github.com/badle0/nano-vllm/actions/runs/34048871011)
passed both syntax/evidence jobs, but Python 3.10 CPU-test collection failed:
the older chunked-prefill benchmark helpers imported `datetime.UTC`, an alias
introduced in Python 3.11. Matrix fail-fast then cancelled the Python 3.12
contract job; cancellation is not a passing or failing test result for that lane.
The preceding local Python 3.12 results could not establish 3.10 compatibility.

The follow-up replaces that alias with `timezone.utc` in `common.py`,
`aggregate_certification.py`, `full_completion_cert.py` and
`scheduler_roofline.py`. Timestamp timezone/format is unchanged. It keeps
Python 3.10 in the matrix and adds a subprocess regression that removes the
newer alias before importing all four helpers. Matrix fail-fast is disabled so
both Python versions can report independently.

This narrow follow-up does change four benchmark helper source files; the
earlier "unchanged benchmarks" check describes the original integration
checkpoint, not this follow-up. No retained JSON, manifests or provenance hashes
are rewritten. Both chunk archives still validate, and the complete `nanovllm`
tree remains identical to the qualified runtime. The affected test modules plus
the regression passed locally (44 tests). The complete CPU-CI selection then
passed on local Python 3.12/Torch 2.10: 992 passed, 26 skipped, 14 existing
TorchScript deprecation warnings. This includes the missing-UTC simulation,
not an actual local Python 3.10 execution. A successful rerun of the remote
dependency matrix is still required before merging.

### Second remote CI follow-up: fixture and compiler environment

[Run 34049297973](https://github.com/badle0/nano-vllm/actions/runs/34049297973)
passed syntax/evidence on both Python versions and got past the previous UTC
import failure. Each CPU-contract lane then reported 979 passed, 26 skipped,
four failures and nine setup errors:

- Nine real-tokenizer detokenizer cases tried the instance-specific path
  `/workspace/models/Qwen3-0.6B`, absent on hosted runners. The earlier shallow
  clone was isolated from Git history, not from that model directory.
- Four ordinary sampler tests hit Torch 2.4.1 CPU Inductor's
  `Tried to erase Node div but it still had 1 users ... copy_` failure. They are
  top-k boundary ties, top-k=1 ties, top-p crossing and peaked-row collapse.
  This is a compiler failure, not a sampled-support assertion failure.

The follow-up changes the CPU job to Torch 2.10.0, aligning its release with the
validated runtime stack. Official CPU wheels are available for both Python 3.10
and 3.12 in the [PyTorch CPU index](https://download.pytorch.org/whl/cpu/torch/).
Both Python versions, all four sampler tests and compilation remain enabled;
there is no `suppress_errors`, eager replacement or newly excluded test module.
This change does not fix Torch 2.4.1, qualify that older CPU compiler, change the
package's broad dependency declaration, or establish GPU compatibility for 2.4.1.

CI now downloads exactly `config.json`, `tokenizer_config.json` and
`tokenizer.json` from Qwen/Qwen3-0.6B at immutable revision
`c1899de289a04d12100db370d81485cdf75e47ca`, into the runner's temporary directory.
The test fixture takes `NANOVLLM_TEST_TOKENIZER_PATH`, requires a local directory
and uses `local_files_only=True`. Fixture setup may use the network; tests remain
offline. No model weights or tokenizer payloads are committed. Added regression
tests check the pinned three-file download, missing-file failure, configured
fixture path and explicit missing-directory failure.

Validation uses a temporary environment with Transformers 4.51.3, tokenizers
0.21.4, NumPy 1.26.4, safetensors 0.5.3 and the downloaded tokenizer, without
changing the main venv. It inherits the installed Torch 2.10.0+cu128 and runs
with CUDA hidden. It is therefore a closer local CPU-contract check, **not** an
exact test of GitHub's CPU-only wheel or its Python 3.10 interpreter. Remote CI
still must pass after pushing. The `nanovllm` runtime remains unchanged.

The full CPU-CI selection in that environment passed: **996 passed, 26 skipped,
14 existing TorchScript deprecation warnings** (131.91 seconds). This includes
all nine formerly missing-tokenizer cases, all four formerly failing compiled
sampler cases, and four new fixture/download regression tests. The new scripts
also parse under the Python 3.10 grammar. No inference runtime, benchmark code,
retained evidence or package metadata changed in this second follow-up.
