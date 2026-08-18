# Chunked-prefill tail diagnostics

This directory contains release-pinned observers and contract gates. It does
not change scheduler, model, attention, or graph-routing production code. Run
each command from a clean committed checkout in a fresh process; nano-vllm owns
its process group and selected GPU allocation until process exit.

## Evidence contract

Every retained JSON contains the exact argv, clean Git commit/tree/branch,
aggregate SHA-256 plus per-file hashes for the executable source surface, model
path plus content hashes, Python/package/CUDA/GPU environment, and UTC start
time. The source identity is checked before and after execution. Output uses
exclusive creation (`O_EXCL`) and mode `0444`, so an existing result is never
silently replaced.

The step diagnostic records, for every full `LLMEngine._step()`:

- synchronized wall and CUDA-event time;
- actual prefill/decode/total tokens and scheduled rows/segment count;
- candidate and selected CUDA graph key, model route, and graph-miss delta;
- cold/steady label, queue state, mid-chunk owner, emitted events, and finishes;
- prepare-to-model, model, and post-model/sampler wall and CUDA-event spans;
- allocated/reserved CUDA memory and used/free KV blocks, plus run peaks.

The phase CUDA values are spans, not isolated kernel sums. In particular, they
intentionally retain host/runtime gaps between work enqueued at the two marker
events. This distinguishes two plausible tail causes: real model/attention
growth as a long prefix advances, and a slow outer step whose model and sampler
spans remain ordinary. A zero graph-miss counter alone cannot distinguish them.

`cold` means the first `--cold-steps` immediately after the long request is
admitted; later measured steps are `steady`. Conditioning the interactive cohort
before admission is not included in either timing population.

## Decision gates

Treat a cell as a correctness pass only when all of the following hold:

1. The selected key matches the recorded live token/segment shape; graph-backed
   cells have zero miss delta. Tau 64 deliberately has no captured bucket, so
   exactly one eager miss is expected instead.
2. The graph/eager greedy token vectors match. The 4x511 boundary additionally
   requires exactly 2,044 real tokens, four unpadded 511-token segments,
   `max_model_len=512`, selected key `(2048, 5)`, zero misses, and matching
   end-to-end tokens.
3. Actual per-step tokens never exceed tau, decodes remain present on mixed
   steps, and queue/mid-chunk state is consistent with the intended workload.
4. Judge a latency promise from steady route-specific p95/max, not a pooled
   median. Attribute a threshold breach using the phase spans and prefix state.
   Historical tau-256 evidence met the latency SLO below; tau 512 is explicitly
   a throughput/TTFT-only historical reference, not a sub-10-ms latency fix.
5. Peak allocated/reserved memory and KV-block consumption must fit the release
   headroom, with no monotonic per-step growth unexplained by longer live KV.

These are diagnostic gates. They identify the limiting component before any
production optimization is proposed.

## Release classification

Five retained A100-SXM4-40GB processes per tau, seeds 20260821 through 20260825,
ran with Python GC disabled manually before engine construction at evidence commit
`4a5742ab4003c1ecf7e442a812cbba3e04e06450`. Its `nanovllm/` tree is identical
to this branch's production base
`1ffe033bfc60dab6875634e90ffe816d772ec338`. The release SLO is strict maximum
interactive ITL below 10 ms. Each process used 16 64-token interactive requests,
40 steps before admitting two 2,048-token requests, temperature 0.6, and 256
completion tokens per request (`max_model_len=4096`, GPU utilization 0.8).

| tau | per-run maximum ITL (ms) | median / worst (ms) | SLO runs | historical role |
| ---: | --- | ---: | ---: | --- |
| 256 | 8.057, 6.924, 7.436, 7.575, 7.302 | 7.436 / 8.057 | 5/5 | **latency SLO met at evidence commit** |
| 512 | 9.752, 17.763, 9.202, 13.003, 16.355 | 13.003 / 17.763 | 2/5 | **throughput/TTFT only** |

Tau 512 slightly raises observed completion throughput (~3.18–3.24k token/s)
and lowers long-request maximum TTFT (~65.9–78.5 ms versus tau 256's
~109.3–122.7 ms), but its repeated ITL breaches remain after GC suppression.
Therefore GC control fixes one outlier cause, not tau 512's latency contract.
Other tau values remain performance-unverified by the retained evidence.

The exact arrays, seeds, artifact basenames, hardware, commits, and historical
role are in `release_policy.py`; every new step-diagnostic JSON embeds that reference plus
a recorded-field `matches_historical_recorded_configuration` verdict. The
legacy files record a commit and model path but no self-pinned source or
model-content hash, so
`applies_to_current_run` is always false: neither a matching configuration, a
bounded diagnostic, nor a run using the new engine option is relabeled as
certified by prior manual-GC evidence. The retained inputs are named
`roofline_chunk_tip_tau{256,512}_seed<seed>_gcdisabled.json` under the external
evidence directory `/workspace/.feat_bench/results/`.

## Optional Python GC control

`LLM(..., disable_python_gc=True)` opts one engine into process-wide cyclic-GC
suppression after initialization succeeds. The default is `False`. The engine
records whether GC was enabled. A locked reference-counted lease keeps GC
disabled while any opted-in engine remains; the final idempotent `exit()`/atexit
cleanup restores the pre-first-acquire state even if model-runner cleanup
raises. This is cooperative process-global ownership: code sharing the process
must not call `gc.enable()` while a lease is active. The option currently rejects
`tensor_parallel_size>1`; worker-process GC behavior is not certified. Nano-vllm
GPU/process-group diagnostics still use one engine per fresh process for
independent measurements.

The step harness exposes the same choice as `--disable-python-gc` and retains
GC state before init, after init, and after explicit engine exit. Omit the flag
for the default-enabled control run.

## Full-completion certification

`full_completion_cert.py` is the only harness whose output is eligible for a
current latency verdict. It reproduces the historical request and timed-workload
protocol exactly, including the 16x64-token/4-token warmup, then measures
16x64-token interactive requests, 40 engine steps before admitting two
2,048-token long requests, and 256 completion tokens per request at temperature
0.6. It pins
`max_model_len=4096`, `max_num_seqs=min(512,tau)`, GPU utilization 0.8, eager
off, and TP1 (256 slots at tau 256 and 512 slots at tau 512). The current harness
intentionally replaces the historical process-level `gc.disable()` before engine
construction with `disable_python_gc=True`, whose engine-owned lease begins only
after successful initialization. GC is disabled for the same warmup and measured
workload, but initialization-time GC state and ownership are not historical matches.

Every single-run artifact contains:

- clean commit/tree plus aggregate and per-file source hashes before and after;
- the complete model file/size/SHA-256 manifest, rehashed after the run;
- exact argv, seed assignments, prompt/output hashes, package/CUDA/GPU/driver
  environment, and allowlisted execution-affecting environment variables;
- all 18 raw per-request metric dictionaries, including all 255 ITLs and the
  raw 256-token completion vector for each request;
- GC enabled state before init, disabled state after successful init, and exact
  restoration after explicit exit;
- configured graph buckets and before/after graph-miss counters, plus CUDA
  allocated/reserved peaks and KV-block counts outside the measured loop.

Per-step routes and selected graph keys are intentionally not observed in this
harness: even lightweight StepOutput processing between scheduler timestamps
changes the next ITL. The separate phase diagnostic performs that intrusive
attribution. A single full-completion run always records
`single_run_latency_certified=false`.

`aggregate_certification.py` accepts exactly five read-only artifacts with five
distinct seeds. It rejects any bounded/phase diagnostic, duplicate content,
dirty or mismatched source pins, mismatched model/environment/workload manifests,
incomplete raw metrics, altered summaries, or a broken GC lifecycle. Tau 256 is
current-certified only when every run's maximum interactive ITL is strictly
below 10 ms. Tau 512 is never latency-certified by this policy, even if five
observations happen to pass; it remains throughput/TTFT-only. Neither a single
run nor any phase diagnostic can promote either profile.

Choose `--output` for one immutable aggregate JSON or `--archive-dir` for a new
self-contained archive. An archive copies all five raw inputs byte-for-byte,
adds `aggregate.json` and a hash manifest, and verifies read-only files and
directories before publishing a sibling `<archive>.COMPLETE` marker. An archive
without that marker is incomplete. Existing outputs, archives, and markers are
never overwritten.

## Decode-only jitter attribution (non-certifying)

`decode_jitter_diagnostic.py` is an intentionally intrusive follow-up for rare,
engine-wide decode pulses. It runs 1,000 decode-only steps at batch size 16 and
then 1,000 at batch size 18 in the same TP1 engine. Those are the two observed
routes: the production decode graphs select capture buckets 16 and 32,
respectively. Python cyclic GC is disabled through the engine-owned lease.

Each raw step splits API wall and current-thread CPU time across scheduler,
decode preparation, model dispatch, sampler dispatch, and scheduler
postprocessing, with an explicit API residual and caller gap. Linux
`RUSAGE_THREAD` voluntary/involuntary context-switch deltas distinguish CPU
descheduling. CUDA events separately split preparation, model, sampler, and the
whole runner stream. The final CUDA event is queried only after production's
existing `tokens.tolist()` synchronization; the harness does not add a per-step
CUDA synchronize. These are stream-boundary elapsed spans: host delay between
two event records can appear as idle time inside a CUDA span, so interpret them
together with the corresponding phase wall, thread-CPU, and context-switch
deltas rather than labeling every CUDA-span outlier as kernel time.

This observer changes the timed path and therefore always writes
`certification.eligible=false`. Its JSON can attribute a pulse, but cannot
replace the uninstrumented five-run full-completion policy.

From a clean committed diagnostic branch, retain one immutable run outside the
source and model trees:

```bash
cd /workspace/nano-vllm-chunk-decode-jitter
PIN_COMMIT="$(git rev-parse HEAD)"
PIN_SOURCE="$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /venv/main/bin/python \
  benchmarks/chunked_prefill_tail/decode_jitter_diagnostic.py \
  --print-source-sha256)"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
TORCHINDUCTOR_CACHE_DIR=/tmp/nv_chunk_decode_jitter \
/venv/main/bin/python \
  benchmarks/chunked_prefill_tail/decode_jitter_diagnostic.py \
  --model /workspace/models/Qwen3-0.6B \
  --tau 256 \
  --steps-per-batch 1000 \
  --seed 20260836 \
  --expected-commit "$PIN_COMMIT" \
  --expected-source-sha256 "$PIN_SOURCE" \
  --output /workspace/.feat_bench/chunk-tail/decode_jitter_tau256_seed20260836.json
```

The default 4,096-token context permits up to 2,008 steps per batch. Use a new
output path for every run; immutable evidence is never overwritten.

The retained A100 run at `d9639fc` is archived under
`evidence/2026-08-18-a100-decode-jitter-d9639fc/`. Eight of 2,000 API steps
exceeded 10 ms. Seven kept runner CUDA below the corresponding profile p99 but
had a greater-than-10 ms post-enqueue/`tokens.tolist()` API residual; one had a
10.027 ms model CUDA span. Every stall row had zero context switches and thread
CPU matched wall within 0.000209 ms, consistent with busy driver polling rather
than descheduling. This intrusive result is attribution only: it neither
certifies latency nor establishes a production regression or fix target.

## Reproducible commands

First pin the exact committed source. `git status --porcelain` must print
nothing. Results belong outside this worktree.

```bash
cd /workspace/nano-vllm-chunk-tail
git status --porcelain
export PIN_COMMIT="$(git rev-parse HEAD)"
export PIN_SOURCE="$(PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python benchmarks/chunked_prefill_tail/full_completion_cert.py --print-source-sha256)"
mkdir -p /workspace/.feat_bench/chunk-tail
```

Retain five fresh tau-256 full-completion processes. These seeds intentionally
do not reuse the legacy evidence seeds:

```bash
for SEED in 20260826 20260827 20260828 20260829 20260830; do
  PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/nv_chunk_full_tau256 \
    /venv/main/bin/python \
      benchmarks/chunked_prefill_tail/full_completion_cert.py \
      --model /workspace/models/Qwen3-0.6B --tau 256 --seed "$SEED" \
      --expected-commit "$PIN_COMMIT" \
      --expected-source-sha256 "$PIN_SOURCE" \
      --output "/workspace/.feat_bench/chunk-tail/full_tau256_seed${SEED}.json"
done

PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /venv/main/bin/python \
    benchmarks/chunked_prefill_tail/aggregate_certification.py \
    --tau 256 --expected-commit "$PIN_COMMIT" \
    --expected-source-sha256 "$PIN_SOURCE" \
    --input \
      /workspace/.feat_bench/chunk-tail/full_tau256_seed20260826.json \
      /workspace/.feat_bench/chunk-tail/full_tau256_seed20260827.json \
      /workspace/.feat_bench/chunk-tail/full_tau256_seed20260828.json \
      /workspace/.feat_bench/chunk-tail/full_tau256_seed20260829.json \
      /workspace/.feat_bench/chunk-tail/full_tau256_seed20260830.json \
    --archive-dir \
      /workspace/.feat_bench/chunk-tail/cert_tau256_seeds20260826_20260830
```

Use five separate fresh seeds to retain the tau-512 throughput/TTFT comparison.
The validator will still set `latency_certified=false`:

```bash
for SEED in 20260831 20260832 20260833 20260834 20260835; do
  PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/nv_chunk_full_tau512 \
    /venv/main/bin/python \
      benchmarks/chunked_prefill_tail/full_completion_cert.py \
      --model /workspace/models/Qwen3-0.6B --tau 512 --seed "$SEED" \
      --expected-commit "$PIN_COMMIT" \
      --expected-source-sha256 "$PIN_SOURCE" \
      --output "/workspace/.feat_bench/chunk-tail/full_tau512_seed${SEED}.json"
done

PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /venv/main/bin/python \
    benchmarks/chunked_prefill_tail/aggregate_certification.py \
    --tau 512 --expected-commit "$PIN_COMMIT" \
    --expected-source-sha256 "$PIN_SOURCE" \
    --input \
      /workspace/.feat_bench/chunk-tail/full_tau512_seed20260831.json \
      /workspace/.feat_bench/chunk-tail/full_tau512_seed20260832.json \
      /workspace/.feat_bench/chunk-tail/full_tau512_seed20260833.json \
      /workspace/.feat_bench/chunk-tail/full_tau512_seed20260834.json \
      /workspace/.feat_bench/chunk-tail/full_tau512_seed20260835.json \
    --archive-dir \
      /workspace/.feat_bench/chunk-tail/cert_tau512_seeds20260831_20260835
```

Run aggregation before changing the checkout: all five run pins must match the
aggregator's current clean HEAD and source hash.

Run the high-segment mixed-tail comparison. Tau 256 is the historical latency
reference; tau 512 is retained only to characterize its throughput/TTFT tradeoff.
These commands build 63 live decode rows, then admit one 2,048-token prompt,
reproducing 64-segment mixed steps while holding every other workload parameter
constant:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/nv_chunk_tail_tau256 \
  /venv/main/bin/python benchmarks/chunked_prefill_tail/step_diagnostics.py \
  --model /workspace/models/Qwen3-0.6B --tau 256 --max-num-seqs 64 \
  --max-model-len 4096 --interactive-count 63 --long-count 1 \
  --long-prompt-len 2048 --pre-long-steps 80 --measured-steps 96 \
  --max-tokens 256 --cold-steps 3 --seed 20260822 --disable-python-gc \
  --expected-commit "$PIN_COMMIT" --expected-source-sha256 "$PIN_SOURCE" \
  --output /workspace/.feat_bench/chunk-tail/tail_tau256_seed20260822.json

PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/nv_chunk_tail_tau512 \
  /venv/main/bin/python benchmarks/chunked_prefill_tail/step_diagnostics.py \
  --model /workspace/models/Qwen3-0.6B --tau 512 --max-num-seqs 64 \
  --max-model-len 4096 --interactive-count 63 --long-count 1 \
  --long-prompt-len 2048 --pre-long-steps 80 --measured-steps 96 \
  --max-tokens 256 --cold-steps 3 --seed 20260822 --disable-python-gc \
  --expected-commit "$PIN_COMMIT" --expected-source-sha256 "$PIN_SOURCE" \
  --output /workspace/.feat_bench/chunk-tail/tail_tau512_seed20260822.json
```

Run the explicit maximum-length boundary regression:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/nv_chunk_4x511 \
  /venv/main/bin/python tests/run_varlen_511_contract.py \
  --model /workspace/models/Qwen3-0.6B --seed 20260818 \
  --expected-commit "$PIN_COMMIT" --expected-source-sha256 "$PIN_SOURCE" \
  --output /workspace/.feat_bench/chunk-tail/contract_4x511_maxlen512.json
```

Run the retained tau `{64,128}` by max-model-length `{512,1024,4096}` matrix.
The orchestrator launches one fresh child process per cell and revalidates the
source pin on both sides of every child:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/nv_chunk_contract_matrix \
  /venv/main/bin/python benchmarks/chunked_prefill_tail/contract_matrix.py \
  --model /workspace/models/Qwen3-0.6B \
  --expected-commit "$PIN_COMMIT" --expected-source-sha256 "$PIN_SOURCE" \
  --output /workspace/.feat_bench/chunk-tail/tau64_128_maxlen_matrix.json
```

Each output path must be new. Choose a new seed/name for a repeat; do not chmod
and overwrite retained evidence.
