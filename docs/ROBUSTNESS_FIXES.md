# Post-RC Robustness Fixes

This document records correctness and edge-case work prepared after
`v0.3.0-rc.1`. It distinguishes repaired defects from intentional resource
contracts so later releases do not accidentally reverse either.

## Resolution Summary

| Area | Resolution | Preserved contract |
|---|---|---|
| Exact top-p with FP32 logits | Nucleus support is computed in a guaranteed-private FP32 workspace, so retained logits are not pre-scaled before final sampling. | Top-p still masks logits in place, and temperature is applied exactly once. |
| Oversized high-level batches | The 512-default `max_num_seqs` admission limit and caller-side windowing procedure are now explicit. | Admission remains atomic, bounded, and fail-before-tokenization. |
| Persistent U+FFFD output | Short fragments remain hidden, but a replacement suffix that reaches `W + O` is emitted as correctable text so an exact split can advance the frontier. | Incremental decode work remains bounded; `flush()` remains exact. |
| Empty prompts | Public admission rejects a zero-token prompt with `ValueError`, and `Sequence` enforces the same invariant defensively. | Failed batch admission remains transactional and releases session ownership. |

## Exact Top-p Temperature Semantics

`filter_top_p()` needs temperature-scaled FP32 values to determine nucleus
support. Those values are temporary. The logits passed to `Sampler.forward()`
must retain their original finite values because `forward()` performs the one
sampling-temperature division.

`Tensor.float()` can alias an already-FP32 tensor. The implementation therefore
uses:

```python
chunk_logits.to(dtype=torch.float32, copy=True)
```

This costs the FP32 path one necessary private workspace. BF16 already required
an FP32 conversion, so it does not add a second conversion allocation there.
Tests cover all-active and indexed-row filtering and a fixed-seed observable
sampling comparison at non-unit temperature.

## Bounded Admission

`max_num_seqs` limits all requests owned by the scheduler, not only requests
selected in one model step. A high-level `generate()` or `stream()` call above
the available capacity raises `SchedulerCapacityError` before tokenization and
without partial admission.

Caller-side windows are intentional. They retain hard backpressure and make the
accepted-work bound visible. A transparent rolling implementation would need to
tokenize and stage every overflow request outside the scheduler to preserve
whole-call validation and stable `StreamSession.seq_ids`; that would restore
unbounded host-side accepted work. Such a mode should be a separately reviewed,
opt-in API rather than a silent bug fix.

When per-prompt `SamplingParams` are supplied, slice them at the same indices as
the prompts. Fully drain or close one stream window before opening another.
Every window has its own submission and final-delivery timestamps.

## Detokenizer Progress and Limits

Let `W` be `window_size` and `O` be `boundary_overlap`. Ordinary trailing
U+FFFD output is withheld while the retained tail is shorter than:

```text
F = W + O
```

At `F`, the replacement output is emitted through `TextUpdate` and the normal
exact-split search runs. For additive invalid-byte output this advances the
frontier and prevents the previous persistent-U+FFFD failure. A later token can
still replace the retained suffix.

The independent hard guard remains:

```text
H = W + 2O
```

A tokenizer whose decoded tail cannot be decomposed at any candidate boundary
still raises at `H`. This is fundamental to the generic interface: guaranteeing
arbitrary whole-history rewrites would require unbounded per-token decoding or
allow incorrect intermediate text. Regardless of intermediate behavior,
`flush()` performs one exact decode of all retained token IDs.

## Empty-prompt Contract

An explicit empty token list, or a string that the configured tokenizer maps to
zero tokens, raises:

```text
ValueError: prompt must contain at least one token
```

The check occurs after scheduler-capacity preflight and string tokenization but
before sequence-ID allocation. Batched `generate()` and `stream()` admission
continues to roll back any earlier IDs from the same failed batch.

## Verification

Run the focused CPU/CUDA-optional regressions first:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /venv/main/bin/pytest -q \
  -p no:cacheprovider \
  tests/test_sampler.py tests/test_detokenizer.py tests/test_streaming.py
```

Then run the complete suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /venv/main/bin/pytest -q \
  -p no:cacheprovider
git diff --check
```

Retained benchmark artifacts remain valid for the exact commits and source
hashes they record. Because the sampler and detokenizer source files change,
those artifacts are ancestry evidence rather than byte-identical certification
of a release containing these fixes. Re-run the relevant sampling and streaming
certification protocols before making performance claims for a new release
candidate.

## Post-Review Hardening

Prepared after the external review of the first certification pass. It addresses
N1, N3, N4, N8, N9 (admission validation), N2 (session finalizer), N6
(engine lifecycle and allocation errors), and N7 (configuration validation).

### Request Admission Contract

`add_request`, `generate`, and `stream` validate every request after
tokenization and before sequence-ID allocation:

- prompts are either strings or explicit `list[int]`; every normalized token ID,
  including tokenizer output, must be an integer in `[0, vocab_size)`;
- `len(prompt) <= max_model_len` and
  `len(prompt) + max_tokens - 1 <= max_model_len`;
- `ceil((len(prompt) + max_tokens - 1) / block_size) <= num_kvcache_blocks`, so
  one cold request can never exceed the whole KV pool.

The `- 1` is intentional rather than permissive guesswork. Prefill processes
the prompt, and only the first `max_tokens - 1` sampled IDs are fed back through
decode. The final sampled ID is returned but never processed or placed in KV
cache. Charging `len(prompt) + max_tokens` would reject the safe boundaries
`P=max_model_len,N=1` and `P=block_size,N=1` without preventing any crash.

`SamplingParams` now validates `max_tokens` (non-boolean integer, >= 1) and
`ignore_eos` (boolean) both at construction and from a fresh value snapshot at
admission, so later caller mutation cannot bypass the checks. Independently
known vocabulary, model-length, and KV-pool limits remain active when a test
double omits another limit. The scheduler's length stop is `>=` rather than
`==` as defense in depth, and empty/no-progress scheduling paths now have
distinct typed `RuntimeError`s.

### Abandoned Stream Sessions

A `StreamSession` collected without `close()` now runs a `weakref.finalize`
callback that cancels only its own request IDs, releases the engine lease, and
then emits a best-effort `RuntimeWarning`. Cleanup therefore still completes
when runtime warnings are configured as exceptions. Finalizer registration is
inside the admission transaction, registration failure rolls back admitted IDs,
and `atexit` execution is disabled because `LLMEngine` owns process teardown.
`close()` and the context manager detach the finalizer, so correctly closed
sessions are idempotent and warning-free. Explicit close remains the contract;
finalization cannot be timely for an uncollected cycle when Python GC is off.

### Engine Lifecycle

`ModelRunner` construction now owns a transaction from process-group creation
through model loading, KV allocation, graph capture, and TP transport setup.
Failure restores the incoming Torch default device/dtype, resets global model
context, releases partial graph/model/KV objects, destroys its process group,
collects Python objects, and empties the CUDA allocator cache. Normal `exit()`
performs the same release idempotently, including module KV views and CUDA graph
static buffers, so a later TP1 engine can start in the same process.

KV sizing still resets peak statistics immediately before its warmup/profile;
an additional earlier reset would be overwritten and cannot release memory.
If sizing yields no usable block, construction raises a detailed `RuntimeError`
with total/free/budget/current/peak/transient/usable/block byte counts. Worker
joins are bounded and partially started workers are terminated on abort. TP
process-group initialization has a finite timeout; real TP2 failure recovery
remains outside the single-A100 verification scope.

`Config` rejects invalid `kvcache_block_size`,
`tensor_parallel_size`, `max_model_len`, `gpu_memory_utilization`,
`enforce_eager`, `num_kvcache_blocks`, and model paths with typed errors that
survive `python -O`; existing `os.PathLike` model paths remain supported.
`num_kvcache_blocks=-1` selects automatic sizing. A positive value is honored
as an exact override when it fits the profiled memory budget and otherwise
raises a typed sizing error.

### Hardening Verification

The hermetic CPU suite includes boundary oracles for `P+N-1`, KV block edges,
tokenizer-produced OOV IDs, mutated sampling parameters, independently missing
limits, transactional batch rollback, warning-as-error finalization, finalizer
registration/cancellation failures, scheduler no-progress errors, constructor
default restoration, and idempotent resource release.

Run it with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /venv/main/bin/pytest -q \
  -p no:cacheprovider
```

Run the standalone lifecycle matrix on a GPU with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /venv/main/bin/python tests/run_engine_lifecycle.py \
  --model /path/to/Qwen3-0.6B
```

On the A100-SXM4-40GB review host, the failure-first test released NCCL and
restored Torch defaults before a successful retry. Three normal restarts passed
for graph -> eager -> graph, with stable post-exit allocator footprints. This
certifies sequential TP1 lifecycle behavior; it does not certify TP2 recovery.
