# Speculative V3 draft-discard A100 evidence

This archive is the narrow retained certificate for nano-vLLM's V3
`draft-discard` milestone. The runtime implementation is commit
`7fec9993d5e4e0e06fec22e3973dfc203fdbd2d8` (tree
`5820fb685d76b549fe21117baa31b3b32ebae14b`; `nanovllm` subtree
`52398af379f767708a0b804646f4b490fa8323ad`). The evidence harness is commit
`e8e0452f99727958077b51f340a5375a090e6884`; it has the identical `nanovllm`
subtree, so the harness-only commit did not alter the certified runtime.

## Claim boundary

On one A100, using the same Qwen3-0.6B checkpoint as target and draft, this
archive certifies:

- transactional draft-cache catch-up, proposal execution, and discard;
- exhaustive exercise of every registered finite `draft-discard-v1` route in
  the retained configuration;
- compiler, CPU/CUDA RNG, and attention-context neutrality inside explicitly
  guarded draft intervals;
- draft-KV fill neutrality for the registered 255/256/257 block-boundary and
  shared-prefix comparisons; and
- speculation-off/on parity of public sequence IDs, authoritative target
  events and token IDs, and all registered CPU/CUDA RNG checkpoints.

It does **not** certify target verification, acceptance/rejection, residual
correction, the all-accepted target bonus, multi-token commit, speculative
streaming or metrics, latency, throughput, acceptance rate, speedup, tensor
parallelism greater than one, FlashInfer, heterogeneous target/draft models, or
every route a future implementation might add. Aggregate compiler state outside
the guarded draft intervals is not claimed unchanged.

The graph cache-neutrality and graph output-control producer runs emitted a
TorchDynamo recompile-limit warning outside the dedicated route-proof windows.
Those artifacts support their stated numerical and authoritative-output oracles
only; they are not an independent compile-completeness certificate. The route
artifacts and their retained logs are the compile-route proof.

## Registered environment

- GPU: NVIDIA A100-SXM4-40GB, compute capability 8.0; `nvidia-smi`
  capacity 40,960 MiB; PyTorch registered total 42,406,903,808 bytes
  (40,442.375 MiB)
- GPU UUID: `773e0633-edb0-6c38-1d2b-d232f9109126`
- Driver: 570.133.20
- Python: 3.12.13
- PyTorch: 2.10.0+cu128; CUDA build 12.8; cuDNN 91002
- Target and draft fixture: the same local Qwen3-0.6B safetensors artifact
- Common policy: TP=1 and the exact top-p backend
- Seed: 20260828

The artifacts bind the complete model-weight and tokenizer metadata hashes, the
selected environment, source and import snapshots, invocation, and the hardware
observations before and after each producer. Offline validation trusts those
registered hashes; it does not require the local model or a GPU.

## Protocol and results

### Finite route and guarded-window proof

The route producer used configured K=2, batch cap 4, model/token limits 512/512,
`gpu_memory_utilization=0.5`, isolated initially empty Inductor and Triton cache
roots, and two repetitions. It exercised every registered cold paged-catch-up
key and every registered zero-catch-up key.

| Mode | Registered/visited routes | Guarded intervals | CUDA-graph ledger init/runtime | Pretouch peak |
|---|---:|---:|---:|---:|
| eager | 4/4 | 32 | 0/0 contexts and objects | 41,995,264 bytes |
| graph | 12/12 | 32 | 18/18 contexts and objects | 41,970,688 bytes |

All 64 run-bound log intervals are structurally complete and contain no
forbidden recompile, graph-break, guard-miss, or capture event. Each interval's
compiler snapshot, CPU/CUDA RNG state, and attention context is unchanged. The
offline route validator independently recomputes these relationships from the
two raw JSON artifacts and their exact stderr logs.

### Draft-KV fill neutrality

Separate fresh processes initialized every reserved physical draft slot with
zeros or NaNs. Each producer first executed the same declared zero-filled
255/256/257 warmup, whose three sampler records were discarded, before measuring
the fill comparison. This removes first-use shape/kernel state from the
zero-versus-NaN variable under test.

For both eager and graph modes, all nine measured records—three boundary steps,
three cold shared-prefix steps, and three prefix-hit steps—have bit-identical
BF16 logits and FP32 probability rows between zero and NaN fills. Every tensor
has shape `[1, 151936]`; maximum absolute difference is zero, all rows are finite,
and the host-side cache/block-table oracles match. The four `.tensors.pt`
sidecars retain the exact compared tensors and are bound by both whole-file and
per-tensor hashes.

### Speculation-off/on authoritative target control

Fresh eager and graph pairs used two prompts, K=2, `temperature=0.8`, `top_k=8`,
`top_p=0.9`, `max_tokens=6`, and `ignore_eos=true`. In each mode, speculation
off and draft-discard on have exact public sequence IDs, authoritative target
events, target token IDs, and CPU/CUDA RNG state at post-init, post-prefill,
first-target-decode, and repeated-target-decode checkpoints.

The off side enters none of the five draft-construction phases, owns no live
draft resources, and executes no draft interval. The on side exercises one cold
catch-up route and one warm zero-catch-up route. This is authoritative target
output parity for a compute-then-discard control; it is not target-verification
or acceptance evidence.

## Rejected diagnostic attempts

Rejected diagnostics are not present under `raw/`, are not listed in the
manifest, and do not contribute to this certificate:

1. A pre-certificate route harness revision failed closed because its runtime
   source registry named the wrong module for `LLM`. The registry and its
   real-object regression test were corrected before `e8e0452` was frozen.
2. The first cache-neutrality diagnostic exposed ordinary BF16 first-use
   variation in the first producer, while a same-fill repeat was bit-identical
   to the other fill. The final harness added the symmetric declared warmup
   above and reran all four cells from the clean producer commit.
3. The first clean graph route attempt at `e8e0452` failed with `ENOSPC` while
   writing its isolated compiler cache and produced no result JSON. Exact
   generated caches were cleared, and a fresh isolated retry passed. Its failed
   diagnostic log is excluded from this archive.

## Offline validation

From a full-history repository checkout, without CUDA, nano-VLLM imports, or the
model fixture:

```bash
PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python \
  benchmarks/speculative_v3/validate_retained_evidence.py \
  benchmarks/speculative_v3/evidence/2026-08-28-a100-v3-e8e0452
```

The standard-library validator rejects unregistered files, symlinks, hardlinks,
duplicate JSON keys, non-finite JSON numbers, path escapes, artifact/hash drift,
producer or implementation drift, historical runner drift, inconsistent model
or environment identity, malformed route logs, and any failed route, cache, or
output-control invariant. The trusted registry is compiled into the validator;
`manifest.json` is an index, not the root of trust.
