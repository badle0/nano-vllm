# Speculative V2 inert-lifecycle A100 evidence

This archive certifies the inert V2 dual-model ownership and recovery rung at
implementation commit
`d87f168b804778fbb5888a662dc8a0defccfd660` (tree
`4e98646e71c4550f58df291cf8efbbab71d9e2cb`). The historical control is the
detached V0 commit `480a3b26c5a4e465aac06d1dabd34e1230686feb`.

The certificate covers configuration, target/draft identity, transactional
construction and teardown, independent physical KV ownership, eager and CUDA
graph lifecycle parity, exact capacity preflight, and all eight registered
constructor-failure recovery points. V2 deliberately performs zero draft
forwards during generation. This archive therefore does **not** certify draft
proposal execution, target verification, acceptance/rejection, multi-token
commit, streaming/metrics integration, heterogeneous target/draft geometry,
tensor parallelism, FlashInfer, latency, throughput, acceptance rate, or
speedup. The modeled speculative workspace remains `gpu_certified=false`.

## Registered environment

- GPU: NVIDIA A100-SXM4-40GB, compute capability 8.0, 40,960 MiB, 108 SMs
- GPU UUID: `773e0633-edb0-6c38-1d2b-d232f9109126`
- Driver: 570.133.20
- PyTorch: 2.10.0+cu128; CUDA build 12.8
- cuDNN: 91002; NCCL: 2.27.5; Transformers: 5.14.1
- Target and draft fixture: the same local Qwen3-0.6B safetensors artifact
- Protocol: TP=1, exact top-p backend, K=2, seed 20260828, model/batch limits
  512/512/4, `gpu_memory_utilization=0.5`, 256-token KV blocks

GPU ownership is established by identifying the sole new `nvidia-smi` process
and matching its 390 MiB NVML increase to a 389 MiB allocator challenge within
the registered 1 MiB rounding tolerance. Container PID aliases are diagnostic
only. Isolation is checked at the before/after endpoints; the archive makes no
claim that a transient mid-run consumer could not appear.

## Results

| Cell | Result | Boundary or recovery result | Raw SHA-256 |
|---|---:|---|---|
| V0 eager | PASS | canonical greedy output/trace | `8e6e7e4edd500ca98010c7bae1b6265d270d9c952fabcb1cd85cdcf80efebcda` |
| V0 graph | PASS | canonical greedy output/trace | `24f5cfcaf667a61913a4018e3d7a9de48b9c38d5da03f0805f3a520eeb6c8a61` |
| V2 eager | PASS | N=305 succeeds; N+1=306 typed-fails with the same ceiling | `f8d617ef25eb6389e4c9912efe1b893ea68abc883efa49afe54e51bfda25dba9` |
| V2 graph | PASS | N=303 succeeds; N+1=304 typed-fails with the same ceiling | `462f14488147a7dae513043190401eb3ae4a36f2881169d06a2667d84d14d9d7` |

Both V2 cells are bit-identical to their matching canonical V0 output and
scheduler trace, bit-identical between speculation off and inert ownership on,
RNG-identical at construction and post-generation endpoints, and observe zero
draft forwards during generation. Automatic and explicit memory states
reconcile independently. Graph ownership is positive only in graph mode; eager
graph fields are zero.

All eight fresh-process recovery cells pass:

| Injected phase | Mode | Real phase calls | Raw SHA-256 |
|---|---:|---:|---|
| `draft_construct` | eager | 1 | `8d9184185a5e29be192dd64063c40e1f6d3ba86ba3dbba36d230a9c061564516` |
| `draft_load` | eager | 1 | `44f33219000ce8c248f53b7bb431462f46f926baaa7bf218c85b4ea485cff59f` |
| `draft_warmup` | eager | 1 | `90efcf20415a720b9956990e85c89911451b2c853cc69fed068dae5d28a1daff` |
| `graph_profile` | graph | 1 | `01c61ffff5adbf04370c5ebaec369d9b1c113695766101c44e73b78d0ba46c3d` |
| `joint_allocate` | eager | 1 | `b11bf9a1be24b98bbd4dd03ab80bbbf2ffa2a9837a7afa7514893c76ae6b439e` |
| `draft_graph` | graph | 2 | `566d2f8a5e15484a0215e5e8c6d89a828714d55d477fb04296540bf9e9486ee9` |
| `draft_pretouch` | graph | 1 | `29ac449e19a23642165de0664cda80fe697b91e73d0e04ea46419a5fe0059165` |
| `memory_finalize` | eager | 1 | `21e25666cbaf3c3fbffae44b152a803db6f1237bc26b8851c42c7ba7c185cce2` |

Each cell injects exactly once after the selected real phase, restores Torch
defaults/context, destroys the failed process group, reconstructs a healthy
four-block engine, reproduces the registered output/trace, and exits within the
32 MiB allocated / 64 MiB reserved compiled-runtime ceilings. The observed
maximum after either failure or recovery exit was 17,039,360 allocated bytes and
41,943,040 reserved bytes.

## Rejected-attempt ledger

Rejected runs are not present under `raw/` and are not included in the manifest:

1. At 2026-08-28 13:05 UTC, the first retained graph attempt selected N=303,
   then a later same-process reconstruction saw only 277 blocks of live driver
   capacity and failed closed. It wrote no evidence JSON. An idle fresh-process
   replay produced byte-identical automatic/explicit baselines at N=303 and
   typed-rejected N+1=304.
2. At 2026-08-28 13:14 UTC, the first `joint_allocate` recovery attempt rejected
   its final isolation gate after observing a foreign `nvidia-smi` process using
   1,438 MiB. It wrote no evidence JSON. Its idle fresh-process replay passed.

The second observation directly supports transient external driver pressure as
the likely cause of the first ~1.45 GiB capacity shift, but this attribution is
an inference because the lifecycle gate samples isolation only at endpoints.
Production behavior was correct in both cases: it re-read live capacity and
failed closed. No rejected output was renamed or retained.

## Offline validation

From the repository root, without a GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python \
  benchmarks/speculative_v2/validate_retained_evidence.py \
  benchmarks/speculative_v2/evidence/2026-08-28-a100-v2-d87f168/manifest.json
```

The validator checks the exact file set and hashes, rejects duplicate JSON keys
and non-finite numbers, re-derives memory-planner and capacity arithmetic without
importing production code, and cross-checks source, model, tokenizer, GPU,
historical V0, lifecycle, and recovery invariants.

Every raw artifact also retains its exact producer argv, selected environment
variables, start/finish timestamps, source/model snapshots, and A100 ownership
observations; these fields are semantic validator inputs, not informal metadata.
