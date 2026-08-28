# PR8 V1 sampling-law certification

Status: **PASS locally** on 2026-08-28. The branch has not been pushed, so its
new GitHub Actions matrix has not run remotely. This certificate covers only the
V1 exact-sampling seam and modified-rejection reference law; it does not claim
that end-to-end speculative decoding exists.

## 1. Provenance and exact scope

- Design base: `origin/fork-main@663753b99131945c297c1fbe02341108f422dce7`.
- Frozen V0 packet: `480a3b2` (`docs: plan speculative decoding v2`).
- V1 implementation under test:
  `59229861e42b9d02636ed521609f96a95b04895f`.
- Fallback-certificate preregistration committed before the reported rerun:
  `83a45044cc4153c48cd3ec8121cbfd9c62b0e5f9`.
- Sampler SHA256:
  `e7a558e933668108c3d114189ae33694cb368c8c658d372930e140e3f2b2a1fb`.
- V1 test SHA256:
  `4ce9818320d2ee432614fadcff4949e4e591c14bc91183a2fc8b08ad99adfb0e`.

The implementation commit changes only the sampler layer, its standalone oracle
test, the targeted CPU workflow, and PR8 status/contract documentation. It does
not change `Config`, `SamplingParams`, `Sequence`, `LLMEngine`, `Scheduler`,
`ModelRunner`, KV-cache code, streaming, metrics, or request admission.

The pre-existing compiled `Sampler.forward` decorator and body are byte-identical
to V0. A behavioral regression also compares its emitted tokens and post-call CPU
RNG state with the frozen V0 expression.

## 2. Implemented contract

V1 adds:

- non-mutating canonical FP32 probability construction using the current
  nano-vLLM temperature, top-k, exact top-p, mixed-active-row, and tie rules;
- greedy rows represented as exact target/draft point masses;
- sampling that returns the precise retained probability rows used by the draw;
- typed validation for malformed target/draft rows, selected `q(d)`, injected
  uniforms, correction draws, shapes, devices, dtypes, and empty dimensions;
- vectorized longest-prefix acceptance with controlled `[B,K]` uniforms;
- first-rejection correction with controlled independent `[B,V]` exponential
  draws and `-1` for full-accept rows;
- an explicit per-row `target_fallback` mask for the later
  `spec_residual_numerical_fallbacks` metric;
- targeted CPU CI on Python 3.10 and 3.12 with CPU PyTorch 2.4.1.

The correctness-first V1 correction law always normalizes the original retained
FP32 target/draft rows in FP64. The reference race also scores retained weights
and valid positive FP32 noise in FP64. This is intentional: V1 is the oracle rung,
not the performance kernel. A faster dtype may replace it only after proving
categorical-law equivalence for every routed input class.

Acceptance uses the strict half-open predicate
`u < min(1, p(d) / q(d))`. Thus `p(d) == 0` rejects even when the representable
uniform draw is exactly zero. A rejected robust-zero residual samples normalized
target `p` with the independent correction draw and marks exactly one fallback;
invalid/non-finite rows raise rather than selecting an arbitrary or uniform token.

## 3. Environment

```text
Python:       3.12.13
PyTorch:      2.10.0+cu128
pytest:       9.1.1
CUDA runtime: 12.8
GPU:          NVIDIA A100-SXM4-40GB
```

The standalone CPU gate hides CUDA and imports `sampler.py` directly, so it needs
only PyTorch and pytest. The optional CUDA BF16 seam case is separately exercised
on the A100; it is not a throughput benchmark.

## 4. Commands and results

Preregistered CPU-only V1 gate:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /venv/main/bin/pytest -q -p no:cacheprovider \
  tests/test_speculative_sampler.py
```

```text
58 passed, 1 skipped, 14 warnings in 4.50s
```

The skip is the explicitly CUDA-only BF16 seam case. With the A100 visible, all
59 V1 cases pass. The warnings are inherited `torch.jit.script_method`
deprecation warnings from the installed PyTorch stack.

Full fork regression on the exact post-preregistration tip:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /venv/main/bin/pytest -q -p no:cacheprovider
```

```text
368 passed, 1 skipped, 14 warnings in 59.03s
```

Additional gates passed:

- workflow YAML parsing;
- `python -m compileall -q nanovllm tests`;
- `git diff --check` and cached-diff checks;
- independent scalar-oracle/code/branch reviews;
- source comparison confirming ordinary `Sampler.forward` is unchanged.

## 5. Numerical fallback certificate

The preregistered population is four fixed one-token Monte Carlo workloads:
temperature, top-k, top-p, and combined top-k/top-p, each with 100,000 cycles.
The deliberately injected robust-zero fallback test is excluded.

```text
non-injected cycles:   400,000
target fallbacks:      0
observed rate:         0
one-sided 95% exact upper bound:
  1 - 0.05 ** (1 / 400000) = 7.489302638941098e-6
fallback TV bound:     <= 7.489302638941098e-6 per cycle (95% confidence)
```

Every workload also passes its preregistered Bernstein count bounds for emitted
target-token frequencies and acceptance counts, including zero target-support
violations. The direct robust-zero hook samples target support deterministically
under the injected draw and marks one fallback, proving that the guard is
observable without relabeling it as an analytic rejection event.

## 6. Adversarial issues found and resolved during V1

1. A near-null FP32 residual could appear all-zero while independently
   normalizing the retained FP32 rows in FP64 produced positive residual mass.
2. A positive FP32 residual could still drop FP64-positive support. A conditioned
   eight-token fixture had residual-law TV about 0.0967 despite fast mass
   `8.618808e-7`.
3. Clamping an otherwise valid tiny positive injected exponential draw changed
   its selected token.
4. Dividing FP32 probabilities by valid subnormal noise could overflow multiple
   scores to `inf` and turn the result into an argmax tie.
5. Empty batch/proposal/vocabulary dimensions initially passed vacuous
   validation.
6. The V0 theory used `<=`, which is measure-equivalent only for an ideal
   continuous uniform and incorrectly accepts a zero-probability token at the
   representable draw `u == 0`.

The retained-row FP64 reference law, unclamped FP64 race scoring, strict typed
non-empty validation, and documented half-open comparison resolve these cases.
Each has a deterministic regression fixture.

## 7. Explicit exclusions and next rung

This certificate does **not** cover:

- draft-model construction, ownership, teardown, or tokenizer compatibility;
- draft KV-cache sizing, catch-up, or rollback;
- effective-K scheduling, reservations, preemption, or burst commit;
- target verification, bonus tokens, EOS/max-token truncation, streaming, or
  speculative metrics;
- FlashInfer speculative semantics, TP>1, graph/warmup shapes, A100 engine
  throughput, memory rooflines, or end-to-end distribution parity.

There is no user-facing flag that enables speculative decoding at V1. The next
branch is V2, `feat/spec-v2-dual-runner`, and must start from this green V1 tip
only after the V1 branch is pushed/reviewed. V2 adds inert typed configuration and
transactional dual-runner/KV lifecycle while preserving speculation-off behavior.
