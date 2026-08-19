# Branch Policy

This repository is a maintained fork of
[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

The fork preserves the original project's authorship, MIT license, and history.
Fork-specific development remains entirely within
[badle0/nano-vllm](https://github.com/badle0/nano-vllm).

## Canonical Branches

| Branch | Purpose |
| --- | --- |
| `main` | Comparison point kept aligned with the original upstream repository |
| `fork-main` | Canonical maintained integration branch for this fork |
| `release/*` | Short-lived branches used to prepare release candidates |
| `fix/*` | Focused repair, certification, evidence, or diagnostic branches |
| `feat/*` | Original feature-contribution branches retained for history |

New fork releases are prepared through pull requests targeting `fork-main`.
Release candidates are identified by annotated tags such as
`v0.3.0-rc.1`.

`main` is not the fork's production integration branch and should not receive
fork-specific fixes.

## Integration Flow

The conceptual integration flow is:

```text
focused fix and certification branches
                  |
                  v
              fix/dev
                  |
                  v
    fix/chunked-prefill-upstream
                  |
                  v
              fork-main
                  |
                  v
             release tags
```

`fix/chunked-prefill` has the same production source and tests as
`fix/chunked-prefill-upstream`, but additionally retains the larger PR6 audit
and evidence bundle.

`fix/chunked-prefill-tail` is a separate diagnostic line. It is not a newer
production superset and must not be used as a release base.

## Sampling Branches

| Branch | Role | Status |
| --- | --- | --- |
| `fix/greedy-sampling` | Homogeneous greedy fast path and validation repairs | Functional repair complete |
| `fix/topk-sampling` | Active-row top-k filtering built on repaired greedy sampling | Functional repair complete |
| `fix/topp-sampling` | Exact top-p semantics and documentation of rejected fast paths | Correctness repaired; exact all-active performance remains expensive |
| `fix/topp-performance` | Optional FlashInfer top-p backend and release evidence | Fast opt-in backend complete under its documented semantic contract |
| `fix/sampling-evidence` | Fresh-process repaired greedy and top-k release evidence | Evidence branch |
| `fix/dev` | Combined sampling, metrics, and streaming integration | Canonical non-chunk integration branch |

The default `top_p_backend="exact"` preserves the audited Transformers boundary,
tie, and nano-vLLM fixed-seed RNG behavior.

The optional `top_p_backend="flashinfer"` is faster but uses a different
boundary-tie and RNG contract. It is not a bitwise or fixed-seed replacement for
the exact backend.

## Request-Metrics Branches

| Branch | Role | Status |
| --- | --- | --- |
| `fix/request-metrics` | Request timing semantics, legacy step compatibility, and provenance | Functional repair complete |
| `fix/request-metrics-cert` | Eight-pair A/B certification and retained evidence | Certification complete |

The implementation branch remains focused on the code repair. The certification
branch extends it with the accepted performance evidence. Both are integrated
into `fix/dev` and `fork-main`.

## Token-Streaming Branches

| Branch | Role | Status |
| --- | --- | --- |
| `fix/token-streaming` | Request-scoped synchronous streaming, cancellation, metrics, and bounded detokenization | Functional repair complete for the documented synchronous API |
| `fix/token-streaming-cert` | Matched-work streaming certification and retained evidence | Certification complete |

Streaming intentionally supports one synchronous engine owner. It is not a
generally concurrent or asynchronous dispatcher.

Both branches are integrated into `fix/dev` and `fork-main`.

## Chunked-Prefill Branches

| Branch | Role | Status |
| --- | --- | --- |
| `fix/chunk-scheduler-cert` | Scheduler backlog roofline and retained evidence | Certification complete |
| `fix/tp-cert-tests` | Spawn-boundary tensor-parallel transport test | Transport test complete; real TP2 execution unverified |
| `fix/chunk-valid-unpadded-oracle` | Ragged graph comparison with valid unpadded eager execution | Correctness oracle complete |
| `fix/chunked-prefill-upstream` | Repaired chunked-prefill runtime plus current core integrations | Canonical full-feature runtime source |
| `fix/chunked-prefill` | Same runtime and tests plus the full PR6 evidence archive | Audit-rich integration branch |
| `fix/chunked-prefill-tail` | GC, full-completion, and decode-jitter diagnostics | Diagnostic branch; not a release candidate |

The production source and tests in `fix/chunked-prefill-upstream` and
`fix/chunked-prefill` are byte-identical at the v0.3.0 release-candidate
preparation point.

## Branch Retention

Focused `fix/*` branches are retained because they provide:

- Reviewable contribution-specific diffs
- Reproducible benchmark and certification ancestry
- Stable references for manifests and documentation
- Clear separation between implementation, certification, and diagnostics

They are treated as frozen after integration. They should not be advanced to
`fork-main`, force-pushed, or reused for unrelated work.

New changes should start from `fork-main` on a newly named branch.

## Pull-Request Policy

Fork-specific pull requests must target this fork explicitly:

```text
base repository: badle0/nano-vllm
base branch:     fork-main
```

Before merging, verify the target with the GitHub API. No fork-specific pull
request should target `GeeeekExplorer/nano-vllm`.

Release-preparation branches follow this pattern:

```text
release/<version>-rc<number>
```

For example:

```text
release/0.3.0-rc1
```

After the release-preparation pull request is merged, the resulting
`fork-main` commit is tagged with an annotated tag such as:

```text
v0.3.0-rc.1
```

## Known Certification Limits

The following limits must remain visible in release notes:

- Exact all-active top-p remains expensive.
- The FlashInfer backend is opt-in because it has different tie and RNG
  semantics.
- Real two-GPU tensor-parallel NCCL inference remains unverified.
- Chunked-prefill tau 256 passed only three of five strict full-completion
  latency runs and is not latency certified.
- Tau 512 is a throughput/TTFT profile and is not eligible for latency
  certification.
- Performance evidence applies only to the recorded hardware, model, software,
  and workload configuration.

These limits do not invalidate the repaired correctness behavior, but they
define the boundaries of the release claims.

## Upstream Synchronization

The `upstream` remote should be fetch-only:

```bash
git remote add upstream https://github.com/GeeeekExplorer/nano-vllm.git
git remote set-url --push upstream DISABLED
git config remote.pushDefault origin
```

If `upstream` already exists, do not add it again; only verify its fetch URL and
disable its push URL.

Updating `main` from upstream and updating `fork-main` are separate maintenance
operations. Upstream changes should first be reviewed on a temporary integration
branch before being merged into `fork-main`.


