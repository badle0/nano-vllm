# PR 8 design packet: speculative decoding v2

Status: the V0 design packet was frozen in commit `480a3b2`, and V1 sampling-law
work was implemented and locally certified through `8989e44`. V2's inert
dual-model lifecycle is implemented at `d87f168b804778fbb5888a662dc8a0defccfd660`
and retained-certified on A100 by the versioned archive under
`benchmarks/speculative_v2/evidence/2026-08-28-a100-v2-d87f168/`. The certificate
covers typed configuration, target/draft identity, transactional
load/warmup/KV/graph ownership, modeled speculative-workspace reservation, and
rollback/teardown. Its offline validator resolves the registered runner bytes
from the immutable `d87f168` Git object, rather than the evolving current
worktree; CI therefore checks out full history. The V2 archive, manifest, and
raw artifacts remain byte-for-byte unchanged as V3 evolves. V3 draft catch-up,
pure-decode planning, transactional
proposal-write reservation, direct retained-q proposal execution, and
compute-then-discard are implemented on `feat/spec-v2-draft-path`. The current
V3 change set also contains the finite draft-only route/workspace/warm
registry, a 32-token ready-route K cap, a 512-row graph-batch admission cap,
structurally unreachable-route pruning, post-default-restoration constructor
pretouch, host-only fail-closed route admission, and the allocator/scheduler/session
rollback and cancellation fences required by discard execution. The route code
and tensor-bearing-error wrapper remain Python 3.10 compatible. Dirty-worktree
A100 exploration has passed the configured-K=2, batch-cap=4 eager and graph
route gate and both eager and graph zero-versus-NaN draft-cache-neutrality
comparisons. Fresh eager and graph speculation-off/on output controls also match
public sequence IDs, authoritative target events/tokens, and all four CPU/CUDA
RNG checkpoints under combined top-k/top-p sampling. The control exposed and the
worktree fixed a constructor identity leak: draft warmup now uses a counter-free
`ScheduledSequence` DTO instead of consuming a public `Sequence` ID. These runs
are useful development observations, but V3 remains **uncertified**: they are
not retained clean-SHA evidence, and the complete memory,
boundary, end-to-end, regression, and archive-validation gates remain pending.
Target verification, burst commit, speculative streaming/metrics, and
performance routing remain V4-V7 work.

Design base: `origin/fork-main` at
`663753b99131945c297c1fbe02341108f422dce7`.

Historical prototype: `feat/speculative-decoding` at
`a632b59` (PR7 C1-C5). Preserve it as read-only provenance. It is not a release
candidate for the current fork because it was built on the pre-hardening
chunked-prefill line, its C6 performance gate was never completed, and its
scheduler, transport, streaming, metrics, and lifecycle assumptions no longer
match `fork-main`.

## Reading order

1. [01_theory.md](01_theory.md) defines the mathematical contract, exact
   rejection sampler, performance model, variants, and limits.
2. [02_code_map.md](02_code_map.md) traces the current engine from construction
   through teardown and maps every required change to current functions.
3. [03_design_map.md](03_design_map.md) records the selected architecture,
   alternatives, invariants, risk points, and decisions that measurements may
   reopen.
4. [04_implementation_validation_plan.md](04_implementation_validation_plan.md)
   is the branch/commit ladder, test gates, A100 benchmark plan, and release
   criteria.
5. [05_v1_sampling_law_certification.md](05_v1_sampling_law_certification.md)
   records the exact V1 commits, local environment, adversarial findings,
   commands, results, fallback bound, and remaining exclusions.
6. [06_v2_dual_model_lifecycle.md](06_v2_dual_model_lifecycle.md) records the
   live V2 implementation delta, capacity policy, failure transaction, test
   protocol, evidence status, and the exact boundary to V3 and later work.

The request's fourth list item was blank. This packet interprets it as the
implementation, validation, and benchmark rollout plan because that is the
fourth document in the earlier PR7 packet and is the necessary bridge between
design and code.

## v2 headline contract

- Implement classic draft-target speculative decoding first.
- Preserve the target model's transformed distribution for the exact sampling
  backend; sampled equivalence is distributional, not same-seed identity.
- Keep speculation disabled by default and preserve baseline behavior when it
  is disabled.
- Keep proposal tokens provisional: they never enter public sequence state,
  streaming output, metrics, or prefix hashes before commit.
- Start with pure-decode cycles, one target and one smaller draft model, a
  shared logical block-ID space with separate physical KV tensors, and TP=1.
- Include the standard all-accepted bonus token in the complete v1 path, while
  allowing a no-bonus correctness milestone during development.
- Reject unsupported combinations before GPU/process ownership: initially
  speculative sampling with FlashInfer top-p and tensor parallelism greater
  than one.
- Let measured cost and acceptance—not a promised headline speedup—decide the
  default lookahead and the batch-size crossover at which speculation bypasses
  to baseline decoding.

## Branch disposition

Do not delete or force-rewrite `feat/speculative-decoding` while v2 is being
designed. Its old worktree is currently dirty: `benchmarks/pr7/status.md` is
modified, `benchmarks/feat_audit/` is untracked, and two audit documents are also
untracked. A tag or bundle of Git refs preserves only committed objects; by itself
it does **not** preserve these dirty or untracked files.

Before eventual deletion, record both SHAs and have the owner review and preserve
the complete WIP state either in an explicit archive commit or in a checksummed
external snapshot. Then verify recovery into a separate location, including the
modified status file, the untracked benchmark directory, and both untracked audit
documents. Only after that recovery check may an immutable archive tag or bundle
be treated as sufficient branch-history preservation. Deletion remains optional;
a clearly named archive ref plus the separately verified WIP archive is safer and
preserves useful engineering history.

The intended sequence is:

```text
origin/fork-main @ 663753b
  -> docs/speculative-decoding-v2       (this packet)
  -> feat/spec-v2-sampling-law          (V1 exact sampling-law seam)
  -> feat/spec-v2-dual-runner           (V2 inert dual-model lifecycle)
  -> feat/spec-v2-draft-path ...        (V3-V7 gated implementation branches)
  -> release/speculative-decoding-v2    (only after correctness + A100 gates)

feat/speculative-decoding @ a632b59     (preserved prototype; never merged wholesale)
```

No command in this packet deletes a branch, tag, worktree, or evidence artifact.
