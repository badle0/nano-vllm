# PR 8 design packet: speculative decoding v2

Status: the V0 design packet was frozen in commit `480a3b2`, and V1 sampling-law
work was implemented and locally certified through `8989e44`. V2's inert
dual-model lifecycle is implemented at `d87f168b804778fbb5888a662dc8a0defccfd660`
and retained-certified on A100 by the versioned archive under
`benchmarks/speculative_v2/evidence/2026-08-28-a100-v2-d87f168/`.

V3's draft-discard runtime is frozen at
`7fec9993d5e4e0e06fec22e3973dfc203fdbd2d8`. Its retained-evidence harness is
`e8e0452f99727958077b51f340a5375a090e6884`; both commits contain the identical
`nanovllm` subtree `52398af379f767708a0b804646f4b490fa8323ad`. The clean-SHA
A100 archive is under
`benchmarks/speculative_v3/evidence/2026-08-28-a100-v3-e8e0452/`.

V3 is retained-certified for the draft-discard milestone only. On one A100,
using the same checkpoint as target and draft, the evidence certifies
transactional draft catch-up, proposal execution and discard; exhaustive
exercise of every registered finite `draft-discard-v1` route in the retained
K=2, batch-cap=4 configuration; compiler, RNG,
and attention-context neutrality inside explicitly guarded draft intervals;
draft-KV fill neutrality for the registered boundary and shared-prefix
comparisons; and speculation-off/on parity of public IDs, authoritative target
events/tokens, and registered RNG checkpoints.

This is not a certificate for target verification, acceptance/rejection, bonus
or burst commit, speculative streaming or metrics, performance or speedup,
tensor parallelism greater than one, FlashInfer, or heterogeneous target/draft
models. Aggregate compiler state outside guarded draft intervals is not claimed
unchanged. The graph cache-neutrality and output-control runs emitted a Dynamo
recompile-limit warning outside the route proof, so those artifacts support only
their stated numerical and output oracles. V4 planning, reservation and shadow
integration are implemented as a local checkpoint; its full retained certificate
and V5-V7 remain pending. See [07_v4_scheduler_transactions.md](07_v4_scheduler_transactions.md)
for the implementation, test results and explicit unfinished gates.

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
7. [07_v4_scheduler_transactions.md](07_v4_scheduler_transactions.md) records
   V4's full-cycle planning, shadow transaction integration and remaining gates.

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
  -> feat/spec-v2-draft-path            (V3 narrow retained certificate)
  -> feat/spec-v2-scheduler-plan ...    (V4-V7 gated implementation branches)
  -> release/speculative-decoding-v2    (only after correctness + A100 gates)

feat/speculative-decoding @ a632b59     (preserved prototype; never merged wholesale)
```

No command in this packet deletes a branch, tag, worktree, or evidence artifact.
