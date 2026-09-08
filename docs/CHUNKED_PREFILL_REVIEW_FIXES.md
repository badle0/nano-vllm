# Chunked-prefill review: verified defects and focused repairs

## Scope and provenance

This repair is based on `integrate/speculative-decoding` at
`80a59490e2ed54ed16d5f5e0fabfcd0f0844f2f0`. Its pre-repair `nanovllm` subtree
is `c2c9fbce937217ad2227955fbb4451441a721dd3`, identical to the archived
`feat/spec-v2-performance` runtime. The supplied 2026-09-06 review instead
patched an older chunk branch and checked `fork-main` at `663753b`.
Its reported patch commit and standalone review harness are not present in
this checkout; the reproductions here are independent. The supplied review
is a reference, not an instruction to apply an unavailable patch.

The focused repair lives on `fix/chunk-prefill-review`. It does not change
`fork-main`, release tags, remote branches, or the original upstream project.

| Finding | Verification and disposition |
| --- | --- |
| Nonmultiple decode cap has no CUDA graph | Confirmed on `663753b`: 472 affected caps and 3,748 uncovered cap/batch pairs. Already fixed on the speculative integration by the shared target/draft bucket policy; exhaustive caps 1–512 have no holes or oversized labels. Preserve that fix. |
| Decode lookup raises when no capture is available | Reproduced with missing/empty captures. Add ordinary eager fallback; no recovery from arbitrary malformed graph buffers is claimed. |
| Partial prefill displaces ongoing decode under KV pressure | Reproduced on the integration runtime. Reclaim the partial waiting head first, recheck capacity, then retain existing decoder-victim fallback. |
| Entire known prompt receives physical KV blocks immediately | Confirmed; intentionally unchanged in this focused repair. Incremental allocation requires a separate allocator/admission/resumption design. |
| Ragged graph coverage ends at fixed token buckets | Confirmed: budget 300 can miss above 256, budgets below 128 use eager, and saturated 16,384-token steps exceed the 2,048 graph ceiling. These are correct eager routes, not incorrect outputs. |
| Segment count 65 can route to 513 slots | Confirmed with max_num_seqs=512. Padding overhead requires GPU measurement before selecting more tiers. |
| Intermediate prompt chunks compute discarded logits/samples | Confirmed by the runner's per-sequence LM head and the scheduler's emission gate. Keep existing sampling/RNG behavior in this repair. |
| Decode-first implies a constant wall-time ITL bound | Incorrect claim. Correct the scheduler comments and user-facing contract. |
| Existing evidence certifies strict sub-10-ms ITL | Not established: retained tau-256 is 3/5 passing; tau-512 is a throughput/TTFT profile, not latency eligible. Preserve the negative evidence. |
| Arbitrary graph/eager greedy streams are identical | Additional check found two divergent requests out of 17 synthetic prompts. First differences involve tied eager maxima; pre-repair graph outputs reproduce identically. This numerical limitation is not repaired by changing scheduler priority. |

## Why this fix, rather than a KV allocator rewrite

Decode admission happens before prefill scheduling in the synchronous
schedule/run/postprocess loop. The existing invariant identifies the only
partially computed waiting request at the queue head. Removing that head and
calling `preempt()` clears its cache coverage and partial pointer, releases
references, and appends it behind existing waiters. Capacity is checked again
before scheduling the decoder. Existing speculative transaction guards still
prevent scheduling over an active proposal/verification lease; preemption
continues to reset draft coverage as well as target coverage.

Sampling kernels, temperature/top-k/top-p rules and RNG ownership are unchanged.
The changed schedule under memory pressure can itself change batch composition
and global RNG consumption order, so fixed-seed stochastic outputs need not
match the old eviction policy. That is distinct from skipping intermediate
sampling draws, which this repair does not do.

This establishes decode-priority victim selection, not immunity from
preemption, a fairness guarantee under unbounded arrivals, or a throughput
improvement. If no partial prefill can supply the needed capacity, ordinary
decoder preemption remains. An asynchronous runner would require explicit
in-flight KV lifetime tracking before using this reclamation rule.

An incremental allocator could reduce this contention at its source, but
changing only the initial allocation loop is unsafe: resumed chunks and
speculative snapshots currently rely on the full block table. That work is
kept separate to make this behavioral repair small and independently testable.

## Reproduction and regression coverage

The pressure workload has two distinct prompts: A has 250 input and 16 output
tokens; B has 4,096 input and one output token. The pool contains 17 blocks
of 256 tokens. At B's first admission, it reserves 16 blocks despite only
six tokens being scheduled in that step.

| Query-token budget | A's largest emission gap: before → after | Total scheduler steps: before → after |
| --- | --- | --- |
| 64 | 59 → 1 | 77 → 79 |
| 128 | 27 → 1 | 43 → 43 |
| 256 | 11 → 1 | 26 → 26 |

These are **scheduler iterations, not milliseconds**. The before values were
reproduced on the unmodified integration. At budget 256, A finishes at
zero-based step 15 instead of 25, while B finishes at step 25 instead of 16.
At budget 64, B recomputes 128 prompt tokens; A no longer needs its own
recomputation. This accounts for the extra two total steps.

Maintained regressions in `tests/test_chunk_prefill_review.py` cover these
three cases, waiting-order preservation and repeated capacity checks, all
decode caps/batches through 512, and eager fallback without graph buffers.
A seeded 1,000-workload host sweep uses block size four, shared prefixes,
constrained pools, completion and cancellation. It compares cached positions
against exact logical-prefix tags, checks allocator partitions/reference
counts after every step, and requires complete release on drain. This is
an ownership oracle, **not a floating-point attention oracle**.

Run focused host tests in the maintained CPU environment:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH=.github/ci/speculative_v2:. \
  python -m pytest -q -p no:cacheprovider \
  tests/test_chunk_prefill_review.py tests/test_scheduler.py \
  tests/test_scheduler_cancel.py tests/test_block_manager_temporary.py
```

The CPU import shim must not be used for actual GPU/model tests. Fresh GPU
checks must cover a real 17-row decode capture, the 17-block pressure case,
graph/eager execution, and the existing ragged/unpadded attention oracles.
The complete speculative contract suite must also pass after this scheduler
change. Correctness checks do not replace a new latency certification.

## Validation performed on 2026-09-08

- Before the runtime edit, the new focused regressions produced seven failures
  and one pass (the already-correct bucket policy). After the edit, the initial
  focused group passed all 35 tests, including the 1,000-workload ownership sweep.
- Full maintained CPU suite: **1,005 passed, 26 skipped**, using the existing
  pinned CI-compatible dependency environment and its CPU import shim.
- Full GPU-enabled suite: **1,035 passed, one skipped** on A100-SXM4-40GB,
  Torch 2.10.0+cu128. This includes the existing padded/unpadded ragged oracles.
- Fresh budget-256 Qwen3-0.6B checks: real 17-row decode graphs, 17-block pressure,
  complete request drain, and forced missing-decode-graph fallback all ran.
  Pressure-case greedy output IDs match between graph and eager execution;
  both preserve consecutive short-request emissions and 26 total steps.
- Existing active-speculation graph smoke: **49 cycles passed**, including
  streaming/GC cleanup, mixed and stochastic sampling, and injected-failure
  rollback/retry. It used the available Qwen3-0.6B as both target and draft;
  it is not a new different-model performance certificate.
- Both historical chunk-evidence validators passed without altering the old
  archives or their negative latency verdicts.

Two subsequently added host safeguards check the worker's `python -O`
rejection and preserve the negative GPU-comparison artifact. The full-suite
counts above describe the runs before those two safeguards were added.
The final focused group, including both safeguards, passed **37 tests**.

### Additional numerical finding: keep the failed comparison visible

The synthetic 17-request workload repeats token `1000 + row` four times per
prompt. Strict end-to-end graph/eager equality fails for requests 9 and 12
(zero-based). At their first identical-prefix disagreements:

| Request | Position | Graph decision | Eager decision |
| --- | --- | --- | --- |
| 9 | First prompt output | Token 3988 at 8.0, ahead of 9/353 at 7.96875 | Tokens 9/353/3988 tied at 8.375; argmax chooses 9 |
| 12 | First decode after prefill | Token 38297 at 14.0, ahead of 25 at 13.875 | Tokens 25/38297 tied at 14.0; argmax chooses 25 |

There were 80 comparable identical-prefix emission rows across the two
workloads. Subsequent tokens after a divergence no longer compare identical
prefixes. This is evidence of BF16-sensitive decision boundaries, not proof
that all graph/eager differences are harmless. Restoring the two pre-repair
methods from `80a5949` **in memory** reproduced the same graph outputs; the
17-prompt graph also matched the existing same-shape padded-eager oracle
bit-for-bit. No change was made to the greedy tie-breaking contract to force
an apparent pass. Universal cross-shape token parity remains unclaimed.

The graph run also logged a Dynamo recompile-limit warning for `rms_forward`
(rank-2/rank-3 specialization). These runs are neither recompile-free nor
latency-certified; the warning belongs in future shape/warmup performance work.

Raw artifacts, including `eager256.json` with `passed: false`, are retained in
[`2026-09-08-review-fixes`](../benchmarks/chunked_prefill_tail/evidence/2026-09-08-review-fixes).
They record modified-working-tree source hashes: the HEAD field names the
base commit, **not a clean committed version of the repair**. The host
regression pins both comparison files by SHA-256. The source aggregate is
`8c6c4bf0470976c12b6a91c8da822c5ed22430fba757250090cde87398c558e5`.

Reproduce in separate fresh processes (choose unused output paths):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python tests/run_chunk_prefill_review.py \
  --budget 256 --output /tmp/chunk-review-graph.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python tests/run_chunk_prefill_review.py \
  --budget 256 --eager --compare /tmp/chunk-review-graph.json \
  --output /tmp/chunk-review-eager.json
```

The second command deliberately returns failure if strict token parity differs,
**after retaining the negative result and logit traces**. Do not treat that
comparison as a passing release gate. The worker refuses `python -O` and will
not overwrite an existing evidence file. Budgets 64 and 128 can be selected
for additional GPU experiments; only 256 was exercised in the new GPU pair.

## Performance work that remains open

1. **Incremental physical KV allocation:** separate full-request-fit admission,
   prefix lookup/pinning and per-step extension. Extend on every resumed chunk;
   define extension-failure retry/requeue behavior before changing allocation.
   Preserve shared-prefix references and speculative reservation rollback.
2. **Ragged graph routing:** benchmark the existing policy against an exact
   token-budget endpoint within the current ceiling, then intermediate slot
   tiers. Record capture memory/time, graph misses, padded work and complete
   request timings. Do not assume a 4,096/16,384-token capture is faster: existing
   measurements motivated the current ceiling.
3. **Emission-only LM head/sampling:** introduce explicit output-row mapping
   before changing the emission gate. Define the fixed-seed RNG compatibility
   contract first; skipping intermediate draws is not a seeded-output-neutral
   change. Keep all-position speculative verification separate from ordinary
   chunk emission and cover rank-consistent TP gathers with host tests.
4. **Latency investigation:** measure scheduler, CPU preparation, model GPU
   span, sampling and token-transfer synchronization separately. For a chunk
   of q tokens following C cached tokens, causal attention processes
   q*C + q*(q+1)/2 query/key pairs per head. Leading attention FLOPs per layer
   are approximately 4*H*d times that count (QK and attention-times-V), excluding
   projections, MLP, softmax and memory/launch effects. A query budget does not
   bound context-dependent work or host/device stalls.

For any promoted performance change, rerun the retained full-completion
protocol on new source/model/environment pins with five fresh processes per
profile. Report raw ITLs, p50/p95/p99/max, long-prompt TTFT, output throughput,
preemptions/recomputed tokens and graph coverage. Keep tau-256's existing
strict five-run gate and tau-512's non-certifying profile unchanged. Never
reuse the historical A100 artifacts as evidence of this repair's speed.
