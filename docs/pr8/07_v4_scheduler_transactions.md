# V4 scheduler transactions: implementation checkpoint

Date: 2026-09-06. Branch: `feat/spec-v2-scheduler-plan`.
Runtime checkpoint: `22b63e8`.

Update: the V4-specific fixed-pool eager/graph GPU archive is now retained and
validated. See [08_v4_retained_gpu_validation.md](08_v4_retained_gpu_validation.md).
The exploratory runs below are historical; they are not the new certificate.

This checkpoint resumes the four unfinished V4 planning/allocator files left
after V3's retained-evidence commit `d760c1c2`. It adds engine/runner integration,
failure-path tests, and CPU CI. It does not change `fork-main`, emit accepted
proposals, or claim completion of speculative decoding.

## Implemented contract

For B selected decode rows, common proposal count K, catch-up work C, and token
budget M, admission enforces both:

```text
verifier inputs = B * (K + 1) <= M
full planned cycle = C + B * K + B * (K + 1) <= M
K_budget = max(floor((M - C - B) / (2 * B)), 0)
```

K also respects configured K, the ready route/workspace cap, the smallest
remaining completion allowance minus one, and the smallest model-position
headroom. With L committed tokens, the draft may write through L+K-2 and the
future target verifier through L+K-1. The selected batch falls back as a whole;
there is no speculative subgrouping. Mixed prefill/decode uses the existing
baseline path.

`SpecStepPlan` is immutable, pickle-safe, and contains no `Sequence`, tensor,
allocator lease, or rollback snapshot. It records full planned draft/verifier
counts and a recomputable conservative workspace certificate including both
q and p. Memory sizing covers the registered route's batch bucket, not just the
live rows. Its fingerprint binds B, C, K, route and modeled memory fields.
`gpu_certified` must remain false: modeled bytes are not measured CUDA peaks.
The route is still the V3 **draft** route key; it does not certify a target
verifier kernel or future verifier graph shape.

The runner revalidates the certificate and full target-write capacity before
executing draft work, then reuses V3's independent live-row, token, cache and
route validation. V4 actually executes C+BK draft inputs and B ordinary target
inputs. `total_model_positions` describes planned V5 work;
`total_scheduled_tokens` is the compatibility view of actual V4 shadow work.
Public output remains one authoritative target token per decode row.

## Ownership and failure handling

```text
ordinary schedule -> capture baseline append undo
  -> validate full plan -> reserve extra verifier suffix
  -> validate runner plan -> catch up/propose/discard
  -> release extra suffix -> stage draft coverage
  -> ordinary target execution -> ordinary postprocess -> release undo record
```

`ActiveSpecTransaction` stays private to the scheduler and survives the handoff
to target execution. Exceptions before target commit release the speculative
lease, clear staged coverage, and undo the ordinary decode-boundary append.
Cleanup failure retains the transaction so a second schedule cannot silently
reuse uncertain state. Targeted cancellation first aborts the batch lease, then
cancels only named requests. The surviving rows remain retryable.

`BlockManager.finalize_temporary_append()` is a **synthetic physical primitive**:
it retains a per-row prefix of newly allocated blocks, restores the trailing
suffix's metadata/free-list order, leaves logical coverages unchanged, and
consumes the lease once. Validation precedes mutation; injected mutation failure
restores the active lease. This is not yet the atomic V5 accept/commit protocol.
In particular, restoring pre-reservation hash metadata is safe for V4's
draft-only writes, not an authorization to restore cached target hashes after
V5 has overwritten the corresponding target KV. V5 must explicitly invalidate
or reconstruct such cache contents and coordinate physical/logical/hash commit.

The ordinary target postprocess is unchanged except for transaction fencing and
successful handoff completion. This checkpoint does **not** promise rollback
after arbitrary failure midway through a multi-row postprocess mutation. V5's
unified atomic commit/undo protocol must cover that separate gate.

## Validation at this checkpoint

Final full CPU regression: **1,041 passed, 31 skipped, 14 warnings in 89.15s**
with CUDA hidden, against runtime commit `22b63e8` plus the historical-provenance
test correction described below. Skipped GPU-dependent tests are not certified
by this result. The workflow's V3/V4 test selection also passed locally:
**324 passed** with the CPU shim and offline environment. The frozen 21-artifact
V3 archive still validates; its claim boundary and pinned runtime are unchanged.

The focused V4 suites plus scheduler cancellation pass: **212 tests** on the
local Python 3.12 environment with CUDA hidden. Coverage includes:

- immutable transport and q+p workspace accounting; exact budget boundaries;
- an independent enumeration of feasible K values for budgets 2 through 29;
- block sizes 4/256, lengths around each boundary, B=1/2, K=1/2/4, and varied
  request tails with a tight model limit;
- full target-write reservations, insufficient pool fallback and exact abort
  restoration of free order, used membership, refcounts, hashes and coverages;
- rejection of forged counts/fingerprints and unavailable target capacity before
  draft model execution or RNG use;
- injected route, reservation, draft, result, handoff, staging and target errors;
- cancellation before/after handoff, retry after cleanup failure, mixed queues,
  and 20 repeated cycles through request completion without allocated-block leaks;
- exhaustive two-row physical suffix retention vectors through three appended
  blocks, including shared-prefix and mutation-failure cases;
- target event/RNG parity with speculation disabled versus shadow-enabled CPU
  model doubles. These doubles do not validate attention or CUDA kernels.

The initial full CPU run found a missing attribute in a deliberately minimal
scheduler double; `abort_speculative_step()` now treats absent transaction state
as inactive. The remaining dirty-checkout failure was the provenance test
correctly requiring runtime file hashes to match HEAD. Do not weaken that gate;
run it after committing the implementation. That run additionally exposed a
historical test assuming HEAD must forever have V3's runtime tree. The test now
checks both frozen V3 producer commits and explicitly requires rejection of a
changed runtime (including V4 HEAD). The evidence validator and its pinned trees
are unchanged.

A fresh-process **exploratory** A100-SXM4-40GB eager off/on control also ran with
the same Qwen3-0.6B checkpoint on both sides, B=2, K=2, model/token limits 128,
and independent compiler caches. Six authoritative target events, token lists,
and all registered CPU/CUDA RNG snapshots matched exactly. Two shadow intervals
executed, with catch-up counts 7 then 0. The raw artifacts are local temporary
files under `/tmp/nanovllm-v4-smoke.BTaGMi/`; they are not retained release evidence.
The frozen V3 comparator rejected this altered configuration as intended. Direct
field comparisons established only the narrow parity result above, not a new
V3 or V4 retained certificate.

An attempted graph-mode run on clean `22b63e8` was rejected **before model
execution** by the unchanged V3 producer/runtime binding. No graph-mode parity
result is claimed for V4. This confirms the need for a separate V4 provenance
harness, rather than extending the old archive's claim boundary.

Reproduce CPU checks from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' \
  /venv/main/bin/pytest -q -p no:cacheprovider \
  tests/test_speculative_v4_workspace_plan.py \
  tests/test_speculative_v4_block_finalizer.py \
  tests/test_speculative_v4_integration.py tests/test_scheduler_cancel.py
```

CI now runs the V3/V4 shadow transaction suites in its existing Python 3.10/3.12
CPU matrix. Local success is not a claim that remote CI has run.

## Unfinished gates / next work

1. Completed for the registered B<=4/K<=2/64-block configuration: V4-specific
   clean-SHA eager/graph controls, boundary/cache-fill cases and admitted-route
   coverage. Automatic KV sizing failed one startup experiment and remains
   uncertified; investigate it separately (document 08).
2. Review coverage against every V4 hard gate in document 04 before marking the
   rung complete. In particular, expand joint queue/pool/route-cap boundary
   coverage beyond the separate tests above.
3. Only then start V5 target verification, exact rejection/bonus handling and
   atomic logical/physical/cache-hash commit. Acceptance, bursts, speculative
   streaming/metrics, measured workspace peaks, and speedup remain unimplemented.
