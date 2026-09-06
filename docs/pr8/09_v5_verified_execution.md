# V5: executable speculative decoding (experimental)

This branch replaces production shadow/discard execution with real draft
proposal, target verification, modified rejection, correction/bonus sampling,
and atomic burst commit. The old V3/V4 entry points remain for historical tests;
an initialized speculative engine selects the verified path, not shadow mode.

## Scope and opt-in

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/workspace/models/Qwen3-0.6B",
    draft_model="/workspace/models/Qwen3-0.6B",
    num_speculative_tokens=4,
    max_num_seqs=4,
    max_num_batched_tokens=1024,
    max_model_len=512,
    gpu_memory_utilization=0.5,
    enforce_eager=False,
)
try:
    result = llm.generate(["The capital of France is"],
                          SamplingParams(temperature=0.8, max_tokens=32))
finally:
    llm.exit()
```

The same-model example tests functionality, not useful acceleration. V7 adds a
Qwen3-4B target / Qwen3-0.6B draft graph-mode qualification; see
[the measured scope and limits](12_v7_experimental_qualification.md).
Config validates model family, vocabulary and
tokenizer compatibility, position limits, and safetensors before execution.
Speculation requires TP=1 and the exact sampling backend; FlashInfer and TP>1
are rejected. Omit both draft options to retain ordinary decoding.

The correctness-first policy caps live speculative batches at **4** and effective
K at **4**, additionally bounded by the configured workspace, complete-batch
input budget, remaining completion/model positions, ready draft routes and free
KV blocks. Larger batches fall back as a whole to ordinary decoding; configured
K above four is capped rather than allocated. These are safety/validation caps,
not measured performance crossover points. Draft weights/KV remain resident
during fallback. Bounded request admission (`max_num_seqs`) is unchanged.

## Two target lanes: design amendment to 03/04

Pure stochastic batches execute one causal paged pass over
`[last_committed_token, d1, ..., dK]`, producing all `B*(K+1)` logits. The original
LM-head behavior (last query only for prefill) is unchanged unless the new
`compute_logits_all` entry point is explicitly selected.

**Batches containing a greedy row use K+1 ordinary one-token target calls.**
Experiments found a non-tied greedy discrepancy between parallel BF16 verification
and ordinary decode, including a matching graph-mode baseline margin of 0.25.
The earlier 0.9375 comparison was graph-versus-eager and is not attributed to this
feature. A same-mode control is mandatory. Rather than widen the tie threshold,
the compatibility lane preserves the target's decode kernel/GEMM geometry.
It deliberately claims no greedy speedup. Replacing it with faster verification
requires fresh matching-mode numerical evidence; it is not an unimplemented
acceptance path. Sampling laws still use exact retained FP32 weights and robust
FP64 modified rejection in both lanes.

`target-verifier-v1-both-lanes` readiness is the immutable set of `(live B,K)`
pairs for which both lanes were warmed. Warmup covers all-query plain, greedy,
top-k, top-p and combined transformations, acceptance, residual recovery, and
bonus draws. The registry is intersected with draft admission **before** the
scheduler reserves blocks. Construction scopes Dynamo's specialization limit
to 32; it does not disable compilation, suppress errors, or change the
speculation-off initialization path. Cold-cache runtime checks must establish
that these warmed specializations are actually reused.

## Commit and rollback contract

For a committed burst of `n` tokens from old logical length `L`:

- new target coverage is `L+n-1`;
- new draft coverage is `min(L+n-1, L+K-1)`;
- the final correction/bonus is the unprocessed tail;
- only physical blocks covering retained target positions survive;
- complete target blocks are hashed from committed tokens only;
- EOS truncates only when `ignore_eos` is false, and max-token bounds always apply.

The scheduler validates every host-only result before mutation. Physical trim,
logical append, coverage, hashes, metrics, queues, deallocation and event creation
share one undo record. Failures restore host state and retain a valid lease for
the outer abort to release. CPU/CUDA RNG are restored on failed cycles, including
commit failures. Successful cycles consume proposal, acceptance and corrective/
bonus RNG normally; rewinding successful proposal RNG would introduce dependence.

Reclaimed free target-cache entries are deliberately evicted before target writes.
Rollback restores ownership/free-list order but cannot republish the old hashes
of overwritten physical KV. This is valid conservative cache eviction, not exact
restoration of stale cached contents. Live committed prefixes remain untouched.

## Streaming and metrics

Existing request-owned `StreamSession` pending queues deliver one event per
committed token in row order; only a request's last terminal event is finished.
Close/abandonment drops undelivered events and cancels only owned requests. A
finished request can have undelivered pending tokens; delivery timestamps remain
distinct from compute/finish timestamps. Tokens committed in one burst have the
same compute timestamp, hence zero intra-burst compute ITL is expected.

Speculation-enabled requests expose additive counters, initialized to zero even
if the request never becomes eligible:

| Counter | Meaning |
|---|---|
| `spec_cycles` | Successfully committed speculative cycles |
| `spec_proposed_draft_tokens` | Proposed draft tokens, including rejected suffixes |
| `spec_accepted_draft_tokens` | Accepted draft tokens actually committed after EOS truncation |
| `spec_committed_tokens` | Emitted tokens from speculative cycles, excluding ordinary steps |
| `spec_bonus_tokens` | Bonus tokens actually committed |
| `spec_draft_positions` | Draft catch-up plus proposal input positions |
| `spec_target_verification_positions` | Target query positions, not emitted tokens |
| `spec_residual_numerical_fallbacks` | Forced/observed robust-zero residual recoveries |

Failed cycles do not add counters. Speculation-off metrics retain their prior
keys. `StepOutput.num_decode_tokens` and the legacy negative `step()` count remain
decode **rows**; the progress display counts actual pure-decode emissions.

## Validation status

CPU tests cover actual engine-to-commit stochastic laws, result validation,
rejection positions, EOS/ignore-EOS, bonus/tails, partial-batch failure injection,
RNG rollback, overwritten-cache eviction, and numerical-fallback accounting.
An independent named-owner ledger checks phase peaks and the sequential lane's
fit within the parallel workspace envelope. It checks owner omission/duplication
per phase; it does not claim an arbitrary non-dominant owner changes global peak.

The first exhaustive GPU sweep exposed an underestimated CUDA top-p sort-scratch
term (about 151 MB measured versus 122 MB priced for 20 rows at V=151936).
The workspace model now allows two value/index scratch payloads, not one;
its independent ledger and raw private-scratch probe are checked together.
The 64 MiB allocator margin is not used to hide a live-tensor underestimate.

The [final retained V5/V6 certificate](10_v6_retained_validation.md) includes
matching-mode greedy controls, cached replay, token stream parity, stochastic
mixtures, automatic KV sizing (including graph K=2), exhaustive registered routes,
failure/lifecycle cases and zero compiler deltas inside speculative intervals.
Performance is qualified separately in [V7](12_v7_experimental_qualification.md).
The old auto-sizing failure remains
retained as historical evidence, not erased or retrospectively labeled a pass.

Runtime `gpu_certified=False` remains intentional: a narrow retained experiment
does not certify every valid model, device, dtype or configuration. No general
speedup, exact graph/eager identity, sampled same-seed identity, or certification
of all possible issues is claimed.
