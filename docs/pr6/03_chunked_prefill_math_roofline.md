# Chunked Prefill — Mathematics and Roofline

*Formulas first, then instantiated for your exact setup: Qwen3-0.6B bf16 on
A100-SXM4-40GB, nano-vllm at `dev` (317e6f0). Predictions at the end are
pre-registered: written before implementation, to be tested by measurement, per house
style. Model claims are labeled; the honest gaps between model and machine are called
out rather than smoothed over.*

## 0. Notation and instantiation

| symbol | meaning | Qwen3-0.6B value |
|---|---|---|
| P | parameter count | ≈ 0.6 × 10⁹ |
| W | weight bytes (bf16) | 2P ≈ 1.19 GB |
| L | layers | 28 |
| d_model | hidden size | 1024 |
| n_q, n_kv | query / KV heads | 16 / 8 |
| d_h | head dim (explicit in Qwen3) | 128 |
| a = n_q·d_h | attention width (≠ d_model here) | 2048 |
| s | bytes per element (bf16) | 2 |
| T | prompt length (tokens) | scenario-dependent |
| B | decoding sequences in a step | scenario-dependent |
| C | prefill chunk size in a step | design knob |
| τ | token budget `max_num_batched_tokens` | default 16384 |
| n_c | chunks per prompt = ⌈T/C⌉ | — |
| π_c | peak bf16 compute (A100) | 312 TFLOP/s |
| π_m | HBM bandwidth: spec / Vast-measured | 1555 / ≈1320 GB/s |

**Per-token KV footprint** (`allocate_kv_cache` line 112 in code form):

    kv_tok = 2 · L · n_kv · d_h · s = 2·28·8·128·2 = 114,688 B ≈ 112 KiB/token
    block (256 tok) = 28 MiB; a 4096-token context = 448 MiB of KV

On 40 GB at 0.9 utilization minus weights and activation headroom, expect order
~1.1–1.2 × 10⁶ cacheable tokens (~4.3–4.7k blocks) — read the actual
`num_kvcache_blocks` at engine start rather than trusting this estimate.

## 1. Work: FLOPs per step

**Linear/MLP work** (all GEMMs against weights), for N tokens processed in a step:

    F_lin(N) ≈ 2 · P · N

**Attention work**, per query token at key-context length K, per layer: QKᵀ costs
2·a·K and A·V another 2·a·K, so 4·a·K; summed over layers, 4·L·a·K per query.

- Monolithic prefill of T tokens: F_att = Σᵢ 4·L·a·i ≈ 2·L·a·T².
  At T = 4096: 2·28·2048·4096² ≈ 1.9 TFLOP — versus F_lin(4096) ≈ 4.9 TFLOP.
  Attention is ~28% of prefill FLOPs at 4k context for this model; linears dominate.
- One decode token at context K: 4·L·a·K ≈ 0.23·K MFLOP — negligible next to
  F_lin(1) = 1.2 GFLOP until K reaches tens of thousands.
- **Chunking conserves attention FLOPs exactly**: every (query, key) causal pair is
  computed once regardless of how the queries are grouped into chunks. Chunking
  redistributes work in time; it does not create FLOPs.

## 2. Traffic: bytes per step

**Weights dominate.** Every forward pass streams W ≈ 1.19 GB once, independent of how
many tokens ride in it. This single fact generates the whole theory.

**KV traffic.** Writes: kv_tok per new token (small). Reads, decode: each of B
sequences reads its full prefix per step, B·K·kv_tok — at the stall demo's scale
(B≈16, K≈100) about 180 MB, i.e. ~15% of the weight stream; at long contexts it
grows toward parity.

**KV reads under chunking — the naive story and its correction.** Naively: chunk j
re-reads its (j−1)·C-token prefix from cache, totaling ≈ kv_tok·T²/(2C) extra bytes
versus a monolithic pass — apparently a strong argument against small C. The
correction: FlashAttention *already* re-reads K/V once per query tile in the
monolithic case. With row-tile size B_r, monolithic causal traffic is
≈ (T/B_r)·(T/2)·kv₁ per layer; chunked totals Σⱼ (C/B_r)·((j−1)C + C/2)·kv₁ =
the same leading term T²/(2B_r)·kv₁. **To first order, chunking does not change
attention HBM traffic** — the causal structure forces quadratic tiled re-reads either
way. What chunking actually costs, traffic-wise: (i) the paged, block-table-gathered
reads are somewhat less coalesced than contiguous fresh tensors, (ii) the chunk's own
K/V round-trips through HBM (stored line 62–63, immediately re-read via line 66),
(iii) per-chunk kernel-launch and Python overheads. All second-order; measure, don't
assume. The **first-order chunking tax lives elsewhere** — §4.

## 3. Roofline

Arithmetic intensity of the weight-bound portion of a step carrying N tokens:

    AI(N) = F_lin(N) / W = 2PN / 2P = N   [FLOP per weight-byte]

Ridge point: N* = π_c / π_m = 312e12 / 1.32e12 ≈ **236 tokens** (spec bandwidth:
201). Interpretation:

- N ≲ 200: the step is **bandwidth-bound**; duration ≈ W/π_m regardless of N —
  tokens are nearly free.
- N ≳ 240: **compute-bound**; duration grows ≈ linearly, ≈ 2PN/π_c.

Floors, instantiated:

    T_bw  = W/π_m  ≈ 1.19 GB / 1.32 TB/s ≈ 0.90 ms      (any small step's floor)
    T_c(N) = 2PN/π_c ≈ N · 3.8 µs                        (compute floor)

**Model vs machine, stated honestly.** Your measured pure-decode steps are 3.0–3.6 ms
at B=8 (slow-consumer baseline, this host) and ~5 ms at B≈16–18 (stall demo, prior
host) against a 0.9 ms floor — 25–30% of roofline. The gap is real machinery: kernel
launches for 28 layers × several ops, the sampler, the `.tolist()` sync,
small-N GEMM inefficiency, Python. Treat the roofline as the *shape* of the curve and
calibrate its scale with one measured point:

    T_step(N) ≈ T₀ + max(T_bw, T_c(N)) · 1/η
    with T₀ + T_bw/η ≈ 3.0–3.6 ms measured ⇒ effective overhead+efficiency lump ≈ 2.2–2.7 ms

A cleaner empirical route (recommended before design): sweep pure-prefill step time
vs C ∈ {64,…,4096} once, fit T₀ and the slope, and use *that* T_step everywhere
below. The formulas here then become the interpretation, not the estimate.

## 4. The piggyback theorem and the true chunking tax

**Piggyback claim.** For a mixed step with B decode tokens and a C-token chunk,
N = B + C:

    T_mix(B, C) ≈ T₀ + max(W/π_m, 2P(B+C)/π_c) + T_att(step)

In the bandwidth regime (B + C ≲ N*): ∂T_mix/∂B ≈ ∂T_mix/∂C ≈ 0 — **decode tokens
ride free on the chunk's weight stream, and vice versa.** This is SARATHI's
piggybacking as one derivative.

**The true first-order chunking tax = weight re-streaming.** A monolithic T-token
prefill streams W once. Chunked into n_c *dedicated* steps, it streams W n_c times:
overhead (n_c − 1)·W/π_m ≈ (n_c − 1)·0.9 ms. **But** if decode steps were going to
run anyway — the serving regime this feature targets — those weight streams are
shared, and the tax vanishes into work already being paid for. Two regimes, cleanly:

- Offline, empty decode queue: chunking costs ≈ (n_c−1)·(T₀ + W/π_m/η); keep C large
  (or τ untouched) — nothing to gain.
- Online, decodes in flight: the tax is absorbed; chunk freely down to the latency
  target. This is why the feature helps `bench_latency.py` and must not hurt
  `bench.py` (whose workload keeps steps full either way).

**Budget accounting caveat** (scheduler line 47–48 charges Q-tokens only): a
late-position chunk carries more attention work than an early one at equal C, since
T_att grows with prefix length. Equal-τ steps are equal-cost only up to the attention
term — at 4k contexts on this model that term is ≤ ~30% of FLOPs, so τ is a good but
not exact cost proxy. A cost-aware budget (charging C + α·prefix) is a refinement,
not a requirement; note it, defer it.

## 5. Latency formulas: what the feature changes

**Today (homogeneous steps, prefill-first).** A long prompt of length T arriving
mid-decode freezes B decoders for its entire prefill:

    stall = TTFT_long ≈ Σ_chunks T_step(chunk)  ≈ T_prefill(T)
    max_ITL_interactive ≈ TTFT_long + T_decode          ← your measured identity

Measured: T=4096 (two 2048 prompts) ⇒ stall ≈ 38–41 ms; max_ITL 43–47 ms; ~9–10×
the 5 ms mean. Chunking-as-it-exists slices the prefill but the decodes still wait
through every slice: same total stall.

**After (mixed steps, decode-first budget).** Decodes advance every step; the worst
gap any decoder sees is one mixed step at full budget:

    max_ITL ≤ T_mix(B, τ − B)                            ← the new bound, set by τ
    TTFT_long ≈ n_c · T_mix(B, C),   n_c = ⌈T / (τ − B)⌉

**Choosing τ against an SLO S** (Sarathi-Serve's rule, in this notation):

    τ* = max { τ : T_mix(B_max, τ − B_max) ≤ S }

using the *measured* T_step fit from §3. Smaller τ tightens the ITL bound and
inflates n_c (hence TTFT_long) — the entire design compressed into one monotone
trade-off.

## 6. Pre-registered predictions for the stall demo (to be tested, not fitted)

Scenario: 16 interactive decoders (ctx ≈ 100), two 2048-token prompts injected;
this host (A100-SXM4-40GB, measured π_m ≈ 1.32 TB/s); model T_step calibrated at
T(step≈16 tok) ≈ 3.5–5 ms. Take τ = 512 for concreteness (C ≈ 494 per step,
N ≈ 512 ≈ 2×N* ⇒ mildly compute-bound):

    T_mix ≈ T₀ + 2P·512/π_c/η_c + small T_att ≈ 4–7 ms        (prediction band)
    n_c ≈ ⌈4096/494⌉ = 9 steps
    TTFT_long ≈ 9 · T_mix ≈ 36–63 ms        vs ~38 ms today   (comparable, slightly worse tail)
    max_ITL_interactive ≈ T_mix ≈ 4–7 ms    vs 43–47 ms today (≈ 7–10× collapse)
    bench.py throughput: within the ±1% same-session band     (steps stay ≥ ridge-full)

Falsifiers, stated in advance: max_ITL failing to drop below ~10 ms indicates the
scheduler is still serializing somewhere (or eager-mixed-step overhead dominates —
see MIXING BLOCKER 3, the CUDA-graph question); TTFT_long blowing past ~2× today's
indicates per-chunk overhead T₀ is larger than the 1–2 ms modeled and τ should rise;
a bench.py regression beyond noise indicates the mixed path broke the fast decode
path (graphs) for pure-decode steps, which the design must preserve unconditionally.

## 7. Formula index (for the design doc's margins)

    kv_tok            = 2·L·n_kv·d_h·s                       = 112 KiB (Qwen3-0.6B)
    F_lin(N)          = 2PN
    F_att(prefill T)  = 2·L·a·T²           (a = n_q·d_h; conserved under chunking)
    AI(N)             = N  FLOP/weight-byte
    N*                = π_c/π_m ≈ 236      (measured-bandwidth ridge, this host)
    T_bw              = W/π_m ≈ 0.90 ms    (floor; measured steps ≈ 3–5 ms ⇒ η≈0.25–0.3)
    chunk weight tax  = (n_c−1)·W/π_m      (dedicated steps only; absorbed when mixed)
    chunk attn traffic≈ monolithic         (flash tiling; first order)
    max_ITL (after)   ≤ T_mix(B, τ−B)
    TTFT_long (after) ≈ ⌈T/(τ−B)⌉ · T_mix
    τ* for SLO S      = max{τ : T_mix(B_max, τ−B_max) ≤ S}
