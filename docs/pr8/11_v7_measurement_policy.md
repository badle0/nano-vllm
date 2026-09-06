# V7 measurement policy (registered before headline runs)

The V5/V6 runtime is frozen at `a165b654660e60ca60cecd47b89c95de1d65735b`.
No performance claim is licensed by the instrumented correctness archive.
This branch qualifies that runtime before proposing any optimization.

Required model pair: Qwen3-4B target at HF revision
`1cfa9a7208912126459214e8b04321603b3df60c`, Qwen3-0.6B draft from the certified
local checkpoint. The 4B weights occupy an isolated RAM-backed directory under
`/dev/shm` because the workspace lacks disk capacity. This checkpoint is temporary
and must be downloaded again after reboot; retained artifacts include weight
and configuration hashes. No workspace data is deleted to make room.

## Controlled primary matrix

- A=speculation off; B=speculation on with configured K=4, graph-enabled engine.
- Batches 1, 4, 8; contexts 32 and 256; 64 completion tokens, `ignore_eos=True`.
- Greedy, temperature .8, top-k 50, top-p .95, and combined top-k 50/top-p .95.
  Code prompts are used for plain/top-p; prose prompts for the other families.
- Five fresh-process pairs; order AB, BA, AB, BA, AB. Every cell uses seeds
  17, 23 and 41. Two complete workload warmups precede its three measured seeds.
- Fixed 64-block pools for both sides, model limit 4096, input-token budget 4096,
  memory utilization .8. Equal physical block capacity isolates execution costs
  from auto-sizing capacity differences; the enabled side still owns two KV pools.
- The headline timer contains the public generation/stream call and final CUDA
  synchronization, but no internal phase timing hooks, allocator snapshots,
  compiler-file scans or GPU-monitor subprocesses. Hardware snapshots occur
  outside the timer. Existing API metrics supply request timing and work counts.

Reject (but retain) a sample if more or fewer than one GPU process is visible,
one-minute host load exceeds .75 per logical CPU, GPU temperature reaches 85 C,
or the timed sample creates a new Dynamo graph. Do not silently discard slow
samples that satisfy these predeclared conditions. GPU clocks, power, temperature,
memory occupancy, load, seeds and raw samples are retained.

Compare pairs/identical seeds, report medians and paired ratios. Five independent
pairs are the replication unit; the three seeds within a process are not falsely
counted as 15 independent machines/processes. Report bootstrap intervals as
descriptive uncertainty, not universal guarantees. A +/-5% noise band is used for
speculation-off regression comparisons against the frozen pre-V5 runtime.

## Supplemental qualification

The extended matrix covers batches 2/16/32/64/128, contexts 1024/2048, completion
tails 1/2/4/5 and 256, and null/2-ms-per-token slow stream consumers. Configured
K=5/6 is capped to the registered K<=4 policy, not mislabeled an implemented
five/six-token verifier. The initial route caps B<=4/K<=4 are **validation caps**,
not measured crossover thresholds. Large-batch requests remain supported through
ordinary decoding; bounded admission and finite KV capacity still apply.

Required numerical gate: compare same-mode, same-seed greedy continuations for
the 4B target. Sampled runs need not match token IDs or RNG streams; their
acceptance and fixed-length law checks are separate. Any unexplained non-tied
greedy mismatch stops a release claim. Heterogeneous-model smoke data are
exploratory and cannot substitute for the paired primary matrix.

## Roofline and interpretation

Measure this host's attainable bandwidth with 256-MiB device-copy buffers (both
read and write bytes counted) and dense BF16 compute with a 4096-square GEMM.
Use 10 warmups and 50 CUDA-event-timed repetitions for each. These are calibrated
ceilings/lower-bound inputs, not an assertion that small decode kernels attain
them. Do not copy old PR7 bandwidth or datasheet compute numbers.

Approximate dense model work by `2*parameter_count*query_tokens`, with attention
work/context reads stated separately. Compare weight/KV/probability byte floors
against calibration bandwidth and compute work against calibration FLOP/s.
The steady-state break-even condition is

`emitted_tokens * ordinary_step_time > draft + verify + accept + commit + catchup`.

Greedy compatibility verification uses K+1 target calls, so the parallel-verifier
speedup formula must not be applied to it. Report actual committed/proposed
ratios and emitted tokens/cycle; the former is a prefix acceptance statistic,
not an unbiased estimate of every conditional per-position acceptance probability.
Prefill, tail clipping, changing effective K, host synchronization, graph padding,
and consumer delays must not be hidden in a fitted acceptance scalar.

If no supported cell improves credibly, ship only as **experimental with no
default performance claim**. Do not waive the required 4B pair or reinterpret
same-model correctness controls as proof of acceleration. A broader Cartesian
matrix/model fleet remains future qualification, not a claim of this initial
bounded experimental implementation.
