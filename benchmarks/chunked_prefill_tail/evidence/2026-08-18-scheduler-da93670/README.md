# Scheduler backlog roofline — 2026-08-18

This directory preserves five fresh-process CPU observations from harness commit
`da93670990725e5aef69ebdad09d76150ddc2642`, source SHA-256
`c56e5b5ff4c95021197f01fce2278f10eb1455be4006262a7f81bb2f56ba8872`.
Every JSON is byte-identical to its immutable external original.

The median of process medians was 0.67 us at zero injected waiters, 0.76 us at
100,000, and 0.76 us at 500,000. The 500k/zero ratio was 1.1343x and remained
well below the predeclared `max(2x zero, 5 us)` threshold. All five raw files
also passed bounded-admission, nonpositive-budget, read-only-budget, GC-state,
and per-process timing gates.

The injected waiting deque is deliberately unreachable through the bounded
public admission API; it isolates the scheduling algorithm's complexity. Queue
construction is outside every timed interval.

Hashes:

- manifest: `e1fe1cb0cb8b781771c113f516a542f139ce7b89d4283b5192f5d9a7594efeba`
- seed 20260818: `1a1b71dc8e654871f97b473a77ee151d531760b8af1770ceac465e8632f5da2c`
- seed 20260819: `790958ecba9eee33af6dfce69dba0f5dbfd3a7459526c158debce37891c8f399`
- seed 20260820: `52f9b9af5072c9f519912075da44624fca50a311d963c3d0f628f1d4d639ccb5`
- seed 20260821: `24bdd56332975766716d378d9ae454b396937281292ba424776073d4db7a9d44`
- seed 20260822: `b5778d7e5023422d83589eb2e6a6a8a61f1103d5671c1a7157bc8b44dfcb4738`

The manifest records the original external absolute paths as run provenance;
the archive validator accepts these byte-identical copies by content and
recomputes every median from the retained nanosecond samples.
