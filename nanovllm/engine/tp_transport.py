from __future__ import annotations

from typing import NamedTuple


class ScheduledSequence(NamedTuple):
    """Compact, pickle-safe view of one sequence for tensor-parallel workers.

    Rank zero keeps the owning :class:`Sequence`; workers only need the tokens in
    this step plus the offsets and block table used by input preparation.  A
    module-level ``NamedTuple`` is intentionally used so multiprocessing's
    ``spawn`` start method can resolve the type while unpickling.
    """

    scheduled_token_ids: tuple[int, ...]
    is_prefill: bool
    num_cached_tokens: int
    num_scheduled_tokens: int
    num_tokens: int
    last_token: int
    block_table: tuple[int, ...]

    @classmethod
    def from_sequence(cls, seq) -> "ScheduledSequence":
        start = seq.num_cached_tokens
        count = seq.num_scheduled_tokens
        end = start + count
        if start < 0 or count <= 0 or end > seq.num_tokens:
            raise ValueError(
                "invalid scheduled sequence bounds: "
                f"cached={start}, scheduled={count}, total={seq.num_tokens}"
            )
        scheduled_token_ids = tuple(seq.token_ids[start:end])
        if len(scheduled_token_ids) != count:
            raise ValueError(
                "scheduled token slice does not match sequence accounting: "
                f"expected {count}, got {len(scheduled_token_ids)}"
            )
        return cls(
            scheduled_token_ids=scheduled_token_ids,
            is_prefill=seq.is_prefill,
            num_cached_tokens=start,
            num_scheduled_tokens=count,
            num_tokens=seq.num_tokens,
            last_token=seq.last_token,
            block_table=tuple(seq.block_table),
        )

    def __len__(self) -> int:
        return self.num_tokens


def compact_run_args(args: tuple) -> tuple:
    """Convert only ``run`` sequence arguments into the worker wire format."""

    if len(args) != 2:
        raise ValueError(f"run expects two arguments, got {len(args)}")
    seqs, is_prefill = args
    return ([ScheduledSequence.from_sequence(seq) for seq in seqs], is_prefill)
