"""Host-only verification results; tensors never cross the scheduler boundary."""
from dataclasses import dataclass
from nanovllm.engine.sequence import SequenceStatus


SPEC_METRIC_KEYS = (
    "spec_cycles", "spec_proposed_draft_tokens", "spec_accepted_draft_tokens",
    "spec_committed_tokens", "spec_bonus_tokens", "spec_draft_positions",
    "spec_target_verification_positions", "spec_residual_numerical_fallbacks",
)


@dataclass(frozen=True, slots=True)
class SpecRowResult:
    seq_id: int
    proposed_token_ids: tuple[int, ...]
    committed_token_ids: tuple[int, ...]
    accepted_draft_tokens: int
    used_bonus: bool
    residual_numerical_fallbacks: int = 0


@dataclass(frozen=True, slots=True)
class SpecExecutionResult:
    cycle_id: int
    workspace_fingerprint: str
    rows: tuple[SpecRowResult, ...]
    target_verification_positions: int
    draft_positions: int


def validate_result(plan, result, seqs, vocab_size):
    """Validate the entire batch before any physical/logical commit mutation."""
    if not isinstance(result, SpecExecutionResult):
        raise TypeError("speculative execution must return SpecExecutionResult")
    if type(result.cycle_id) is not int or result.cycle_id != plan.cycle_id:
        raise ValueError("speculative result cycle mismatch")
    if result.workspace_fingerprint != plan.workspace_fingerprint:
        raise ValueError("speculative result workspace mismatch")
    if type(result.target_verification_positions) is not int or result.target_verification_positions != plan.target_query_tokens:
        raise ValueError("speculative target work mismatch")
    if type(result.draft_positions) is not int or result.draft_positions != plan.draft_catchup_tokens + plan.draft_query_tokens:
        raise ValueError("speculative draft work mismatch")
    if not isinstance(result.rows, tuple) or len(result.rows) != len(plan.rows) or len(seqs) != len(plan.rows):
        raise ValueError("speculative result row count mismatch")
    k = plan.effective_k
    for snapshot, row, seq in zip(plan.rows, result.rows, seqs, strict=True):
        if not isinstance(row, SpecRowResult) or type(row.seq_id) is not int or row.seq_id != snapshot.seq_id or row.seq_id != seq.seq_id:
            raise ValueError("speculative result row identity/order mismatch")
        if (len(seq) != snapshot.committed_tokens or seq.num_cached_tokens != snapshot.target_cached_tokens
                or seq.num_draft_cached_tokens != snapshot.draft_cached_tokens
                or tuple(seq.block_table) != snapshot.block_table or seq.num_scheduled_tokens != 1
                or seq.is_prefill or seq.status is not SequenceStatus.RUNNING
                or seq.max_tokens - seq.num_completion_tokens != snapshot.remaining_completion_tokens):
            raise ValueError("sequence changed before speculative commit")
        accepted = row.accepted_draft_tokens
        if type(accepted) is not int or not 0 <= accepted <= k:
            raise ValueError("invalid speculative accepted count")
        if type(row.used_bonus) is not bool or row.used_bonus != (accepted == k):
            raise ValueError("invalid speculative bonus flag")
        if type(row.residual_numerical_fallbacks) is not int or row.residual_numerical_fallbacks not in (0, 1):
            raise ValueError("invalid residual fallback count")
        if row.used_bonus and row.residual_numerical_fallbacks:
            raise ValueError("full acceptance cannot use a residual fallback")
        if not isinstance(row.proposed_token_ids, tuple) or len(row.proposed_token_ids) != k:
            raise ValueError("incomplete speculative proposals")
        if not isinstance(row.committed_token_ids, tuple) or len(row.committed_token_ids) != accepted + 1:
            raise ValueError("speculative burst must contain accepted prefix plus correction/bonus")
        if row.committed_token_ids[:accepted] != row.proposed_token_ids[:accepted]:
            raise ValueError("speculative burst does not match accepted proposals")
        for token in (*row.proposed_token_ids, *row.committed_token_ids):
            if type(token) is not int or token < 0 or (vocab_size is not None and token >= vocab_size):
                raise ValueError("invalid speculative token ID")
