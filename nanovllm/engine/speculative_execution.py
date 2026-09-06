"""Exact, eager paged target verification; draft decode may use CUDA graphs.

This correctness-first route is deliberately bounded to B<=4, K<=4. Larger
batches use ordinary decoding; larger configured K is capped, never silently
allocated. All admitted verifier shapes are touched before serving requests.
"""
import torch

from nanovllm.engine.speculative_result import SpecExecutionResult, SpecRowResult
from nanovllm.layers.sampler import _sample_exponential_race
from nanovllm.utils.context import set_context, reset_context

MAX_VERIFY_BATCH = 4
MAX_VERIFY_K = 4
VERIFIER_ROUTE_SCHEMA = "target-verifier-v1-both-lanes"


@torch.inference_mode()
def sequential_target_probabilities(runner, rows, proposals):
    """Compatibility lane for batches containing greedy rows.

    BF16 paged-prefill and one-token-decode kernels can disagree even away from
    an exact argmax tie. Use the ordinary target decode kernel/GEMM geometry for
    these batches. This lane makes K+1 target calls and claims no acceleration.
    """
    batch, k = proposals.shape
    device = runner.kv_cache.device
    storage = torch.empty((k + 1, batch, runner.config.hf_config.vocab_size),
                          device=device, dtype=torch.float32)
    metadata = runner._prepare_draft_sample_metadata(rows)
    try:
        for step in range(k + 1):
            # Match ordinary prepare_decode's non-inference input tensors.
            with torch.inference_mode(False):
                inputs = (torch.tensor([row.last_token for row in rows], device=device, dtype=torch.int64)
                          if step == 0 else proposals[:, step - 1].clone())
                inputs, positions = runner._prepare_draft_decode(rows, inputs, step)
            logits = runner.run_model(inputs, positions, False)
            runner.sampler.prepare_exact_probabilities(
                logits, metadata[0], top_k_buckets=metadata[1], top_p_plan=metadata[2],
                probabilities_out=storage[step],
            )
        return storage.permute(1, 0, 2)
    finally:
        reset_context()


@torch.inference_mode()
def target_probabilities(runner, rows, proposals):
    """p(d1|prefix), ..., p(dK|prefix,d<K), p(bonus|prefix,d<=K)."""
    batch, k = proposals.shape
    device = runner.kv_cache.device
    positions, slots, cumulative_k = [], [], [0]
    for row in rows:
        for position in range(len(row) - 1, len(row) + k):
            positions.append(position)
            slots.append(row.block_table[position // runner.block_size] * runner.block_size
                         + position % runner.block_size)
        cumulative_k.append(cumulative_k[-1] + len(row) + k)
    tail = torch.tensor([row.last_token for row in rows], dtype=torch.int64, device=device)
    inputs = torch.cat((tail[:, None], proposals), dim=1).flatten()
    positions = torch.tensor(positions, dtype=torch.int64, device=device)
    set_context(
        True,
        torch.arange(batch + 1, dtype=torch.int32, device=device) * (k + 1),
        torch.tensor(cumulative_k, dtype=torch.int32, device=device),
        k + 1, max(len(row) + k for row in rows),
        torch.tensor(slots, dtype=torch.int32, device=device),
        block_tables=runner._prepare_draft_block_tables(rows, device=device),
    )
    try:
        hidden = runner.model(inputs, positions)
        logits = runner.model.compute_logits_all(hidden)
        metadata = runner._prepare_draft_sample_metadata(tuple(row for row in rows for _ in range(k + 1)))
        probabilities = runner.sampler.prepare_exact_probabilities(
            logits, metadata[0], top_k_buckets=metadata[1], top_p_plan=metadata[2],
        )
        return probabilities.reshape(batch, k + 1, -1)
    finally:
        reset_context()


@torch.inference_mode()
def execute(runner, plan, seqs):
    reset_context()
    rows, k, route_key = runner._validate_speculative_step_plan(plan, seqs)
    if (len(rows), k) not in runner.speculative_verifier_shapes:
        raise RuntimeError("target verifier route was not warmed")
    try:
        # Unlike shadow mode, successful proposal draws MUST NOT be rewound.
        draft = runner._execute_draft_proposals_validated(rows, k, route_key)
        verifier = (sequential_target_probabilities if any(row.temperature == 0. for row in rows)
                    else target_probabilities)
        p = verifier(runner, rows, draft.proposal_token_ids)
        accepted = runner.speculative_rejection_sampler.accept(
            p[:, :k], draft.proposal_token_ids, draft.q_probabilities,
        )
        # Independent bonus draws; rejected rows simply discard their draw.
        noise = torch.empty_like(p[:, k]).exponential_()
        bonuses = _sample_exponential_race(p[:, k], noise).tolist()
        proposals = draft.proposal_token_ids.tolist()
        counts = accepted.accepted_counts.tolist()
        corrections = accepted.corrective_token_ids.tolist()
        fallbacks = accepted.target_fallback.tolist()
        results = tuple(SpecRowResult(
            row.seq_id, tuple(tokens),
            tuple(tokens[:count]) + (bonus if count == k else correction,),
            count, count == k, int(fallback),
        ) for row, tokens, count, bonus, correction, fallback in zip(
            rows, proposals, counts, bonuses, corrections, fallbacks, strict=True))
        return SpecExecutionResult(plan.cycle_id, plan.workspace_fingerprint, results,
                                   plan.target_query_tokens,
                                   plan.draft_catchup_tokens + plan.draft_query_tokens)
    finally:
        reset_context()


@torch.inference_mode()
def warm_verifier(runner):
    from nanovllm.engine.model_runner import DraftCycleRow

    ready = set()
    max_batch = min(MAX_VERIFY_BATCH, runner.speculative_memory_plan.batch_size)
    max_k = min(MAX_VERIFY_K, runner.speculative_memory_plan.max_effective_k)
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for batch in range(1, max_batch + 1):
        for k in range(1, max_k + 1):
            blocks_per_row = (k + 1 + runner.block_size - 1) // runner.block_size
            if (batch * blocks_per_row > runner.kv_cache.size(2)
                    or batch * (2 * k + 1) > runner.config.max_num_batched_tokens
                    or k + 1 > runner.config.max_model_len):
                continue
            rows = tuple(DraftCycleRow(
                i, (0,), 1, 0, 0,
                tuple(range(i * blocks_per_row, (i + 1) * blocks_per_row)),
                1.0, -1, 1.0,
            ) for i in range(batch))
            proposals = torch.zeros((batch, k), dtype=torch.int64, device=runner.kv_cache.device)
            p = target_probabilities(runner, rows, proposals)
            # Exercise acceptance/bonus entry points without model sampling.
            tokens = p[:, :k].argmax(-1)
            runner.speculative_rejection_sampler.accept(p[:, :k], tokens, p[:, :k])
            _sample_exponential_race(p[:, k], torch.empty_like(p[:, k]).exponential_())
            # Sampler is eager ATen, but pretouch every filtering/recovery family
            # at the same all-query row count to exercise CUDA library scratch.
            logits = p.reshape(batch * (k + 1), -1).log()
            vocab = logits.size(1)
            for temperature, top_k, top_p in ((0., -1, 1.), (.7, 8, 1.),
                                               (.8, -1, .9), (1.1, 8, .8)):
                temperatures = torch.full((logits.size(0),), temperature,
                                          device=logits.device, dtype=torch.float32)
                buckets = ((min(top_k, vocab), None),) if top_k > 0 else ()
                nucleus = ((None, torch.full_like(temperatures, 1. - top_p))
                           if top_p < 1 else None)
                weights = runner.sampler.prepare_exact_probabilities(
                    logits, temperatures, top_k_buckets=buckets, top_p_plan=nucleus)
                del weights
            del logits
            # Equal-law forced correction is only a warmup/test seam; no claim
            # that exact equal laws can naturally reject is made.
            runner.speculative_rejection_sampler.sample_correction(p[:, 0], p[:, 0])
            runner.speculative_rejection_sampler.sample_correction(p[:, 0], p[:, 0].roll(1, -1))
            del p
            greedy_rows = tuple(DraftCycleRow(
                row.seq_id, row.token_ids, row.committed_len, row.target_cached_tokens,
                row.draft_cached_tokens, row.block_table, 0., row.top_k, row.top_p,
            ) for row in rows)
            sequential_target_probabilities(runner, greedy_rows, proposals)
            ready.add((batch, k))
    if not ready:
        raise RuntimeError("configuration has no legal target verifier route")
    torch.cuda.synchronize(runner.kv_cache.device)
    runner._verifier_pretouch_peak_bytes = torch.cuda.max_memory_allocated() - baseline
    if runner._verifier_pretouch_peak_bytes > runner.speculative_memory_plan.reservation_bytes + runner._warmup_transient_bytes:
        raise RuntimeError("target verifier pretouch exceeds the reserved runtime envelope")
    runner.speculative_verifier_shapes = frozenset(ready)
    runner.speculative_verifier_ready = True
