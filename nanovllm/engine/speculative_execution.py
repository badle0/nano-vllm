"""Bounded speculative execution with reusable workspace and parallel verification."""
from hashlib import sha256

import torch

from nanovllm.engine.speculative_result import SpecExecutionResult, SpecRowResult
from nanovllm.layers.sampler import _sample_exponential_race
from nanovllm.utils.context import set_context, reset_context

MAX_VERIFY_BATCH = 4
MAX_VERIFY_K = 4
VERIFIER_ROUTE_SCHEMA = "target-verifier-v2-invariant-greedy"


def _tensor_readiness_identity(value):
    if not isinstance(value, torch.Tensor):
        return None
    storage = value.untyped_storage()
    return (
        id(value),
        storage.data_ptr(),
        storage.nbytes(),
        value.storage_offset(),
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        str(value.device),
    )


def _mapping_readiness_identity(value):
    if not isinstance(value, dict):
        return (type(value).__qualname__, id(value))
    return tuple(
        (
            repr(key),
            id(item),
            _tensor_readiness_identity(item),
        )
        for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
    )


def _verifier_readiness_fingerprint(runner) -> str:
    """Bind verifier readiness to its backend and exact resource ownership."""

    registry = getattr(runner, "draft_route_registry", None)
    payload = (
        VERIFIER_ROUTE_SCHEMA,
        id(registry),
        getattr(registry, "plan_fingerprint", None),
        getattr(runner, "numerical_mode", "fast"),
        bool(getattr(runner, "enforce_eager", False)),
        tuple(
            (
                name,
                _mapping_readiness_identity(getattr(runner, name, None)),
            )
            for name in ("graphs", "varlen_graphs", "draft_graphs")
        ),
        tuple(
            (
                name,
                _mapping_readiness_identity(getattr(runner, name, None)),
            )
            for name in ("graph_vars", "varlen_vars", "draft_graph_vars")
        ),
        tuple(
            (
                name,
                _tensor_readiness_identity(getattr(runner, name, None)),
            )
            for name in (
                "kv_cache",
                "draft_kv_cache",
                "_spec_q_rows",
                "_spec_proposal_ids",
                "_spec_target_probability_rows",
                "_spec_bonus_noise",
                "_spec_result_rows",
            )
        ),
    )
    return sha256(repr(payload).encode("utf-8")).hexdigest()


def _verifier_workspaces_ready(runner) -> bool:
    """Validate persistent shapes and exclusive storage without a GPU sync."""

    plan = getattr(runner, "speculative_memory_plan", None)
    cache = getattr(runner, "kv_cache", None)
    if plan is None or not isinstance(cache, torch.Tensor):
        return False
    batch = min(MAX_VERIFY_BATCH, plan.batch_size)
    k = min(MAX_VERIFY_K, plan.max_effective_k)
    vocab = plan.vocab_size
    specs = (
        ("_spec_q_rows", torch.float32, 2, batch * k, vocab),
        ("_spec_proposal_ids", torch.int64, 1, batch * k, None),
        (
            "_spec_target_probability_rows",
            torch.float32,
            2,
            batch * (k + 1),
            vocab,
        ),
        ("_spec_bonus_noise", torch.float32, 2, batch, vocab),
        ("_spec_result_rows", torch.int64, 2, batch, k + 3),
    )
    tensors = []
    for name, dtype, ndim, rows, columns in specs:
        tensor = getattr(runner, name, None)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != cache.device
            or tensor.dtype != dtype
            or tensor.ndim != ndim
            or tensor.shape[0] < rows
            or (columns is not None and tensor.shape[1] < columns)
            or not tensor.is_contiguous()
        ):
            return False
        tensors.append(tensor)
    return not any(
        torch._C._overlaps(left, right)
        for index, left in enumerate(tensors)
        for right in tensors[index + 1:]
    )


def verifier_route_ready(runner, batch: int, k: int) -> bool:
    """Return whether an exact verifier route still matches warmup state."""

    return bool(
        getattr(runner, "speculative_verifier_ready", False)
        and (batch, k) in getattr(runner, "speculative_verifier_shapes", ())
        and getattr(runner, "speculative_verifier_fingerprint", None)
        == _verifier_readiness_fingerprint(runner)
        and _verifier_workspaces_ready(runner)
    )


def _probability_rows(runner, row_count, vocab_size, device):
    storage = getattr(runner, "_spec_target_probability_rows", None)
    if (
        isinstance(storage, torch.Tensor)
        and storage.device == device
        and storage.dtype == torch.float32
        and storage.shape[0] >= row_count
        and storage.shape[1] == vocab_size
    ):
        return storage[:row_count]
    return torch.empty((row_count, vocab_size), device=device, dtype=torch.float32)


def _prepare_probabilities(
    runner,
    logits,
    metadata,
    probabilities_out,
    *,
    all_greedy,
):
    trusted = getattr(
        runner.sampler, "prepare_exact_probabilities_trusted", None
    )
    if trusted is None:
        return runner.sampler.prepare_exact_probabilities(
            logits,
            metadata[0],
            top_k_buckets=metadata[1],
            top_p_plan=metadata[2],
            probabilities_out=probabilities_out,
        )
    return trusted(
        logits,
        metadata[0],
        top_k_buckets=metadata[1],
        top_p_plan=metadata[2],
        probabilities_out=probabilities_out,
        all_greedy=all_greedy,
    )


@torch.inference_mode()
def sequential_target_probabilities(runner, rows, proposals):
    """Compatibility lane for fast-mode batches containing greedy rows."""

    batch, k = proposals.shape
    device = runner.kv_cache.device
    vocab_size = runner.config.hf_config.vocab_size
    storage = _probability_rows(
        runner, (k + 1) * batch, vocab_size, device
    ).view(k + 1, batch, vocab_size)
    metadata = runner._prepare_draft_sample_metadata(rows)
    all_greedy = all(row.temperature == 0.0 for row in rows)
    try:
        for step in range(k + 1):
            with torch.inference_mode(False):
                inputs = (
                    torch.tensor(
                        [row.last_token for row in rows],
                        device=device,
                        dtype=torch.int64,
                    )
                    if step == 0
                    else proposals[:, step - 1].clone()
                )
                inputs, positions = runner._prepare_draft_decode(
                    rows, inputs, step
                )
            logits = runner.run_model(inputs, positions, False)
            _prepare_probabilities(
                runner,
                logits,
                metadata,
                storage[step],
                all_greedy=all_greedy,
            )
        return storage.permute(1, 0, 2)
    finally:
        reset_context()


@torch.inference_mode()
def target_logits(runner, rows, proposals):
    """Compute all K proposal laws and the bonus law in one causal target pass."""

    batch, k = proposals.shape
    device = runner.kv_cache.device
    positions, slots, cumulative_k = [], [], [0]
    for row in rows:
        for position in range(len(row) - 1, len(row) + k):
            positions.append(position)
            slots.append(
                row.block_table[position // runner.block_size] * runner.block_size
                + position % runner.block_size
            )
        cumulative_k.append(cumulative_k[-1] + len(row) + k)
    tail = torch.tensor(
        [row.last_token for row in rows], dtype=torch.int64, device=device
    )
    inputs = torch.cat((tail[:, None], proposals), dim=1).flatten()
    positions = torch.tensor(positions, dtype=torch.int64, device=device)
    set_context(
        True,
        torch.arange(batch + 1, dtype=torch.int32, device=device) * (k + 1),
        torch.tensor(cumulative_k, dtype=torch.int32, device=device),
        k + 1,
        max(len(row) + k for row in rows),
        torch.tensor(slots, dtype=torch.int32, device=device),
        block_tables=runner._prepare_draft_block_tables(rows, device=device),
        query_sequence_ids=tuple(
            row for row in range(batch) for _ in range(k + 1)
        ),
        query_context_lengths=tuple(
            len(row) + step for row in rows for step in range(k + 1)
        ),
    )
    try:
        run_all_positions = getattr(runner, "run_model_all_positions", None)
        logits = (
            run_all_positions(inputs, positions)
            if callable(run_all_positions)
            else runner.model.compute_logits_all(
                runner.model(inputs, positions)
            )
        )
        return logits.reshape(batch, k + 1, -1)
    finally:
        reset_context()


@torch.inference_mode()
def target_probabilities(runner, rows, proposals):
    """Return canonical p rows using one parallel target execution."""

    logits = target_logits(runner, rows, proposals)
    batch, positions, vocab_size = logits.shape
    flat_logits = logits.reshape(batch * positions, vocab_size)
    repeated_rows = tuple(row for row in rows for _ in range(positions))
    metadata = runner._prepare_draft_sample_metadata(repeated_rows)
    flat_probabilities = _probability_rows(
        runner,
        batch * positions,
        vocab_size,
        logits.device,
    )
    _prepare_probabilities(
        runner,
        flat_logits,
        metadata,
        flat_probabilities,
        all_greedy=all(row.temperature == 0.0 for row in rows),
    )
    return flat_probabilities.view(batch, positions, vocab_size)


def _host_results(runner, rows, proposals, counts, terminal_tokens, fallbacks):
    """Use one device-to-host transfer for every cycle result field."""

    batch, k = proposals.shape
    storage = getattr(runner, "_spec_result_rows", None)
    if (
        not isinstance(storage, torch.Tensor)
        or storage.device != proposals.device
        or storage.dtype != torch.int64
        or storage.ndim != 2
        or storage.shape[0] < batch
        or storage.shape[1] < k + 3
        or not storage.is_contiguous()
    ):
        raise RuntimeError("speculative result workspace is unavailable")
    packed_device = storage[:batch, :k + 3]
    packed_device[:, :k].copy_(proposals)
    packed_device[:, k].copy_(counts)
    packed_device[:, k + 1].copy_(terminal_tokens)
    packed_device[:, k + 2].copy_(fallbacks)
    packed = packed_device.cpu().tolist()
    results = []
    for row, values in zip(rows, packed, strict=True):
        proposed = tuple(values[:k])
        count, terminal, fallback = values[k:]
        results.append(
            SpecRowResult(
                row.seq_id,
                proposed,
                proposed[:count] + (terminal,),
                count,
                count == k,
                fallback,
            )
        )
    return tuple(results)


@torch.inference_mode()
def execute(runner, plan, seqs):
    reset_context()
    rows, k, route_key = runner._validate_speculative_step_plan(plan, seqs)
    if not verifier_route_ready(runner, len(rows), k):
        raise RuntimeError("target verifier route was not warmed")
    try:
        draft = runner._execute_draft_proposals_validated(rows, k, route_key)
        all_greedy = all(row.temperature == 0.0 for row in rows)
        invariant_greedy = (
            all_greedy
            and getattr(runner, "numerical_mode", "fast") == "invariant"
        )
        if invariant_greedy:
            target_token_ids = target_logits(
                runner, rows, draft.proposal_token_ids
            ).argmax(dim=-1)
            matches = target_token_ids[:, :k] == draft.proposal_token_ids
            counts = matches.to(torch.int64).cumprod(dim=1).sum(dim=1)
            terminal = target_token_ids.gather(1, counts[:, None]).squeeze(1)
            fallbacks = torch.zeros_like(counts, dtype=torch.bool)
        else:
            verifier = (
                sequential_target_probabilities
                if any(row.temperature == 0.0 for row in rows)
                and getattr(runner, "numerical_mode", "fast") != "invariant"
                else target_probabilities
            )
            p = verifier(runner, rows, draft.proposal_token_ids)
            trusted_accept = getattr(
                runner.speculative_rejection_sampler, "accept_trusted", None
            )
            accepted = (
                trusted_accept(
                    p[:, :k],
                    draft.proposal_token_ids,
                    draft.q_probabilities,
                )
                if trusted_accept is not None
                else runner.speculative_rejection_sampler.accept(
                    p[:, :k],
                    draft.proposal_token_ids,
                    draft.q_probabilities,
                )
            )
            noise = getattr(runner, "_spec_bonus_noise", None)
            if not isinstance(noise, torch.Tensor) or noise.shape[0] < len(rows):
                noise = torch.empty_like(p[:, k])
            else:
                noise = noise[: len(rows)]
            noise.exponential_(1)
            bonuses = _sample_exponential_race(p[:, k], noise)
            counts = accepted.accepted_counts
            terminal = torch.where(
                counts == k, bonuses, accepted.corrective_token_ids
            )
            fallbacks = accepted.target_fallback
        results = _host_results(
            runner,
            rows,
            draft.proposal_token_ids,
            counts,
            terminal,
            fallbacks,
        )
        return SpecExecutionResult(
            plan.cycle_id,
            plan.workspace_fingerprint,
            results,
            plan.target_query_tokens,
            plan.draft_catchup_tokens + plan.draft_query_tokens,
        )
    finally:
        reset_context()


@torch.inference_mode()
def warm_verifier(runner):
    from nanovllm.engine.model_runner import DraftCycleRow

    # Re-warm is transactional: a failure must never leave an earlier
    # certificate available for a partially replaced resource set.
    runner.speculative_verifier_ready = False
    runner.speculative_verifier_shapes = frozenset()
    runner.speculative_verifier_fingerprint = None
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
    if not _verifier_workspaces_ready(runner):
        raise RuntimeError("target verifier workspaces failed readiness validation")
    runner.speculative_verifier_fingerprint = _verifier_readiness_fingerprint(runner)
    runner.speculative_verifier_ready = True
