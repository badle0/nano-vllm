"""Deterministic memory planning for speculative decoding.

This module deliberately contains no CUDA queries and allocates no tensors.  It
turns model geometry and configured work limits into auditable byte counts that
the model runner can combine with its measured persistent/activation footprint.

The workspace model follows the PR8 correctness-first implementation:

* proposal laws retain FP32 ``q[B, K, V]``;
* verifier laws materialize FP32 ``p[B, K + 1, V]`` at the same time;
* draft and verifier logits, canonical probability transforms, exact top-k and
  top-p workspaces, exponential-race sampling, and FP64 rejection correction
  are priced;
* filtering, probability materialization, and categorical-race subphases are
  priced from their actual tensor lifetimes; mutually exclusive subphases and
  draft, verifier, rejection, and bonus phases are combined with ``max`` rather
  than summed;
* the allocator margin is the larger of 64 MiB and 10 percent of the modeled
  live peak, rounded up to a 2 MiB CUDA allocation boundary.

Library-internal selection/sort scratch and graph-pool/static allocations are
not fully knowable without executing the selected CUDA routes. Until the A100
peak-memory certificate replaces this provisional model, top-p reserves an
additional payload equal to its sort outputs and the diagnostic marks the
remaining quantities as audit-required rather than silently reporting zero.
"""

from dataclasses import dataclass

import torch


FP32_BYTES = 4
FP64_BYTES = 8
INT64_BYTES = 8
BOOL_BYTES = 1
TOP_P_CHUNK_SIZE = 64

ALLOCATOR_MARGIN_MIN_BYTES = 64 * 1024**2
ALLOCATOR_MARGIN_NUMERATOR = 1
ALLOCATOR_MARGIN_DENOMINATOR = 10
ALLOCATOR_MARGIN_ALIGNMENT_BYTES = 2 * 1024**2


class SpeculativeMemoryPlanningError(ValueError):
    """The supplied geometry cannot produce a valid deterministic plan."""


@dataclass(frozen=True, slots=True)
class SpeculativeMemoryPlan:
    """Immutable configured-maximum speculative workspace diagnostic."""

    # This is also the maximum speculative batch the runtime may admit for
    # *every* effective K covered by this plan. A smaller effective K does not
    # license a larger batch without building and reserving a different plan.
    batch_size: int
    configured_k: int
    max_effective_k: int
    vocab_size: int
    draft_rows: int
    verifier_rows: int
    target_logits_itemsize: int
    draft_logits_itemsize: int

    draft_probability_bytes: int
    target_probability_bytes: int
    probability_floor_bytes: int

    draft_logits_bytes: int
    verifier_logits_bytes: int
    draft_transform_bytes: int
    verifier_transform_bytes: int
    draft_top_k_workspace_bytes: int
    verifier_top_k_workspace_bytes: int
    draft_top_p_workspace_bytes: int
    verifier_top_p_workspace_bytes: int
    draft_sampling_workspace_bytes: int
    bonus_sampling_workspace_bytes: int
    rejection_correction_workspace_bytes: int
    metadata_bytes: int

    draft_filter_phase_bytes: int
    draft_softmax_phase_bytes: int
    draft_race_phase_bytes: int
    verifier_filter_phase_bytes: int
    verifier_softmax_phase_bytes: int

    # These cannot be inferred honestly before V2 defines concrete graph keys
    # and runs backend-specific peak measurements. None is intentional:
    # reporting zero would turn an unmeasured quantity into a false claim.
    graph_static_workspace_bytes: int | None
    backend_library_workspace_bytes: int | None
    audit_required_components: tuple[str, ...]

    draft_phase_bytes: int
    verifier_phase_bytes: int
    rejection_phase_bytes: int
    bonus_phase_bytes: int
    modeled_live_peak_bytes: int
    allocator_margin_bytes: int
    reservation_bytes: int


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise SpeculativeMemoryPlanningError(f"{name} must be positive")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise SpeculativeMemoryPlanningError(
            f"{name} must be non-negative"
        )
    return value


def _positive_config_int(config, name: str) -> int:
    try:
        value = getattr(config, name)
    except AttributeError as error:
        raise SpeculativeMemoryPlanningError(
            f"model config is missing {name}"
        ) from error
    return _positive_int(name, value)


def _floating_itemsize(name: str, dtype: torch.dtype) -> int:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if not dtype.is_floating_point:
        raise SpeculativeMemoryPlanningError(
            f"{name} must be a floating-point torch.dtype"
        )
    return dtype.itemsize


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _round_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def kv_cache_block_bytes(
    hf_config,
    *,
    block_size: int,
    tensor_parallel_size: int = 1,
    dtype: torch.dtype | None = None,
) -> int:
    """Return exact bytes for one model's K+V cache at one logical block.

    The returned number covers both key and value tensors for every local layer
    and KV head.  For tensor parallelism the total KV-head count must divide the
    world size exactly.  PR8 speculative execution is TP1, but keeping this
    helper general prevents hidden arithmetic assumptions in target-only code.
    """

    block_size = _positive_int("block_size", block_size)
    tensor_parallel_size = _positive_int(
        "tensor_parallel_size", tensor_parallel_size
    )
    num_layers = _positive_config_int(hf_config, "num_hidden_layers")
    num_kv_heads = _positive_config_int(hf_config, "num_key_value_heads")
    if num_kv_heads % tensor_parallel_size:
        raise SpeculativeMemoryPlanningError(
            "num_key_value_heads must be divisible by tensor_parallel_size"
        )

    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        hidden_size = _positive_config_int(hf_config, "hidden_size")
        num_attention_heads = _positive_config_int(
            hf_config, "num_attention_heads"
        )
        if hidden_size % num_attention_heads:
            raise SpeculativeMemoryPlanningError(
                "hidden_size must be divisible by num_attention_heads when "
                "head_dim is absent"
            )
        head_dim = hidden_size // num_attention_heads
    else:
        head_dim = _positive_int("head_dim", head_dim)

    if dtype is None:
        try:
            dtype = hf_config.dtype
        except AttributeError as error:
            raise SpeculativeMemoryPlanningError(
                "model config is missing dtype"
            ) from error
    itemsize = _floating_itemsize("dtype", dtype)
    local_kv_heads = num_kv_heads // tensor_parallel_size
    return (
        2
        * num_layers
        * block_size
        * local_kv_heads
        * head_dim
        * itemsize
    )


def _top_p_workspace_bytes(
    *,
    rows: int,
    vocab_size: int,
    logits_itemsize: int,
) -> int:
    """Conservative private workspace for the current exact top-p transform.

    Worst-case heterogeneous routing first gathers active logits for all rows.
    Sorting is chunked exactly like ``Sampler.filter_top_p``.  Per live chunk we
    price FP32 scaled and sorted values, int64 sort indices, FP32 softmax and
    cumulative-probability buffers, two boolean masks, and an additional
    value+index payload for opaque backend sort scratch.
    """

    dense_elements = rows * vocab_size
    chunk_elements = min(rows, TOP_P_CHUNK_SIZE) * vocab_size
    active_logits_copy = dense_elements * logits_itemsize
    explicit_chunk_payload = chunk_elements * (
        # scaled, sorted, softmax temporary, cumulative probabilities
        4 * FP32_BYTES
        + INT64_BYTES
        + 2 * BOOL_BYTES
    )
    sort_library_scratch = chunk_elements * (FP32_BYTES + INT64_BYTES)
    return active_logits_copy + explicit_chunk_payload + sort_library_scratch


def _top_k_workspace_bytes(
    *,
    rows: int,
    vocab_size: int,
    logits_itemsize: int,
) -> int:
    """Worst-case private workspace for the current exact top-k transform.

    A heterogeneous active-row plan materializes an active-logits copy. The
    configured-max sampler signature must also cover top_k == vocab_size:
    torch.topk returns dense values and int64 indices, followed by one threshold
    value per row. Backend-internal selection workspace remains an
    audit-required quantity rather than being mislabeled as zero.
    """

    dense_elements = rows * vocab_size
    active_logits_copy = dense_elements * logits_itemsize
    values = dense_elements * logits_itemsize
    indices = dense_elements * INT64_BYTES
    thresholds = rows * logits_itemsize
    return active_logits_copy + values + indices + thresholds


def _categorical_race_workspace_bytes(rows: int, vocab_size: int) -> int:
    """FP32 noise plus FP64 weights/noise/scores used by V1's exact race."""

    elements = rows * vocab_size
    return elements * (FP32_BYTES + 3 * FP64_BYTES)


def _rejection_correction_workspace_bytes(batch_size: int, vocab_size: int) -> int:
    """Worst-case all-row FP64 correction workspace.

    The V1 reference path retains two gathered FP32 rows and FP32 correction
    noise, plus FP64 normalized target, normalized draft, residual, normalized
    correction law, one indexing/normalization temporary, converted noise, and
    race scores.
    """

    elements = batch_size * vocab_size
    return elements * (3 * FP32_BYTES + 7 * FP64_BYTES)


def _metadata_bytes(batch_size: int, configured_k: int, verifier_rows: int) -> int:
    """Worst-case device metadata for proposal, routing, and acceptance.

    Per proposal position this counts token IDs, two FP64 row masses, two FP32
    selected weights, two FP64 normalized weights, an FP64 acceptance
    probability, FP32 uniform, boolean accept/prefix masks, and two int64 prefix
    intermediates.  Per sampler row it counts temperature/top-p cutoff plus
    top-k/top-p row indices.  Per sequence it counts accepted/corrective IDs,
    rejected row/position indices, and three boolean result masks.
    """

    proposal_elements = batch_size * configured_k
    proposal_metadata = proposal_elements * (
        INT64_BYTES
        + 2 * FP64_BYTES
        + 2 * FP32_BYTES
        + 2 * FP64_BYTES
        + FP64_BYTES
        + FP32_BYTES
        + 2 * BOOL_BYTES
        + 2 * INT64_BYTES
    )
    sampler_rows = batch_size + verifier_rows
    sampler_metadata = sampler_rows * (
        2 * FP32_BYTES + 2 * INT64_BYTES
    )
    per_sequence = batch_size * (4 * INT64_BYTES + 3 * BOOL_BYTES)
    return proposal_metadata + sampler_metadata + per_sequence


def _allocator_margin(live_peak_bytes: int) -> int:
    fractional = _ceil_div(
        live_peak_bytes * ALLOCATOR_MARGIN_NUMERATOR,
        ALLOCATOR_MARGIN_DENOMINATOR,
    )
    raw_margin = max(ALLOCATOR_MARGIN_MIN_BYTES, fractional)
    return _round_up(raw_margin, ALLOCATOR_MARGIN_ALIGNMENT_BYTES)


def plan_speculative_workspace(
    *,
    vocab_size: int,
    configured_k: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    target_logits_dtype: torch.dtype,
    draft_logits_dtype: torch.dtype,
) -> SpeculativeMemoryPlan:
    """Build the configured-maximum speculative workspace reservation.

    ``B`` is intentionally internal policy rather than a public tuning knob.
    It is a fixed maximum-admission cap for every runtime ``effective_k`` that
    this plan covers; clipping K never permits the router to expand B:

    The configured K is first clipped to the largest value that a one-row cycle
    could actually execute: min(configured_k, max_num_batched_tokens - 1,
    max_model_len - 1). B is then min(max_num_seqs,
    max_num_batched_tokens // (max_effective_k + 1)).

    If either global limit leaves max_effective_k == 0, no speculative route can
    run. The returned diagnostic has B=0 and a zero reservation; the engine may
    remain correct by routing all work through ordinary decode.
    """

    vocab_size = _positive_int("vocab_size", vocab_size)
    configured_k = _positive_int("configured_k", configured_k)
    max_num_seqs = _positive_int("max_num_seqs", max_num_seqs)
    max_num_batched_tokens = _positive_int(
        "max_num_batched_tokens", max_num_batched_tokens
    )
    max_model_len = _positive_int("max_model_len", max_model_len)
    target_itemsize = _floating_itemsize(
        "target_logits_dtype", target_logits_dtype
    )
    draft_itemsize = _floating_itemsize(
        "draft_logits_dtype", draft_logits_dtype
    )

    max_effective_k = min(
        configured_k,
        max_num_batched_tokens - 1,
        max_model_len - 1,
    )
    audit_required_components = (
        "model activation and attention-library workspace",
        "CUDA graph-pool and route-specific static buffers",
        "top-k/top-p backend-internal selection/sort workspace beyond the "
        "documented payload proxy",
        "measured CUDA allocator fragmentation beyond the fixed margin",
    )
    if max_effective_k == 0:
        return SpeculativeMemoryPlan(
            batch_size=0,
            configured_k=configured_k,
            max_effective_k=0,
            vocab_size=vocab_size,
            draft_rows=0,
            verifier_rows=0,
            target_logits_itemsize=target_itemsize,
            draft_logits_itemsize=draft_itemsize,
            draft_probability_bytes=0,
            target_probability_bytes=0,
            probability_floor_bytes=0,
            draft_logits_bytes=0,
            verifier_logits_bytes=0,
            draft_transform_bytes=0,
            verifier_transform_bytes=0,
            draft_top_k_workspace_bytes=0,
            verifier_top_k_workspace_bytes=0,
            draft_top_p_workspace_bytes=0,
            verifier_top_p_workspace_bytes=0,
            draft_sampling_workspace_bytes=0,
            bonus_sampling_workspace_bytes=0,
            rejection_correction_workspace_bytes=0,
            metadata_bytes=0,
            draft_filter_phase_bytes=0,
            draft_softmax_phase_bytes=0,
            draft_race_phase_bytes=0,
            verifier_filter_phase_bytes=0,
            verifier_softmax_phase_bytes=0,
            graph_static_workspace_bytes=None,
            backend_library_workspace_bytes=None,
            audit_required_components=audit_required_components,
            draft_phase_bytes=0,
            verifier_phase_bytes=0,
            rejection_phase_bytes=0,
            bonus_phase_bytes=0,
            modeled_live_peak_bytes=0,
            allocator_margin_bytes=0,
            reservation_bytes=0,
        )

    batch_size = min(
        max_num_seqs,
        max_num_batched_tokens // (max_effective_k + 1),
    )
    draft_rows = batch_size
    verifier_rows = batch_size * (max_effective_k + 1)

    draft_probability_bytes = (
        batch_size * max_effective_k * vocab_size * FP32_BYTES
    )
    target_probability_bytes = verifier_rows * vocab_size * FP32_BYTES
    probability_floor_bytes = (
        draft_probability_bytes + target_probability_bytes
    )

    draft_logits_bytes = draft_rows * vocab_size * draft_itemsize
    verifier_logits_bytes = verifier_rows * vocab_size * target_itemsize

    # Canonical preparation privately clones logits in their production dtype,
    # then creates an FP32 scaled workspace.  The returned FP32 probabilities
    # are already counted in q/p above.
    draft_transform_bytes = draft_rows * vocab_size * (
        draft_itemsize + FP32_BYTES
    )
    verifier_transform_bytes = verifier_rows * vocab_size * (
        target_itemsize + FP32_BYTES
    )
    draft_top_k_workspace_bytes = _top_k_workspace_bytes(
        rows=draft_rows,
        vocab_size=vocab_size,
        logits_itemsize=draft_itemsize,
    )
    verifier_top_k_workspace_bytes = _top_k_workspace_bytes(
        rows=verifier_rows,
        vocab_size=vocab_size,
        logits_itemsize=target_itemsize,
    )
    draft_top_p_workspace_bytes = _top_p_workspace_bytes(
        rows=draft_rows,
        vocab_size=vocab_size,
        logits_itemsize=draft_itemsize,
    )
    verifier_top_p_workspace_bytes = _top_p_workspace_bytes(
        rows=verifier_rows,
        vocab_size=vocab_size,
        logits_itemsize=target_itemsize,
    )
    draft_sampling_workspace_bytes = _categorical_race_workspace_bytes(
        draft_rows, vocab_size
    )
    bonus_sampling_workspace_bytes = _categorical_race_workspace_bytes(
        batch_size, vocab_size
    )
    rejection_correction_workspace_bytes = (
        _rejection_correction_workspace_bytes(batch_size, vocab_size)
    )
    metadata_bytes = _metadata_bytes(
        batch_size, max_effective_k, verifier_rows
    )

    # q is preallocated at its configured maximum and remains live for every
    # draft subphase. The current probability row must be written/retained in
    # that allocation without an additional full-row result copy; an execution
    # strategy that cannot satisfy this lifetime contract needs a larger plan.
    # Filtering, softmax, and race are sequential and their private workspaces
    # therefore combine by maximum, not sum.
    draft_filtered_clone_bytes = draft_logits_bytes
    draft_scaled_logits_bytes = batch_size * vocab_size * FP32_BYTES
    draft_filter_phase_bytes = (
        draft_probability_bytes
        + draft_logits_bytes
        + draft_filtered_clone_bytes
        + max(draft_top_k_workspace_bytes, draft_top_p_workspace_bytes)
        + metadata_bytes
    )
    draft_softmax_phase_bytes = (
        draft_probability_bytes
        + draft_logits_bytes
        + draft_filtered_clone_bytes
        + draft_scaled_logits_bytes
        + metadata_bytes
    )
    draft_race_phase_bytes = (
        draft_probability_bytes
        + draft_logits_bytes
        + draft_sampling_workspace_bytes
        + metadata_bytes
    )
    draft_phase_bytes = max(
        draft_filter_phase_bytes,
        draft_softmax_phase_bytes,
        draft_race_phase_bytes,
    )

    # Verification retains all q rows. p and the final scaled logits are born
    # only after filtering, so they do not overlap the filter-private top-k/top-p
    # peak. The original verifier logits and filtered clone persist in both.
    verifier_filtered_clone_bytes = verifier_logits_bytes
    verifier_scaled_logits_bytes = target_probability_bytes
    verifier_filter_phase_bytes = (
        draft_probability_bytes
        + verifier_logits_bytes
        + verifier_filtered_clone_bytes
        + max(verifier_top_k_workspace_bytes, verifier_top_p_workspace_bytes)
        + metadata_bytes
    )
    verifier_softmax_phase_bytes = (
        draft_probability_bytes
        + verifier_logits_bytes
        + verifier_filtered_clone_bytes
        + verifier_scaled_logits_bytes
        + target_probability_bytes
        + metadata_bytes
    )
    verifier_phase_bytes = max(
        verifier_filter_phase_bytes,
        verifier_softmax_phase_bytes,
    )
    rejection_phase_bytes = (
        probability_floor_bytes
        + rejection_correction_workspace_bytes
        + metadata_bytes
    )
    bonus_phase_bytes = (
        probability_floor_bytes
        + bonus_sampling_workspace_bytes
        + metadata_bytes
    )
    modeled_live_peak_bytes = max(
        draft_phase_bytes,
        verifier_phase_bytes,
        rejection_phase_bytes,
        bonus_phase_bytes,
    )
    allocator_margin_bytes = _allocator_margin(modeled_live_peak_bytes)
    reservation_bytes = modeled_live_peak_bytes + allocator_margin_bytes

    return SpeculativeMemoryPlan(
        batch_size=batch_size,
        configured_k=configured_k,
        max_effective_k=max_effective_k,
        vocab_size=vocab_size,
        draft_rows=draft_rows,
        verifier_rows=verifier_rows,
        target_logits_itemsize=target_itemsize,
        draft_logits_itemsize=draft_itemsize,
        draft_probability_bytes=draft_probability_bytes,
        target_probability_bytes=target_probability_bytes,
        probability_floor_bytes=probability_floor_bytes,
        draft_logits_bytes=draft_logits_bytes,
        verifier_logits_bytes=verifier_logits_bytes,
        draft_transform_bytes=draft_transform_bytes,
        verifier_transform_bytes=verifier_transform_bytes,
        draft_top_k_workspace_bytes=draft_top_k_workspace_bytes,
        verifier_top_k_workspace_bytes=verifier_top_k_workspace_bytes,
        draft_top_p_workspace_bytes=draft_top_p_workspace_bytes,
        verifier_top_p_workspace_bytes=verifier_top_p_workspace_bytes,
        draft_sampling_workspace_bytes=draft_sampling_workspace_bytes,
        bonus_sampling_workspace_bytes=bonus_sampling_workspace_bytes,
        rejection_correction_workspace_bytes=(
            rejection_correction_workspace_bytes
        ),
        metadata_bytes=metadata_bytes,
        draft_filter_phase_bytes=draft_filter_phase_bytes,
        draft_softmax_phase_bytes=draft_softmax_phase_bytes,
        draft_race_phase_bytes=draft_race_phase_bytes,
        verifier_filter_phase_bytes=verifier_filter_phase_bytes,
        verifier_softmax_phase_bytes=verifier_softmax_phase_bytes,
        graph_static_workspace_bytes=None,
        backend_library_workspace_bytes=None,
        audit_required_components=audit_required_components,
        draft_phase_bytes=draft_phase_bytes,
        verifier_phase_bytes=verifier_phase_bytes,
        rejection_phase_bytes=rejection_phase_bytes,
        bonus_phase_bytes=bonus_phase_bytes,
        modeled_live_peak_bytes=modeled_live_peak_bytes,
        allocator_margin_bytes=allocator_margin_bytes,
        reservation_bytes=reservation_bytes,
    )


def speculative_route_fits_plan(
    plan: SpeculativeMemoryPlan,
    *,
    batch_size: int,
    effective_k: int,
) -> bool:
    """Return whether a positive speculative route fits a reserved plan.

    ``plan.batch_size`` is fixed at configured-plan time. The scheduler may
    reduce ``effective_k`` for token/model/request headroom, but it must not use
    the newly available token budget to admit more sequences: that larger batch
    has a different correction/sampling peak and is not covered here. Zero B or
    K describes ordinary-decode fallback and therefore returns ``False``.

    Invalid types or negative geometry are programmer/configuration errors and
    raise typed exceptions. Capacity misses are normal routing decisions and
    return ``False``.
    """

    if not isinstance(plan, SpeculativeMemoryPlan):
        raise TypeError("plan must be a SpeculativeMemoryPlan")
    batch_size = _nonnegative_int("batch_size", batch_size)
    effective_k = _nonnegative_int("effective_k", effective_k)
    if batch_size == 0 or effective_k == 0:
        return False
    return (
        batch_size <= plan.batch_size
        and effective_k <= plan.max_effective_k
        and batch_size * effective_k
        <= plan.batch_size * plan.max_effective_k
        and batch_size * (effective_k + 1) <= plan.verifier_rows
    )


__all__ = [
    "ALLOCATOR_MARGIN_ALIGNMENT_BYTES",
    "ALLOCATOR_MARGIN_MIN_BYTES",
    "SpeculativeMemoryPlan",
    "SpeculativeMemoryPlanningError",
    "kv_cache_block_bytes",
    "plan_speculative_workspace",
    "speculative_route_fits_plan",
]
