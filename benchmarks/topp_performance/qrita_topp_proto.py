# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 nano-vllm top-p performance contributors
"""Development-only BF16 top-p prototype inspired by vLLM's Qrita kernel.

This is a focused adaptation of the standalone top-p path in:

    https://github.com/vllm-project/vllm/blob/
    e0e5a7fb2808504ba86c94f7b379e38496002fd0/
    vllm/v1/sample/ops/topk_topp_triton.py

The upstream implementation is Apache-2.0 and is based on *Qrita:
High-performance Top-k and Top-p Algorithm for GPUs using Pivot-based
Truncation and Selection* by Park et al. (https://arxiv.org/abs/2602.01518).

Differences from upstream are intentional and keep this experiment bounded:

* only standalone top-p is included;
* input and retained output are raw BF16 logits;
* per-row FP32 temperatures are applied internally using reciprocal multiply
  (which is intentionally included in the measured semantic delta);
* the outlier heuristic is specialized to the requested ``top_p == 0.9``;
* workspace ownership is explicit so benchmarks can report scratch memory; and
* this module is not imported by :mod:`nanovllm.layers.sampler`.

Like upstream Qrita, this is a sorting-free pivot algorithm. It does not claim
bit-for-bit support equivalence with the repository's Transformers-compatible
full-sort filter; the companion benchmark measures that semantic delta.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 8192
_BLOCK_SIZE_TRUNC = 4096

# vLLM's pinned 200-entry table has entry 1.135 at int(0.9 * 200) == 180,
# followed by its sigma -= abs(sigma) * 0.25 conservative adjustment.
_OUTLIER_SIGMA_P90 = 0.85125


@triton.jit
def _update_min_larger_stats(
    data,
    above_mask,
    min_larger,
    num_min_larger,
    SENTINEL: tl.constexpr,
):
    tile_min = tl.min(tl.where(above_mask, data, SENTINEL))
    tile_equal = above_mask & (tl.abs(data - tile_min) < 1e-9)
    tile_count = tl.sum(tile_equal)
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < 1e-9
    num_min_larger = tl.where(
        is_new,
        tile_count,
        num_min_larger + tile_count * is_same,
    )
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def _qrita_topp_p90_bf16_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    TEMPERATURES,
    TOP_PS,
    BUFFER,
    BATCH_SIZE,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
    OUTLIER_SIGMA: tl.constexpr,
):
    """Qrita-style standalone top-p selection specialized for p=0.9."""

    num_tiles: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_trunc_tiles: tl.constexpr = (
        VOCAB_SIZE + BLOCK_SIZE_TRUNC - 1
    ) // BLOCK_SIZE_TRUNC
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_id in tl.range(program_id, BATCH_SIZE, num_programs):
        logits_row = LOGITS + row_id.to(tl.int64) * LOGITS_STRIDE_0
        buffer_row = BUFFER + program_id * VOCAB_SIZE
        inverse_temperature = 1.0 / tl.load(TEMPERATURES + row_id)
        top_p = tl.load(TOP_PS + row_id)

        # Qrita zeroth pass: estimate the distribution from one tile.
        offsets = tl.arange(0, BLOCK_SIZE)
        valid = offsets < VOCAB_SIZE
        raw_sample = tl.load(logits_row + offsets, mask=valid, other=-float("inf"))
        sample = raw_sample.to(tl.float32) * inverse_temperature
        finite = (sample > -float("inf")) & valid
        num_finite = tl.sum(finite)
        finite_sample = tl.where(finite, sample, 0.0)
        mean = tl.where(num_finite > 0, tl.sum(finite_sample) / num_finite, 0.0)
        square_mean = tl.where(
            num_finite > 0,
            tl.sum(finite_sample * finite_sample) / num_finite,
            0.0,
        )
        standard_deviation = tl.sqrt(
            tl.maximum(square_mean - mean * mean, 0.0)
        )
        max_sample = mean + standard_deviation * 10.0
        outlier_pivot = mean + standard_deviation * OUTLIER_SIGMA

        max_logit = -float("inf")
        min_logit = float("inf")
        sum_exp_logits = 0.0

        # First pass: softmax normalization and search bounds.
        for tile in range(0, num_tiles):
            tile_offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            tile_valid = tile_offsets < VOCAB_SIZE
            raw_logits = tl.load(
                logits_row + tile_offsets,
                mask=tile_valid,
                other=-float("inf"),
            )
            scaled_logits = raw_logits.to(tl.float32) * inverse_temperature
            max_logit = tl.maximum(max_logit, tl.max(scaled_logits))
            finite_logits = tl.where(
                scaled_logits > -float("inf"), scaled_logits, float("inf")
            )
            min_logit = tl.minimum(min_logit, tl.min(finite_logits))
            exponentials = tl.exp(scaled_logits - max_sample)
            sum_exp_logits += tl.sum(tl.where(tile_valid, exponentials, 0.0))

        min_logit = tl.minimum(min_logit, max_logit)
        outlier_probability = (
            tl.exp(outlier_pivot - max_sample) / sum_exp_logits
        )
        sum_outlier_probabilities = 0.0
        num_outliers = tl.zeros((), dtype=tl.uint32)

        # Second pass: compact the high-probability tail to program-local scratch.
        for tile in range(0, num_tiles):
            tile_offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            tile_valid = tile_offsets < VOCAB_SIZE
            raw_logits = tl.load(
                logits_row + tile_offsets,
                mask=tile_valid,
                other=-float("inf"),
            )
            scaled_logits = raw_logits.to(tl.float32) * inverse_temperature
            probabilities = tl.exp(scaled_logits - max_sample) / sum_exp_logits
            is_outlier = (probabilities > outlier_probability) & tile_valid
            sum_outlier_probabilities += tl.sum(
                tl.where(is_outlier, probabilities, 0.0)
            )
            compact_positions = tl.cast(
                tl.cumsum(is_outlier) - 1 + num_outliers, tl.int32
            )
            num_outliers += tl.sum(is_outlier)
            tl.store(
                buffer_row + compact_positions,
                probabilities,
                mask=is_outlier,
            )

        max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
        min_range = tl.exp(min_logit - max_sample) / sum_exp_logits
        search_range = tl.cast(num_outliers, tl.int32)
        search_iterations = tl.cast(
            (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
            tl.int32,
        )

        # Preserve Qrita's full-vocabulary fallback for non-Gaussian inputs.
        if sum_outlier_probabilities > top_p:
            min_range = outlier_probability
        else:
            for tile in range(0, num_tiles):
                tile_offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                tile_valid = tile_offsets < VOCAB_SIZE
                raw_logits = tl.load(
                    logits_row + tile_offsets,
                    mask=tile_valid,
                    other=-float("inf"),
                )
                scaled_logits = raw_logits.to(tl.float32) * inverse_temperature
                probabilities = (
                    tl.exp(scaled_logits - max_sample) / sum_exp_logits
                )
                tl.store(buffer_row + tile_offsets, probabilities, mask=tile_valid)
            search_range = VOCAB_SIZE
            search_iterations = num_trunc_tiles

        probability_pivot = 1.0
        min_larger_probability = 1.0
        num_min_larger = tl.zeros((), dtype=tl.uint32)
        pivot_sum = 0.0
        search_count = 0
        found_pivot = 0

        # Third pass: probability-mass pivot search, replacing a full sort.
        while found_pivot == 0:
            candidate_pivot = (max_range - min_range) * 0.5 + min_range
            candidate_sum = 0.0
            candidate_min_larger = 1.0
            candidate_num_min_larger = tl.zeros((), dtype=tl.uint32)

            for tile in range(0, search_iterations):
                tile_offsets = tile * BLOCK_SIZE_TRUNC + tl.arange(
                    0, BLOCK_SIZE_TRUNC
                )
                tile_valid = tile_offsets < search_range
                probabilities = tl.load(
                    buffer_row + tile_offsets,
                    mask=tile_valid,
                    other=0.0,
                )
                above = probabilities > candidate_pivot
                candidate_sum += tl.sum(tl.where(above, probabilities, 0.0))
                candidate_min_larger, candidate_num_min_larger = (
                    _update_min_larger_stats(
                        probabilities,
                        above,
                        candidate_min_larger,
                        candidate_num_min_larger,
                        SENTINEL=1.0,
                    )
                )

            if candidate_sum >= top_p and (
                candidate_sum
                - candidate_min_larger * candidate_num_min_larger
                < top_p
            ):
                probability_pivot = candidate_pivot
                min_larger_probability = candidate_min_larger
                num_min_larger = candidate_num_min_larger
                pivot_sum = candidate_sum
                found_pivot = 1

            if candidate_sum > top_p:
                min_range = candidate_pivot
            elif candidate_sum < top_p:
                max_range = candidate_pivot

            search_count += 1
            if max_range - min_range < 1e-9 or search_count >= 18:
                probability_pivot = (max_range + min_range) * 0.5
                min_larger_probability = candidate_min_larger
                num_min_larger = candidate_num_min_larger
                pivot_sum = candidate_sum
                found_pivot = 1

        duplicate_logit = (
            tl.log(min_larger_probability * sum_exp_logits) + max_sample
        )
        num_duplicate_logits = num_min_larger
        num_duplicate_keep = num_duplicate_logits - tl.cast(
            (pivot_sum - top_p) / min_larger_probability, tl.uint32
        )
        num_duplicates_kept = tl.zeros((), dtype=tl.uint32)
        final_pivot = tl.log(probability_pivot * sum_exp_logits) + max_sample

        # Final pass: decide support in scaled space but retain raw BF16 values.
        if final_pivot < max_logit:
            for tile in range(0, num_tiles):
                tile_offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                tile_valid = tile_offsets < VOCAB_SIZE
                raw_logits = tl.load(
                    logits_row + tile_offsets,
                    mask=tile_valid,
                    other=-float("inf"),
                )
                scaled_logits = raw_logits.to(tl.float32) * inverse_temperature
                keep = (scaled_logits > final_pivot) & tile_valid

                if num_duplicate_keep < num_duplicate_logits:
                    duplicate = (
                        tl.abs(scaled_logits - duplicate_logit) < 1e-9
                    ) & tile_valid
                    duplicate_rank = tl.cumsum(duplicate) + num_duplicates_kept
                    keep_duplicate = duplicate & (
                        duplicate_rank <= num_duplicate_keep
                    )
                    remove_duplicate = duplicate & ~keep_duplicate
                    num_duplicates_kept += tl.sum(keep_duplicate)
                    keep = keep & ~remove_duplicate

                output = tl.where(keep, raw_logits, -float("inf"))
                tl.store(logits_row + tile_offsets, output, mask=tile_valid)


def allocate_qrita_workspace(logits: torch.Tensor) -> torch.Tensor:
    """Allocate explicit Qrita scratch for ``logits`` on its CUDA device."""

    if logits.device.type != "cuda" or logits.ndim != 2:
        raise ValueError("logits must be a two-dimensional CUDA tensor")
    num_programs = min(
        torch.cuda.get_device_properties(logits.device).multi_processor_count,
        logits.size(0),
    )
    return torch.empty(
        (num_programs, logits.size(1)),
        dtype=torch.float32,
        device=logits.device,
    )


@torch.inference_mode()
def qrita_top_p_bf16_raw_(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Mask raw BF16 ``logits`` in place using the p=0.9 prototype.

    The caller must supply p=0.9 for every row. Avoiding a value check here
    prevents a device synchronization inside the latency measurement.
    """

    if logits.device.type != "cuda" or logits.dtype is not torch.bfloat16:
        raise TypeError("logits must be a CUDA torch.bfloat16 tensor")
    if logits.ndim != 2 or logits.stride(1) != 1:
        raise ValueError("logits must be two-dimensional and row-contiguous")
    rows, columns = logits.shape
    if rows == 0 or columns == 0:
        return logits
    for name, tensor in (("temperatures", temperatures), ("top_ps", top_ps)):
        if (
            tensor.device != logits.device
            or tensor.dtype is not torch.float32
            or tensor.shape != (rows,)
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA float32 with shape {(rows,)}"
            )
    num_programs = min(
        torch.cuda.get_device_properties(logits.device).multi_processor_count,
        rows,
    )
    if (
        workspace.device != logits.device
        or workspace.dtype is not torch.float32
        or workspace.shape != (num_programs, columns)
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace must match allocate_qrita_workspace(logits)")

    _qrita_topp_p90_bf16_kernel[(num_programs,)](
        logits,
        logits.stride(0),
        temperatures,
        top_ps,
        workspace,
        rows,
        VOCAB_SIZE=columns,
        BLOCK_SIZE=_BLOCK_SIZE,
        BLOCK_SIZE_TRUNC=_BLOCK_SIZE_TRUNC,
        OUTLIER_SIGMA=_OUTLIER_SIGMA_P90,
        num_warps=8,
    )
    return logits


__all__ = ["allocate_qrita_workspace", "qrita_top_p_bf16_raw_"]
