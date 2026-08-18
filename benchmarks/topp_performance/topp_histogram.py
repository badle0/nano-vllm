"""Development-only BF16 counting-sort primitive for exact top-p work.

This module is intentionally not wired into :class:`Sampler` yet.  It provides
the first independently testable building block for replacing the expensive
full-vocabulary value sort while retaining the exact FP32 sorted-value tensor
used by the current Transformers-compatible top-p implementation.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl


_NUM_BF16_KEYS = 1 << 16


@triton.jit
def _bf16_histogram_kernel(
    logits_ptr,
    counts_ptr,
    num_columns,
    row_stride,
    NUM_KEYS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    columns = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = columns < num_columns
    values = tl.load(logits_ptr + row * row_stride + columns, mask=mask)
    bits = values.to(tl.uint16, bitcast=True).to(tl.int32)
    negative = (bits & 0x8000) != 0
    ordered_keys = tl.where(negative, (~bits) & 0xFFFF, bits ^ 0x8000)
    tl.atomic_add(
        counts_ptr + row * NUM_KEYS + ordered_keys,
        1,
        mask=mask,
    )


@triton.jit
def _bf16_expand_counts_kernel(
    counts_ptr,
    starts_ptr,
    ordered_values_ptr,
    output_ptr,
    num_columns,
    NUM_KEYS: tl.constexpr,
    BLOCK_KEYS: tl.constexpr,
    BLOCK_OFFSETS: tl.constexpr,
):
    row = tl.program_id(0)
    key_block = tl.program_id(1)
    keys = key_block * BLOCK_KEYS + tl.arange(0, BLOCK_KEYS)
    key_mask = keys < NUM_KEYS

    counts = tl.load(counts_ptr + row * NUM_KEYS + keys, mask=key_mask, other=0)
    starts = tl.load(starts_ptr + row * NUM_KEYS + keys, mask=key_mask, other=0)
    values = tl.load(ordered_values_ptr + keys, mask=key_mask, other=0.0)
    max_count = tl.max(tl.where(key_mask, counts, 0), axis=0)
    offsets = tl.arange(0, BLOCK_OFFSETS)

    offset_base = 0
    while offset_base < max_count:
        range_offsets = offset_base + offsets[None, :]
        active = key_mask[:, None] & (range_offsets < counts[:, None])
        tl.store(
            output_ptr
            + row * num_columns
            + starts[:, None]
            + range_offsets,
            values[:, None],
            mask=active,
        )
        offset_base += BLOCK_OFFSETS


@lru_cache(maxsize=None)
def _ordered_bf16_values(device_index: int) -> torch.Tensor:
    device = torch.device("cuda", device_index)
    ordered_keys = torch.arange(_NUM_BF16_KEYS, dtype=torch.int32, device=device)
    negative_bits = (~ordered_keys) & 0xFFFF
    positive_bits = ordered_keys ^ 0x8000
    bits = torch.where(ordered_keys < 0x8000, negative_bits, positive_bits)
    return bits.to(torch.int16).view(torch.bfloat16).float()


@torch.inference_mode()
def sort_bf16_scaled_values(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Return the ascending FP32 scaled-value tensor without a radix sort.

    The output is intended to equal ``torch.sort(logits.float() / temp).values``
    bit-for-bit for finite CUDA BF16 logits.  Equal values may be written in any
    order because their expanded FP32 bits are identical.  Index recovery for a
    cutoff tie is deliberately outside this primitive and must be proven before
    it is used by the production sampler.
    """

    if logits.device.type != "cuda" or temperatures.device.type != "cuda":
        raise ValueError("logits and temperatures must be CUDA tensors")
    if logits.dtype is not torch.bfloat16:
        raise TypeError("logits must have dtype torch.bfloat16")
    if temperatures.dtype is not torch.float32:
        raise TypeError("temperatures must have dtype torch.float32")
    if logits.ndim != 2:
        raise ValueError("logits must be a two-dimensional tensor")
    if temperatures.shape != (logits.size(0),):
        raise ValueError("temperatures must have one value per logit row")
    if not logits.is_contiguous():
        raise ValueError("logits must be contiguous")
    if not temperatures.is_contiguous():
        raise ValueError("temperatures must be contiguous")
    if logits.size(0) == 0 or logits.size(1) == 0:
        raise ValueError("logits must have non-empty batch and vocabulary dimensions")
    if logits.numel() >= 2**31:
        raise ValueError("the prototype supports fewer than 2**31 logits per call")

    rows, columns = logits.shape
    counts = torch.zeros(
        (rows, _NUM_BF16_KEYS),
        dtype=torch.int32,
        device=logits.device,
    )
    block_size = 256
    grid = (rows, triton.cdiv(columns, block_size))
    _bf16_histogram_kernel[grid](
        logits,
        counts,
        columns,
        logits.stride(0),
        NUM_KEYS=_NUM_BF16_KEYS,
        BLOCK_SIZE=block_size,
    )

    starts = torch.cumsum(counts, dim=-1, dtype=torch.int32)
    starts.sub_(counts)
    ordered_values = _ordered_bf16_values(logits.device.index)
    output = torch.empty((rows, columns), dtype=torch.float32, device=logits.device)
    block_keys = 256
    block_offsets = 16
    expansion_grid = (rows, triton.cdiv(_NUM_BF16_KEYS, block_keys))
    _bf16_expand_counts_kernel[expansion_grid](
        counts,
        starts,
        ordered_values,
        output,
        columns,
        NUM_KEYS=_NUM_BF16_KEYS,
        BLOCK_KEYS=block_keys,
        BLOCK_OFFSETS=block_offsets,
        num_warps=8,
    )
    output.div_(temperatures.unsqueeze(1))
    return output
