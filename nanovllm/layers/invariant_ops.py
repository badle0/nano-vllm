"""Shape-stable numerical primitives for the opt-in invariant backend.

Each output element has one owner and a fixed reduction order.  These kernels
trade some peak throughput for results that do not depend on the number or
position of other live token rows.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_wn: tl.constexpr, stride_wk: tl.constexpr,
    stride_om: tl.constexpr, stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(0, k, BLOCK_K):
        offsets = start + reduction
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + offsets[None, :] * stride_xk,
            mask=(rows[:, None] < m) & (offsets[None, :] < k),
            other=0.0,
        )
        w = tl.load(
            w_ptr + cols[None, :] * stride_wn + offsets[:, None] * stride_wk,
            mask=(cols[None, :] < n) & (offsets[:, None] < k),
            other=0.0,
        )
        accumulator = tl.dot(x, w, accumulator)
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + cols, mask=cols < n, other=0.0)[None, :]
    tl.store(
        out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,
        accumulator,
        mask=(rows[:, None] < m) & (cols[None, :] < n),
    )


def invariant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``x @ weight.T`` with reduction geometry independent of row count."""

    if x.device.type != "cuda" or weight.device.type != "cuda":
        return torch.nn.functional.linear(x, weight, bias)
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("invariant linear supports BF16, FP16, and FP32 tensors")
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1]).contiguous()
    weight_2d = weight.contiguous()
    m, k = x_2d.shape
    n = weight_2d.shape[0]
    if weight_2d.shape[1] != k:
        raise ValueError("invariant linear received incompatible matrix shapes")
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m == 0:
        return output.reshape(*original_shape[:-1], n)
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 64))
    _linear_kernel[grid](
        x_2d, weight_2d, bias, output,
        m, n, k,
        x_2d.stride(0), x_2d.stride(1),
        weight_2d.stride(0), weight_2d.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    return output.reshape(*original_shape[:-1], n)


@triton.jit
def _rms_kernel(
    x_ptr, residual_ptr, weight_ptr, output_ptr, residual_out_ptr,
    rows: tl.constexpr, width: tl.constexpr,
    stride_x: tl.constexpr, stride_residual: tl.constexpr,
    eps: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < width
    values = tl.load(x_ptr + row * stride_x + columns, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        values += tl.load(
            residual_ptr + row * stride_residual + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(residual_out_ptr + row * width + columns, values, mask=mask)
    variance = tl.sum(values * values, axis=0) / width
    scale = tl.rsqrt(variance + eps)
    weights = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + row * width + columns, values * scale * weights, mask=mask)


def invariant_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
):
    """RMSNorm with one fixed-width reduction program per logical row."""

    if x.device.type != "cuda":
        values = x.float() if residual is None else x.float() + residual.float()
        residual_out = None if residual is None else values.to(x.dtype)
        variance = values.square().mean(dim=-1, keepdim=True)
        output = (values * torch.rsqrt(variance + eps)).to(x.dtype) * weight
        return output if residual is None else (output, residual_out)
    original_shape = x.shape
    width = original_shape[-1]
    x_2d = x.reshape(-1, width).contiguous()
    residual_2d = None if residual is None else residual.reshape(-1, width).contiguous()
    output = torch.empty_like(x_2d)
    residual_out = torch.empty_like(x_2d) if residual is not None else None
    block = triton.next_power_of_2(width)
    _rms_kernel[(x_2d.shape[0],)](
        x_2d, residual_2d, weight, output, residual_out,
        x_2d.shape[0], width,
        x_2d.stride(0), 0 if residual_2d is None else residual_2d.stride(0),
        eps, HAS_RESIDUAL=residual is not None, BLOCK=block,
        num_warps=8 if block >= 4096 else 4,
    )
    output = output.reshape(original_shape)
    if residual_out is None:
        return output
    return output, residual_out.reshape(original_shape)
