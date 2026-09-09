"""Independent FP64 references for invariant numerical primitives."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from benchmarks.invariant_qualification import _implementation_sha256
from nanovllm.layers.attention import Attention
from nanovllm.layers.invariant_ops import invariant_linear, invariant_rms_norm
from nanovllm.utils.context import reset_context, set_context


SEED = 20260908


def _hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _errors(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    actual_float = actual.float()
    reference_float = reference.float()
    difference = (actual_float - reference_float).abs()
    reference_peak = reference_float.abs().max().clamp_min(1e-12)
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "reference_max_abs": float(reference_peak.item()),
        "relative_peak": float((difference.max() / reference_peak).item()),
        "actual_sha256": _hash(actual),
        "reference_fp32_sha256": _hash(reference_float),
    }


def linear_checks() -> list[dict]:
    reports = []
    for index, (rows, input_width, output_width) in enumerate(
        ((1, 96, 130), (7, 1024, 1536))
    ):
        torch.manual_seed(SEED + index)
        x = torch.randn(
            rows, input_width, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            output_width, input_width, device="cuda", dtype=torch.bfloat16
        )
        actual = invariant_linear(x, weight)
        reference = x.double() @ weight.double().T
        metrics = _errors(actual, reference)
        reports.append(
            {
                "shape": [rows, input_width, output_width],
                **metrics,
                "threshold_relative_peak": 0.005,
                "passed": metrics["relative_peak"] < 0.005,
            }
        )
    return reports


def rms_checks() -> list[dict]:
    reports = []
    for index, (rows, width) in enumerate(((1, 128), (7, 1024), (2, 2048))):
        torch.manual_seed(SEED + 100 + index)
        x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
        actual, actual_residual = invariant_rms_norm(
            x, weight, 1e-6, residual
        )
        values = x.double() + residual.double()
        reference = (
            values
            * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)
            * weight.double()
        )
        metrics = _errors(actual, reference)
        residual_metrics = _errors(actual_residual, values)
        reports.append(
            {
                "shape": [rows, width],
                **metrics,
                "residual": residual_metrics,
                "threshold_relative_peak": 0.01,
                "threshold_residual_max_abs": 0.016,
                "passed": (
                    metrics["relative_peak"] < 0.01
                    and residual_metrics["max_abs"] <= 0.016
                ),
            }
        )
    return reports


def attention_checks() -> list[dict]:
    torch.manual_seed(SEED + 200)
    block_size, num_blocks = 256, 32
    num_kv_heads, num_heads, head_dim = 2, 4, 32
    attention = Attention(
        num_heads, head_dim, head_dim**-0.5, num_kv_heads
    ).cuda()
    attention.k_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    attention.v_cache = torch.randn_like(attention.k_cache)
    block_tables = torch.stack(
        (torch.arange(16), torch.arange(16, 32))
    ).to(device="cuda", dtype=torch.int32)
    lengths = (1, 255, 256, 257, 4096)
    owners = (0, 0, 0, 1, 1)
    queries = torch.randn(
        len(lengths),
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    set_context(
        True,
        block_tables=block_tables,
        query_sequence_ids=owners,
        query_context_lengths=lengths,
    )
    reports = []
    try:
        actual = attention._invariant_paged_attention(queries)
        for index, (owner, length) in enumerate(
            zip(owners, lengths, strict=True)
        ):
            ids = block_tables[
                owner, :math.ceil(length / block_size)
            ].long()
            keys = attention.k_cache.index_select(0, ids).flatten(0, 1)[:length]
            values = attention.v_cache.index_select(0, ids).flatten(0, 1)[:length]
            groups = num_heads // num_kv_heads
            keys = keys[:, :, None, :].expand(
                -1, -1, groups, -1
            ).reshape(length, num_heads, head_dim)
            values = values[:, :, None, :].expand(
                -1, -1, groups, -1
            ).reshape(length, num_heads, head_dim)
            scores = (
                queries[index].double()[:, None, :]
                * keys.double().permute(1, 0, 2)
            ).sum(dim=-1) * attention.scale
            probabilities = torch.softmax(scores, dim=-1)
            reference = (
                probabilities[:, :, None]
                * values.double().permute(1, 0, 2)
            ).sum(dim=1)
            metrics = _errors(actual[index], reference)
            reports.append(
                {
                    "context_length": length,
                    **metrics,
                    "threshold_rtol": 0.01,
                    "threshold_atol": 0.001,
                    "passed": bool(
                        torch.allclose(
                            actual[index].float(),
                            reference.float(),
                            rtol=0.01,
                            atol=0.001,
                        )
                    ),
                }
            )
    finally:
        reset_context()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    linear = linear_checks()
    rms = rms_checks()
    attention = attention_checks()
    passed = all(
        item["passed"] for group in (linear, rms, attention) for item in group
    )
    root = Path(__file__).resolve().parents[1]
    report = {
        "kind": "invariant_fp64_primitive_reference_v1",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": SEED,
        "implementation_sha256": _implementation_sha256(root),
        "linear": linear,
        "rmsnorm": rms,
        "attention": attention,
        "passed": passed,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": passed,
                "max_linear_relative_peak": max(
                    item["relative_peak"] for item in linear
                ),
                "max_rms_relative_peak": max(
                    item["relative_peak"] for item in rms
                ),
                "max_attention_relative_peak": max(
                    item["relative_peak"] for item in attention
                ),
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit("independent FP64 qualification failed")


if __name__ == "__main__":
    main()

