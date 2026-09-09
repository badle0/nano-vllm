"""Matched-resource feasibility analysis for prefill/decode disaggregation.

This tool does not implement serving or infer isolation from a single GPU. It
validates separately collected two-GPU reports and quantifies KV handoff cost.
"""

import argparse
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class KVGeometry:
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2

    @property
    def bytes_per_token(self) -> int:
        for value in (self.layers, self.kv_heads, self.head_dim, self.dtype_bytes):
            if type(value) is not int or value < 1:
                raise ValueError("KV geometry values must be positive integers")
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes


@dataclass(frozen=True, slots=True)
class HandoffDescriptor:
    model_identity: str
    numerical_backend: str
    token_ids: tuple[int, ...]
    processed_tokens: int
    kv_layout: str
    logical_block_order: tuple[int, ...]
    first_token_owner: str

    def validate(self) -> None:
        if not self.model_identity or not self.numerical_backend or not self.kv_layout:
            raise ValueError("handoff identities and KV layout must be present")
        if self.first_token_owner not in {"prefill", "decode"}:
            raise ValueError("first_token_owner must be prefill or decode")
        if type(self.processed_tokens) is not int or not 0 < self.processed_tokens <= len(self.token_ids):
            raise ValueError("processed-token coverage is invalid")
        if any(type(token) is not int or token < 0 for token in self.token_ids):
            raise ValueError("handoff token IDs must be non-negative integers")
        if not self.logical_block_order or len(set(self.logical_block_order)) != len(self.logical_block_order):
            raise ValueError("logical KV block order must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class HandoffProgress:
    """Acknowledgments that make KV ownership transfer explicit and auditable."""

    receiver_allocation_ack: bool = False
    transfer_completion_ack: bool = False
    receiver_install_ack: bool = False
    cancellation_requested: bool = False
    cancellation_ack: bool = False
    retry_count: int = 0
    source_release_ack: bool = False

    def validate(self) -> None:
        for name in (
            "receiver_allocation_ack",
            "transfer_completion_ack",
            "receiver_install_ack",
            "cancellation_requested",
            "cancellation_ack",
            "source_release_ack",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if type(self.retry_count) is not int or self.retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")
        if self.transfer_completion_ack and not self.receiver_allocation_ack:
            raise ValueError("transfer completion requires receiver allocation")
        if self.receiver_install_ack and not self.transfer_completion_ack:
            raise ValueError("receiver install requires transfer completion")
        if self.cancellation_ack and not self.cancellation_requested:
            raise ValueError("cancellation acknowledgment requires a cancellation request")
        if self.retry_count and not self.cancellation_ack:
            raise ValueError("retry requires acknowledgment of the prior cancellation")
        release_safe = self.receiver_install_ack or self.cancellation_ack
        if self.source_release_ack and not release_safe:
            raise ValueError(
                "source release requires receiver install or cancellation acknowledgment"
            )


def transfer_estimate(
    geometry: KVGeometry,
    *,
    prompt_tokens: int,
    bandwidth_gbps: float,
) -> dict:
    if type(prompt_tokens) is not int or prompt_tokens < 1:
        raise ValueError("prompt_tokens must be a positive integer")
    if isinstance(bandwidth_gbps, bool) or not isinstance(bandwidth_gbps, (int, float)):
        raise TypeError("bandwidth_gbps must be a number")
    if not isfinite(bandwidth_gbps) or bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be finite and positive")
    total_bytes = geometry.bytes_per_token * prompt_tokens
    return {
        "bytes_per_token": geometry.bytes_per_token,
        "prompt_tokens": prompt_tokens,
        "total_bytes": total_bytes,
        "total_mib": total_bytes / 2**20,
        "wire_time_ms_at_bandwidth": total_bytes * 8 / (bandwidth_gbps * 1e9) * 1000,
    }


def compare_matched_resources(
    disaggregated: dict,
    colocated: dict,
    *,
    minimum_goodput_improvement_fraction: float = 0.10,
) -> dict:
    required = {
        "gpu_count",
        "duration_s",
        "completed_requests",
        "requests_meeting_both_slos",
        "p95_ttft_ms",
        "p99_itl_ms",
        "max_itl_ms",
        "cost_usd",
    }
    for name, report in (("disaggregated", disaggregated), ("colocated", colocated)):
        missing = required - report.keys()
        if missing:
            raise ValueError(f"{name} report is missing {sorted(missing)}")
        if report["gpu_count"] < 1 or report["duration_s"] <= 0 or report["cost_usd"] <= 0:
            raise ValueError(f"{name} report has invalid resource/time/cost values")
    if disaggregated["gpu_count"] != colocated["gpu_count"]:
        raise ValueError("feasibility comparison requires the same total GPU count")
    if (
        isinstance(minimum_goodput_improvement_fraction, bool)
        or not isinstance(minimum_goodput_improvement_fraction, (int, float))
        or not isfinite(minimum_goodput_improvement_fraction)
        or minimum_goodput_improvement_fraction < 0
    ):
        raise ValueError("minimum goodput improvement must be finite and non-negative")

    def metrics(report):
        goodput = report["requests_meeting_both_slos"] / report["duration_s"]
        return {
            "throughput_rps": report["completed_requests"] / report["duration_s"],
            "latency_constrained_goodput_rps": goodput,
            "goodput_per_dollar": report["requests_meeting_both_slos"] / report["cost_usd"],
            "p95_ttft_ms": report["p95_ttft_ms"],
            "p99_itl_ms": report["p99_itl_ms"],
            "max_itl_ms": report["max_itl_ms"],
        }

    disagg = metrics(disaggregated)
    coloc = metrics(colocated)
    baseline = coloc["latency_constrained_goodput_rps"]
    improvement = (
        None
        if baseline == 0
        else disagg["latency_constrained_goodput_rps"] / baseline - 1
    )
    cost_covered = (
        disagg["goodput_per_dollar"] >= coloc["goodput_per_dollar"]
    )
    latency_not_regressed = all(
        disagg[field] <= coloc[field]
        for field in ("p95_ttft_ms", "p99_itl_ms", "max_itl_ms")
    )
    goodput_gate_passed = (
        improvement is not None
        and improvement >= minimum_goodput_improvement_fraction
    )
    return {
        "matched_gpu_count": disaggregated["gpu_count"],
        "disaggregated": disagg,
        "colocated_replicas": coloc,
        "goodput_improvement_fraction": improvement,
        "minimum_goodput_improvement_fraction": minimum_goodput_improvement_fraction,
        "goodput_gate_passed": goodput_gate_passed,
        "operational_cost_covered": cost_covered,
        "latency_not_regressed": latency_not_regressed,
        "proceed_to_serving_plan": (
            goodput_gate_passed and cost_covered and latency_not_regressed
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--bandwidth-gbps", type=float, default=200.0)
    parser.add_argument("--disaggregated", type=Path)
    parser.add_argument("--colocated", type=Path)
    args = parser.parse_args()
    report = {
        "target_transfer": transfer_estimate(
            KVGeometry(args.layers, args.kv_heads, args.head_dim, args.dtype_bytes),
            prompt_tokens=args.prompt_tokens,
            bandwidth_gbps=args.bandwidth_gbps,
        )
    }
    if (args.disaggregated is None) != (args.colocated is None):
        raise ValueError("provide both matched-resource reports or neither")
    if args.disaggregated is not None:
        report["matched_resource_comparison"] = compare_matched_resources(
            json.loads(args.disaggregated.read_text()),
            json.loads(args.colocated.read_text()),
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
