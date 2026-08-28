"""Compare paired zero/NaN V3 cache-neutrality A100 artifacts."""

import argparse
import json
from pathlib import Path

import torch


SCHEMA = "nano-vllm-speculative-v3-cache-neutrality-v1"
LOGIT_ATOL = 0.03125
LOGIT_RTOL = 0.01


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("zero_json")
    parser.add_argument("nan_json")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def load_json(path, expected_fill):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(value.get("schema") == SCHEMA, f"unexpected schema in {path}")
    require(
        value.get("draft_cache_fill") == expected_fill,
        f"unexpected cache fill in {path}",
    )
    return value


def cycle_oracle(value):
    return {
        "route": value["route"],
        "effective_k": value["effective_k"],
        "catchup_tokens": value["catchup_tokens"],
        "proposal_input_positions": value["proposal_input_positions"],
        "proposed_token_ids": value["proposed_token_ids"],
        "target_token_ids": value["target_token_ids"],
        "graph_decode_steps": value["graph_decode_steps"],
        "eager_decode_steps": value["eager_decode_steps"],
        "probability_hashes": [
            record["probabilities_sha256"]
            for record in value["sampler_records"]
        ],
        "rng_before_draft": value["rng_before_draft"],
        "rng_after_draft": value["rng_after_draft"],
    }


def artifact_oracle(value):
    return {
        "mode": value["mode"],
        "boundary": cycle_oracle(value["boundary"]),
        "shared_cold": cycle_oracle(value["shared_prefix"]["cold"]),
        "shared_hit": cycle_oracle(
            value["shared_prefix"]["shared_prefix"]
        ),
        "all_logits_finite": value["all_logits_finite"],
        "all_probabilities_finite": value["all_probabilities_finite"],
    }


def load_tensors(value):
    payload = torch.load(
        value["tensor_artifact"],
        map_location="cpu",
        weights_only=True,
    )
    require(payload.get("schema") == SCHEMA, "tensor artifact schema drifted")
    require(payload.get("mode") == value["mode"], "tensor mode drifted")
    require(
        payload.get("draft_cache_fill") == value["draft_cache_fill"],
        "tensor fill mode drifted",
    )
    return payload["records"]


def compare_tensor_records(zero_records, nan_records):
    require(len(zero_records) == len(nan_records), "tensor record count drifted")
    comparisons = []
    for index, (zero, nan) in enumerate(zip(zero_records, nan_records, strict=True)):
        zero_logits = zero["logits"]
        nan_logits = nan["logits"]
        require(zero_logits.shape == nan_logits.shape, "logit shape drifted")
        difference = (zero_logits - nan_logits).abs()
        tolerance = LOGIT_ATOL + LOGIT_RTOL * nan_logits.abs()
        require(
            bool((difference <= tolerance).all()),
            f"logit record {index} exceeds the BF16 graph numerical contract",
        )
        require(
            torch.equal(zero["probabilities"], nan["probabilities"]),
            f"probability record {index} differs",
        )
        denominator = nan_logits.abs().clamp_min(torch.finfo(torch.float32).tiny)
        comparisons.append(
            {
                "index": index,
                "shape": list(zero_logits.shape),
                "bitwise_equal": bool(torch.equal(zero_logits, nan_logits)),
                "max_abs_difference": float(difference.max().item()),
                "max_relative_difference": float(
                    (difference / denominator).max().item()
                ),
                "probabilities_bitwise_equal": True,
            }
        )
    return comparisons


def main(argv=None):
    args = parse_args(argv)
    zero = load_json(args.zero_json, "zero")
    nan = load_json(args.nan_json, "nan")
    require(
        artifact_oracle(zero) == artifact_oracle(nan),
        "zero/NaN host route, proposal, target, probability, or RNG oracle differs",
    )
    comparisons = compare_tensor_records(
        load_tensors(zero),
        load_tensors(nan),
    )
    output = {
        "schema": "nano-vllm-speculative-v3-cache-neutrality-comparison-v1",
        "mode": zero["mode"],
        "zero_json": str(Path(args.zero_json).resolve()),
        "nan_json": str(Path(args.nan_json).resolve()),
        "logit_atol": LOGIT_ATOL,
        "logit_rtol": LOGIT_RTOL,
        "host_oracles_equal": True,
        "tensor_comparisons": comparisons,
    }
    payload = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
