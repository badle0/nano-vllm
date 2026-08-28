"""Compare fresh speculation-off/on V3 output-control artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INPUT_SCHEMA = "nano-vllm-speculative-v3-output-control-v1"
OUTPUT_SCHEMA = "nano-vllm-speculative-v3-output-control-comparison-v1"
DRAFT_PHASES = (
    "_construct_draft_model",
    "warmup_draft_model",
    "capture_draft_cudagraph",
    "_pretouch_draft_eager_prefill",
    "_pretouch_draft_routes",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Require exact off/on V3 target-output and RNG parity."
    )
    parser.add_argument("--off", required=True)
    parser.add_argument("--on", required=True)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact(path: Path, side: str) -> dict:
    path = path.expanduser().resolve()
    artifact = json.loads(path.read_text(encoding="utf-8"))
    require(artifact.get("schema") == INPUT_SCHEMA, f"invalid {side} schema")
    require(artifact.get("side") == side, f"expected {side} artifact")
    return artifact


def exact_equal(off: dict, on: dict, field: str) -> None:
    require(off.get(field) == on.get(field), f"off/on {field} mismatch")


def write_once(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite output: {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    args = parse_args(argv)
    off_path = Path(args.off).expanduser().resolve()
    on_path = Path(args.on).expanduser().resolve()
    require(off_path != on_path, "off and on artifacts must be distinct files")
    off = load_artifact(off_path, "off")
    on = load_artifact(on_path, "on")

    for field in (
        "mode",
        "seed",
        "model",
        "draft_model_argument",
        "configured_k",
        "configuration",
        "workload",
        "runner_script_sha256",
        "environment",
        "source",
    ):
        exact_equal(off, on, field)

    off_phase_calls = off.get("draft_phase_calls", {})
    require(
        set(off_phase_calls) == set(DRAFT_PHASES),
        "off artifact has an incomplete phase ledger",
    )
    require(
        all(off_phase_calls[name] == 0 for name in DRAFT_PHASES),
        f"off artifact entered a draft phase: {off_phase_calls}",
    )
    require(
        not any(off.get("live_draft_resource_attributes", {}).values()),
        "off artifact retained a draft resource attribute",
    )
    require(
        not off.get("draft_owned_instance_attributes"),
        "off artifact created a draft-owned runner attribute",
    )
    require(not off.get("runtime_draft_calls"), "off artifact ran V3 draft work")

    on_calls = on.get("runtime_draft_calls")
    require(isinstance(on_calls, list) and len(on_calls) == 2, "on artifact needs two V3 intervals")
    require(on_calls[0]["catchup_tokens"] > 0, "on artifact lacks cold catch-up")
    require(on_calls[1]["catchup_tokens"] == 0, "on artifact lacks warm no-catch-up route")
    require(
        all(call.get("rng_neutral") is True for call in on_calls),
        "an on-side V3 interval was not RNG neutral",
    )
    require(
        all(call["rng_before"] == call["rng_after"] for call in on_calls),
        "on-side V3 RNG endpoint evidence is inconsistent",
    )

    exact_equal(off, on, "rng_snapshots")
    exact_equal(off, on, "steps")
    exact_equal(off, on, "authoritative_target_events")
    exact_equal(off, on, "target_token_ids_by_seq")

    retention_eligible = bool(
        off.get("retention_eligible")
        and on.get("retention_eligible")
        and off["source"]["clean"]
        and on["source"]["clean"]
    )
    result = {
        "schema": OUTPUT_SCHEMA,
        "mode": off["mode"],
        "seed": off["seed"],
        "source_head": off["source"]["head"],
        "nanovllm_python_tree_sha256": off["source"][
            "nanovllm_python_tree_sha256"
        ],
        "off_artifact": str(off_path),
        "off_artifact_sha256": file_sha256(off_path),
        "on_artifact": str(on_path),
        "on_artifact_sha256": file_sha256(on_path),
        "retention_eligible": retention_eligible,
        "off_draft_phase_calls_zero": True,
        "off_draft_resources_absent": True,
        "on_real_v3_intervals": len(on_calls),
        "on_cold_and_warm_routes_exercised": True,
        "target_events_exact": True,
        "target_tokens_exact": True,
        "rng_endpoints_exact": True,
        "verdict": "pass",
    }

    if args.output:
        write_once(Path(args.output), result)
    print(
        "PASS: off/on authoritative target events, tokens, and all four "
        f"RNG endpoints match exactly ({off['mode']}); "
        f"retention_eligible={retention_eligible}",
        flush=True,
    )


if __name__ == "__main__":
    main()
