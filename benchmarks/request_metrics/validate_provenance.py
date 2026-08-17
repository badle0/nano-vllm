#!/usr/bin/env python3
"""Validate request-metrics raw evidence against its provenance manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("provenance.json")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    expected_environment = {
        "torch": manifest["environment"]["pytorch"]["value"],
        "cuda": manifest["environment"]["cuda_wheel"]["value"],
        "gpu": manifest["environment"]["gpu"]["value"],
    }

    checked = 0
    for run in manifest["runs"]:
        for side, code_key in (("baseline", "baseline"), ("repair", "repair")):
            artifact = run[side]
            path = ROOT / artifact["path"]
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != artifact["sha256"]:
                raise SystemExit(
                    f"SHA-256 mismatch for {path}: {digest} != {artifact['sha256']}"
                )

            result = json.loads(payload)
            expected_label = manifest["code_under_test"][code_key]["label"]
            if result["label"] != expected_label:
                raise SystemExit(
                    f"label mismatch for {path}: {result['label']} != {expected_label}"
                )
            if result["seed"] != run["seed"]:
                raise SystemExit(
                    f"seed mismatch for {path}: {result['seed']} != {run['seed']}"
                )
            for field, expected in expected_environment.items():
                if result[field] != expected:
                    raise SystemExit(
                        f"{field} mismatch for {path}: {result[field]} != {expected}"
                    )
            checked += 1

    harness = ROOT / manifest["benchmark"]["harness"]["path"]
    harness_digest = hashlib.sha256(harness.read_bytes()).hexdigest()
    expected_harness_digest = manifest["benchmark"]["harness"]["sha256"]
    if harness_digest != expected_harness_digest:
        raise SystemExit(
            f"harness SHA-256 mismatch: {harness_digest} != {expected_harness_digest}"
        )

    print(f"validated {checked} raw files and the historical harness")


if __name__ == "__main__":
    main()
