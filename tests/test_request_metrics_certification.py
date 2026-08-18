import json
import statistics
import sys
from pathlib import Path

import pytest


BENCHMARK_DIR = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "request_metrics"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from certification_common import (  # noqa: E402
    BENCHMARK_NAME,
    SCHEMA_VERSION,
    derive_observation_seed,
    exclusive_json_dump,
    sha256_file,
    summarize_observations,
)
from validate_certification import (  # noqa: E402
    recompute_pair,
    validate_manifest,
)


def make_payload(pair_id, side, order, source, model, harness, started):
    position = order.split("-").index(side) + 1
    seed = 1000 + pair_id
    observations = []
    elapsed_values = [1.1, 1.0] if side == "baseline" else [1.08, 0.99]
    for repetition, elapsed in enumerate(elapsed_values):
        observations.append(
            {
                "repetition": repetition,
                "cold": repetition == 0,
                "seed": derive_observation_seed(seed, pair_id, 1, 32, repetition),
                "elapsed_s": elapsed,
                "output_tokens": 32,
                "output_tokens_per_s": 32 / elapsed,
                "token_ids_sha256": "a" * 64,
                "resources": {
                    "host_rss_before_kib": 100,
                    "host_rss_after_kib": 101,
                    "host_peak_rss_kib": 102,
                    "cuda_allocated_before_bytes": 200,
                    "cuda_reserved_before_bytes": 300,
                    "cuda_peak_allocated_bytes": 210,
                    "cuda_peak_reserved_bytes": 310,
                },
            }
        )
    argv = ["synthetic-harness", str(pair_id), side]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "identity": {
            "run_id": "synthetic",
            "pair_id": pair_id,
            "side": side,
            "order": order,
            "order_position": position,
            "label": side,
            "seed": seed,
        },
        "invocation": {
            "argv": argv,
            "cwd": "/synthetic",
            "started_utc": f"2026-08-18T00:00:{started:02d}+00:00",
            "finished_utc": f"2026-08-18T00:00:{started + 1:02d}+00:00",
            "harness_path": str(harness),
            "harness_sha256": sha256_file(harness),
        },
        "source": source,
        "model": model,
        "environment": {
            "python": "3.12",
            "python_executable": "/venv/main/bin/python",
            "platform": "synthetic-linux",
            "torch": "2.10",
            "torch_cuda": "12.8",
            "cudnn": 90000,
            "transformers": "5.14",
            "flash_attn": "2.8",
            "gpu": "synthetic-a100",
            "gpu_capability": [8, 0],
            "gpu_total_memory_bytes": 40_000_000_000,
            "gpu_multiprocessor_count": 108,
            "visible_cuda_devices": 1,
            "nvidia_driver_versions": ["570.0"],
        },
        "configuration": {
            "batches": [1],
            "output_lengths": [32],
            "prompt_length": 8,
            "repetitions": 2,
            "cold_observations": 1,
        },
        "cases": {
            "b1_o32": {
                "batch": 1,
                "output_length": 32,
                "prompt_sha256": "b" * 64,
                "observations": observations,
                "summary": summarize_observations(observations),
            }
        },
    }


def test_validator_recomputes_balanced_pairs_and_rejects_tampering(tmp_path):
    harness = tmp_path / "harness.py"
    runner = tmp_path / "runner.py"
    harness.write_text("# synthetic harness\n")
    runner.write_text("# synthetic runner\n")
    sources = {
        "baseline": {"root": "/base", "head": "1" * 40, "clean": True},
        "repair": {"root": "/repair", "head": "2" * 40, "clean": True},
    }
    model = {"root": "/model", "fingerprint_sha256": "c" * 64}
    pair_runs = []
    payloads = {}
    for pair_id, order in ((1, "baseline-repair"), (2, "repair-baseline")):
        pair = {
            "pair_id": pair_id,
            "seed": 1000 + pair_id,
            "order": order,
            "runs": {},
        }
        for side in order.split("-"):
            position = order.split("-").index(side) + 1
            payload = make_payload(
                pair_id,
                side,
                order,
                sources[side],
                model,
                harness,
                started=(pair_id - 1) * 4 + (position - 1) * 2,
            )
            path = tmp_path / f"pair{pair_id}_{side}.json"
            path.write_text(json.dumps(payload, sort_keys=True) + "\n")
            payloads[(pair_id, side)] = payload
            pair["runs"][side] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "command": ["python", *payload["invocation"]["argv"]],
            }
        pair["comparisons"] = recompute_pair(
            payloads[(pair_id, "baseline")], payloads[(pair_id, "repair")]
        )
        pair_runs.append(pair)

    efficiencies = [pair["comparisons"]["b1_o32"]["efficiency"] for pair in pair_runs]
    aggregate = {
        "b1_o32": {
            "pair_efficiencies": efficiencies,
            "median_efficiency": statistics.median(efficiencies),
            "minimum_efficiency": min(efficiencies),
            "median_gate_pass": True,
            "individual_gate_pass": True,
        }
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "artifact_kind": "balanced_pair_manifest",
        "harness": {"path": str(harness), "sha256": sha256_file(harness)},
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "sources": sources,
        "model": model,
        "protocol": {
            "pairs": 2,
            "median_efficiency_floor": 0.98,
            "individual_efficiency_floor": 0.95,
        },
        "pair_runs": pair_runs,
        "aggregate": aggregate,
        "gate_pass": True,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    assert validate_manifest(manifest_path) == {
        "pairs": 2,
        "raw_files": 4,
        "cases": 1,
        "gate_pass": True,
    }

    raw_path = tmp_path / "pair1_baseline.json"
    tampered = json.loads(raw_path.read_text())
    tampered["cases"]["b1_o32"]["observations"][1]["output_tokens_per_s"] = 1.0
    raw_path.write_text(json.dumps(tampered, sort_keys=True) + "\n")
    manifest["pair_runs"][0]["runs"]["baseline"]["sha256"] = sha256_file(raw_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="throughput"):
        validate_manifest(manifest_path)


def test_exclusive_output_refuses_overwrite(tmp_path):
    output = tmp_path / "artifact.json"
    exclusive_json_dump(output, {"value": 1})
    with pytest.raises(ValueError, match="refusing to overwrite"):
        exclusive_json_dump(output, {"value": 2})
