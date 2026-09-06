"""CPU-only qualification and tamper checks for the experimental V7 archive."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks/speculative_v7/evidence/2026-09-06-a100-v7-89829e6"


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks/speculative_v7" / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


V7 = module("v7_archive", "validate_retained_evidence.py")
ANALYSIS = module("v7_analysis", "analyze.py")


def test_archive_passes_without_gpu_or_model():
    result = V7.validate(EVIDENCE)
    assert result["runs"] == 32
    assert result["samples"] == 1308
    assert result["cold_cycles"] == 271


def test_untrusted_manifest_fails(tmp_path):
    (tmp_path / "manifest.json").write_bytes((EVIDENCE / "manifest.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="untrusted"):
        V7.validate(tmp_path)


def test_sealing_refuses_overwrite(tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        V7.seal(tmp_path, tmp_path)


@pytest.mark.parametrize("kind", ["matrix", "duplicate", "source", "harness", "capacity", "gpu", "tokens", "metrics", "fallback", "work", "time", "roof", "pair", "contended"])
def test_benchmark_semantic_tampering_is_rejected(kind):
    value = V7.load_json((EVIDENCE / "primary-p0-on.json").read_bytes())
    row = value["records"][0]
    metrics = row["outputs"][0]["metrics"]
    if kind == "matrix": value["records"].pop()
    elif kind == "duplicate": value["records"][-1] = copy.deepcopy(row)
    elif kind == "source": value["source_sha256"].pop(next(iter(value["source_sha256"])))
    elif kind == "harness": value["harness_sha256"] = "0" * 64
    elif kind == "capacity": value["config"]["num_kvcache_blocks"] = 128
    elif kind == "gpu": value["gpu"] = "CPU"
    elif kind == "tokens": row["outputs"][0]["token_ids"].pop()
    elif kind == "metrics": metrics["num_completion_tokens"] += 1
    elif kind == "fallback": metrics["spec_residual_numerical_fallbacks"] = 1
    elif kind == "work": metrics["spec_target_verification_positions"] += 1
    elif kind == "time": row["seconds"] = -1
    elif kind == "roof": value["calibration"]["copy_bytes"] //= 2
    elif kind == "pair": row["pair"] = 4
    elif kind == "contended": row["hardware_before"]["gpu_processes"].append("999999")
    with pytest.raises(ValueError):
        V7.check_benchmark(value, "primary-p0-on")


@pytest.mark.parametrize("kind", ["headline", "phase", "source", "storage", "count"])
def test_phase_semantic_tampering_is_rejected(kind):
    value = V7.load_json((EVIDENCE / "phases-graph.json").read_bytes())
    if kind == "headline": value["headline"] = True
    elif kind == "phase": value["cycles"][0]["phases"].pop("draft")
    elif kind == "source": value["source_sha256"] = {}
    elif kind == "storage": value["weight_ledger"]["target"]["unique_storage_bytes"] += 2
    elif kind == "count": value["cells"][0]["cycles"] += 1
    with pytest.raises(ValueError):
        V7.check_phases(value, "phases-graph")


def test_pair_order_cannot_be_relabelled():
    values = {role: V7.load_json((EVIDENCE / f"{role}.json").read_bytes())
              for role in V7.ROLES if role.startswith(("primary-", "regression-", "graph-"))}
    values["primary-p1-on"]["process_started_ns"] = values["primary-p1-off"]["samples_finished_ns"] + 1
    with pytest.raises(ValueError, match="ordering"):
        V7.cross_checks(values)


def test_exact_bootstrap_is_deterministic_and_uses_pairs():
    assert ANALYSIS.interval([2.] * 5) == [2., 2.]
    assert ANALYSIS.interval([1., 2., 3., 4., 5.]) == ANALYSIS.interval([5., 4., 3., 2., 1.])
    results = ANALYSIS.analyze(EVIDENCE)
    assert len(results["primary"]) == 30
    assert len(results["off_regression"]) == 3
    assert all(r["pairs"] == len(r["pair_ratios"]) == 5 for r in results["primary"])
