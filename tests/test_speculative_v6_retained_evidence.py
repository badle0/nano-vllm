"""Tamper tests for the CPU/model-free V5/V6 certificate."""
import copy
from functools import lru_cache
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v56_archive", ROOT / "benchmarks/speculative_v5/validate_retained_evidence.py")
ARCHIVE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ARCHIVE)
EVIDENCE = ROOT / "benchmarks/speculative_v5/evidence/2026-09-06-a100-v5-v6-a165b65"
PRODUCER = "a165b654660e60ca60cecd47b89c95de1d65735b"


def test_retained_v5_v6_archive_passes_without_cuda_or_model():
    result = ARCHIVE.validate(EVIDENCE)
    assert result["route_cells"] == 160
    assert result["enabled_cycles"] == 335
    assert result["natural_fallbacks"] == 0
    assert result["automatic_kv_graph_k2"] is True


def test_untrusted_manifest_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_bytes((EVIDENCE / "manifest.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="untrusted"):
        ARCHIVE.validate(tmp_path)


def test_sealing_refuses_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        ARCHIVE.seal(tmp_path, tmp_path, PRODUCER)


@pytest.mark.parametrize("kind", ["role", "source", "compiler", "peak", "burst", "fallback", "sweep", "scratch", "causality"])
def test_semantic_tampering_fails_even_before_hash_sealing(monkeypatch, kind):
    value = ARCHIVE.load_json((EVIDENCE / "eager-on.json").read_bytes())
    if kind == "role": value["args"]["enabled"] = False
    elif kind == "source": value["source_sha256"].pop(next(iter(value["source_sha256"])))
    elif kind == "compiler": value["cycles"][0]["compiler_unchanged"] = False
    elif kind == "peak": value["cycles"][0]["peak_increment"] = 10**12
    elif kind == "burst": value["cycles"][0]["result"]["rows"][0]["committed_token_ids"] = []
    elif kind == "fallback": value["forced_metrics"]["spec_residual_numerical_fallbacks"] = 0
    elif kind == "sweep": value["sweep_cells"].pop()
    elif kind == "scratch": value["sort_scratch"][0]["measured"] = 10**12
    elif kind == "causality": value["causality_probe"] = False
    with pytest.raises(ValueError):
        ARCHIVE.check_run(value, "eager-on", PRODUCER)


def test_changed_payload_is_rejected(monkeypatch):
    original = ARCHIVE.regular
    def changed(path):
        raw = original(path)
        return raw + b" " if path.name == "eager-on.json" else raw
    monkeypatch.setattr(ARCHIVE, "regular", changed)
    with pytest.raises(ValueError, match="digest"):
        ARCHIVE.validate(EVIDENCE)
