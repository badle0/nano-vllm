import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v4_archive", ROOT / "benchmarks/speculative_v4/validate_retained_evidence.py")
ARCHIVE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ARCHIVE)


def plan():
    return dict(rows=[dict(seq_id=0, committed_tokens=5, target_cached_tokens=4,
                          draft_cached_tokens=0, remaining_completion_tokens=8,
                          model_position_headroom=507, highest_draft_write_position=5,
                          highest_target_write_position=6, block_table=[0])],
                effective_k=2, gpu_certified=False, workspace_fingerprint="1"*64,
                bypass_reason=None, draft_catchup_tokens=4, draft_query_tokens=2,
                target_query_tokens=3, total_model_positions=9,
                route_key=dict(effective_k=2, batch_bucket=1),
                modeled_live_peak_bytes=64*1024**2, reservation_bytes=128*1024**2)


def test_independent_geometry_validator_accepts_valid_plan():
    ARCHIVE.check_plan(plan())


@pytest.mark.parametrize("field,value", [
    ("effective_k", True), ("gpu_certified", True), ("target_query_tokens", 1),
    ("draft_query_tokens", 1), ("draft_catchup_tokens", 0),
    ("total_model_positions", 5), ("modeled_live_peak_bytes", 1),
    ("workspace_fingerprint", "not-a-digest"),
])
def test_geometry_validator_rejects_tampering(field, value):
    value_plan = plan()
    value_plan[field] = value
    with pytest.raises(ValueError):
        ARCHIVE.check_plan(value_plan)


@pytest.mark.parametrize("field,value", [
    ("highest_target_write_position", 512), ("highest_draft_write_position", 8),
    ("remaining_completion_tokens", 2), ("draft_cached_tokens", 5),
    ("target_cached_tokens", 3), ("block_table", []),
])
def test_geometry_validator_rejects_invalid_rows(field, value):
    value_plan = plan()
    value_plan["rows"][0][field] = value
    with pytest.raises(ValueError):
        ARCHIVE.check_plan(value_plan)


@pytest.mark.parametrize("payload", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'])
def test_json_is_strict(payload):
    with pytest.raises(ValueError):
        ARCHIVE.load_json(payload)


def test_regular_rejects_symlink_and_hardlink(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text("{}")
    linked = tmp_path / "linked.json"
    linked.symlink_to(raw)
    with pytest.raises(ValueError, match="symlink"):
        ARCHIVE.regular(linked)
    hard = tmp_path / "hard.json"
    hard.hardlink_to(raw)
    with pytest.raises(ValueError, match="singly linked"):
        ARCHIVE.regular(hard)


def test_harness_runtime_pin_matches_frozen_v4():
    import _speculative_v4_evidence as evidence
    assert evidence.IMPLEMENTATION_COMMIT == ARCHIVE.IMPLEMENTATION
    assert evidence.IMPLEMENTATION_NANOVLLM_TREE == ARCHIVE.RUNTIME_TREE
    assert ARCHIVE.git(ROOT, "rev-parse", f"{ARCHIVE.IMPLEMENTATION}:nanovllm").decode().strip() == ARCHIVE.RUNTIME_TREE


EVIDENCE = ROOT / "benchmarks/speculative_v4/evidence/2026-09-06-a100-v4-6ca56b4"


def test_retained_v4_archive_passes_without_cuda_or_model():
    result = ARCHIVE.validate(EVIDENCE, ROOT)
    assert result["modes"] == ["eager", "graph"]
    assert result["all_intervals_per_enabled_run"] == 46
    assert result["accepted_tokens"] is False
    assert result["verifier_memory_peak_certified"] is False


def test_manifest_is_a_pinned_trust_root(tmp_path):
    payload = (EVIDENCE / "manifest.json").read_bytes()
    (tmp_path / "manifest.json").write_bytes(payload + b" ")
    with pytest.raises(ValueError, match="untrusted archive manifest"):
        ARCHIVE.validate(tmp_path, ROOT)


def test_payload_byte_change_is_rejected_even_if_json_still_parses(monkeypatch):
    original = ARCHIVE.regular
    def changed(path):
        payload = original(path)
        return payload + b" " if path.name == "eager-off.json" else payload
    monkeypatch.setattr(ARCHIVE, "regular", changed)
    with pytest.raises(ValueError, match="pinned artifact bytes changed"):
        ARCHIVE.validate(EVIDENCE, ROOT)


def test_cross_producer_output_oracle_rejects_changed_public_token(monkeypatch):
    original = ARCHIVE.load_json
    def changed(payload):
        value = original(payload)
        if value.get("mode") == "graph" and value.get("side") == "nan":
            value["control"]["events"][1][0][1] ^= 1
        return value
    monkeypatch.setattr(ARCHIVE, "load_json", changed)
    with pytest.raises(ValueError, match="target outputs or RNG diverged"):
        ARCHIVE.check_set(EVIDENCE, ROOT)


@pytest.mark.parametrize("mutation", ["dirty", "producer", "foreign_source", "compiler_disabled"])
def test_provenance_validator_rejects_forged_raw_record(mutation):
    value = ARCHIVE.load_json((EVIDENCE / "eager-off.json").read_bytes())
    p = value["provenance"]
    # Mutate both sides: equality flags alone must not manufacture provenance.
    for snapshot in (p["before"], p["after"]):
        if mutation == "dirty":
            snapshot["source"]["clean"] = False
        elif mutation == "foreign_source":
            snapshot["files"]["tests/run_speculative_v4_gpu.py"]["sha256"] = "0" * 64
        elif mutation == "compiler_disabled":
            snapshot["environment"]["software"]["torch_dynamo_disable"] = True
    if mutation == "producer":
        p["producer_commit"] = "7fec9993d5e4e0e06fec22e3973dfc203fdbd2d8"
    with pytest.raises(ValueError):
        ARCHIVE.check_provenance(value, ROOT)


def test_graph_interval_log_cannot_be_missing_or_reordered():
    value = ARCHIVE.load_json((EVIDENCE / "graph-zero.json").read_bytes())
    with pytest.raises(ValueError, match="log/JSON interval coverage"):
        ARCHIVE.check_raw(value, "graph-zero", b"", ROOT)


def test_registered_route_coverage_cannot_be_dropped():
    value = ARCHIVE.load_json((EVIDENCE / "eager-zero.json").read_bytes())
    value["records"] = [r for r in value["records"] if r["label"] != "route/3/2/1/warm"]
    with pytest.raises(ValueError, match="incomplete route sweep"):
        ARCHIVE.check_raw(value, "eager-zero", b"", ROOT)


def test_empty_probability_observations_cannot_pass_normalization_vacuously():
    value = ARCHIVE.load_json((EVIDENCE / "eager-zero.json").read_bytes())
    value["records"][0]["samples"][0]["sums"] = []
    with pytest.raises(ValueError, match="incomplete probability observations"):
        ARCHIVE.check_raw(value, "eager-zero", b"", ROOT)
