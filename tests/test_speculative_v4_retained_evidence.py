import copy
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


@pytest.mark.parametrize("payload", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
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


def test_harness_runtime_pin_is_current_v4():
    import _speculative_v4_evidence as evidence
    assert evidence.IMPLEMENTATION_COMMIT == ARCHIVE.IMPLEMENTATION
    assert evidence.IMPLEMENTATION_NANOVLLM_TREE == ARCHIVE.RUNTIME_TREE
    assert ARCHIVE.git(ROOT, "rev-parse", "HEAD:nanovllm").decode().strip() == ARCHIVE.RUNTIME_TREE
