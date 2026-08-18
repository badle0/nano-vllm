import json
import runpy
import stat
from pathlib import Path

import pytest

from benchmarks.chunked_prefill_tail import common
from benchmarks.chunked_prefill_tail.contract_matrix import MATRIX


def test_timing_summary_uses_nearest_rank_p95_and_true_median():
    rows = [
        {"wall_ms": wall, "cuda_ms": cuda}
        for wall, cuda in ((4, 40), (1, 10), (3, 30), (2, 20))
    ]
    assert common.timing_summary(rows) == {
        "count": 4,
        "wall_ms": {"median": 2.5, "p95": 4.0, "min": 1.0, "max": 4.0},
        "cuda_ms": {
            "median": 25.0,
            "p95": 40.0,
            "min": 10.0,
            "max": 40.0,
        },
    }
    assert common.timing_summary([]) is None


def test_immutable_json_is_read_only_and_refuses_overwrite(tmp_path):
    output = tmp_path / "retained" / "result.json"
    common.immutable_write_json(output, {"answer": 42})
    assert json.loads(output.read_text()) == {"answer": 42}
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        common.immutable_write_json(output, {"answer": 43})


def test_source_identity_is_deterministic_and_covers_contract_entrypoints():
    first = common.source_identity()
    second = common.source_identity()
    assert first == second
    paths = {entry["path"] for entry in first["files"]}
    assert {
        "benchmarks/chunked_prefill_tail/common.py",
        "benchmarks/chunked_prefill_tail/step_diagnostics.py",
        "benchmarks/chunked_prefill_tail/contract_matrix.py",
        "tests/run_varlen_graph_config.py",
        "tests/run_varlen_511_contract.py",
        "pyproject.toml",
    } <= paths
    assert len(first["aggregate_sha256"]) == 64


def test_boundary_prompt_and_matrix_contracts():
    namespace = runpy.run_path(
        str(common.ROOT / "tests/run_varlen_511_contract.py"),
        run_name="chunk_tail_boundary_contract",
    )
    prompts = namespace["boundary_prompts"](32000)
    assert len(prompts) == 4
    assert all(len(prompt) == 511 for prompt in prompts)
    assert len({tuple(prompt) for prompt in prompts}) == 4
    assert all(8 <= token < 32000 for prompt in prompts for token in prompt)
    assert namespace["MAX_MODEL_LEN"] == 512
    assert namespace["EXPECTED_GRAPH_KEY"] == (2048, 5)
    assert MATRIX == (
        (64, 512),
        (64, 1024),
        (64, 4096),
        (128, 512),
        (128, 1024),
        (128, 4096),
    )


def test_model_identity_is_content_addressed(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    (model / "weights.bin").write_bytes(b"weights")
    first = common.model_identity(model)
    second = common.model_identity(model)
    assert first == second
    assert first["file_count"] == 2
    assert first["total_bytes"] == 10
    assert Path(first["resolved_path"]) == model.resolve()
