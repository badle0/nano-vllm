import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).with_name("run_speculative_v0_golden.py")
SPEC = importlib.util.spec_from_file_location(
    "run_speculative_v0_golden",
    RUNNER_PATH,
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_require_is_an_explicit_runtime_check():
    with pytest.raises(AssertionError, match="sentinel failure"):
        RUNNER.require(False, "sentinel failure")


def test_runner_rejects_optimized_python_before_argument_parsing():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-O", str(RUNNER_PATH)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    assert "refuses optimized Python" in output
    assert "the following arguments are required" not in output


def test_parser_defaults_to_the_frozen_v0_contract():
    runner_commit = "1" * 40
    args = RUNNER.parse_args([
        "--model",
        "/model",
        "--output",
        "/golden",
        "--expected-runner-commit",
        runner_commit,
    ])

    assert args.expected_commit == RUNNER.CANONICAL_V0_COMMIT
    assert args.expected_runner_commit == runner_commit
    assert args.seed == 20260828
    assert args.mode == "eager"
    assert args.gpu_memory_utilization == 0.5
    assert args.max_model_len == 512
    assert args.max_num_batched_tokens == 512
    assert args.max_num_seqs == 4
    assert args.kvcache_block_size == 256
    assert "--overwrite" not in RUNNER.build_parser().format_help()


def test_main_rejects_noncanonical_v0_commit_before_model_or_gpu_access():
    with pytest.raises(AssertionError, match="canonical frozen commit"):
        RUNNER.main([
            "--model",
            "/does/not/matter",
            "--output",
            "/also/unused",
            "--expected-commit",
            "0" * 40,
            "--expected-runner-commit",
            "1" * 40,
        ])


def test_exclusive_writer_never_replaces_existing_evidence(tmp_path):
    output = tmp_path / "golden.json"
    RUNNER.write_json_exclusive(output, {"value": 1})
    original = output.read_bytes()

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        RUNNER.write_json_exclusive(output, {"value": 2})

    assert output.read_bytes() == original
    assert json.loads(original) == {"value": 1}


def test_model_snapshot_hashes_same_size_weight_substitutions(tmp_path):
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"weights")

    _, before = RUNNER.model_snapshot(str(tmp_path))
    weight.write_bytes(b"WEIGHTS")
    _, after = RUNNER.model_snapshot(str(tmp_path))

    assert before["weight_files"][0]["size_bytes"] == 7
    assert after["weight_files"][0]["size_bytes"] == 7
    assert (
        before["weight_files"][0]["sha256"]
        != after["weight_files"][0]["sha256"]
    )


def _git(repo, *args):
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_git_snapshot_requires_exact_clean_detached_source(tmp_path):
    _git(tmp_path, "init", "--quiet")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=V0 Golden Test",
        "-c",
        "user.email=v0-golden@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "frozen",
    )
    head = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "--quiet", "--detach", head)

    before = RUNNER.git_snapshot(tmp_path)
    RUNNER.validate_v0_snapshot(before, head)
    assert before["detached"] is True
    assert before["dirty"] is False
    assert before["head"] == head

    (tmp_path / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    after = RUNNER.git_snapshot(tmp_path)
    assert after["dirty"] is True
    with pytest.raises(AssertionError, match="must be clean"):
        RUNNER.validate_v0_snapshot(after, head)
    with pytest.raises(AssertionError, match="worktree state changed"):
        RUNNER.validate_unchanged(before, after)
