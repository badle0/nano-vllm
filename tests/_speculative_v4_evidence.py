"""V4 provenance, independent of the frozen V3 implementation binding.

Only version-neutral file/model/environment observation helpers are reused.
Neither V3's constants nor its prepare/finalize/validation gates are modified.
"""
from pathlib import Path
import sys

from _speculative_v3_evidence import (
    environment_snapshot, git_snapshot, model_snapshot, runtime_import_identities,
    source_file_identity, require, require_retained_a100_sxm4_40gb,
    require_retained_runtime_environment,
)

IMPLEMENTATION_COMMIT = "22b63e8e24db3c7bc9c24b61aedd93b25d76d289"
IMPLEMENTATION_NANOVLLM_TREE = "922d81417cf72ec912da13267fbb024c145a6a15"
HARNESS_FILES = (
    "tests/run_speculative_v4_gpu.py",
    "tests/_speculative_v4_evidence.py",
    "tests/_speculative_v3_evidence.py",
    "tests/run_speculative_v3_route_compile.py",
)


def snapshot(repo, model, imports):
    return {
        "source": git_snapshot(repo),
        "files": {name: source_file_identity(repo / name, repo) for name in HARNESS_FILES},
        "imports": runtime_import_identities(repo, imports),
        "model": model_snapshot(model)[1],
        "environment": environment_snapshot(),
    }


def prepare(model, expected_commit, imports):
    repo = Path(__file__).resolve().parents[1]
    require_retained_a100_sxm4_40gb()
    require_retained_runtime_environment()
    require(len(expected_commit) == 40, "expected commit must be a full SHA")
    before = snapshot(repo, model, imports)
    require(before["source"]["clean"], "V4 retained source must be clean")
    require(before["source"]["head"] == expected_commit, "unexpected producer commit")
    require(before["source"]["nanovllm_tree"] == IMPLEMENTATION_NANOVLLM_TREE,
            "producer runtime is not frozen V4")
    require(all(item["matches_head"] for item in (*before["files"].values(), *before["imports"].values())),
            "uncommitted or foreign runtime/harness import")
    return repo, before


def finalize(repo, model, imports, before):
    after = snapshot(repo, model, imports)
    require(before == after, "source, model, imports or environment changed during run")
    return {
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "implementation_nanovllm_tree": IMPLEMENTATION_NANOVLLM_TREE,
        "producer_commit": before["source"]["head"],
        "before": before, "after": after,
        "argv": sys.argv, "retention_eligible": True,
    }
