#!/usr/bin/env python3
"""Validate byte identity and non-certifying verdicts of retained evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path, PurePosixPath


SCHEMA_VERSION = 1
FULL_KIND = "chunked_prefill_full_completion_repository_archive"
DIAGNOSTIC_KIND = "chunked_prefill_contract_phase_repository_archive"
IGNORED_REPOSITORY_FILES = {"README.md", "archive_provenance.json"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_payload(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and relative, "artifact path is invalid")
    pure = PurePosixPath(relative)
    _require(
        not pure.is_absolute() and ".." not in pure.parts,
        f"artifact path is not repository-relative: {relative}",
    )
    path = (root / relative).resolve(strict=True)
    _require(path.is_relative_to(root), f"artifact escapes archive root: {relative}")
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode), f"artifact is not regular: {relative}")
    _require(not path.is_symlink(), f"artifact must not be a symlink: {relative}")
    return path


def _validate_outer_files(
    root: Path,
    provenance: dict[str, object],
) -> dict[str, Path]:
    files = provenance.get("files")
    _require(isinstance(files, list), "outer files must be a list")
    _require(
        provenance.get("artifact_count") == len(files),
        "outer artifact count mismatch",
    )
    expected_paths = []
    retained = {}
    original_root = provenance.get("original_root")
    _require(
        isinstance(original_root, str) and PurePosixPath(original_root).is_absolute(),
        "original_root must be absolute",
    )
    for row in files:
        _require(isinstance(row, dict), "outer file row must be an object")
        relative = row.get("path")
        _require(isinstance(relative, str), "outer file path must be a string")
        path = _resolve_payload(root, relative)
        expected_paths.append(relative)
        _require(row.get("original_mode") == "0444", f"mode mismatch: {relative}")
        _require(path.stat().st_size == row.get("bytes"), f"size mismatch: {relative}")
        _require(_sha256(path) == row.get("sha256"), f"hash mismatch: {relative}")
        expected_original = str(PurePosixPath(original_root) / relative)
        _require(
            row.get("original_path") == expected_original,
            f"original path mismatch: {relative}",
        )
        retained[relative] = path
    _require(
        expected_paths == sorted(expected_paths)
        and len(expected_paths) == len(set(expected_paths)),
        "outer artifact paths must be unique and sorted",
    )
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in IGNORED_REPOSITORY_FILES
    }
    _require(
        actual_paths == set(expected_paths),
        "outer manifest does not cover exactly the retained payload files",
    )
    return retained


def _validate_pin(
    payload: dict[str, object],
    provenance: dict[str, object],
    *,
    after_name: str,
) -> None:
    before = payload.get("provenance")
    after = payload.get(after_name)
    _require(isinstance(before, dict) and before == after, "source pin changed")
    git = before.get("git")
    source = before.get("source")
    _require(isinstance(git, dict) and isinstance(source, dict), "source pin missing")
    _require(git.get("clean") is True and git.get("status") == [], "pin is dirty")
    _require(
        git.get("commit") == provenance.get("evidence_commit"),
        "evidence commit mismatch",
    )
    _require(git.get("tree") == provenance.get("evidence_tree"), "tree mismatch")
    _require(
        source.get("aggregate_sha256") == provenance.get("evidence_source_sha256"),
        "source hash mismatch",
    )
    model = payload.get("model")
    _require(isinstance(model, dict), "model manifest missing")
    _require(
        model.get("aggregate_sha256") == provenance.get("model_sha256"),
        "model hash mismatch",
    )


def _validate_inner_manifest(
    archive_root: Path,
    manifest: dict[str, object],
) -> dict[str, Path]:
    _require(manifest.get("schema_version") == 1, "inner manifest schema mismatch")
    _require(
        manifest.get("kind") == "chunked_prefill_certification_archive",
        "inner manifest kind mismatch",
    )
    rows = manifest.get("files")
    _require(isinstance(rows, list) and len(rows) == 6, "inner manifest needs 6 files")
    paths = []
    retained = {}
    for row in rows:
        _require(isinstance(row, dict), "inner file row must be an object")
        relative = row.get("path")
        _require(isinstance(relative, str), "inner file path must be a string")
        path = _resolve_payload(archive_root, relative)
        paths.append(relative)
        _require(path.stat().st_size == row.get("bytes"), f"inner size mismatch: {relative}")
        _require(_sha256(path) == row.get("sha256"), f"inner hash mismatch: {relative}")
        retained[relative] = path
    _require(len(paths) == len(set(paths)), "inner paths are duplicated")
    _require(
        set(paths) == {
            "aggregate.json",
            *{f"runs/{path.name}" for path in (archive_root / "runs").glob("*.json")},
        },
        "inner manifest does not cover exactly aggregate plus five runs",
    )
    return retained


def _validate_full_archive(
    root: Path,
    provenance: dict[str, object],
) -> dict[str, object]:
    rows = provenance.get("archives")
    _require(isinstance(rows, list) and len(rows) == 2, "two tau archives required")
    reports = []
    for row in rows:
        _require(isinstance(row, dict), "archive row must be an object")
        tau = row.get("tau")
        _require(tau in (256, 512), "unexpected tau")
        archive_root = (root / str(row.get("directory"))).resolve(strict=True)
        _require(archive_root.parent == root, "nested archive path invalid")
        manifest_path = archive_root / "manifest.json"
        _require(
            _sha256(manifest_path) == row.get("inner_manifest_sha256"),
            "inner manifest hash mismatch",
        )
        inner = _validate_inner_manifest(archive_root, _load_json(manifest_path))
        aggregate_path = inner["aggregate.json"]
        _require(
            _sha256(aggregate_path) == row.get("aggregate_sha256"),
            "aggregate hash mismatch",
        )
        marker_path = root / str(row.get("complete_marker"))
        marker = _load_json(marker_path)
        _require(marker.get("aggregate_sha256") == row.get("aggregate_sha256"), "marker hash mismatch")
        _require(marker.get("manifest") == "manifest.json", "marker manifest mismatch")
        _require(
            marker.get("archive")
            == str(PurePosixPath(str(provenance.get("original_root"))) / archive_root.name),
            "marker original archive path mismatch",
        )
        aggregate = _load_json(aggregate_path)
        _require(
            aggregate.get("kind") == "chunked_prefill_full_completion_aggregate",
            "aggregate kind mismatch",
        )
        _require(aggregate.get("tau") == tau, "aggregate tau mismatch")
        _validate_pin(
            aggregate,
            provenance,
            after_name="provenance_after_validation",
        )
        policy = aggregate.get("policy")
        _require(isinstance(policy, dict), "aggregate policy missing")
        _require(policy.get("latency_certified") is False, "archive must be non-certifying")
        _require(policy.get("classification") == row.get("classification"), "classification mismatch")
        _require(policy.get("runs_meeting_slo") == row.get("runs_meeting_slo"), "SLO count mismatch")
        _require(row.get("latency_certified") is False, "outer verdict must be false")
        inputs = aggregate.get("inputs")
        _require(isinstance(inputs, list) and len(inputs) == 5, "five inputs required")
        run_hashes = {
            _sha256(path) for relative, path in inner.items() if relative.startswith("runs/")
        }
        _require({item.get("sha256") for item in inputs} == run_hashes, "aggregate input hashes mismatch")
        for relative, run_path in inner.items():
            if not relative.startswith("runs/"):
                continue
            run = _load_json(run_path)
            _require(run.get("kind") == "chunked_prefill_full_completion_run", "run kind mismatch")
            _require(run.get("arguments", {}).get("tau") == tau, "run tau mismatch")
            _require(run.get("summary", {}).get("single_run_latency_certified") is False, "single run promoted itself")
            _validate_pin(run, provenance, after_name="provenance_after_run")
        if tau == 256:
            _require(row.get("classification") == "latency_not_certified", "tau256 verdict changed")
            _require(row.get("runs_meeting_slo") == 3, "tau256 must retain 3/5 result")
        else:
            _require(
                row.get("classification")
                == "throughput_ttft_only_not_latency_certified",
                "tau512 verdict changed",
            )
            _require(policy.get("tau512_can_certify") is False, "tau512 policy changed")
        reports.append({"tau": tau, "classification": row.get("classification")})
    _require({row["tau"] for row in reports} == {256, 512}, "tau archive set mismatch")
    return {"kind": FULL_KIND, "archives": reports}


def _validate_diagnostics(
    provenance: dict[str, object],
    retained: dict[str, Path],
) -> dict[str, object]:
    classifications = provenance.get("classifications")
    _require(isinstance(classifications, dict), "diagnostic classifications missing")
    _require(set(classifications) == set(retained), "diagnostic classification set mismatch")
    for name, path in retained.items():
        payload = _load_json(path)
        _validate_pin(payload, provenance, after_name="provenance_after_run")
        classification = classifications[name]
        if classification == "correctness_contract_pass":
            if payload.get("kind") == "varlen_graph_4x511_maxlen512_contract":
                _require(payload.get("contract", {}).get("passed") is True, "4x511 contract failed")
            elif payload.get("kind") == "varlen_graph_tau_maxlen_contract_matrix":
                _require(payload.get("matrix", {}).get("passed") is True, "contract matrix failed")
            else:
                raise ValueError(f"unexpected correctness artifact: {name}")
        elif classification == "phase_diagnostic_non_certifying":
            _require(payload.get("kind") == "chunked_prefill_tail_step_diagnostic", "phase kind mismatch")
            release = payload.get("release_policy")
            _require(isinstance(release, dict), "phase release policy missing")
            _require(release.get("current_run_latency_certified") is False, "phase run promoted itself")
            gc_state = payload.get("python_gc")
            _require(isinstance(gc_state, dict), "phase GC state missing")
            _require(gc_state.get("enabled_before_engine") is True, "GC pre-state mismatch")
            _require(gc_state.get("enabled_after_engine_init") is False, "GC lease mismatch")
            _require(gc_state.get("enabled_after_engine_exit") is True, "GC restore mismatch")
        else:
            raise ValueError(f"unknown diagnostic classification: {classification}")
    return {"kind": DIAGNOSTIC_KIND, "classifications": classifications}


def validate_archive(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve(strict=True)
    _require(root.is_dir(), f"archive root is not a directory: {root}")
    provenance = _load_json(root / "archive_provenance.json")
    _require(provenance.get("schema_version") == SCHEMA_VERSION, "outer schema mismatch")
    _require(provenance.get("copied_byte_for_byte") is True, "byte-copy declaration missing")
    retained = _validate_outer_files(root, provenance)
    kind = provenance.get("kind")
    if kind == FULL_KIND:
        return _validate_full_archive(root, provenance)
    if kind == DIAGNOSTIC_KIND:
        return _validate_diagnostics(provenance, retained)
    raise ValueError(f"unknown retained evidence kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="+")
    args = parser.parse_args()
    for archive in args.archive:
        report = validate_archive(archive)
        print(f"validated {archive}: {json.dumps(report, sort_keys=True)}")


if __name__ == "__main__":
    main()
