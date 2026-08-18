#!/usr/bin/env python3
"""Validate retained decode-jitter bytes and recompute non-certifying attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "evidence" / "2026-08-18-a100-decode-jitter-d9639fc"
ARTIFACT_NAME = "decode_jitter_tau256_seed20260836_d9639fc.json"
ARTIFACT_SHA256 = "6e8236dfb68c4ab42047c39ade251bad27d0697d2767d8bac215abd57092de79"
ARTIFACT_BYTES = 5_571_547
SOURCE_COMMIT = "d9639fc5604f88069edbf0c36e71d0f0c83b3c91"
SOURCE_TREE = "c98567308092051083d248edb5cc0a5d84c01069"
SOURCE_SHA256 = "5b2a4946700b72a0e1472f8dc21410cd7477396f6c975a8c86a9311d44b7b55e"
MODEL_SHA256 = "0c659d1dba2804b0943c24bbece1858e93273147f7a11693fa99062df8c5997b"
EXPECTED_STALLS = {
    "bs16": {519, 622, 787, 952},
    "bs18": {49, 119, 677, 757},
}


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


def _load(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_bytes(), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    _require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pin(payload: dict[str, object]) -> None:
    before = payload.get("provenance")
    after = payload.get("provenance_after_run")
    _require(isinstance(before, dict) and before == after, "source pin changed")
    git = before.get("git")
    source = before.get("source")
    _require(isinstance(git, dict), "git pin missing")
    _require(isinstance(source, dict), "source pin missing")
    _require(git.get("clean") is True and git.get("status") == [], "pin is dirty")
    _require(git.get("commit") == SOURCE_COMMIT, "commit pin mismatch")
    _require(git.get("tree") == SOURCE_TREE, "tree pin mismatch")
    _require(source.get("aggregate_sha256") == SOURCE_SHA256, "source hash mismatch")
    model = payload.get("model")
    _require(isinstance(model, dict), "model manifest missing")
    _require(model.get("aggregate_sha256") == MODEL_SHA256, "model hash mismatch")
    _require(payload.get("model_after_run_matches") is True, "model changed")


def _validate_profile(
    name: str,
    profile: dict[str, object],
    batch_size: int,
    graph_size: int,
) -> list[dict[str, object]]:
    rows = profile.get("steps")
    summary = profile.get("summary")
    routing = profile.get("routing")
    memory = profile.get("memory")
    _require(isinstance(rows, list) and len(rows) == 1000, f"{name} row count")
    _require(isinstance(summary, dict), f"{name} summary missing")
    _require(summary.get("step_count") == len(rows), f"{name} summary count")
    _require(isinstance(routing, dict), f"{name} routing missing")
    _require(routing.get("decode_only") is True, f"{name} not decode-only")
    _require(
        routing.get("decode_graph_eager_fallback_possible") is False,
        f"{name} eager fallback possible",
    )
    _require(routing.get("varlen_miss_delta") == 0, f"{name} graph miss")
    _require(isinstance(memory, dict), f"{name} memory missing")
    _require(
        memory.get("allocated_before_bytes") == memory.get("allocated_after_bytes"),
        f"{name} allocated memory changed",
    )
    _require(
        memory.get("reserved_before_bytes") == memory.get("reserved_after_bytes"),
        f"{name} reserved memory changed",
    )
    _require(
        profile.get("selected_decode_graph_batch_size") == graph_size,
        f"{name} graph bucket mismatch",
    )

    stalls = []
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"{name} row is not an object")
        _require(row.get("step") == index, f"{name} step ordering changed")
        _require(row.get("batch_size") == batch_size, f"{name} batch changed")
        _require(row.get("signed_tokens") == -batch_size, f"{name} signed tokens")
        _require(row.get("finished_outputs") == 0, f"{name} request finished")
        _require(
            row.get("cuda_events_queried_after_api_return") is True,
            f"{name} CUDA completion contract changed",
        )
        route = row.get("route")
        _require(isinstance(route, dict), f"{name} route missing")
        _require(route.get("is_ragged") is False, f"{name} ragged row")
        _require(route.get("scheduled_prefill_tokens") == 0, f"{name} prefill row")
        _require(
            route.get("scheduled_decode_tokens") == batch_size,
            f"{name} decode count changed",
        )
        _require(
            route.get("selected_decode_graph_batch_size") == graph_size,
            f"{name} row graph mismatch",
        )
        timings = row.get("timings_ms")
        _require(isinstance(timings, dict), f"{name} timings missing")
        if float(timings.get("api_wall_ms", 0.0)) > 10.0:
            stalls.append(row)

    _require(
        {int(row["step"]) for row in stalls} == EXPECTED_STALLS[name],
        f"{name} stall set changed",
    )
    _require(summary.get("spike_count") == len(stalls), f"{name} spike count")
    _require(
        {int(row["step"]) for row in summary.get("spikes", [])}
        == EXPECTED_STALLS[name],
        f"{name} summary spike set changed",
    )
    return stalls


def validate_archive(archive: Path = ARCHIVE) -> dict[str, object]:
    archive = archive.expanduser().resolve(strict=True)
    _require(archive.is_dir(), "archive is not a directory")
    _require(
        {path.name for path in archive.iterdir()} == {
            ARTIFACT_NAME,
            "README.md",
            "archive_provenance.json",
        },
        "archive file set changed",
    )
    artifact = archive / ARTIFACT_NAME
    _require(artifact.is_file() and not artifact.is_symlink(), "artifact is not regular")
    info = artifact.stat()
    _require(info.st_size == ARTIFACT_BYTES, "artifact size mismatch")
    _require(_sha256(artifact) == ARTIFACT_SHA256, "artifact hash mismatch")

    provenance = _load(archive / "archive_provenance.json")
    _require(provenance.get("schema_version") == 1, "provenance schema mismatch")
    _require(
        provenance.get("kind")
        == "chunked_prefill_decode_jitter_repository_archive",
        "provenance kind mismatch",
    )
    _require(provenance.get("copied_byte_for_byte") is True, "copy flag missing")
    _require(
        provenance.get("classification")
        == "intrusive_diagnostic_only_non_certifying",
        "provenance classification mismatch",
    )
    manifest_artifact = provenance.get("artifact")
    _require(isinstance(manifest_artifact, dict), "manifest artifact missing")
    _require(manifest_artifact.get("archive_path") == ARTIFACT_NAME, "manifest path")
    _require(
        manifest_artifact.get("original_path")
        == "/workspace/.feat_bench/chunk-tail/"
        "decode_jitter_tau256_seed20260836_d9639fc.json",
        "manifest original path",
    )
    _require(manifest_artifact.get("bytes") == ARTIFACT_BYTES, "manifest size")
    _require(manifest_artifact.get("sha256") == ARTIFACT_SHA256, "manifest hash")
    _require(manifest_artifact.get("original_mode") == "0444", "original mode")

    payload = _load(artifact)
    _require(
        payload.get("kind") == "chunked_prefill_decode_jitter_diagnostic",
        "diagnostic kind mismatch",
    )
    _require(
        payload.get("protocol") == "decode_phase_attribution_bs16_bs18_v1",
        "protocol mismatch",
    )
    certification = payload.get("certification")
    _require(isinstance(certification, dict), "certification block missing")
    _require(certification.get("eligible") is False, "diagnostic promoted itself")
    _require(
        certification.get("classification") == "intrusive_diagnostic_only",
        "diagnostic classification changed",
    )
    _validate_pin(payload)
    source_run = provenance.get("source_run")
    _require(isinstance(source_run, dict), "manifest source run missing")
    _require(source_run.get("commit") == SOURCE_COMMIT, "manifest commit")
    _require(source_run.get("tree") == SOURCE_TREE, "manifest tree")
    _require(source_run.get("source_sha256") == SOURCE_SHA256, "manifest source")
    _require(source_run.get("model_sha256") == MODEL_SHA256, "manifest model")
    _require(source_run.get("started_at_utc") == payload.get("started_at_utc"), "manifest start")
    _require(source_run.get("completed_at_utc") == payload.get("completed_at_utc"), "manifest end")
    _require(source_run.get("gpu") == payload.get("environment", {}).get("gpu"), "manifest GPU")
    _require(source_run.get("seed") == payload.get("randomness", {}).get("seed"), "manifest seed")
    gc_state = payload.get("python_gc")
    _require(isinstance(gc_state, dict), "GC state missing")
    _require(gc_state.get("enabled_before_engine") is True, "GC pre-state")
    _require(gc_state.get("enabled_after_engine_init") is False, "GC lease")
    _require(gc_state.get("enabled_after_engine_exit") is True, "GC restore")
    contract = payload.get("measurement_contract")
    _require(isinstance(contract, dict), "measurement contract missing")
    _require(contract.get("added_per_step_cuda_synchronize") is False, "added sync")
    spec = payload.get("workload", {}).get("spec")
    _require(isinstance(spec, dict), "workload spec missing")
    _require(spec.get("tau") == 256, "tau changed")
    _require(spec.get("batch_sizes") == [16, 18], "batch profiles changed")
    _require(spec.get("steps_per_batch") == 1000, "step count changed")
    _require(spec.get("disable_python_gc") is True, "GC mode changed")
    _require(source_run.get("tau") == spec.get("tau"), "manifest tau")
    _require(
        source_run.get("steps_per_batch") == spec.get("steps_per_batch"),
        "manifest steps",
    )
    _require(source_run.get("batch_sizes") == spec.get("batch_sizes"), "manifest batches")
    profiles = payload.get("profiles")
    _require(isinstance(profiles, dict) and set(profiles) == {"bs16", "bs18"}, "profiles changed")
    stalls = [
        *_validate_profile("bs16", profiles["bs16"], 16, 16),
        *_validate_profile("bs18", profiles["bs18"], 18, 32),
    ]

    normal_sync = []
    model_cuda = []
    max_cpu_wall_difference_ms = 0.0
    for row in stalls:
        timings = row["timings_ms"]
        context = row["context_switches"]
        _require(
            context.get("api") == {"involuntary": 0, "voluntary": 0},
            "stall API context switch observed",
        )
        _require(
            all(
                phase == {"involuntary": 0, "voluntary": 0}
                for phase in context.get("phases", {}).values()
            ),
            "stall phase context switch observed",
        )
        difference = abs(
            float(timings["api_wall_ms"])
            - float(timings["api_thread_cpu_ms"])
        )
        max_cpu_wall_difference_ms = max(max_cpu_wall_difference_ms, difference)
        _require(difference <= 0.001, "stall thread CPU no longer matches wall")
        if float(timings["model_cuda_ms"]) > 10.0:
            model_cuda.append(row)
            _require(float(timings["runner_cuda_ms"]) > 10.0, "model spike runner normal")
            _require(float(timings["api_residual_wall_ms"]) < 1.0, "model spike residual")
        else:
            normal_sync.append(row)
            profile = profiles["bs16" if row["batch_size"] == 16 else "bs18"]
            runner_p99 = profile["summary"]["timings_ms"]["runner_cuda_ms"]["p99"]
            _require(
                float(timings["runner_cuda_ms"]) <= float(runner_p99),
                "post-enqueue stall has abnormal runner CUDA",
            )
            _require(
                float(timings["api_residual_wall_ms"]) > 10.0,
                "post-enqueue stall residual changed",
            )
    _require(len(stalls) == 8, "combined stall count changed")
    _require(len(normal_sync) == 7, "post-enqueue attribution count changed")
    _require(
        len(model_cuda) == 1
        and model_cuda[0]["batch_size"] == 16
        and model_cuda[0]["step"] == 622,
        "model CUDA attribution changed",
    )
    result = provenance.get("result")
    _require(isinstance(result, dict), "manifest result missing")
    _require(result.get("latency_certified") is False, "manifest promoted latency")
    _require(result.get("production_performance_claim") is False, "production claim")
    _require(result.get("stall_count") == len(stalls), "manifest stall count")
    _require(
        result.get("normal_runner_cuda_post_enqueue_sync_residual_count")
        == len(normal_sync),
        "manifest residual attribution count",
    )
    _require(result.get("model_cuda_spike_count") == len(model_cuda), "manifest model count")
    return {
        "artifact_sha256": ARTIFACT_SHA256,
        "classification": "intrusive_diagnostic_only_non_certifying",
        "stall_count": len(stalls),
        "post_enqueue_sync_residual_count": len(normal_sync),
        "model_cuda_spike_count": len(model_cuda),
        "max_stall_thread_cpu_wall_difference_ms": max_cpu_wall_difference_ms,
        "latency_certified": False,
        "production_performance_claim": False,
    }


def main() -> None:
    print(json.dumps(validate_archive(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
