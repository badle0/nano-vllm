#!/usr/bin/env python3
"""Validate the byte integrity and semantic shape of the PR6 repair bundle."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
MANAGED_DIRECTORIES = ("tau", "stock", "tokens", "harnesses", "smoke")


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str):
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def value_at(row, dotted_path: str):
    value = row
    for key in dotted_path.split("."):
        value = value[key]
    return value


def require_close(actual: float, expected: float, label: str) -> None:
    require(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{label}: expected {expected!r}, got {actual!r}",
    )


def require_sha(value: str, label: str) -> None:
    require(bool(re.fullmatch(r"[0-9a-f]{64}", value)), f"invalid SHA-256 for {label}")


def validate_artifact_inventory(manifest: dict) -> dict[str, dict]:
    artifacts = manifest["artifacts"]
    require(
        manifest["bundle"]["artifact_count"] == len(artifacts),
        "manifest artifact_count does not match artifacts",
    )
    indexed: dict[str, dict] = {}
    for entry in artifacts:
        relative = entry["path"]
        path = Path(relative)
        require(not path.is_absolute(), f"absolute artifact path: {relative}")
        require(".." not in path.parts, f"escaping artifact path: {relative}")
        require(path.parts[0] in MANAGED_DIRECTORIES, f"unmanaged artifact: {relative}")
        require(relative not in indexed, f"duplicate artifact entry: {relative}")
        target = ROOT / path
        require(target.is_file(), f"missing artifact: {relative}")
        require(not target.is_symlink(), f"artifact must not be a symlink: {relative}")
        require(target.stat().st_size == entry["bytes"], f"byte count mismatch: {relative}")
        require_sha(entry["sha256"], relative)
        require(sha256(target) == entry["sha256"], f"SHA-256 mismatch: {relative}")
        if target.suffix == ".json":
            load_json(relative)
        indexed[relative] = entry

    discovered = {
        path.relative_to(ROOT).as_posix()
        for directory in MANAGED_DIRECTORIES
        for path in (ROOT / directory).rglob("*")
        if path.is_file()
    }
    require(discovered == set(indexed), "managed files and manifest artifact list differ")
    return indexed


TAU_KEYS = {
    "after_long_admission_s",
    "commit",
    "completion_tokens_per_s",
    "cuda",
    "gpu",
    "groups",
    "max_num_seqs",
    "model",
    "seed",
    "tau",
    "torch",
    "whole_run_s",
}
GROUP_KEYS = {
    "count",
    "ttft_p50_ms",
    "ttft_max_ms",
    "mean_itl_p50_ms",
    "max_itl_p50_ms",
    "max_itl_max_ms",
}
TAU_PATTERN = re.compile(
    r"tau/chunk_(base|fix)_tau(\d+)_(?:seed([123])(b)?|pair([123]))\.json"
)
METRICS = {
    "throughput_tokens_per_s": "completion_tokens_per_s",
    "interactive_ttft_p50_ms": "groups.interactive.ttft_p50_ms",
    "interactive_max_itl_p50_ms": "groups.interactive.max_itl_p50_ms",
    "long_ttft_p50_ms": "groups.long.ttft_p50_ms",
}


def validate_tau_payload(relative: str) -> dict:
    match = TAU_PATTERN.fullmatch(relative)
    require(match is not None, f"unexpected tau filename: {relative}")
    branch, tau_text, seed_index, _rerun, pair_index = match.groups()
    index = int(seed_index or pair_index)
    row = load_json(relative)
    require(set(row) == TAU_KEYS, f"unexpected tau schema: {relative}")
    tau = int(tau_text)
    require(row["tau"] == tau, f"tau/filename mismatch: {relative}")
    require(row["seed"] == 20260813 + index, f"seed/filename mismatch: {relative}")
    require(row["max_num_seqs"] == min(512, tau), f"max_num_seqs mismatch: {relative}")
    expected_commit = "1ef1ab5" if branch == "base" else "631a346"
    require(row["commit"] == expected_commit, f"commit label mismatch: {relative}")
    require(row["torch"] == "2.10.0+cu128", f"torch mismatch: {relative}")
    require(row["cuda"] == "12.8", f"CUDA mismatch: {relative}")
    require(row["gpu"] == "NVIDIA A100-SXM4-40GB", f"GPU mismatch: {relative}")
    require(row["model"] == "/workspace/models/Qwen3-0.6B", f"model mismatch: {relative}")
    require(set(row["groups"]) == {"interactive", "long"}, f"group mismatch: {relative}")
    for group_name, count in (("interactive", 16), ("long", 2)):
        group = row["groups"][group_name]
        require(set(group) == GROUP_KEYS, f"group schema mismatch: {relative}/{group_name}")
        require(group["count"] == count, f"group count mismatch: {relative}/{group_name}")
        require(
            all(isinstance(value, (int, float)) and value >= 0 for value in group.values()),
            f"invalid group measurement: {relative}/{group_name}",
        )
    require(row["whole_run_s"] > 0, f"invalid run time: {relative}")
    require(0 < row["after_long_admission_s"] <= row["whole_run_s"], f"invalid phase time: {relative}")
    require_close(
        row["completion_tokens_per_s"],
        (18 * 256) / row["whole_run_s"],
        f"throughput derivation {relative}",
    )
    return row


def validate_tau_sets(manifest: dict, indexed: dict[str, dict]) -> None:
    sets = manifest["sets"]
    adjudication = sets["tau_adjudication"]
    require(set(adjudication) == {"128", "1024", "16384"}, "unexpected adjudication taus")
    adjudication_paths: set[str] = set()
    for tau, entry in adjudication.items():
        require(len(entry["baseline"]) == len(entry["fixed"]) == 3, f"A/B count mismatch at tau {tau}")
        baseline_rows = [validate_tau_payload(path) for path in entry["baseline"]]
        fixed_rows = [validate_tau_payload(path) for path in entry["fixed"]]
        adjudication_paths.update(entry["baseline"])
        adjudication_paths.update(entry["fixed"])
        for name, dotted in METRICS.items():
            baseline_median = statistics.median(value_at(row, dotted) for row in baseline_rows)
            fixed_median = statistics.median(value_at(row, dotted) for row in fixed_rows)
            expected = entry["summary"][name]
            require_close(baseline_median, expected["baseline_median"], f"tau {tau} {name} baseline")
            require_close(fixed_median, expected["fixed_median"], f"tau {tau} {name} fixed")
            require_close(
                (fixed_median / baseline_median - 1.0) * 100.0,
                expected["delta_percent"],
                f"tau {tau} {name} delta",
            )

    sweep = sets["constructor_sweep"]
    require(
        set(sweep["artifacts"]) == {"128", "256", "512", "1024", "2048", "16384"},
        "unexpected constructor sweep taus",
    )
    sweep_paths: set[str] = set()
    for tau, paths in sweep["artifacts"].items():
        require(len(paths) == 3, f"constructor sweep count mismatch at tau {tau}")
        rows = [validate_tau_payload(path) for path in paths]
        sweep_paths.update(paths)
        for name, dotted in METRICS.items():
            actual = statistics.median(value_at(row, dotted) for row in rows)
            require_close(actual, sweep["summary_medians"][tau][name], f"sweep tau {tau} {name}")

    tau_inventory = {path for path in indexed if path.startswith("tau/")}
    require(tau_inventory == adjudication_paths | sweep_paths, "tau set membership is incomplete")
    for relative in tau_inventory:
        roles = set(indexed[relative]["roles"])
        require((relative in adjudication_paths) == ("tau_adjudication" in roles), f"A/B role mismatch: {relative}")
        require((relative in sweep_paths) == ("constructor_sweep" in roles), f"sweep role mismatch: {relative}")


STOCK_KEYS = {
    "commit",
    "cuda",
    "elapsed_s",
    "gpu",
    "seed",
    "throughput_tokens_per_s",
    "torch",
    "total_tokens",
}
STOCK_PATTERN = re.compile(r"stock/stock_(dev|chunk)_seed([123])\.json")


def validate_stock(manifest: dict, indexed: dict[str, dict]) -> None:
    stock = manifest["sets"]["stock"]
    rows: dict[str, list[dict]] = {"dev": [], "chunk": []}
    expected_paths: set[str] = set()
    totals = {1: 146430, 2: 146364, 3: 141374}
    for side in ("dev", "chunk"):
        require(len(stock["artifacts"][side]) == 3, f"stock {side} count mismatch")
        for relative in stock["artifacts"][side]:
            match = STOCK_PATTERN.fullmatch(relative)
            require(match is not None and match.group(1) == side, f"stock filename mismatch: {relative}")
            index = int(match.group(2))
            row = load_json(relative)
            require(set(row) == STOCK_KEYS, f"unexpected stock schema: {relative}")
            require(row["seed"] == 20260813 + index, f"stock seed mismatch: {relative}")
            require(row["total_tokens"] == totals[index], f"stock token count mismatch: {relative}")
            expected_commit = "edb997e" if side == "dev" else "631a346"
            require(row["commit"] == expected_commit, f"stock commit label mismatch: {relative}")
            require(row["torch"] == "2.10.0+cu128", f"stock torch mismatch: {relative}")
            require(row["cuda"] == "12.8", f"stock CUDA mismatch: {relative}")
            require(row["gpu"] == "NVIDIA A100-SXM4-40GB", f"stock GPU mismatch: {relative}")
            require(row["elapsed_s"] > 0, f"stock elapsed time invalid: {relative}")
            require_close(
                row["throughput_tokens_per_s"],
                row["total_tokens"] / row["elapsed_s"],
                f"stock throughput derivation {relative}",
            )
            rows[side].append(row)
            expected_paths.add(relative)

    summary = stock["summary"]
    dev_median = statistics.median(row["throughput_tokens_per_s"] for row in rows["dev"])
    chunk_median = statistics.median(row["throughput_tokens_per_s"] for row in rows["chunk"])
    pair_deltas = [
        (chunk["throughput_tokens_per_s"] / dev["throughput_tokens_per_s"] - 1.0) * 100.0
        for dev, chunk in zip(rows["dev"], rows["chunk"])
    ]
    require_close(dev_median, summary["dev_median_tokens_per_s"], "stock dev median")
    require_close(chunk_median, summary["chunk_median_tokens_per_s"], "stock chunk median")
    require_close(
        (chunk_median / dev_median - 1.0) * 100.0,
        summary["delta_ratio_of_medians_percent"],
        "stock ratio-of-medians delta",
    )
    require(len(pair_deltas) == len(summary["paired_deltas_percent"]), "stock paired delta count")
    for index, (actual, expected) in enumerate(zip(pair_deltas, summary["paired_deltas_percent"]), 1):
        require_close(actual, expected, f"stock pair {index} delta")
    require_close(statistics.median(pair_deltas), summary["median_paired_delta_percent"], "stock median paired delta")
    require(
        expected_paths == {path for path in indexed if path.startswith("stock/")},
        "stock set membership is incomplete",
    )


TOKEN_KEYS = {"max_tokens", "prompt_lengths", "seed", "sha256", "temperature", "tokens"}
PROMPT_LENGTHS = [16, 31, 64, 95, 128, 191, 224, 255]


def validate_token_file(relative: str, temperature: float) -> dict:
    row = load_json(relative)
    require(set(row) == TOKEN_KEYS, f"unexpected token schema: {relative}")
    require(row["seed"] == 20260817, f"token seed mismatch: {relative}")
    require(row["prompt_lengths"] == PROMPT_LENGTHS, f"prompt lengths mismatch: {relative}")
    require(row["max_tokens"] == 32, f"max_tokens mismatch: {relative}")
    require(row["temperature"] == temperature, f"temperature mismatch: {relative}")
    require(len(row["tokens"]) == 8, f"token row count mismatch: {relative}")
    require(
        all(len(tokens) == 32 and all(isinstance(token, int) for token in tokens) for tokens in row["tokens"]),
        f"token matrix shape/type mismatch: {relative}",
    )
    canonical = json.dumps(row["tokens"], separators=(",", ":")).encode()
    payload_sha = hashlib.sha256(canonical).hexdigest()
    require_sha(row["sha256"], f"token payload {relative}")
    require(payload_sha == row["sha256"], f"token payload SHA mismatch: {relative}")
    return row


def divergence_summary(left: list[list[int]], right: list[list[int]]) -> list[dict]:
    summary = []
    for row_index, (left_row, right_row) in enumerate(zip(left, right)):
        positions = [
            index for index, (a, b) in enumerate(zip(left_row, right_row)) if a != b
        ]
        if positions:
            summary.append({
                "sequence_index": row_index,
                "first_completion_index_zero_based": positions[0],
                "differing_positions": positions,
                "differing_position_count": len(positions),
            })
    return summary


def validate_tokens(manifest: dict, indexed: dict[str, dict]) -> None:
    token_set = manifest["sets"]["tokens"]
    artifacts = token_set["artifacts"]
    require(set(artifacts) == {"dev_stochastic", "chunk_stochastic", "dev_greedy", "chunk_greedy"}, "token artifact keys")
    expected_paths = set(artifacts.values())
    require(
        expected_paths == {path for path in indexed if path.startswith("tokens/")},
        "token set membership is incomplete",
    )
    for mode, temperature in (("stochastic", 0.6), ("greedy", 0.0)):
        dev = validate_token_file(artifacts[f"dev_{mode}"], temperature)
        chunk = validate_token_file(artifacts[f"chunk_{mode}"], temperature)
        expected = token_set["summary"][mode]
        require(dev["sha256"] == expected["dev_payload_sha256"], f"{mode} dev payload summary")
        require(chunk["sha256"] == expected["chunk_payload_sha256"], f"{mode} chunk payload summary")
        require(
            divergence_summary(dev["tokens"], chunk["tokens"]) == expected["divergent_sequences"],
            f"{mode} divergence summary mismatch",
        )


def validate_smoke_and_provenance(manifest: dict, indexed: dict[str, dict]) -> None:
    smoke = manifest["sets"]["supplemental_smoke"]
    relative = smoke["artifact"]
    require(relative in indexed, "smoke artifact missing from inventory")
    rows = load_json(relative)
    require(isinstance(rows, list) and len(rows) == 4, "smoke output count mismatch")
    require(
        all(isinstance(row, list) and len(row) == 48 and all(isinstance(token, int) for token in row) for row in rows),
        "smoke token matrix shape/type mismatch",
    )
    require(indexed[relative]["sha256"] == smoke["summary"]["sha256"], "smoke summary SHA mismatch")
    require(smoke["summary"]["classification"] == "single-process smoke only", "smoke classification changed")

    protocols = manifest["protocols"]
    for protocol in ("tau_mixed_workload", "stock_throughput"):
        harness = protocols[protocol]["harness"]
        require(harness in indexed, f"missing harness for {protocol}")
        require(indexed[harness]["sha256"] == protocols[protocol]["harness_sha256"], f"harness SHA mismatch for {protocol}")
    require(
        protocols["tau128_post_repair_smoke"]["result_sha256"] == indexed[relative]["sha256"],
        "smoke protocol SHA mismatch",
    )

    for short, entry in manifest["git"]["post_run_resolution"]["objects"].items():
        require(bool(re.fullmatch(r"[0-9a-f]{7}", short)), f"invalid short commit label: {short}")
        require(bool(re.fullmatch(r"[0-9a-f]{40}", entry["full_sha"])), f"invalid full commit SHA: {short}")
        require(entry["full_sha"].startswith(short), f"commit resolution mismatch: {short}")
    model = manifest["model"]["post_run_snapshot_observation"]
    require(bool(re.fullmatch(r"[0-9a-f]{40}", model["huggingface_revision_from_local_metadata"])), "invalid HF revision")
    for entry in model["files"]:
        require_sha(entry["sha256"], f"model observation {entry['path']}")
        require(entry["bytes"] >= 0, f"invalid model byte count: {entry['path']}")


def main() -> None:
    require(MANIFEST_PATH.is_file(), "manifest.json is missing")
    manifest = load_json("manifest.json")
    require(manifest["schema_version"] == 1, "unsupported manifest schema")
    indexed = validate_artifact_inventory(manifest)
    validate_tau_sets(manifest, indexed)
    validate_stock(manifest, indexed)
    validate_tokens(manifest, indexed)
    validate_smoke_and_provenance(manifest, indexed)
    print(
        "validated "
        f"{len(indexed)} artifacts: 18 tau A/B memberships, "
        "18 constructor-sweep memberships, 6 stock runs, 4 token gates, "
        "2 external harnesses, and 1 supplemental smoke"
    )


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as error:
        raise SystemExit(f"validation failed: {error}") from error
