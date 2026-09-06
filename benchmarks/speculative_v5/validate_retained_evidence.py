#!/usr/bin/env python3
"""Model/CUDA-free V5/V6 archive checker. Sealing never overwrites an archive."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("v4_archive_helpers", ROOT / "benchmarks/speculative_v4/validate_retained_evidence.py")
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
require, regular, load_json, git = (_helpers.require, _helpers.regular, _helpers.load_json, _helpers.git)
SCHEMA = "nano-vllm-speculative-v5-v6-retained-v1"
ROLES = ("eager-off", "eager-on", "graph-off", "graph-on", "graph-auto-k2")
TRUSTED_MANIFEST_SHA256 = "1c8e3b81f18e74a1d2f6b89601213c5d239f75b771646e88ef13f6b06f79a302"
MODEL_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def check_run(value, role, producer, repo=ROOT, *, model_files=None,
              work_limits=(1024, 512), memory_utilization=.5):
    require(value["schema"] == "nano-vllm-speculative-v5-gpu-v1", "wrong run schema")
    require(value["revision"] == producer and value["args"]["expected_commit"] == producer, "producer mismatch")
    require(value["args"]["retained"] is True, "exploratory run cannot be retained")
    require("A100-SXM4-40GB" in value["gpu"], "wrong GPU")
    if model_files is None:
        require(value["model_files"]["model.safetensors"]["sha256"] == MODEL_SHA256, "wrong model weights")
    else:
        require(value["model_files"] == model_files, "wrong model weights/configuration")
    tracked = git(repo, "ls-tree", "-r", "--name-only", producer, "nanovllm").decode().splitlines()
    expected_files = {name for name in tracked if name.endswith(".py")}
    require(set(value["source_sha256"]) == expected_files, "incomplete runtime source identities")
    for path, digest in value["source_sha256"].items():
        require(sha(git(repo, "show", f"{producer}:{path}")) == digest, "runtime source hash mismatch")
    mode, side = role.split("-", 1)
    enabled = side != "off"
    require(value["args"]["mode"] == mode and value["args"]["enabled"] is enabled, "role mismatch")
    require((value["config"]["max_num_batched_tokens"], value["config"]["max_model_len"]) == work_limits, "work-limit mismatch")
    require(value["config"]["gpu_memory_utilization"] == memory_utilization, "memory budget mismatch")
    require([(r["length"], r["batch"]) for r in value["results"]] == [(16, 1), (255, 2), (256, 3), (257, 4)], "missing boundary workload")
    for row in value["results"]:
        require(row["cache_repeat"] is True and row["stream_parity"] is True, "cache/stream control failed")
        require(len(row["tokens"]) == row["batch"] and all(len(tokens) == 17 for tokens in row["tokens"]), "wrong completion lengths")
        require(all(type(t) is int and 0 <= t < 151936 for tokens in row["tokens"] for t in tokens), "invalid token")
    require(value["mixed_sample_lengths"] == [13] * 4 and value["stochastic_lengths"] == [13] * 4, "missing sampling families")
    snapshot = value["init_compiler_snapshot"]
    require(snapshot and snapshot["cache_manifest"], "vacuous compiler evidence")
    require(value["init_compiles"].get("stats/unique_graphs", 0) > 0, "compiler was not exercised")
    require(not value["init_compiles"].get("unimplemented/recompile_limit reached", 0), "constructor fell back at compile limit")
    if not enabled:
        require(not value["cycles"] and value["audit"] is None, "off control executed speculation")
        require(not any(k.startswith("spec_") for k in value["completed_metrics"]), "off metrics changed")
        return
    require(value["failure_retry"] is True and value["causality_probe"] is True, "missing failure/causality gate")
    require(value["pending_before_close"] > 0 and value["abandoned_pending"] > 0, "vacuous pending-burst lifecycle test")
    require(value["forced_metrics"]["spec_residual_numerical_fallbacks"] == 1, "forced residual not counted once")
    require(value["completed_metrics"]["spec_residual_numerical_fallbacks"] == 0, "unexpected natural fallback")
    require(0. in value["completed_metrics"]["engine_itls"], "missing intra-burst compute timestamps")
    audit, cycles = value["audit"], value["cycles"]
    require(cycles and audit["modeled_runtime_headroom_bytes"] >= 0, "invalid runtime headroom")
    require(audit["gpu_certified"] is False, "narrow evidence must not imply universal certification")
    plan = audit["workspace_plan"]
    natural_fallbacks = forced_fallbacks = 0
    for cycle in cycles:
        b, k = cycle["batch"], cycle["k"]
        require(type(b) is int and 1 <= b <= 4 and type(k) is int and 1 <= k <= 4, "unregistered execution shape")
        require(cycle["verifier"] in ("paged_parallel", "sequential_greedy"), "unregistered target lane")
        require(cycle["compiler_unchanged"] is True and not cycle["compile_delta"], "runtime compilation/capture changed")
        require(re.fullmatch(r"[0-9a-f]{64}", cycle["compiler_sha256"] or ""), "missing interval compiler digest")
        require(0 <= cycle["peak_increment"] <= plan["reservation_bytes"] + audit["warmup_transient_bytes"], "reservation exceeded")
        if cycle["purpose"] == "normal":
            require(cycle["peak_increment"] <= plan["modeled_live_peak_bytes"], "modeled production live peak exceeded")
        result = cycle["result"]
        require(result["target_verification_positions"] == b * (k + 1), "wrong target position count")
        require(result["draft_positions"] == cycle["catchup"] + b * k, "wrong draft position count")
        require(len(result["rows"]) == b, "wrong result row count")
        for row in result["rows"]:
            count = row["accepted_draft_tokens"]
            require(type(count) is int and 0 <= count <= k, "invalid acceptance count")
            require(row["used_bonus"] is (count == k), "invalid bonus flag")
            require(len(row["proposed_token_ids"]) == k and len(row["committed_token_ids"]) == count + 1, "invalid burst length")
            require(row["committed_token_ids"][:count] == row["proposed_token_ids"][:count], "wrong accepted prefix")
            require(row["residual_numerical_fallbacks"] in (0, 1), "invalid residual count")
            if cycle["purpose"] == "forced_empty_residual":
                forced_fallbacks += row["residual_numerical_fallbacks"]
            else:
                natural_fallbacks += row["residual_numerical_fallbacks"]
    require(natural_fallbacks == 0 and forced_fallbacks == 1, "residual recovery classification mismatch")
    require({c["verifier"] for c in cycles} == {"paged_parallel", "sequential_greedy"}, "target lane not exercised")
    require([s["rows"] for s in value["sort_scratch"]] == [4, 12, 16, 20], "missing sort-scratch probe")
    for probe in value["sort_scratch"]:
        require(0 < probe["measured"] <= probe["priced"] == probe["rows"] * 151936 * 52, "private sort scratch exceeded allowance")
    if side == "on":
        expected = {(b, k, family) for b in range(1, 5) for k in range(1, 5)
                    for family in ("greedy", "plain", "topk", "topp", "combined")}
        actual = {(c["batch"], c["k"], c["sampling"]) for c in value["sweep_cells"]}
        require(actual == expected and len(value["sweep_cells"]) == 80, "incomplete route/sampling sweep")
        require(value["args"]["configured_k"] == 4, "wrong configured K")
    else:
        require(side == "auto-k2" and value["args"]["auto_kv"] is True and value["args"]["configured_k"] == 2, "wrong auto-sizing regression cell")


def validate(archive, repo=ROOT, *, trusted=TRUSTED_MANIFEST_SHA256):
    manifest_bytes = regular(archive / "manifest.json")
    require(trusted and sha(manifest_bytes) == trusted, "untrusted archive manifest")
    manifest = load_json(manifest_bytes)
    require(manifest["schema"] == SCHEMA and set(manifest["runs"]) == set(ROLES), "incomplete archive")
    producer = manifest["producer"]
    require(re.fullmatch(r"[0-9a-f]{40}", producer), "invalid producer")
    require(git(repo, "rev-parse", f"{producer}:nanovllm").decode().strip() == manifest["runtime_tree"], "runtime tree mismatch")
    require(sha(git(repo, "show", f"{producer}:tests/run_speculative_v5_gpu.py")) == manifest["harness_sha256"], "harness mismatch")
    values = {}
    for role, record in manifest["runs"].items():
        for extension in ("json", "log"):
            entry = record[extension]
            require(entry["file"] == f"{role}.{extension}", "unsafe artifact path")
            data = regular(archive / entry["file"])
            require(len(data) == entry["bytes"] and sha(data) == entry["sha256"], "artifact digest mismatch")
            if extension == "json":
                values[role] = load_json(data)
            else:
                require(b"PASS " in data and b"Traceback (most recent call last)" not in data, "unsuccessful run log")
        check_run(values[role], role, producer, repo)
    for mode in ("eager", "graph"):
        off, on = values[f"{mode}-off"], values[f"{mode}-on"]
        require(off["args"]["max_batch"] == on["args"]["max_batch"], "mismatched control configuration")
        require([r["tokens"] for r in off["results"]] == [r["tokens"] for r in on["results"]], "matching-mode greedy control diverged")
    return dict(producer=producer, modes=["eager", "graph"],
                route_cells=160, enabled_cycles=sum(len(v["cycles"]) for v in values.values()),
                natural_fallbacks=0, automatic_kv_graph_k2=True)


def seal(source, destination, producer, repo=ROOT):
    for component in (destination, *destination.parents):
        require(not component.is_symlink(), "archive path must not traverse symlinks")
    require(not destination.exists() and not destination.is_symlink(), "archive destination already exists")
    require(re.fullmatch(r"[0-9a-f]{40}", producer), "invalid producer")
    files, runs = {}, {}
    for role in ROLES:
        record = {}
        for extension in ("json", "log"):
            payload = regular(source / f"spec-v5-final-{role}.{extension}")
            filename = f"{role}.{extension}"
            files[filename] = payload
            record[extension] = dict(file=filename, bytes=len(payload), sha256=sha(payload))
        check_run(load_json(files[f"{role}.json"]), role, producer, repo)
        runs[role] = record
    manifest = dict(schema=SCHEMA, producer=producer,
                    runtime_tree=git(repo, "rev-parse", f"{producer}:nanovllm").decode().strip(),
                    harness_sha256=sha(git(repo, "show", f"{producer}:tests/run_speculative_v5_gpu.py")), runs=runs)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=False)
    for filename, data in {**files, "manifest.json": payload}.items():
        with (destination / filename).open("xb") as handle:
            handle.write(data)
    digest = sha(payload)
    result = validate(destination, repo, trusted=digest)
    return dict(manifest_sha256=digest, **result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--seal-from", type=Path)
    parser.add_argument("--producer")
    args = parser.parse_args()
    result = (seal(args.seal_from, args.archive, args.producer) if args.seal_from
              else validate(args.archive))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
