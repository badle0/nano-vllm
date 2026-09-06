#!/usr/bin/env python3
"""Model/CUDA-free validation of the bounded experimental V7 qualification."""
import argparse
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("v56_evidence", ROOT / "benchmarks/speculative_v5/validate_retained_evidence.py")
V56 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V56)
require, regular, load_json, git, sha = V56.require, V56.regular, V56.load_json, V56.git, V56.sha
SCHEMA = "nano-vllm-speculative-v7-retained-v1"
TRUSTED_MANIFEST_SHA256 = "PENDING"
BENCH_PRODUCER = "89829e6052c17e0ef4fcd65e294d0f1e78139184"
COLD_PRODUCER = "a715a199d413a67ba563271f1b4fa8fe87f00eaa"
OLD_REVISION = "2678d764ad0341bbfbdd2a93ac0e5528959058a4"
ROLES = tuple(f"primary-p{i}-{s}" for i in range(6) for s in ("off", "on"))
ROLES += tuple(f"regression-p{i}-{s}" for i in range(6) for s in ("old", "new"))
ROLES += ("extended-off", "extended-on", "cap5-on", "cap6-on", "graph-off", "graph-on", "phases-graph", "phases-eager")
TARGET_WEIGHTS = {
    "model-00001-of-00003.safetensors": "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    "model-00002-of-00003.safetensors": "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    "model-00003-of-00003.safetensors": "e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1",
}
FAMILIES = ("greedy", "plain", "topk", "topp", "combined")
FAILED_ATTEMPTS = {"attempt-noexec.log": "failed to map segment from shared object",
                   "attempt-first-burst.log": "assert abandoned_pending > 0"}


@lru_cache(maxsize=8)
def runtime_hashes(repo, revision):
    files = git(repo, "ls-tree", "-r", "--name-only", revision, "nanovllm").decode().splitlines()
    return {p: sha(git(repo, "show", f"{revision}:{p}")) for p in files if p.endswith(".py")}


def expected_cells(suite):
    if suite == "primary":
        return {(b, c, f, "generate", 64) for b in (1, 4, 8) for c in (32, 256) for f in FAMILIES}
    if suite == "off-regression":
        return {(b, 32, "plain", "generate", 64) for b in (1, 4, 8)}
    if suite == "smoke":
        return {(b, 32, f, "generate", 16) for b in (1, 4) for f in ("greedy", "plain")}
    require(suite == "extended", "unregistered benchmark suite")
    return ({(b, 32, "plain", "generate", 64) for b in (2, 16, 32, 64, 128)}
            | {(b, c, "plain", "generate", 64) for b in (1, 4) for c in (1024, 2048)}
            | {(1, 32, "plain", s, 16) for s in ("stream", "slow-stream")}
            | {(1, 32, "plain", "generate", n) for n in (1, 2, 4, 5, 256)})


def row_key(row):
    return tuple(row[k] for k in ("batch", "context", "family", "consumer", "completion", "seed"))


def check_models(models):
    require({k: v["sha256"] for k, v in models["target"].items() if k.endswith(".safetensors")} == TARGET_WEIGHTS,
            "wrong target weights")
    require(models["target"]["config.json"]["sha256"] == "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a", "wrong target config")
    require(models["draft"]["model.safetensors"]["sha256"] == V56.MODEL_SHA256, "wrong draft weights")


def check_benchmark(value, role, repo=ROOT):
    require(value["schema"] == "speculative-v7-benchmark-v1", "wrong benchmark schema")
    args = value["args"]
    suite = ("primary" if role.startswith("primary-") else "off-regression" if role.startswith("regression-")
             else "extended" if role.startswith("extended-") else "smoke")
    enabled = role.endswith("-on")
    require(args["suite"] == suite and args["enabled"] is enabled and args["mode"] == "graph", "benchmark role mismatch")
    require(args["configured_k"] == (int(role[3]) if role.startswith("cap") else 4), "wrong configured K")
    revision = OLD_REVISION if role.endswith("-old") else BENCH_PRODUCER
    require(value["runtime_revision"] == revision, "wrong runtime revision")
    expected = runtime_hashes(repo, revision)
    require(value["source_sha256"] == expected, "runtime source hash mismatch")
    require(value["harness_sha256"] == sha(git(repo, "show", f"{BENCH_PRODUCER}:tests/run_speculative_v7_benchmark.py")), "wrong timing harness")
    require(value["warmups_per_cell"] == 2 and value["seeds"] == [17, 23, 41], "warmup/seed mismatch")
    require(value["process_started_ns"] < value["samples_finished_ns"], "invalid process timing")
    require("A100-SXM4-40GB" in value["gpu"], "wrong GPU")
    config = value["config"]
    require(config["max_num_seqs"] == (128 if suite == "extended" else 8), "wrong batch capacity")
    require(config["max_num_batched_tokens"] == config["max_model_len"] == 4096, "wrong work limit")
    require(config["num_kvcache_blocks"] == 64 and config["gpu_memory_utilization"] == .8, "unequal capacity policy")
    require(config["enforce_eager"] is False and bool(value["workspace"]) is enabled, "wrong execution mode")
    check_models(value["models"])
    rows = value["records"]
    keys = [row_key(r) for r in rows]
    expected = {(*cell, seed) for cell in expected_cells(suite) for seed in (17, 23, 41)}
    require(len(keys) == len(expected) and set(keys) == expected, "incomplete or duplicate sample matrix")
    pair = int(role.split("-p")[1].split("-")[0]) if "-p" in role else 0
    require(args["pair"] == pair, "wrong pair ID")
    for row in rows:
        require(row["pair"] == pair, "sample pair ID mismatch")
        require(math.isfinite(row["seconds"]) and row["seconds"] > 0, "invalid sample time")
        require(math.isclose(row["tokens_per_second"], row["batch"] * row["completion"] / row["seconds"], rel_tol=1e-12), "throughput mismatch")
        require(row["accepted"] is (not row["rejection_reasons"]), "inconsistent sample exclusion")
        if row["accepted"]:
            for hardware in (row["hardware_before"], row["hardware_after"]):
                require(len(hardware["gpu_processes"]) == 1, "contended GPU sample")
                require(hardware["load_average"][0] <= .75 * hardware["cpu_count"], "busy host sample")
                require(float(hardware["gpu"].split(",")[4].strip()) < 85, "hot GPU sample")
        require(len(row["outputs"]) == row["batch"], "missing request output")
        for output in row["outputs"]:
            require(len(output["token_ids"]) == row["completion"], "wrong completion count")
            require(all(type(t) is int and 0 <= t < 151936 for t in output["token_ids"]), "invalid token")
            metrics = output["metrics"]
            require(metrics["num_completion_tokens"] == row["completion"] and metrics["num_prompt_tokens"] == row["context"], "wrong metrics counts")
            require(len(metrics["engine_itls"]) == row["completion"] - 1, "wrong token timing count")
            if enabled:
                require(metrics["spec_residual_numerical_fallbacks"] == 0, "natural residual fallback requires qualified law claim")
                if row["batch"] <= 4 and row["context"] <= 256 and row["completion"] > 2:
                    require(metrics["spec_cycles"] > 0, "eligible benchmark silently bypassed speculation")
                require(0 <= metrics["spec_accepted_draft_tokens"] <= metrics["spec_proposed_draft_tokens"] <= 4 * metrics["spec_cycles"], "invalid proposal/cap counts")
                require(metrics["spec_target_verification_positions"] == metrics["spec_proposed_draft_tokens"] + metrics["spec_cycles"], "wrong verifier work")
                if suite == "primary" and row["batch"] == 8:
                    require(metrics["spec_cycles"] == 0, "high-batch fallback not exercised")
            else:
                require(not any(k.startswith("spec_") for k in metrics), "off metrics changed")
    calibration = value["calibration"]
    require(calibration["copy_bytes"] == 2 * 256 * 1024**2 and calibration["matmul_shape"] == [4096] * 3, "wrong roof calibration")
    require(calibration["copy_ms"] > 0 and calibration["matmul_ms"] > 0, "invalid roof timing")
    require(math.isclose(calibration["bandwidth_bytes_per_second"], calibration["copy_bytes"] / (calibration["copy_ms"] / 1000), rel_tol=1e-12), "bandwidth arithmetic mismatch")
    require(math.isclose(calibration["bf16_flops_per_second"], 2 * 4096**3 / (calibration["matmul_ms"] / 1000), rel_tol=1e-12), "compute arithmetic mismatch")


def cross_checks(values):
    require([r["tokens"] for r in values["graph-on"]["results"]] == [r["tokens"] for r in values["graph-off"]["results"]], "cold greedy divergence")
    for prefix, sides in (("primary", ("off", "on")), ("regression", ("old", "new"))):
        previous_end = 0
        for pair in range(6):
            a, b = (values[f"{prefix}-p{pair}-{side}"] for side in sides)
            first, second = (a, b) if pair % 2 == 0 else (b, a)
            require(previous_end < first["process_started_ns"] < first["samples_finished_ns"] < second["process_started_ns"], "paired AB/BA ordering violation")
            previous_end = second["samples_finished_ns"]
            require(a["models"] == b["models"] and a["torch"] == b["torch"] and a["cuda"] == b["cuda"], "mismatched paired environment")
            left, right = ({row_key(r): r for r in run["records"]} for run in (a, b))
            require(left.keys() == right.keys(), "unpaired matrix")
            for key in left:
                if key[2] == "greedy" or prefix == "regression":
                    require([r["token_ids"] for r in left[key]["outputs"]] == [r["token_ids"] for r in right[key]["outputs"]], "baseline output divergence")
        cells = expected_cells("primary" if prefix == "primary" else "off-regression")
        for cell in cells:
            valid_pairs = 0
            for pair in range(6):
                rows = [r for side in sides for r in values[f"{prefix}-p{pair}-{side}"]["records"]
                        if row_key(r)[:-1] == cell]
                valid_pairs += len(rows) == 6 and all(r["accepted"] for r in rows)
            require(valid_pairs >= 5, "fewer than five valid independent pairs")


def check_phases(value, role, repo=ROOT):
    require(value["schema"] == "speculative-v7-phase-diagnostics-v1", "wrong phase schema")
    require(value["headline"] is False and value["synchronized"] is True, "instrumented phases misrepresented as headline")
    require(value["mode"] == role.split("-")[1], "wrong phase mode")
    require("A100-SXM4-40GB" in value["gpu"], "wrong phase GPU")
    check_models(value["models"])
    require(value["source_sha256"] == runtime_hashes(repo, BENCH_PRODUCER) == runtime_hashes(repo, value["revision"]), "phase runtime mismatch")
    require(value["harness_sha256"] == sha(git(repo, "show", f"{value['revision']}:tests/run_speculative_v7_phases.py")), "wrong phase harness")
    for label, ledger in value["weight_ledger"].items():
        require(label in ("target", "draft") and ledger["tied_storage"] is True, "wrong tied-weight model")
        require(ledger["parameter_object_bytes"] - ledger["unique_storage_bytes"] == ledger["embedding_bytes"] > 0, "tied storage counted twice")
        require(0 < 2 * ledger["linear_weight_elements"] <= ledger["unique_storage_bytes"], "invalid dense weight count")
    expected = {(b, f, w) for b in (1, 4) for f in ("greedy", "plain") for w in ("prose", "code", "repetitive", "adversarial")}
    require(len(value["cells"]) == 16 and {tuple(c["cell"]) for c in value["cells"]} == expected, "incomplete phase matrix")
    require(sum(c["cycles"] for c in value["cells"]) == len(value["cycles"]), "phase cycle counts mismatch")
    for cell in value["cells"]:
        actual = [c for c in value["cycles"] if c["cell"] == cell["cell"]]
        require(len(actual) == cell["cycles"] > 0, "missing phase workload")
        require(len(cell["outputs"]) == cell["cell"][0] and all(len(o["token_ids"]) == 32 for o in cell["outputs"]), "phase completion mismatch")
    for cycle in value["cycles"]:
        verify = "verify_greedy" if cycle["cell"][1] == "greedy" else "verify_parallel"
        require(set(cycle["phases"]) == {"draft", verify, "accept", "bonus", "run_speculative", "commit"}, "missing phase component")
        # Sampled rows can finish in different cycles; the live batch shrinks.
        require(1 <= cycle["k"] <= 4 and 1 <= cycle["batch"] <= cycle["cell"][0], "invalid phase shape")
        require(math.isfinite(cycle["total_seconds"]) and cycle["total_seconds"] > 0, "invalid phase total")
        require(all(len(times) == 1 and 0 < times[0] <= cycle["total_seconds"] for times in cycle["phases"].values()), "invalid phase timing")
        result = cycle["result"]
        require(len(result["rows"]) == cycle["batch"] and result["target_verification_positions"] == cycle["batch"] * (cycle["k"] + 1), "wrong phase result")
        for row in result["rows"]:
            require(row["residual_numerical_fallbacks"] == 0 and len(row["committed_token_ids"]) == row["accepted_draft_tokens"] + 1, "phase law violation")


def check_value(value, role, repo):
    if role.startswith("phases-"):
        check_phases(value, role, repo)
    elif role.startswith("graph-"):
        check_models(dict(target=value["model_files"], draft=value["draft_model_files"]))
        V56.check_run(value, role, COLD_PRODUCER, repo, model_files=value["model_files"],
                      work_limits=(4096, 4096), memory_utilization=.8)
        require(value["args"]["max_batch"] == 8, "wrong cold batch capacity")
    else:
        check_benchmark(value, role, repo)


def validate(archive, repo=ROOT, *, trusted=TRUSTED_MANIFEST_SHA256):
    raw = regular(archive / "manifest.json")
    require(sha(raw) == trusted, "untrusted archive manifest")
    manifest = load_json(raw)
    require(manifest["schema"] == SCHEMA and set(manifest["runs"]) == set(ROLES), "incomplete archive roles")
    require(manifest["benchmark_producer"] == BENCH_PRODUCER and manifest["cold_producer"] == COLD_PRODUCER
            and manifest["old_revision"] == OLD_REVISION, "manifest producer mismatch")
    require(manifest["target_hf_revision"] == "1cfa9a7208912126459214e8b04321603b3df60c", "wrong target checkpoint revision")
    require(set(manifest["failed_attempts"]) == set(FAILED_ATTEMPTS), "missing failed-attempt history")
    for name, marker in FAILED_ATTEMPTS.items():
        entry = manifest["failed_attempts"][name]
        require(entry["file"] == name and entry["classification"] == "failed-not-certified", "failed attempt mislabeled")
        data = regular(archive / name)
        require(len(data) == entry["bytes"] and sha(data) == entry["sha256"], "failed-attempt digest mismatch")
        require(marker.encode() in data and b"PASS " not in data, "wrong failed-attempt log")
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
        check_value(values[role], role, repo)
    cross_checks(values)
    return dict(runtime_tree=git(repo, "rev-parse", f"{BENCH_PRODUCER}:nanovllm").decode().strip(),
                runs=len(values), cold_cycles=len(values["graph-on"]["cycles"]),
                samples=sum(len(v["records"]) for v in values.values() if "records" in v),
                scope="bounded experimental; no default speedup claim")


def seal(source, destination):
    for component in (destination, *destination.parents):
        require(not component.is_symlink(), "archive path traverses symlink")
    require(not destination.exists(), "archive destination already exists")
    files, runs, values = {}, {}, {}
    for role in ROLES:
        record = {}
        for extension in ("json", "log"):
            filename = f"{role}.{extension}"
            data = regular(source / filename)
            files[filename] = data
            record[extension] = dict(file=filename, bytes=len(data), sha256=sha(data))
            if extension == "log":
                require(b"PASS " in data and b"Traceback (most recent call last)" not in data, "unsuccessful run log")
        runs[role] = record
        values[role] = load_json(files[f"{role}.json"])
        check_value(values[role], role, ROOT)
    cross_checks(values)
    failures = {}
    for name, marker in FAILED_ATTEMPTS.items():
        data = regular(source / name)
        require(marker.encode() in data and b"PASS " not in data, "wrong failed-attempt log")
        files[name] = data
        failures[name] = dict(file=name, bytes=len(data), sha256=sha(data), classification="failed-not-certified")
    raw = (json.dumps(dict(schema=SCHEMA, benchmark_producer=BENCH_PRODUCER, cold_producer=COLD_PRODUCER,
                           old_revision=OLD_REVISION, target_hf_revision="1cfa9a7208912126459214e8b04321603b3df60c",
                           runs=runs, failed_attempts=failures), indent=2, sort_keys=True) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in {**files, "manifest.json": raw}.items():
        with (destination / name).open("xb") as handle:
            handle.write(data)
    return dict(manifest_sha256=sha(raw), **validate(destination, trusted=sha(raw)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--seal-from", type=Path)
    args = parser.parse_args()
    print(json.dumps(seal(args.seal_from, args.archive) if args.seal_from else validate(args.archive), indent=2))


if __name__ == "__main__":
    main()
