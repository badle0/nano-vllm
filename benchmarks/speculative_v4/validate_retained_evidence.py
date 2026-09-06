#!/usr/bin/env python3
"""CPU/model-free V4 archive verifier and exclusive archive sealer.

The checked-in manifest digest is the release trust root. Sealing is an explicit
new-artifact operation, not validation; it never overwrites an existing archive.
No pickle, Torch, CUDA, model download, or execution of archived code is needed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

SCHEMA = "nano-vllm-speculative-v4-gpu-v1"
IMPLEMENTATION = "22b63e8e24db3c7bc9c24b61aedd93b25d76d289"
RUNTIME_TREE = "922d81417cf72ec912da13267fbb024c145a6a15"
CONFIG = dict(max_num_seqs=4, max_num_batched_tokens=1024, max_model_len=512,
              gpu_memory_utilization=0.5, num_kvcache_blocks=64, configured_k=2, seed=20260906,
              top_p_backend="exact", tensor_parallel_size=1)
NAMES = tuple(f"{mode}-{side}" for mode in ("eager", "graph") for side in ("off", "zero", "nan"))
TRUSTED_MANIFEST_SHA256 = "582e1e112213b2b5bdd796febce683af5d1530ee6cc45678dd374859b7e6bc1e"
HEX64 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def load_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")
    def finite_float(value):
        parsed = float(value)
        require(math.isfinite(parsed), f"non-finite JSON value: {value}")
        return parsed
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)


def regular(path):
    for component in (path, *path.parents):
        require(not component.is_symlink(), f"symlink is not allowed: {component}")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, f"not a singly linked regular file: {path}")
    require(info.st_size <= 32 * 1024**2, f"artifact exceeds 32 MiB: {path}")
    return path.read_bytes()


def git(repo, *args):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return subprocess.run(["git", "--no-replace-objects", *args], cwd=repo, env=env,
                          check=True, capture_output=True).stdout


def check_provenance(value, repo):
    p = value["provenance"]
    require(p["implementation_commit"] == IMPLEMENTATION and p["implementation_nanovllm_tree"] == RUNTIME_TREE,
            "incorrect implementation pin")
    require(p["retention_eligible"] is True and p["before"] == p["after"], "provenance changed")
    producer = p["producer_commit"]
    require(re.fullmatch(r"[0-9a-f]{40}", producer), "invalid producer commit")
    require(git(repo, "rev-parse", f"{IMPLEMENTATION}:nanovllm").decode().strip() == RUNTIME_TREE, "historical V4 runtime drifted")
    require(git(repo, "rev-parse", f"{producer}:nanovllm").decode().strip() == RUNTIME_TREE, "producer altered runtime")
    before = p["before"]
    source = before["source"]
    require(source["head"] == producer and source["clean"] is True and source["status_porcelain_v1"] == [], "dirty producer")
    require(source["nanovllm_tree"] == RUNTIME_TREE, "snapshot runtime mismatch")
    require(source["tree"] == git(repo, "rev-parse", f"{producer}^{{tree}}").decode().strip(), "producer tree mismatch")
    for identities in (before["files"], before["imports"]):
        require(bool(identities), "missing source identities")
        for identity in identities.values():
            path = identity["path"]
            require(path.startswith(("tests/", "nanovllm/")) and ".." not in Path(path).parts, "unsafe source path")
            payload = git(repo, "show", f"{producer}:{path}")
            require(identity["matches_head"] is True and sha(payload) == identity["sha256"], "historical source hash mismatch")
    require(set(before["files"]) == {
        "tests/run_speculative_v4_gpu.py", "tests/_speculative_v4_evidence.py",
        "tests/_speculative_v3_evidence.py", "tests/run_speculative_v3_route_compile.py",
    }, "incomplete harness provenance")
    require(set(before["imports"]) == {"nanovllm", "LLM", "LLMEngine", "ModelRunner", "Scheduler", "SpecStepPlan", "Sampler"},
            "incomplete runtime import provenance")
    hardware = before["environment"]["hardware"]
    require(hardware["cuda_available"] is True and hardware["device_count"] == 1, "not single-GPU CUDA")
    device, = hardware["devices"]
    require(device["name"] == "NVIDIA A100-SXM4-40GB" and device["compute_capability"] == [8, 0], "wrong GPU")
    software = before["environment"]["software"]
    require(software["python_optimize"] == 0 and software["torch_dynamo_disable"] is False
            and software["torch_dynamo_suppress_errors"] is False, "vacuous runtime mode")
    require(before["model"]["weight_files"] and before["model"]["metadata_files"], "missing model identity")
    for item in [*before["model"]["weight_files"], *before["model"]["metadata_files"].values()]:
        require(item["size_bytes"] > 0 and HEX64.fullmatch(item["sha256"]), "invalid model fingerprint")
    return producer


def check_plan(plan):
    rows = plan["rows"]
    batch, k = len(rows), plan["effective_k"]
    require(type(k) is int and 0 <= k <= 2, "invalid K")
    require(plan["gpu_certified"] is False, "modeled plan falsely GPU-certified")
    require(HEX64.fullmatch(plan["workspace_fingerprint"]), "invalid workspace fingerprint")
    if not k:
        require(plan["bypass_reason"] and plan["route_key"] is None, "invalid fallback")
        require(plan["modeled_live_peak_bytes"] == plan["reservation_bytes"] == 0, "fallback reserved workspace")
        require(plan["draft_catchup_tokens"] == plan["draft_query_tokens"] == 0, "fallback drafted")
        require(plan["total_model_positions"] == plan["target_query_tokens"] == batch, "fallback counts drifted")
        require(all(row["highest_target_write_position"] is None and row["highest_draft_write_position"] is None for row in rows), "fallback names writes")
        return
    require(plan["bypass_reason"] is None, "positive bypass")
    require(1 <= batch <= 4 and len({row["seq_id"] for row in rows}) == batch, "invalid batch")
    catchup = sum(row["target_cached_tokens"] - row["draft_cached_tokens"] for row in rows)
    require(plan["draft_catchup_tokens"] == catchup >= 0, "catchup count mismatch")
    require(plan["draft_query_tokens"] == batch*k, "draft count mismatch")
    require(plan["target_query_tokens"] == batch*(k+1) <= 1024, "verifier count mismatch")
    require(plan["total_model_positions"] == catchup+batch*k+batch*(k+1) <= 1024, "aggregate mismatch")
    require(plan["route_key"]["effective_k"] == k, "route K mismatch")
    bucket = plan["route_key"]["batch_bucket"]
    require(batch <= bucket <= 4, "bad route bucket")
    # Both q[B,K,V] and p[B,K+1,V] must fit even before transforms/sort/race.
    require(4*bucket*(2*k+1)*151936 <= plan["modeled_live_peak_bytes"] <= plan["reservation_bytes"], "q+p memory omitted")
    for row in rows:
        length = row["committed_tokens"]
        require(row["target_cached_tokens"] == length-1 and 0 <= row["draft_cached_tokens"] <= length-1, "not pure decode")
        require(row["remaining_completion_tokens"] >= k+1 and row["model_position_headroom"] >= k, "headroom violation")
        require(row["highest_draft_write_position"] == length+k-2, "draft write position drifted")
        require(row["highest_target_write_position"] == length+k-1 < 512, "target write position drifted")
        require(len(row["block_table"]) >= (length+k+255)//256, "missing full target reservation")


def check_raw(value, name, log, repo):
    mode, side = name.split("-")
    require(value["schema"] == SCHEMA and value["mode"] == mode and value["side"] == side, "raw identity mismatch")
    require(value["configuration"] == CONFIG, "configuration drifted")
    producer = check_provenance(value, repo)
    require(value["captures_after_init"] == value["captures_after_runtime"], "new runtime graph capture")
    init = value["compiler_after_init"]
    require(init["counters"]["'stats'"]["'unique_graphs'"] > 0 and init["cache_manifest"], "compiler proof is vacuous")
    if mode == "graph":
        require(value["captures_after_init"]["cuda_graph_objects"] > 0, "graph mode captured no graph")
    control = value["control"]
    require(len(control["events"]) == 6 and all(len(step) == 2 for step in control["events"]), "incomplete output control")
    require(len(control["ids"]) == len(set(control["ids"])) == 2, "invalid control request IDs")
    for index, step in enumerate(control["events"]):
        require([event[0] for event in step] == control["ids"], "control event order drifted")
        require(all(type(event[1]) is int and 0 <= event[1] < 151936
                    and event[2] is (index == 5) for event in step), "invalid control token/finish flags")
    require(len(control["rng"]) == 7, "incomplete RNG checkpoints")
    records = value.get("records", [])
    if side == "off":
        require(not records and "registry" not in value, "off control ran draft work")
    else:
        expected_labels = {f"route/{b}/{k}/{r}/{p}" for b in range(1,5) for k in (1,2)
                           for r in range(2) for p in ("cold", "warm")}
        route_records = [record for record in records if record["label"].startswith("route/")]
        require(len(route_records) == 32 and {r["label"] for r in route_records} == expected_labels, "incomplete route sweep")
        visited = {json.dumps(r["plan"]["route_key"], sort_keys=True) for r in route_records}
        require(visited == {json.dumps(key, sort_keys=True) for key in value["registry"]}, "unvisited ready key")
        require(len(records) == 46, "incomplete enabled interval workload")
        all_labels = (expected_labels | {f"control/{i}" for i in range(1,5)}
                      | {f"{phase}/{length}" for phase in ("boundary", "failure") for length in (254,255,256,257)}
                      | {"prefix/cold", "prefix/hit"})
        require({record["label"] for record in records} == all_labels, "incomplete control/boundary/prefix intervals")
        for record in records:
            plan = record["plan"]
            check_plan(plan)
            require(plan["effective_k"] > 0, "fallback executed draft")
            require(record["compiler_before"] == record["compiler_after"], "draft compiler delta")
            require(record["rng_before"] == record["rng_after"] and record["context_default"] is True, "draft changed RNG/context")
            k, batch = plan["effective_k"], len(plan["rows"])
            require(record["graph_steps"] == (k if mode == "graph" else 0), "graph replay count drifted")
            require(record["eager_steps"] == (k if mode == "eager" else 0), "eager count drifted")
            require(len(record["samples"]) == k and len(record["proposals"]) == batch, "incomplete proposals")
            require(all(len(row) == k for row in record["proposals"]), "incomplete per-row proposals")
            for step, sample in enumerate(record["samples"]):
                require(sample["shape"] == [batch,151936], "probability geometry drifted")
                require(len(sample["sums"]) == len(sample["token_ids"]) == batch, "incomplete probability observations")
                require(all(math.isfinite(x) and abs(x-1) <= 2e-6 for x in sample["sums"]), "invalid probability sums")
                require(all(type(token) is int and 0 <= token < 151936 for token in sample["token_ids"]), "invalid proposal token")
                require(sample["token_ids"] == [row[step] for row in record["proposals"]], "sampler/proposal tokens disagree")
                require(HEX64.fullmatch(sample["q_sha256"]) and HEX64.fullmatch(sample["logits_sha256"]), "missing numerical observation")
            if record["label"].startswith("route/"):
                _, b, expected_k, _, phase = record["label"].split("/")
                require(batch == int(b) and k == int(expected_k) == record["forced_route_cap"], "route geometry coverage mismatch")
                require((plan["draft_catchup_tokens"] > 0) == (phase == "cold"), "route catchup mismatch")
        for item in value["plans"]:
            check_plan(item["plan"])
        require([b["length"] for b in value["boundaries"]] == [254,255,256,257], "boundary matrix missing")
        require(any(b["extra_blocks"] > 0 for b in value["boundaries"]), "no speculative suffix exercised")
        require(all(b["before_sha256"] == b["after_sha256"] for b in value["boundaries"]), "rollback mismatch")
        cold, hit = value["prefix"]
        require(cold["prefill_tokens"] == 257 and hit["prefill_tokens"] == 1, "prefix-cache oracle missing")
        require(cold["samples"] == hit["samples"] and cold["target_tokens"] == hit["target_tokens"], "prefix numerics drifted")
    # Pair run-bound stderr markers, and reject warnings inside draft intervals.
    observed, active = [], None
    for line in log.decode("utf-8").splitlines():
        if line.startswith("V4_DRAFT_INTERVAL_BEGIN "):
            require(active is None, "nested draft interval")
            active = line.split(" ", 1)[1]
        elif line.startswith("V4_DRAFT_INTERVAL_END "):
            require(active == line.split(" ",1)[1], "unmatched draft interval")
            observed.append(active)
            active = None
        elif active is not None:
            require(not re.search(r"recompil|graph.break|warning|traceback", line, re.I), "diagnostic inside guarded draft interval")
    require(active is None and observed == [r["interval"] for r in records], "log/JSON interval coverage mismatch")
    return producer


def check_set(directory, repo):
    data, descriptors, producers, cache_roots = {}, {}, set(), []
    for name in NAMES:
        raw = regular(directory / f"{name}.json")
        log = regular(directory / f"{name}.log")
        value = load_json(raw)
        producers.add(check_raw(value, name, log, repo))
        data[name] = value
        for suffix, payload in (("json", raw), ("log", log)):
            descriptors[f"{name}.{suffix}"] = {"bytes": len(payload), "sha256": sha(payload)}
        env = value["provenance"]["before"]["environment"]["selected_environment"]
        for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            root = Path(env[key])
            require(root.is_absolute(), "cache root not absolute")
            require(all(root != old and root not in old.parents and old not in root.parents for old in cache_roots), "producer caches overlap")
            cache_roots.append(root)
    require(len(producers) == 1, "mixed producer commits")
    identity = None
    for value in data.values():
        before = value["provenance"]["before"]
        current = (before["model"], before["environment"]["hardware"], before["environment"]["software"], before["source"])
        if identity is None:
            identity = current
        require(current == identity, "source/model/hardware/software changed between producers")
    for mode in ("eager", "graph"):
        off, zero, nan = (data[f"{mode}-{side}"] for side in ("off", "zero", "nan"))
        require(off["control"] == zero["control"] == nan["control"], "target outputs or RNG diverged")
        require(len(zero["records"]) == len(nan["records"]), "fill coverage differs")
        for a, b in zip(zero["records"], nan["records"]):
            for field in ("label", "plan", "samples", "proposals", "rng_before", "rng_after"):
                require(a[field] == b[field], f"zero/NaN {field} differs")
        require(zero["boundaries"] == nan["boundaries"] and zero["prefix"] == nan["prefix"], "cache-fill boundary output differs")
    return {"schema": "nano-vllm-speculative-v4-archive-v1", "producer_commit": producers.pop(),
            "implementation_commit": IMPLEMENTATION, "runtime_tree": RUNTIME_TREE,
            "files": descriptors,
            "claims": {"modes": ["eager","graph"], "draft_route_intervals_per_enabled_run": 32,
                       "all_intervals_per_enabled_run": 46, "accepted_tokens": False,
                       "verifier_memory_peak_certified": False, "performance_certified": False}}


def validate(directory, repo):
    payload = regular(directory / "manifest.json")
    require(TRUSTED_MANIFEST_SHA256 is not None and sha(payload) == TRUSTED_MANIFEST_SHA256, "untrusted archive manifest")
    expected = {f"{name}.{ext}" for name in NAMES for ext in ("json", "log")} | {"manifest.json"}
    require({p.name for p in directory.iterdir()} == expected, "unexpected/missing archive members")
    manifest = load_json(payload)
    for name, descriptor in manifest["files"].items():
        raw = regular(directory / name)
        require(descriptor == {"bytes":len(raw), "sha256":sha(raw)}, "pinned artifact bytes changed")
    require(check_set(directory, repo) == manifest, "archive semantic/provenance checks differ")
    return manifest["claims"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seal-to", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.seal_to is None:
        result = validate(args.directory, repo)
    else:
        destination = args.seal_to.absolute()
        require(not destination.exists(), "refusing to overwrite archive")
        for parent in destination.parents:
            require(not parent.is_symlink(), "archive parent is symlinked")
        manifest = check_set(args.directory, repo)
        destination.mkdir(parents=True, exist_ok=False)
        for name in manifest["files"]:
            shutil.copyfile(args.directory / name, destination / name)
        payload = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
        with (destination / "manifest.json").open("xb") as output:
            output.write(payload)
        result = {"sealed_to": str(destination), "manifest_sha256": sha(payload)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
