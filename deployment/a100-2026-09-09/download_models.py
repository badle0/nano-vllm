"""Restore pinned target/draft checkpoints and verify archived content hashes."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil


def fingerprint(path, expected):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"]:
        return None
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {expected['bytes']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(block)
            blob.update(block)
    if "sha256" in expected and sha256.hexdigest() != expected["sha256"]:
        return None
    if "git_blob_sha1" in expected and blob.hexdigest() != expected["git_blob_sha1"]:
        return None
    return {"bytes": expected["bytes"], "sha256": sha256.hexdigest()}


def restore(root, name, expected, verify_only):
    target = root / name
    staging = root / ("." + name + ".download")
    if target.is_symlink() or staging.is_symlink():
        raise RuntimeError(f"Refusing symlink checkpoint/staging path for {name}")
    directory = target if target.exists() or verify_only else staging
    verified = {}
    missing = []
    for filename, details in expected["files"].items():
        result = fingerprint(directory / filename, details)
        if result is None:
            missing.append(filename)
        else:
            verified[filename] = result
    if missing and (verify_only or target.exists()):
        raise RuntimeError(f"Incomplete or invalid checkpoint at {target}: {missing}")
    if missing:
        required = sum(expected["files"][f]["bytes"] for f in missing) + 512 * 1024**2
        free = shutil.disk_usage(root).free
        if free < required:
            raise RuntimeError(f"{root}: need {required:,} bytes free; found {free:,}")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        from huggingface_hub import snapshot_download

        snapshot_download(
            expected["repo_id"], revision=expected["revision"], local_dir=staging,
            allow_patterns=missing, max_workers=1, token=False,
            force_download=any((staging / filename).exists() for filename in missing),
        )
        for filename in missing:
            result = fingerprint(staging / filename, expected["files"][filename])
            if result is None:
                raise RuntimeError(f"Integrity check failed: {staging / filename}")
            verified[filename] = result
    from safetensors import safe_open

    tensor_files = {}
    for filename in verified:
        if filename.endswith(".safetensors"):
            with safe_open(directory / filename, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in tensor_files:
                        raise RuntimeError(f"Duplicate tensor: {key}")
                    tensor_files[key] = filename
    index = directory / "model.safetensors.index.json"
    if index.exists() and json.loads(index.read_text())["weight_map"] != tensor_files:
        raise RuntimeError("Shard index does not match the safetensors headers")
    if not tensor_files:
        raise RuntimeError("No model tensors found")
    if not verify_only:
        if directory == staging:
            if target.exists():
                raise RuntimeError(f"Destination appeared during download: {target}")
            staging.rename(target)
        # Keep verification metadata outside the model directory: the existing
        # benchmark fingerprint includes every file under that directory.
        records = root / ".verification"
        records.mkdir(exist_ok=True)
        record = {
            "repo_id": expected["repo_id"], "revision": expected["revision"],
            "destination": str(target), "files": verified,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "tensor_count": len(tensor_files),
        }
        (records / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    print(f"Verified {name}@{expected['revision']}: {len(tensor_files)} tensors; {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("/workspace/models"))
    parser.add_argument("--model", choices=("Qwen3-0.6B", "Qwen3-4B", "all"), default="all")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    expected = json.loads(Path(__file__).with_name("models.lock.json").read_text())["models"]
    names = ("Qwen3-0.6B", "Qwen3-4B") if args.model == "all" else (args.model,)
    if not args.models_root.is_dir():
        raise RuntimeError(f"Create the model parent directory first: {args.models_root}")
    if args.verify_only:
        for name in names:
            restore(args.models_root, name, expected[name], True)
        return
    with (args.models_root / ".checkpoint-download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in names:
            restore(args.models_root, name, expected[name], False)


if __name__ == "__main__":
    main()
