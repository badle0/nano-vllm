"""Check the preserved package, runtime, GPU, and native-extension contract."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    expected = json.loads((here / "environment.json").read_text())
    errors = []
    installed = {}
    for name, version in expected["packages"].items():
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
        if installed[name] != version:
            errors.append(f"{name}: expected {version}, got {installed[name]}")
    if platform.python_version() != expected["python"]:
        errors.append(f"Python must be {expected['python']}")
    if platform.machine() != expected["architecture"]:
        errors.append(f"Architecture must be {expected['architecture']}")
    digest = hashlib.sha256()
    for path in sorted((root / "nanovllm").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != expected["runtime_sha256"]:
        errors.append("Runtime differs from the reviewed source snapshot")
    # Imports also detect a FlashAttention/Torch ABI mismatch.
    import torch
    from flash_attn import flash_attn_func

    gpu = None
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable")
    else:
        gpu = torch.cuda.get_device_name()
        if gpu != expected["gpu"]:
            errors.append(f"Baseline GPU must be {expected['gpu']}; got {gpu}")
        if list(torch.cuda.get_device_capability()) != expected["gpu_compute_capability"]:
            errors.append("GPU compute capability differs")
        q = torch.zeros((1, 2, 2, 128), device="cuda", dtype=torch.bfloat16)
        output = flash_attn_func(q, q, q, causal=True)
        torch.cuda.synchronize()
        if not torch.equal(output, q):
            errors.append("Native FlashAttention smoke failed")
    if torch.version.cuda != expected["torch_cuda"]:
        errors.append("Torch CUDA version differs")
    if torch.compiled_with_cxx11_abi() != expected["torch_cxx11_abi"]:
        errors.append("Torch C++ ABI differs")
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).strip()
    report = {
        "passed": not errors,
        "errors": errors,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu,
        "driver_version": driver,
        "driver_matches_original": driver == expected["driver_version"],
        "packages": installed,
        "runtime_sha256": digest.hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
