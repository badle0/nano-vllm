"""Retain the tau {64,128} x max_model_len {512,1024,4096} contract matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from benchmarks.chunked_prefill_tail.common import (
    ROOT,
    base_result,
    environment_identity,
    handle_pin_query,
    immutable_write_json,
    model_identity,
    validate_release_pin,
)


DEFAULT_MODEL = "/workspace/models/Qwen3-0.6B"
MATRIX = tuple(
    (tau, max_model_len)
    for tau in (64, 128)
    for max_model_len in (512, 1024, 4096)
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and retain the six-cell low-budget graph contract matrix."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cell-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--print-source-sha256", action="store_true")
    parser.add_argument("--show-pin", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("expected_commit", "expected_source_sha256", "output")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "runtime matrix requires "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    if args.cell_timeout_seconds <= 0:
        raise ValueError("--cell-timeout-seconds must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite retained output: {args.output}")


def _run_cell(args: argparse.Namespace, tau: int, max_model_len: int) -> dict:
    command = [
        sys.executable,
        str(ROOT / "tests/run_varlen_graph_config.py"),
        "--tau",
        str(tau),
        "--max-model-len",
        str(max_model_len),
        "--model",
        args.model,
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = (
        str(ROOT)
        if not environment.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + environment["PYTHONPATH"]
    )
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=args.cell_timeout_seconds,
        check=False,
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(
            f"matrix cell tau={tau}, max_model_len={max_model_len} failed "
            f"with exit {completed.returncode}; stdout={completed.stdout!r}; "
            f"stderr={completed.stderr!r}"
        )
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError(
            f"matrix cell tau={tau}, max_model_len={max_model_len} emitted no JSON"
        )
    try:
        payload = json.loads(stdout_lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"matrix cell tau={tau}, max_model_len={max_model_len} ended with "
            f"non-JSON stdout: {stdout_lines[-1]!r}"
        ) from error
    if not payload.get("passed"):
        raise RuntimeError(
            f"matrix cell tau={tau}, max_model_len={max_model_len} did not pass"
        )
    return {
        "command": command,
        "elapsed_ms": elapsed_ms,
        "worker_stdout_before_json": stdout_lines[:-1],
        "worker_stderr": completed.stderr.splitlines(),
        "result": payload,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if handle_pin_query(args.print_source_sha256, args.show_pin):
        return 0
    _validate_args(args)

    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the retained contract matrix")

    started = perf_counter()
    pin = validate_release_pin(args.expected_commit, args.expected_source_sha256)
    model = model_identity(Path(args.model))
    environment = environment_identity(torch, transformers)
    result = base_result(
        "varlen_graph_tau_maxlen_contract_matrix",
        [sys.executable, *sys.argv],
        pin,
        model,
        environment,
    )
    result["arguments"] = vars(args) | {"output": str(args.output)}
    cells = []
    for tau, max_model_len in MATRIX:
        # Revalidate on both sides of every fresh process. A matrix is valid
        # only if every cell measured the same clean source identity.
        validate_release_pin(args.expected_commit, args.expected_source_sha256)
        cell = _run_cell(args, tau, max_model_len)
        validate_release_pin(args.expected_commit, args.expected_source_sha256)
        cells.append(cell)

    result.update({
        "matrix": {
            "tau_values": [64, 128],
            "max_model_len_values": [512, 1024, 4096],
            "cell_count": len(cells),
            "cells": cells,
            "passed": len(cells) == len(MATRIX)
            and all(cell["result"]["passed"] for cell in cells),
        },
        "elapsed_ms_before_write": (perf_counter() - started) * 1000.0,
        "provenance_after_run": validate_release_pin(
            args.expected_commit, args.expected_source_sha256
        ),
    })
    immutable_write_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "cells": len(cells),
        "passed": result["matrix"]["passed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
