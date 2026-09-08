"""The CPU matrix supports Python 3.10, which has no datetime.UTC alias."""

import os
from pathlib import Path
import subprocess
import sys


def test_chunk_benchmark_imports_without_datetime_utc():
    # Run in a child: changing a stdlib module must not affect other tests.
    # Removing only the newer alias also reproduces the collection failure
    # on a developer machine running Python 3.11+.
    script = """
import datetime
import importlib

if hasattr(datetime, "UTC"):
    del datetime.UTC

for name in (
    "common",
    "full_completion_cert",
    "aggregate_certification",
    "scheduler_roofline",
):
    module = importlib.import_module("benchmarks.chunked_prefill_tail." + name)
    assert module.timezone.utc is datetime.timezone.utc
    assert module.datetime.now(module.timezone.utc).isoformat().endswith("+00:00")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
