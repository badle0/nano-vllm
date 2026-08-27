from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

import pytest

import nanovllm.config as config_module
from nanovllm.config import Config


@pytest.fixture
def config_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda model: SimpleNamespace(max_position_embeddings=8192),
    )
    return tmp_path


@pytest.mark.parametrize("backend", ["exact", "flashinfer"])
def test_top_p_backend_accepts_documented_values(config_dependencies, backend):
    assert Config(
        str(config_dependencies), top_p_backend=backend
    ).top_p_backend == backend


@pytest.mark.parametrize("backend", [None, True, 1])
def test_top_p_backend_rejects_non_strings(config_dependencies, backend):
    with pytest.raises(TypeError, match="must be a string"):
        Config(str(config_dependencies), top_p_backend=backend)


def test_top_p_backend_rejects_unknown_value(config_dependencies):
    with pytest.raises(ValueError, match="exact.*flashinfer"):
        Config(str(config_dependencies), top_p_backend="fast")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kvcache_block_size=0),
        dict(kvcache_block_size=255),
        dict(max_model_len=-5),
        dict(max_model_len=2.5),
        dict(gpu_memory_utilization=7.0),
        dict(gpu_memory_utilization=-0.2),
        dict(gpu_memory_utilization=float("nan")),
        dict(gpu_memory_utilization=float("inf")),
        dict(gpu_memory_utilization=True),
        dict(enforce_eager="maybe"),
        dict(tensor_parallel_size=9),
        dict(tensor_parallel_size=1.0),
        dict(num_kvcache_blocks=0),
    ],
)
def test_config_rejects_invalid_construction_values(config_dependencies, kwargs):
    with pytest.raises((TypeError, ValueError)):
        Config(config_dependencies, **kwargs)


def test_config_preserves_pathlike_model_support(config_dependencies):
    config = Config(Path(config_dependencies))
    assert config.model == str(config_dependencies)


def test_config_rejects_missing_model_dir():
    with pytest.raises(ValueError, match="not a directory"):
        Config("/nonexistent/model/path")


@pytest.mark.parametrize(
    "expression",
    [
        'Config("/nonexistent/model/path")',
        'Config("/nonexistent/model/path", kvcache_block_size=255)',
        'Config("/nonexistent/model/path", tensor_parallel_size=9)',
    ],
)
def test_construction_validation_survives_optimized_mode(expression):
    script = f"""
from nanovllm.config import Config
try:
    {expression}
except (TypeError, ValueError):
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
