from types import SimpleNamespace

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
