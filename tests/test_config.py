from dataclasses import fields
import json
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

import pytest

import nanovllm.config as config_module
from nanovllm.config import Config


def add_fake_safetensors(*directories):
    for directory in directories:
        (directory / "model.safetensors").touch()


@pytest.fixture
def config_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda model: SimpleNamespace(
            max_position_embeddings=8192,
            model_type="qwen3",
            vocab_size=151936,
        ),
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


def test_speculative_config_defaults_are_inert_and_appended(config_dependencies):
    config = Config(config_dependencies)

    assert config.draft_model is None
    assert config.num_speculative_tokens == 0
    assert config.configured_k == 0
    assert config.speculation_enabled is False
    assert config.draft_hf_config is None
    assert config.numerical_mode == "fast"
    assert config.speculative_policy == "fixed"
    assert [field.name for field in fields(Config) if field.init][-5:] == [
        "disable_python_gc",
        "draft_model",
        "num_speculative_tokens",
        "numerical_mode",
        "speculative_policy",
    ]


@pytest.mark.parametrize("invalid", [True, 1.0, "4", None])
def test_num_speculative_tokens_requires_an_integer(
    config_dependencies,
    invalid,
):
    with pytest.raises(
        TypeError,
        match="num_speculative_tokens must be an integer",
    ):
        Config(config_dependencies, num_speculative_tokens=invalid)


def test_num_speculative_tokens_rejects_negative_values(config_dependencies):
    with pytest.raises(ValueError, match="must be >= 0"):
        Config(config_dependencies, num_speculative_tokens=-1)


@pytest.mark.parametrize("invalid", [True, 1, 1.5, object()])
def test_draft_model_requires_a_path_type(config_dependencies, invalid):
    with pytest.raises(TypeError, match="draft_model must be None"):
        Config(
            config_dependencies,
            draft_model=invalid,
            num_speculative_tokens=1,
        )


def test_speculative_config_rejects_dangling_halves(config_dependencies):
    with pytest.raises(ValueError, match="requires draft_model"):
        Config(config_dependencies, num_speculative_tokens=1)
    with pytest.raises(
        ValueError,
        match="draft_model requires num_speculative_tokens > 0",
    ):
        Config(config_dependencies, draft_model=config_dependencies)


def test_speculative_config_accepts_pathlike_and_clamps_both_models(
    monkeypatch,
    tmp_path,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    add_fake_safetensors(target, draft)
    calls = []

    def load_config(model):
        calls.append(str(model))
        limit = 8192 if str(model) == str(target) else 2048
        return SimpleNamespace(
            max_position_embeddings=limit,
            model_type="qwen3",
            vocab_size=151936,
        )

    monkeypatch.setattr(config_module.AutoConfig, "from_pretrained", load_config)

    config = Config(
        target,
        draft_model=draft,
        num_speculative_tokens=4,
        enforce_eager=True,
    )

    assert config.model == str(target)
    assert config.draft_model == str(draft)
    assert config.configured_k == 4
    assert config.speculation_enabled is True
    assert config.enforce_eager is True
    assert config.max_model_len == 2048
    assert config.draft_hf_config.max_position_embeddings == 2048
    assert calls == [str(target), str(draft)]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"tensor_parallel_size": 2}, "tensor_parallel_size=1 only"),
        ({"top_p_backend": "flashinfer"}, "top_p_backend='exact'"),
    ],
)
def test_unsupported_speculative_config_fails_before_hf_loading(
    monkeypatch,
    tmp_path,
    extra,
    message,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    add_fake_safetensors(target, draft)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda model: pytest.fail("HF config loaded before compatibility gate"),
    )

    with pytest.raises(ValueError, match=message):
        Config(
            target,
            draft_model=draft,
            num_speculative_tokens=1,
            **extra,
        )


def test_missing_draft_directory_fails_before_hf_loading(monkeypatch, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    missing = tmp_path / "missing-draft"
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda model: pytest.fail("HF config loaded before draft path gate"),
    )

    with pytest.raises(ValueError, match="draft model path is not a directory"):
        Config(
            target,
            draft_model=missing,
            num_speculative_tokens=1,
        )


@pytest.mark.parametrize(
    ("target_type", "draft_type", "target_vocab", "draft_vocab", "message"),
    [
        ("qwen2", "qwen3", 100, 100, "requires Qwen3"),
        ("qwen3", "qwen2", 100, 100, "requires Qwen3"),
        ("qwen3", "qwen3", 100, 101, "vocab_size must match"),
    ],
)
def test_speculative_model_config_compatibility(
    monkeypatch,
    tmp_path,
    target_type,
    draft_type,
    target_vocab,
    draft_vocab,
    message,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    add_fake_safetensors(target, draft)

    def load_config(model):
        is_target = str(model) == str(target)
        return SimpleNamespace(
            max_position_embeddings=8192,
            model_type=target_type if is_target else draft_type,
            vocab_size=target_vocab if is_target else draft_vocab,
        )

    monkeypatch.setattr(config_module.AutoConfig, "from_pretrained", load_config)

    with pytest.raises(ValueError, match=message):
        Config(
            target,
            draft_model=draft,
            num_speculative_tokens=1,
        )


@pytest.mark.parametrize(
    ("role", "position_limit"),
    [
        ("target", None),
        ("target", True),
        ("target", 2048.0),
        ("target", 0),
        ("target", -1),
        ("draft", None),
        ("draft", True),
        ("draft", 2048.0),
        ("draft", 0),
        ("draft", -1),
    ],
)
def test_speculative_config_requires_positive_integer_position_limits(
    monkeypatch,
    tmp_path,
    role,
    position_limit,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    add_fake_safetensors(target, draft)

    def load_config(model):
        current_role = "target" if str(model) == str(target) else "draft"
        return SimpleNamespace(
            max_position_embeddings=(
                position_limit if current_role == role else 8192
            ),
            model_type="qwen3",
            vocab_size=151936,
        )

    monkeypatch.setattr(config_module.AutoConfig, "from_pretrained", load_config)

    with pytest.raises(
        ValueError,
        match=f"{role} model must define a positive integer",
    ):
        Config(
            target,
            draft_model=draft,
            num_speculative_tokens=1,
        )


def test_speculation_off_keeps_tp_and_flashinfer_config_supported(
    config_dependencies,
):
    config = Config(
        config_dependencies,
        tensor_parallel_size=2,
        top_p_backend="flashinfer",
    )

    assert config.speculation_enabled is False
    assert config.tensor_parallel_size == 2
    assert config.top_p_backend == "flashinfer"


def test_speculative_config_rejects_missing_supported_weight_artifact(
    monkeypatch,
    tmp_path,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda model: pytest.fail("HF config loaded before weight preflight"),
    )

    with pytest.raises(ValueError, match="target model.*safetensors"):
        Config(target, draft_model=draft, num_speculative_tokens=1)

    add_fake_safetensors(target)
    with pytest.raises(ValueError, match="draft model.*safetensors"):
        Config(target, draft_model=draft, num_speculative_tokens=1)


def test_speculative_weight_preflight_rejects_safetensors_directory(
    monkeypatch,
    tmp_path,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "model.safetensors").mkdir()
    add_fake_safetensors(draft)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda model: pytest.fail("HF config loaded before weight preflight"),
    )

    with pytest.raises(ValueError, match="target model.*safetensors"):
        Config(target, draft_model=draft, num_speculative_tokens=1)


def test_oversized_configured_k_is_an_accepted_clipped_maximum(
    monkeypatch,
    tmp_path,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    add_fake_safetensors(target, draft)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda model: SimpleNamespace(
            max_position_embeddings=3,
            model_type="qwen3",
            vocab_size=16,
        ),
    )

    config = Config(
        target,
        draft_model=draft,
        num_speculative_tokens=100,
        max_model_len=3,
        max_num_batched_tokens=4,
        max_num_seqs=1,
    )

    assert config.configured_k == 100
    assert min(
        config.configured_k,
        config.max_model_len - 2,
        config.max_num_batched_tokens - 2,
    ) == 1


@pytest.mark.parametrize(
    "limits",
    [
        {"max_model_len": 1},
        {"max_model_len": 2},
        {"max_num_batched_tokens": 1, "max_num_seqs": 1},
        {"max_num_batched_tokens": 2, "max_num_seqs": 1},
    ],
)
def test_speculative_config_rejects_globally_zero_effective_k(
    monkeypatch,
    tmp_path,
    limits,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    add_fake_safetensors(target, draft)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda model: SimpleNamespace(
            max_position_embeddings=8192,
            model_type="qwen3",
            vocab_size=16,
        ),
    )

    with pytest.raises(ValueError, match="no globally usable proposal slot"):
        Config(
            target,
            draft_model=draft,
            num_speculative_tokens=1,
            **limits,
        )


def test_zero_effective_k_validation_survives_optimized_mode(tmp_path):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    add_fake_safetensors(target, draft)
    hf_config = {
        "model_type": "qwen3",
        "vocab_size": 16,
        "max_position_embeddings": 8192,
    }
    for directory in (target, draft):
        (directory / "config.json").write_text(
            json.dumps(hf_config),
            encoding="utf-8",
        )
    script = f"""
from nanovllm.config import Config
try:
    Config(
        {str(target)!r},
        draft_model={str(draft)!r},
        num_speculative_tokens=100,
        max_num_batched_tokens=2,
        max_num_seqs=1,
    )
except ValueError:
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


@pytest.mark.parametrize(
    "arguments",
    [
        "num_speculative_tokens=True",
        "num_speculative_tokens=-1",
        "draft_model={model}",
    ],
)
def test_speculative_validation_survives_optimized_mode(
    config_dependencies,
    arguments,
):
    model = repr(str(config_dependencies))
    arguments = arguments.format(model=model)
    script = f"""
from nanovllm.config import Config
try:
    Config({model}, {arguments})
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

@pytest.mark.parametrize("value", [None, True, 1, 1.0])
def test_numerical_mode_requires_a_string(config_dependencies, value):
    with pytest.raises(TypeError, match="numerical_mode must be a string"):
        Config(config_dependencies, numerical_mode=value)


def test_numerical_mode_rejects_unknown_value(config_dependencies):
    with pytest.raises(ValueError, match="fast.*invariant"):
        Config(config_dependencies, numerical_mode="deterministic")


@pytest.mark.parametrize("value", [None, True, 1, 1.0])
def test_speculative_policy_requires_a_string(config_dependencies, value):
    with pytest.raises(TypeError, match="speculative_policy must be a string"):
        Config(config_dependencies, speculative_policy=value)


def test_adaptive_policy_requires_enabled_speculation(config_dependencies):
    with pytest.raises(ValueError, match="requires speculative decoding"):
        Config(config_dependencies, speculative_policy="adaptive")


def test_invariant_mode_accepts_qualified_qwen_geometry(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda model: SimpleNamespace(
            max_position_embeddings=8192,
            model_type="qwen3",
            vocab_size=151936,
            dtype=torch.bfloat16,
            hidden_size=1024,
            num_hidden_layers=28,
        ),
    )
    config = Config(tmp_path, numerical_mode="invariant", max_model_len=4096)
    assert config.numerical_mode == "invariant"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dtype", "torch.float16", "requires BF16"),
        ("model_type", "llama", "supports Qwen3"),
        ("hidden_size", 2048, "Qwen3-0.6B and Qwen3-4B"),
    ],
)
def test_invariant_mode_rejects_unqualified_models(
    monkeypatch, tmp_path, field, value, message
):
    import torch

    values = dict(
        max_position_embeddings=8192,
        model_type="qwen3",
        vocab_size=151936,
        dtype=torch.bfloat16,
        hidden_size=1024,
        num_hidden_layers=28,
    )
    values[field] = value
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda model: SimpleNamespace(**values),
    )
    with pytest.raises(ValueError, match=message):
        Config(tmp_path, numerical_mode="invariant")
