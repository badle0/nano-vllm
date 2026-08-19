import pytest

import nanovllm.engine.llm_engine as llm_engine_module
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


@pytest.mark.parametrize("num_params", [1, 3])
def test_generate_rejects_mismatched_sampling_params_before_admission(num_params):
    engine = object.__new__(LLMEngine)

    with pytest.raises(ValueError, match="same length"):
        engine.generate(
            ["first", "second"],
            [SamplingParams()] * num_params,
            use_tqdm=False,
        )


def test_flashinfer_dependency_fails_before_worker_processes_start(monkeypatch):
    monkeypatch.setattr(
        llm_engine_module,
        "fields",
        lambda config_type: [type("Field", (), {"name": "top_p_backend"})()],
    )
    monkeypatch.setattr(
        llm_engine_module,
        "Config",
        lambda model, **kwargs: type(
            "ConfigResult", (), {"top_p_backend": "flashinfer"}
        )(),
    )

    def missing_dependency():
        raise RuntimeError("install nano-vllm[fast-sampling]")

    monkeypatch.setattr(
        llm_engine_module, "require_flashinfer_sampling", missing_dependency
    )
    monkeypatch.setattr(
        llm_engine_module.mp,
        "get_context",
        lambda method: pytest.fail("workers started before dependency validation"),
    )

    with pytest.raises(RuntimeError, match="fast-sampling"):
        LLMEngine("/unused", top_p_backend="flashinfer")
