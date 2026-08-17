import pytest

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
