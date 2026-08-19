import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/topp_performance/topp_histogram.py"
)
SPEC = importlib.util.spec_from_file_location("topp_histogram", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
sort_bf16_scaled_values = MODULE.sort_bf16_scaled_values


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)


@pytest.mark.parametrize("distribution", ["random", "ties", "masked"])
def test_histogram_sort_matches_reference_values(distribution):
    torch.manual_seed(20260821)
    if distribution == "random":
        logits = torch.randn(4, 4097, device="cuda", dtype=torch.bfloat16)
    elif distribution == "ties":
        logits = torch.randint(-8, 9, (4, 4097), device="cuda").to(torch.bfloat16)
    else:
        logits = torch.randn(4, 4097, device="cuda", dtype=torch.bfloat16)
        logits[:, :3072] = float("-inf")
    temperatures = torch.tensor([0.5, 0.6, 1.0, 1.5], device="cuda")

    expected = torch.sort(
        logits.float() / temperatures.unsqueeze(1),
        dim=-1,
        descending=False,
    ).values
    actual = sort_bf16_scaled_values(logits, temperatures)

    assert torch.equal(actual, expected)


def test_histogram_sort_matches_production_vocabulary_values():
    torch.manual_seed(20260822)
    logits = torch.randn(2, 151_936, device="cuda", dtype=torch.bfloat16)
    temperatures = torch.tensor([0.6, 1.0], device="cuda")

    expected = torch.sort(logits.float() / temperatures.unsqueeze(1), dim=-1).values
    actual = sort_bf16_scaled_values(logits, temperatures)

    assert torch.equal(actual, expected)


def test_histogram_sort_rejects_unsupported_inputs():
    cuda_logits = torch.randn(2, 17, device="cuda", dtype=torch.bfloat16)
    cuda_temperatures = torch.ones(2, device="cuda")

    with pytest.raises(TypeError, match="bfloat16"):
        sort_bf16_scaled_values(cuda_logits.float(), cuda_temperatures)
    with pytest.raises(TypeError, match="float32"):
        sort_bf16_scaled_values(cuda_logits, cuda_temperatures.to(torch.float64))
    with pytest.raises(ValueError, match="one value per logit row"):
        sort_bf16_scaled_values(cuda_logits, cuda_temperatures[:1])
    with pytest.raises(ValueError, match="CUDA"):
        sort_bf16_scaled_values(cuda_logits.cpu(), cuda_temperatures.cpu())
