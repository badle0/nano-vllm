import importlib.util
import pathlib
import subprocess
import sys

import pytest
import torch

from nanovllm.sampling_params import SamplingParams

_p = pathlib.Path(__file__).resolve().parents[1] / "nanovllm/layers/sampler.py"
_s = importlib.util.spec_from_file_location("sampler", _p)
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
Sampler = _m.Sampler

def test_temperature_zero_is_permitted():
    SamplingParams(temperature=0.0)

def test_negative_temperature_rejected():
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1.0)

def test_greedy_is_argmax():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    out = Sampler().greedy(logits.clone())
    assert torch.equal(out, logits.argmax(-1))

def test_greedy_argmax_preserves_exact_tie_order():
    logits = torch.tensor([[1.0, 4.0, 4.0, 2.0]])
    assert Sampler().greedy(logits).item() == logits.argmax(-1).item() == 1

def test_greedy_does_not_advance_cpu_rng():
    logits = torch.arange(8000, dtype=torch.float32).reshape(8, 1000)
    sampler = Sampler()
    sampler.greedy(logits)
    state = torch.get_rng_state().clone()
    sampler.greedy(logits)
    assert torch.equal(torch.get_rng_state(), state)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_greedy_does_not_advance_cuda_rng():
    logits = torch.arange(8000, dtype=torch.float32, device="cuda").reshape(8, 1000)
    sampler = Sampler().cuda()
    sampler.greedy(logits)
    torch.cuda.synchronize()
    state = torch.cuda.get_rng_state().clone()
    sampler.greedy(logits)
    torch.cuda.synchronize()
    assert torch.equal(torch.cuda.get_rng_state(), state)

def test_mixed_batch_routes_per_row():
    logits = torch.randn(6, 1000, dtype=torch.bfloat16)
    temps = torch.tensor([0., 0.6, 0., 1.0, 0., 0.6])
    out = Sampler()(logits.clone(), temps)
    g = temps == 0
    assert torch.equal(out[g], logits.argmax(-1)[g])

def test_stochastic_path_is_not_constant():
    logits = torch.randn(1, 1000, dtype=torch.bfloat16)
    draws = {Sampler()(logits.clone(), torch.ones(1)).item() for _ in range(200)}
    assert len(draws) > 1

def _main_sampler(logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

def test_stochastic_path_is_fixed_seed_equivalent_to_main():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    temperatures = torch.tensor([0.6, 1.0, 0.8, 1.2, 0.5, 0.9, 1.1, 0.7])
    reference = torch.compile(_main_sampler)
    torch.manual_seed(42)
    expected = reference(logits.clone(), temperatures.clone())
    torch.manual_seed(42)
    actual = Sampler()(logits.clone(), temperatures.clone())
    assert torch.equal(actual, expected)

def test_validation_survives_optimized_python():
    code = """
from nanovllm.sampling_params import SamplingParams

for temperature in (-1.0, True, "cold"):
    try:
        SamplingParams(temperature=temperature)
    except (TypeError, ValueError):
        continue
    raise SystemExit(f"accepted invalid temperature: {temperature!r}")
"""
    subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        check=True,
    )
