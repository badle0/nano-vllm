import importlib.util, pathlib, torch, pytest

_sp = pathlib.Path(__file__).resolve().parents[1] / "nanovllm/layers/sampler.py"
_ss = importlib.util.spec_from_file_location("sampler", _sp)
_sm = importlib.util.module_from_spec(_ss); _ss.loader.exec_module(_sm)
Sampler = _sm.Sampler

_pp = pathlib.Path(__file__).resolve().parents[1] / "nanovllm/sampling_params.py"
_ps = importlib.util.spec_from_file_location("sampling_params", _pp)
_pm = importlib.util.module_from_spec(_ps); _ps.loader.exec_module(_pm)
SamplingParams = _pm.SamplingParams

def _disabled(n):
    return torch.full((n,), -1, dtype=torch.int64)

def test_temperature_zero_is_permitted():
    SamplingParams(temperature=0.0)

def test_negative_temperature_rejected():
    with pytest.raises(AssertionError):
        SamplingParams(temperature=-1.0)

def test_greedy_is_argmax():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    out = Sampler()(logits.clone(), torch.zeros(8), _disabled(8))
    assert torch.equal(out, logits.argmax(-1))

def test_mixed_batch_routes_per_row():
    logits = torch.randn(6, 1000, dtype=torch.bfloat16)
    temps = torch.tensor([0., 0.6, 0., 1.0, 0., 0.6])
    out = Sampler()(logits.clone(), temps, _disabled(6))
    g = temps == 0
    assert torch.equal(out[g], logits.argmax(-1)[g])

def test_stochastic_path_is_not_constant():
    logits = torch.randn(1, 1000, dtype=torch.bfloat16)
    draws = {Sampler()(logits.clone(), torch.ones(1), _disabled(1)).item() for _ in range(200)}
    assert len(draws) > 1

def _pr1_reference(logits, temperatures):
    greedy = logits.argmax(dim=-1)
    logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sampled = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return torch.where(temperatures == 0, greedy, sampled)

def test_disabled_topk_is_seed_equivalent_to_pr1():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    temps = torch.tensor([0., .6, 1., .6, 1.3, .9, .6, 0.])
    ref = torch.compile(_pr1_reference)
    torch.manual_seed(42); a = ref(logits.clone(), temps.clone())
    torch.manual_seed(42); b = Sampler()(logits.clone(), temps.clone(), torch.full((8,), -1, dtype=torch.int64))
    assert torch.equal(a, b)

def test_topk_support():
    logits = torch.randn(4, 1000, dtype=torch.bfloat16)
    kth = logits.float().sort(-1, descending=True).values[:, 4:5]   # value threshold, not topk indices
    allowed = logits.float() >= kth
    for _ in range(500):
        s = Sampler()(logits.clone(), torch.ones(4), torch.full((4,), 5, dtype=torch.int64))
        assert bool(allowed.gather(1, s.unsqueeze(1)).all())

def test_topk_tie_at_boundary_keeps_all():
    row = torch.full((1, 10), -10.0); row[0, :2] = 5.0; row[0, 2:5] = 3.0
    seen = {int(Sampler()(row.clone(), torch.ones(1), torch.tensor([3]))) for _ in range(2000)}
    assert seen == {0, 1, 2, 3, 4}

def test_topk_one_is_greedy():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    out = Sampler()(logits.clone(), torch.ones(8), torch.ones(8, dtype=torch.int64))
    assert torch.equal(out, logits.argmax(-1))

def test_topk_params_validation():
    SamplingParams(top_k=-1); SamplingParams(top_k=5)
    with pytest.raises(AssertionError):
        SamplingParams(top_k=0)