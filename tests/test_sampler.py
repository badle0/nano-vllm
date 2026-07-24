import importlib.util, pathlib, torch, pytest, math

_sp = pathlib.Path(__file__).resolve().parents[1] / "nanovllm/layers/sampler.py"
_ss = importlib.util.spec_from_file_location("sampler", _sp)
_sm = importlib.util.module_from_spec(_ss); _ss.loader.exec_module(_sm)
Sampler = _sm.Sampler

_pp = pathlib.Path(__file__).resolve().parents[1] / "nanovllm/sampling_params.py"
_ps = importlib.util.spec_from_file_location("sampling_params", _pp)
_pm = importlib.util.module_from_spec(_ps); _ps.loader.exec_module(_pm)
SamplingParams = _pm.SamplingParams

def S(logits, temps, top_ks=None, top_ps=None):
    return Sampler()(logits, temps, top_ks, top_ps)

def _disabled(n):
    return torch.full((n,), -1, dtype=torch.int64)

def test_temperature_zero_is_permitted():
    SamplingParams(temperature=0.0)

def test_negative_temperature_rejected():
    with pytest.raises(AssertionError):
        SamplingParams(temperature=-1.0)

def test_greedy_is_argmax():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    out = S(logits.clone(), torch.zeros(8), _disabled(8))
    assert torch.equal(out, logits.argmax(-1))

def test_mixed_batch_routes_per_row():
    logits = torch.randn(6, 1000, dtype=torch.bfloat16)
    temps = torch.tensor([0., 0.6, 0., 1.0, 0., 0.6])
    out = S(logits.clone(), temps, _disabled(6))
    g = temps == 0
    assert torch.equal(out[g], logits.argmax(-1)[g])

def test_stochastic_path_is_not_constant():
    logits = torch.randn(1, 1000, dtype=torch.bfloat16)
    draws = {S(logits.clone(), torch.ones(1), _disabled(1)).item() for _ in range(200)}
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
    torch.manual_seed(42); b = S(logits.clone(), temps.clone(), torch.full((8,), -1, dtype=torch.int64))
    assert torch.equal(a, b)

def test_none_topk_is_seed_equivalent_to_pr1():
    # fast path: with top_ks=None the compiled graph is op-for-op the PR1 sampler
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    temps = torch.tensor([0., .6, 1., .6, 1.3, .9, .6, 0.])
    ref = torch.compile(_pr1_reference)
    torch.manual_seed(42); a = ref(logits.clone(), temps.clone())
    torch.manual_seed(42); b = S(logits.clone(), temps.clone(), None)
    assert torch.equal(a, b)

def test_none_and_all_disabled_tensor_agree():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    torch.manual_seed(9); a = S(logits.clone(), torch.ones(8), None)
    torch.manual_seed(9); b = S(logits.clone(), torch.ones(8), torch.full((8,), -1, dtype=torch.int64))
    assert torch.equal(a, b)

def test_topk_support():
    logits = torch.randn(4, 1000, dtype=torch.bfloat16)
    kth = logits.float().sort(-1, descending=True).values[:, 4:5]   # value threshold, not topk indices
    allowed = logits.float() >= kth
    for _ in range(500):
        s = S(logits.clone(), torch.ones(4), torch.full((4,), 5, dtype=torch.int64))
        assert bool(allowed.gather(1, s.unsqueeze(1)).all())

def test_topk_tie_at_boundary_keeps_all():
    row = torch.full((1, 10), -10.0); row[0, :2] = 5.0; row[0, 2:5] = 3.0
    seen = {int(S(row.clone(), torch.ones(1), torch.tensor([3]))) for _ in range(2000)}
    assert seen == {0, 1, 2, 3, 4}

def test_topk_one_samples_a_maximum():
    torch.manual_seed(3)
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    out = S(logits.clone(), torch.ones(8), torch.ones(8, dtype=torch.int64))
    lf = logits.float()
    assert torch.equal(lf.gather(1, out.unsqueeze(1)).squeeze(1), lf.max(-1).values)

def test_topk_one_with_tied_maxima_samples_among_ties():
    row = torch.full((1, 10), -10.0); row[0, 2] = 5.0; row[0, 7] = 5.0
    seen = {int(S(row.clone(), torch.ones(1), torch.tensor([1]))) for _ in range(500)}
    assert seen == {2, 7}

def test_topk_params_validation():
    SamplingParams(top_k=-1); SamplingParams(top_k=5)
    with pytest.raises(AssertionError):
        SamplingParams(top_k=0)

def test_topp_off_by_one_keeps_crossing_token():
    row = torch.tensor([[math.log(.5), math.log(.3), math.log(.2)]])
    seen = {int(S(row.clone(), torch.ones(1), top_ps=torch.tensor([0.6]))) for _ in range(2000)}
    assert seen == {0, 1}

def test_topp_support():
    torch.manual_seed(11)
    logits = torch.randn(4, 1000, dtype=torch.bfloat16)
    p_full = torch.softmax(logits.float(), -1)
    sp, si = p_full.sort(-1, descending=True)
    keep_sorted = (sp.cumsum(-1) - sp) <= 0.8
    allowed = torch.zeros_like(keep_sorted).scatter_(-1, si, keep_sorted)
    for _ in range(300):
        s = S(logits.clone(), torch.ones(4), top_ps=torch.full((4,), 0.8))
        assert bool(allowed.gather(1, s.unsqueeze(1)).all())

def test_topp_adaptive_collapse_on_peaked_row():
    row = torch.full((1, 100), -10.0); row[0, 7] = 10.0; row[0, 3] = 2.0
    seen = {int(S(row.clone(), torch.ones(1), top_ps=torch.tensor([0.8]))) for _ in range(500)}
    assert seen == {7}

def test_topk_topp_combined_containment():
    torch.manual_seed(13)
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    kth = logits.float().sort(-1, descending=True).values[:, 4:5]
    allowed = logits.float() >= kth          # value-threshold top-5 set (ties included)
    for _ in range(300):
        s = S(logits.clone(), torch.ones(8),
              top_ks=torch.full((8,), 5, dtype=torch.int64), top_ps=torch.full((8,), 0.9))
        assert bool(allowed.gather(1, s.unsqueeze(1)).all())

def test_greedy_override_with_both_knobs():
    torch.manual_seed(17)
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    temps = torch.tensor([0., 1., 0., 1., 0., 1., 1., 0.])
    out = S(logits.clone(), temps,
            top_ks=torch.full((8,), 5, dtype=torch.int64), top_ps=torch.full((8,), 0.5))
    g = temps == 0
    assert torch.equal(out[g], logits.argmax(-1)[g])

def test_topp_params_validation():
    SamplingParams(top_p=1.0); SamplingParams(top_p=0.9)
    with pytest.raises(AssertionError):
        SamplingParams(top_p=0.0)
    with pytest.raises(AssertionError):
        SamplingParams(top_p=1.5)