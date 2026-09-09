import importlib.util
import math
import pathlib
import sys

import pytest
import torch


_SAMPLER_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "nanovllm/layers/sampler.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "nanovllm_speculative_sampler_under_test",
    _SAMPLER_PATH,
)
_MODULE = importlib.util.module_from_spec(_SPEC)
# Register this standalone test import just as a normal package import would, so
# runtime type machinery observes a stable module identity.
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Sampler = _MODULE.Sampler
ModifiedRejectionSampler = _MODULE.ModifiedRejectionSampler
SpeculativeSamplingInvariantError = _MODULE.SpeculativeSamplingInvariantError


def _as_fp64_probabilities(weights: torch.Tensor) -> torch.Tensor:
    """Independent scalar-oracle normalization of retained FP32 weights."""

    assert weights.ndim == 1
    assert weights.dtype == torch.float32
    values = tuple(float(value) for value in weights.tolist())
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("invalid oracle row")
    mass = math.fsum(values)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("invalid oracle mass")
    return torch.tensor(
        tuple(value / mass for value in values),
        dtype=torch.float64,
    )


def _oracle_race_token(weights: torch.Tensor, noise: torch.Tensor) -> int:
    weights = weights.to(torch.float64)
    noise = noise.to(torch.float64)
    assert bool(torch.isfinite(noise).all())
    assert bool((noise > 0).all())
    return int((weights / noise).argmax().item())


def _oracle_correction(
    target_weights: torch.Tensor,
    draft_weights: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[int, bool]:
    target = _as_fp64_probabilities(target_weights)
    draft = _as_fp64_probabilities(draft_weights)
    residual = (target - draft).clamp_min(0.0)
    residual_mass = math.fsum(float(value) for value in residual.tolist())
    if not math.isfinite(residual_mass):
        raise ValueError("non-finite oracle residual")
    if residual_mass == 0.0:
        return _oracle_race_token(target, noise), True
    return _oracle_race_token(residual, noise), False


def _oracle_accept(
    target_weights: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_weights: torch.Tensor,
    uniforms: torch.Tensor,
    correction_noise: torch.Tensor,
) -> tuple[int, int, bool]:
    accepted = 0
    for position in range(draft_tokens.numel()):
        target = _as_fp64_probabilities(target_weights[position])
        draft = _as_fp64_probabilities(draft_weights[position])
        token = int(draft_tokens[position].item())
        q_selected = float(draft[token])
        if not math.isfinite(q_selected) or q_selected <= 0.0:
            raise ValueError("invalid selected draft probability")
        uniform = float(uniforms[position])
        if not math.isfinite(uniform) or not 0.0 <= uniform < 1.0:
            raise ValueError("invalid oracle uniform")
        acceptance_probability = min(1.0, float(target[token]) / q_selected)
        if uniform < acceptance_probability:
            accepted += 1
            continue
        correction, fallback = _oracle_correction(
            target_weights[position],
            draft_weights[position],
            correction_noise,
        )
        return accepted, correction, fallback
    return accepted, -1, False


def _reference_probability_row(
    logits: torch.Tensor,
    temperature: float,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    logits = logits.to(dtype=torch.float32, copy=True)
    if temperature == 0.0:
        result = torch.zeros_like(logits)
        result[int(logits.argmax().item())] = 1.0
        return result
    logits.div_(temperature)
    if top_k is not None and top_k < logits.numel():
        threshold = logits.topk(top_k, sorted=False).values.amin()
        logits.masked_fill_(logits < threshold, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(descending=False)
        remove_sorted = sorted_logits.softmax(dim=-1).cumsum(dim=-1) <= 1.0 - top_p
        remove_sorted[-1] = False
        remove = torch.zeros_like(remove_sorted).scatter(
            0, sorted_indices, remove_sorted
        )
        logits.masked_fill_(remove, float("-inf"))
    return logits.softmax(dim=-1)


def _assert_bool_tensor(value: torch.Tensor, expected: list[bool]) -> None:
    assert value.dtype == torch.bool
    assert value.tolist() == expected


def test_prepare_exact_probabilities_matches_independent_warp_oracle():
    logits = torch.tensor(
        [
            [1.0, 5.0, 5.0, -2.0, 0.0],
            [2.0, 1.0, 0.0, -1.0, -2.0],
            [0.2, 1.4, -0.3, 0.8, 0.1],
            [1.7, -0.2, 0.5, 1.1, -0.9],
        ],
        dtype=torch.float32,
    )
    original = logits.clone()
    temperatures = torch.tensor([0.0, 0.5, 0.8, 1.2], dtype=torch.float32)
    top_k_buckets = (
        (3, torch.tensor([2], dtype=torch.int64)),
        (4, torch.tensor([3], dtype=torch.int64)),
    )
    top_p_plan = (
        torch.tensor([1, 2, 3], dtype=torch.int64),
        torch.tensor([0.25, 0.2, 0.1], dtype=torch.float32),
    )

    actual = Sampler().prepare_exact_probabilities(
        logits,
        temperatures,
        top_k_buckets=top_k_buckets,
        top_p_plan=top_p_plan,
    )
    expected = torch.stack(
        (
            _reference_probability_row(logits[0], 0.0),
            _reference_probability_row(logits[1], 0.5, top_p=0.75),
            _reference_probability_row(logits[2], 0.8, top_k=3, top_p=0.8),
            _reference_probability_row(logits[3], 1.2, top_k=4, top_p=0.9),
        )
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(4))
    assert torch.equal(logits, original)
    assert actual[0].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]


def test_prepare_exact_probabilities_supports_homogeneous_filter_plans():
    logits = torch.tensor(
        [[2.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, -1.0]],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.7, 1.1])
    actual = Sampler().prepare_exact_probabilities(
        logits,
        temperatures,
        top_k_buckets=((3, None),),
        top_p_plan=(None, torch.tensor([0.15, 0.25])),
    )
    expected = torch.stack(
        (
            _reference_probability_row(logits[0], 0.7, top_k=3, top_p=0.85),
            _reference_probability_row(logits[1], 1.1, top_k=3, top_p=0.75),
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_sample_exact_returns_the_exact_probabilities_used_by_the_draw():
    logits = torch.tensor(
        [[2.0, 1.0, -1.0], [0.5, 1.0, 1.5]], dtype=torch.float32
    )
    original = logits.clone()
    temperatures = torch.tensor([0.6, 1.0])
    race_noise = torch.tensor(
        [[4.0, 0.25, 1.0], [0.5, 2.0, 4.0]], dtype=torch.float32
    )
    expected_probabilities = Sampler().prepare_exact_probabilities(
        logits, temperatures
    )

    result = Sampler().sample_exact_with_probabilities(
        logits,
        temperatures,
        race_noise=race_noise,
    )

    expected_tokens = (expected_probabilities / race_noise).argmax(dim=-1)
    assert torch.equal(result.token_ids, expected_tokens)
    assert torch.equal(result.probabilities, expected_probabilities)
    assert torch.equal(logits, original)


@pytest.mark.parametrize(
    "noise",
    (
        torch.ones(1, 2, dtype=torch.float64),
        torch.ones(1, 3),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, float("nan")]]),
        torch.tensor([[1.0, float("inf")]]),
    ),
)
def test_sample_exact_rejects_invalid_injected_race_noise(noise):
    with pytest.raises(SpeculativeSamplingInvariantError):
        Sampler().sample_exact_with_probabilities(
            torch.tensor([[1.0, 0.0]]),
            torch.ones(1),
            race_noise=noise,
        )


def test_correction_matches_analytic_residual_and_does_not_mutate_inputs():
    target = torch.tensor(
        [[0.6, 0.3, 0.1], [0.1, 0.8, 0.1]], dtype=torch.float32
    )
    draft = torch.tensor(
        [[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]], dtype=torch.float32
    )
    target_before = target.clone()
    draft_before = draft.clone()
    noise = torch.tensor([[2.0, 1.0, 0.5], [1.0, 4.0, 2.0]])

    result = ModifiedRejectionSampler().sample_correction(
        target, draft, correction_noise=noise
    )

    expected = [
        _oracle_correction(target[row], draft[row], noise[row])[0]
        for row in range(target.size(0))
    ]
    assert result.token_ids.tolist() == expected == [0, 1]
    _assert_bool_tensor(result.used_reference, [True, True])
    _assert_bool_tensor(result.target_fallback, [False, False])
    assert torch.equal(target, target_before)
    assert torch.equal(draft, draft_before)


def test_fast_zero_residual_recovers_positive_fp64_residual_without_fallback():
    target = torch.softmax(
        torch.tensor([[0.0, math.log(1e-8)]], dtype=torch.float32), dim=-1
    )
    draft = torch.softmax(
        torch.tensor([[0.0, math.log(2e-8)]], dtype=torch.float32), dim=-1
    )
    fast_residual = (
        target / target.sum(dim=-1, keepdim=True)
        - draft / draft.sum(dim=-1, keepdim=True)
    ).clamp_min(0.0)
    assert fast_residual.sum().item() == 0.0

    result = ModifiedRejectionSampler().sample_correction(
        target,
        draft,
        correction_noise=torch.ones_like(target),
    )

    assert result.token_ids.tolist() == [0]
    _assert_bool_tensor(result.used_reference, [True])
    _assert_bool_tensor(result.target_fallback, [False])


def test_fast_overflow_recovers_a_finite_fp64_residual():
    maximum = torch.finfo(torch.float32).max
    target = torch.tensor([[maximum, maximum, 0.0]], dtype=torch.float32)
    draft = torch.tensor([[0.0, 0.0, maximum]], dtype=torch.float32)
    noise = torch.tensor([[2.0, 1.0, 0.5]])

    result = ModifiedRejectionSampler().sample_correction(
        target, draft, correction_noise=noise
    )

    assert result.token_ids.tolist() == [1]
    _assert_bool_tensor(result.used_reference, [True])
    _assert_bool_tensor(result.target_fallback, [False])


def test_injected_robust_zero_residual_samples_target_support_and_flags_fallback():
    target = torch.tensor([[0.0, 0.25, 0.0, 0.75]], dtype=torch.float32)
    noise = torch.tensor([[1e-3, 4.0, 1e-3, 1.0]], dtype=torch.float32)

    result = ModifiedRejectionSampler().sample_correction(
        target, target.clone(), correction_noise=noise
    )

    assert result.token_ids.tolist() == [3]
    assert target[0, result.token_ids[0]].item() > 0.0
    _assert_bool_tensor(result.used_reference, [True])
    _assert_bool_tensor(result.target_fallback, [True])


def test_all_subnormal_weights_retain_their_residual_law():
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    target = torch.tensor([[3.0 * tiny, tiny]], dtype=torch.float32)
    draft = torch.tensor([[tiny, 3.0 * tiny]], dtype=torch.float32)

    result = ModifiedRejectionSampler().sample_correction(
        target,
        draft,
        correction_noise=torch.ones_like(target),
    )

    assert result.token_ids.tolist() == [0]
    _assert_bool_tensor(result.target_fallback, [False])


@pytest.mark.parametrize(
    ("target", "draft"),
    (
        ([0.0, 0.0], [0.5, 0.5]),
        ([0.5, -0.1], [0.5, 0.5]),
        ([0.5, float("nan")], [0.5, 0.5]),
        ([0.5, float("inf")], [0.5, 0.5]),
        ([0.5, 0.5], [0.0, 0.0]),
        ([0.5, 0.5], [float("nan"), 1.0]),
    ),
)
def test_correction_rejects_invalid_canonical_weight_rows(target, draft):
    with pytest.raises(SpeculativeSamplingInvariantError):
        ModifiedRejectionSampler().sample_correction(
            torch.tensor([target], dtype=torch.float32),
            torch.tensor([draft], dtype=torch.float32),
            correction_noise=torch.ones(1, 2),
        )


@pytest.mark.parametrize(
    "noise",
    (
        torch.ones(1, 2, dtype=torch.float64),
        torch.ones(2, 2),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, float("nan")]]),
        torch.tensor([[1.0, float("inf")]]),
    ),
)
def test_correction_rejects_invalid_noise(noise):
    with pytest.raises(SpeculativeSamplingInvariantError):
        ModifiedRejectionSampler().sample_correction(
            torch.tensor([[0.7, 0.3]], dtype=torch.float32),
            torch.tensor([[0.2, 0.8]], dtype=torch.float32),
            correction_noise=noise,
        )


def test_modified_rejection_stops_at_first_rejection_and_handles_all_accept():
    target = torch.tensor(
        [
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.6, 0.2, 0.2]],
            [[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.1, 0.2, 0.7]],
        ],
        dtype=torch.float32,
    )
    draft = torch.tensor(
        [
            [[0.6, 0.3, 0.1], [0.8, 0.1, 0.1], [0.2, 0.4, 0.4]],
            [[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.1, 0.2, 0.7]],
        ],
        dtype=torch.float32,
    )
    draft_tokens = torch.tensor([[0, 0, 2], [2, 0, 2]], dtype=torch.int64)
    uniforms = torch.tensor([[0.9, 0.5, 0.0], [0.99, 0.99, 0.99]])
    correction_noise = torch.tensor([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])

    result = ModifiedRejectionSampler().accept(
        target,
        draft_tokens,
        draft,
        uniforms=uniforms,
        correction_noise=correction_noise,
    )

    assert result.accepted_counts.tolist() == [1, 3]
    assert result.corrective_token_ids[0].item() == 1
    assert result.corrective_token_ids[1].item() == -1
    _assert_bool_tensor(result.target_fallback, [False, False])


def test_acceptance_uses_a_strict_half_open_uniform_boundary():
    # At equality U == p(d)/q(d), the half-open inverse-CDF contract rejects.
    target = torch.tensor([[[0.25, 0.75]]], dtype=torch.float32)
    draft = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)
    token = torch.tensor([[0]])
    noise = torch.ones(1, 2)

    below = ModifiedRejectionSampler().accept(
        target,
        token,
        draft,
        uniforms=torch.tensor([[torch.nextafter(torch.tensor(0.5), torch.tensor(0.0))]]),
        correction_noise=noise,
    )
    boundary = ModifiedRejectionSampler().accept(
        target,
        token,
        draft,
        uniforms=torch.tensor([[0.5]]),
        correction_noise=noise,
    )

    assert below.accepted_counts.tolist() == [1]
    assert boundary.accepted_counts.tolist() == [0]


def test_p_equals_q_accepts_every_legal_uniform_without_numerical_fallback():
    weights = torch.tensor(
        [[[0.0, 0.2, 0.8], [0.3, 0.4, 0.3]]], dtype=torch.float32
    )
    tokens = torch.tensor([[2, 1]])
    maximum_uniform = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    result = ModifiedRejectionSampler().accept(
        weights,
        tokens,
        weights.clone(),
        uniforms=torch.tensor([[maximum_uniform, maximum_uniform]]),
        correction_noise=torch.ones(1, 3),
    )

    assert result.accepted_counts.tolist() == [2]
    assert result.corrective_token_ids.tolist() == [-1]
    _assert_bool_tensor(result.target_fallback, [False])


def test_selected_zero_draft_weight_is_a_typed_invariant_error():
    with pytest.raises(SpeculativeSamplingInvariantError):
        ModifiedRejectionSampler().accept(
            torch.tensor([[[0.4, 0.6]]], dtype=torch.float32),
            torch.tensor([[0]]),
            torch.tensor([[[0.0, 1.0]]], dtype=torch.float32),
            uniforms=torch.zeros(1, 1),
            correction_noise=torch.ones(1, 2),
        )


def test_selected_positive_subnormal_draft_weight_is_not_treated_as_zero():
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    target = torch.tensor([[[tiny, 3.0 * tiny]]], dtype=torch.float32)
    draft = torch.tensor([[[tiny, 3.0 * tiny]]], dtype=torch.float32)
    result = ModifiedRejectionSampler().accept(
        target,
        torch.tensor([[0]]),
        draft,
        uniforms=torch.zeros(1, 1),
        correction_noise=torch.ones(1, 2),
    )
    assert result.accepted_counts.tolist() == [1]


@pytest.mark.parametrize(
    "uniforms",
    (
        torch.zeros(1, 1, dtype=torch.float64),
        torch.zeros(1, 2),
        torch.tensor([[-0.1]]),
        torch.tensor([[1.0]]),
        torch.tensor([[float("nan")]]),
        torch.tensor([[float("inf")]]),
    ),
)
def test_accept_rejects_invalid_injected_uniforms(uniforms):
    with pytest.raises(SpeculativeSamplingInvariantError):
        ModifiedRejectionSampler().accept(
            torch.tensor([[[0.4, 0.6]]], dtype=torch.float32),
            torch.tensor([[0]]),
            torch.tensor([[[0.8, 0.2]]], dtype=torch.float32),
            uniforms=uniforms,
            correction_noise=torch.ones(1, 2),
        )


def test_vectorized_acceptance_matches_independent_scalar_oracle():
    generator = torch.Generator().manual_seed(20260828)
    batch_size, proposal_length, vocab_size = 37, 4, 7
    target = torch.rand(
        batch_size, proposal_length, vocab_size, generator=generator
    ).add_(0.01)
    draft = torch.rand(
        batch_size, proposal_length, vocab_size, generator=generator
    ).add_(0.01)
    draft_tokens = torch.multinomial(
        draft.reshape(-1, vocab_size),
        1,
        generator=generator,
    ).reshape(batch_size, proposal_length)
    uniforms = torch.rand(batch_size, proposal_length, generator=generator)
    correction_noise = torch.empty(
        batch_size, vocab_size
    ).exponential_(1.0, generator=generator)

    result = ModifiedRejectionSampler().accept(
        target,
        draft_tokens,
        draft,
        uniforms=uniforms,
        correction_noise=correction_noise,
    )

    for row in range(batch_size):
        expected_count, expected_token, expected_fallback = _oracle_accept(
            target[row],
            draft_tokens[row],
            draft[row],
            uniforms[row],
            correction_noise[row],
        )
        assert result.accepted_counts[row].item() == expected_count
        assert result.corrective_token_ids[row].item() == expected_token
        assert result.target_fallback[row].item() is expected_fallback


def test_independent_positive_row_scaling_preserves_acceptance_and_correction():
    target = torch.tensor(
        [[[0.5, 0.25, 0.125, 0.125], [0.125, 0.625, 0.125, 0.125]]],
        dtype=torch.float32,
    )
    draft = torch.tensor(
        [[[0.125, 0.375, 0.25, 0.25], [0.625, 0.125, 0.125, 0.125]]],
        dtype=torch.float32,
    )
    tokens = torch.tensor([[1, 0]])
    uniforms = torch.tensor([[0.1, 0.9]])
    noise = torch.tensor([[1.0, 0.5, 2.0, 4.0]])
    sampler = ModifiedRejectionSampler()

    baseline = sampler.accept(
        target,
        tokens,
        draft,
        uniforms=uniforms,
        correction_noise=noise,
    )
    scaled = sampler.accept(
        target * 8.0,
        tokens,
        draft * 0.25,
        uniforms=uniforms,
        correction_noise=noise,
    )

    assert torch.equal(scaled.accepted_counts, baseline.accepted_counts)
    assert torch.equal(scaled.corrective_token_ids, baseline.corrective_token_ids)
    assert torch.equal(scaled.target_fallback, baseline.target_fallback)


def test_rejection_identity_holds_for_mismatched_distributions():
    target = torch.tensor([0.55, 0.25, 0.15, 0.05], dtype=torch.float32)
    draft = torch.tensor([0.05, 0.15, 0.30, 0.50], dtype=torch.float32)
    p = _as_fp64_probabilities(target)
    q = _as_fp64_probabilities(draft)
    residual = (p - q).clamp_min(0.0)
    z = math.fsum(float(value) for value in residual.tolist())
    alpha = math.fsum(min(float(pv), float(qv)) for pv, qv in zip(p, q))

    assert alpha + z == pytest.approx(1.0, abs=2e-15)
    reconstructed = torch.minimum(p, q) + residual
    torch.testing.assert_close(reconstructed, p, rtol=0.0, atol=2e-15)


MC_SAMPLES = 100_000
MC_FAMILYWISE_ALPHA = 1e-6
MC_TOTAL_CELLS = 28  # 4 cases * (six token cells plus one acceptance cell)


def _bernstein_count_tolerance(sample_count: int, probability: float) -> int:
    log_term = math.log(2.0 * MC_TOTAL_CELLS / MC_FAMILYWISE_ALPHA)
    variance = sample_count * probability * (1.0 - probability)
    return math.ceil(
        math.sqrt(2.0 * variance * log_term) + (2.0 / 3.0) * log_term
    )


@pytest.mark.parametrize(
    ("top_k", "top_p"),
    ((None, None), (3, None), (None, 0.75), (4, 0.8)),
    ids=("temperature", "top-k", "top-p", "top-k-plus-top-p"),
)
def test_modified_rejection_monte_carlo_emits_the_target_law(top_k, top_p):
    target_logits = torch.tensor(
        [[2.1, 1.4, 0.8, 0.1, -0.5, -1.1]], dtype=torch.float32
    )
    draft_logits = torch.tensor(
        [[0.7, 2.0, -0.2, 1.2, 0.3, -0.8]], dtype=torch.float32
    )
    temperature = torch.tensor([0.85], dtype=torch.float32)
    top_k_buckets = () if top_k is None else ((top_k, None),)
    top_p_plan = (
        None
        if top_p is None
        else (None, torch.tensor([1.0 - top_p], dtype=torch.float32))
    )
    exact_sampler = Sampler()
    target = exact_sampler.prepare_exact_probabilities(
        target_logits,
        temperature,
        top_k_buckets=top_k_buckets,
        top_p_plan=top_p_plan,
    )
    draft = exact_sampler.prepare_exact_probabilities(
        draft_logits,
        temperature,
        top_k_buckets=top_k_buckets,
        top_p_plan=top_p_plan,
    )
    p = _as_fp64_probabilities(target[0])
    q = _as_fp64_probabilities(draft[0])
    expected_acceptance = math.fsum(
        min(float(pv), float(qv)) for pv, qv in zip(p, q)
    )

    seed = 20260828 + (0 if top_k is None else top_k) + int((top_p or 0.0) * 100)
    generator = torch.Generator().manual_seed(seed)
    proposals = torch.multinomial(
        draft[0], MC_SAMPLES, replacement=True, generator=generator
    )
    uniforms = torch.rand(MC_SAMPLES, 1, generator=generator)
    correction_noise = torch.empty(
        MC_SAMPLES, target.size(1)
    ).exponential_(1.0, generator=generator)
    target_batch = target.view(1, 1, -1).expand(MC_SAMPLES, 1, -1)
    draft_batch = draft.view(1, 1, -1).expand(MC_SAMPLES, 1, -1)

    result = ModifiedRejectionSampler().accept(
        target_batch,
        proposals.view(-1, 1),
        draft_batch,
        uniforms=uniforms,
        correction_noise=correction_noise,
    )
    emitted = torch.where(
        result.accepted_counts == 1,
        proposals,
        result.corrective_token_ids,
    )
    counts = torch.bincount(emitted, minlength=target.size(1))

    for token, probability in enumerate(p.tolist()):
        if probability == 0.0:
            assert counts[token].item() == 0
            continue
        expected_count = MC_SAMPLES * probability
        assert abs(counts[token].item() - expected_count) <= (
            _bernstein_count_tolerance(MC_SAMPLES, probability)
        )
    accepted_count = int((result.accepted_counts == 1).sum().item())
    assert abs(accepted_count - MC_SAMPLES * expected_acceptance) <= (
        _bernstein_count_tolerance(MC_SAMPLES, expected_acceptance)
    )
    assert not bool(result.target_fallback.any())


def test_positive_fast_residual_cannot_drop_reference_positive_support():
    # FP32 row normalization loses token 0 from the residual even though the
    # retained-FP32-to-FP64 law gives tokens 0 and 1 equal residual weight.
    target = torch.tensor([[1.0, 3e-8, 0.0]], dtype=torch.float32)
    draft = torch.tensor([[1.0, 2e-8, 2e-8]], dtype=torch.float32)
    noise = torch.tensor([[0.1, 1.0, 1.0]], dtype=torch.float32)

    fast_target = target / target.sum(dim=-1, keepdim=True)
    fast_draft = draft / draft.sum(dim=-1, keepdim=True)
    fast_residual = (fast_target - fast_draft).clamp_min(0.0)
    assert fast_residual.sum().item() > 0.0
    assert fast_residual[0, 0].item() == 0.0

    expected, fallback = _oracle_correction(target[0], draft[0], noise[0])
    result = ModifiedRejectionSampler().sample_correction(
        target, draft, correction_noise=noise
    )

    assert expected == result.token_ids.item() == 0
    assert fallback is False
    _assert_bool_tensor(result.used_reference, [True])
    _assert_bool_tensor(result.target_fallback, [False])


def test_conditioned_near_equal_residual_matches_retained_fp64_oracle():
    # Minimized from the independent V1 review. The old "mass >= tiny" fast
    # predicate had conditional residual TV ~= 0.0967 and selected token 7.
    target = torch.tensor(
        [[
            0.2553751468658447,
            0.24542509019374847,
            0.11769459396600723,
            0.04408375918865204,
            0.09875188767910004,
            0.2061999887228012,
            0.031808074563741684,
            0.0006614328012801707,
        ]],
        dtype=torch.float32,
    )
    draft = torch.tensor(
        [[
            0.25537505745887756,
            0.24542491137981415,
            0.11769469827413559,
            0.0440841019153595,
            0.09875205904245377,
            0.20620037615299225,
            0.03180774673819542,
            0.0006610550917685032,
        ]],
        dtype=torch.float32,
    )
    noise = torch.ones_like(target)
    noise[0, 0] = 0.1

    expected, fallback = _oracle_correction(target[0], draft[0], noise[0])
    result = ModifiedRejectionSampler().sample_correction(
        target, draft, correction_noise=noise
    )

    assert expected == result.token_ids.item() == 0
    assert fallback is False
    _assert_bool_tensor(result.used_reference, [True])
    _assert_bool_tensor(result.target_fallback, [False])


def test_exact_race_preserves_valid_tiny_positive_noise():
    logits = torch.log(torch.tensor([[0.1, 0.9]], dtype=torch.float32))
    noise = torch.tensor([[1e-12, 5e-11]], dtype=torch.float32)

    result = Sampler().sample_exact_with_probabilities(
        logits,
        torch.ones(1),
        race_noise=noise,
    )

    expected = (result.probabilities / noise).argmax(dim=-1)
    assert expected.item() == result.token_ids.item() == 0


def test_exact_race_avoids_fp32_overflow_ties_on_subnormal_noise():
    logits = torch.log(torch.tensor([[0.1, 0.9]], dtype=torch.float32))
    smallest = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    noise = torch.tensor([[smallest, 2.0 * smallest]], dtype=torch.float32)

    result = Sampler().sample_exact_with_probabilities(
        logits,
        torch.ones(1),
        race_noise=noise,
    )

    fp32_scores = result.probabilities / noise
    assert torch.isinf(fp32_scores).all()
    expected = (
        result.probabilities.to(torch.float64) / noise.to(torch.float64)
    ).argmax(dim=-1)
    assert expected.item() == result.token_ids.item() == 1


def test_correction_rejects_an_empty_row_batch():
    empty = torch.ones(0, 3, dtype=torch.float32)
    with pytest.raises(SpeculativeSamplingInvariantError):
        ModifiedRejectionSampler().sample_correction(
            empty,
            empty.clone(),
            correction_noise=torch.ones_like(empty),
        )


@pytest.mark.parametrize(
    ("weight_shape", "token_shape"),
    (
        ((0, 1, 3), (0, 1)),
        ((1, 0, 3), (1, 0)),
        ((1, 1, 0), (1, 1)),
    ),
    ids=("empty-batch", "empty-proposal", "empty-vocabulary"),
)
def test_accept_rejects_empty_dimensions(weight_shape, token_shape):
    weights = torch.ones(weight_shape, dtype=torch.float32)
    tokens = torch.zeros(token_shape, dtype=torch.int64)
    with pytest.raises(SpeculativeSamplingInvariantError):
        ModifiedRejectionSampler().accept(
            weights,
            tokens,
            weights.clone(),
        )


def test_invalid_inputs_preserve_rng_and_caller_tensors():
    target = torch.tensor([[[0.4, 0.6]]], dtype=torch.float32)
    draft = torch.tensor([[[0.8, 0.2]]], dtype=torch.float32)
    target_before = target.clone()
    draft_before = draft.clone()
    torch.manual_seed(20260828)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(SpeculativeSamplingInvariantError):
        ModifiedRejectionSampler().accept(
            target,
            torch.tensor([[0]]),
            draft,
            correction_noise=torch.tensor([[1.0, 0.0]]),
        )

    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(target, target_before)
    assert torch.equal(draft, draft_before)


def test_invalid_selected_q_preserves_rng_and_caller_tensors():
    target = torch.tensor([[[0.4, 0.6]]], dtype=torch.float32)
    draft = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
    target_before = target.clone()
    draft_before = draft.clone()
    torch.manual_seed(20260829)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(SpeculativeSamplingInvariantError):
        ModifiedRejectionSampler().accept(
            target,
            torch.tensor([[0]]),
            draft,
        )

    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(target, target_before)
    assert torch.equal(draft, draft_before)


def test_full_accept_with_supplied_uniforms_consumes_no_rng():
    weights = torch.tensor(
        [[[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]], dtype=torch.float32
    )
    tokens = torch.tensor([[1, 0, 1]])
    uniforms = torch.tensor([[0.9, 0.8, 0.7]])
    torch.manual_seed(20260830)
    rng_before = torch.get_rng_state().clone()

    result = ModifiedRejectionSampler().accept(
        weights,
        tokens,
        weights.clone(),
        uniforms=uniforms,
    )

    assert result.accepted_counts.tolist() == [3]
    assert result.corrective_token_ids.tolist() == [-1]
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_full_accept_generates_only_the_acceptance_uniforms():
    batch_size, proposal_length, vocab_size = 2, 3, 4
    weights = torch.ones(
        batch_size, proposal_length, vocab_size, dtype=torch.float32
    )
    tokens = torch.zeros(batch_size, proposal_length, dtype=torch.int64)

    torch.manual_seed(20260831)
    torch.rand((batch_size, proposal_length), dtype=torch.float32)
    expected_state = torch.get_rng_state().clone()
    torch.manual_seed(20260831)

    result = ModifiedRejectionSampler().accept(
        weights,
        tokens,
        weights.clone(),
    )

    assert result.accepted_counts.tolist() == [proposal_length] * batch_size
    assert torch.equal(torch.get_rng_state(), expected_state)


def test_all_greedy_exact_sampling_consumes_no_rng():
    logits = torch.tensor([[0.1, 0.7, 0.2], [3.0, 1.0, 2.0]])
    temperatures = torch.zeros(2)
    torch.manual_seed(20260901)
    rng_before = torch.get_rng_state().clone()

    result = Sampler().sample_exact_with_probabilities(logits, temperatures)

    assert result.token_ids.tolist() == [1, 0]
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_greedy_match_accepts_and_mismatch_corrects_to_target_argmax():
    target = torch.tensor(
        [[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=torch.float32
    )
    draft = torch.tensor(
        [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=torch.float32
    )
    result = ModifiedRejectionSampler().accept(
        target,
        torch.tensor([[0], [1]]),
        draft,
        uniforms=torch.zeros(2, 1),
        correction_noise=torch.ones(2, 3),
    )

    assert result.accepted_counts.tolist() == [0, 1]
    assert result.corrective_token_ids.tolist() == [1, -1]
    _assert_bool_tensor(result.target_fallback, [False, False])


def _v0_sampler_reference(logits: torch.Tensor, temperatures: torch.Tensor):
    greedy_tokens = logits.argmax(dim=-1)
    logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
    probabilities = torch.softmax(logits, dim=-1)
    sampled_tokens = probabilities.div_(
        torch.empty_like(probabilities).exponential_(1).clamp_min_(1e-10)
    ).argmax(dim=-1)
    return torch.where(temperatures == 0, greedy_tokens, sampled_tokens)


def test_ordinary_sampler_remains_v0_token_and_rng_identical():
    logits = torch.tensor(
        [[0.2, 0.5, -0.1], [1.0, 0.0, 0.5], [-0.2, 0.4, 0.1]],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.0, 0.7, 1.1])
    torch.manual_seed(20260902)
    expected = _v0_sampler_reference(logits.clone(), temperatures)
    expected_state = torch.get_rng_state().clone()
    torch.manual_seed(20260902)

    actual = Sampler.forward.__wrapped__(
        Sampler(), logits.clone(), temperatures
    )

    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_state)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_bfloat16_canonical_seam_is_finite_and_nonmutating(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    logits = torch.tensor(
        [[1.0, 0.5, -0.5, 2.0], [0.1, 1.2, 0.7, -0.3]],
        dtype=torch.bfloat16,
        device=device,
    )
    original = logits.clone()
    temperatures = torch.tensor([0.8, 1.1], device=device)
    rows = torch.tensor([1], dtype=torch.int64, device=device)
    cutoffs = torch.tensor([0.2], device=device)

    probabilities = Sampler().prepare_exact_probabilities(
        logits,
        temperatures,
        top_k_buckets=((3, rows),),
        top_p_plan=(rows, cutoffs),
    )

    assert probabilities.dtype == torch.float32
    assert probabilities.device == logits.device
    assert bool(torch.isfinite(probabilities).all())
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2, device=device),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.equal(logits, original)



def test_trusted_probability_path_matches_validated_exact_oracle():
    logits = torch.tensor(
        [[1.0, 5.0, 5.0, -2.0], [0.2, 1.4, -0.3, 0.8]],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.0, 0.8], dtype=torch.float32)
    rows = torch.tensor([1], dtype=torch.int64)
    top_k = ((3, rows),)
    top_p = (rows, torch.tensor([0.2], dtype=torch.float32))
    sampler = Sampler()

    expected = sampler.prepare_exact_probabilities(
        logits,
        temperatures,
        top_k_buckets=top_k,
        top_p_plan=top_p,
    )
    storage = torch.empty_like(expected)
    actual = sampler.prepare_exact_probabilities_trusted(
        logits,
        temperatures,
        top_k_buckets=top_k,
        top_p_plan=top_p,
        probabilities_out=storage,
        all_greedy=False,
    )

    assert actual.data_ptr() == storage.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_trusted_rejection_preserves_exact_law_for_certain_outcomes():
    target = torch.tensor(
        [[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=torch.float32
    )
    draft = torch.tensor(
        [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=torch.float32
    )
    tokens = torch.tensor([[0], [1]], dtype=torch.int64)

    result = ModifiedRejectionSampler().accept_trusted(target, tokens, draft)

    assert result.accepted_counts.tolist() == [0, 1]
    assert result.corrective_token_ids[0].item() == 1
    _assert_bool_tensor(result.used_reference, [True, False])
    _assert_bool_tensor(result.target_fallback, [False, False])


def _assert_rejection_results_equal(actual, expected):
    assert torch.equal(actual.accepted_counts, expected.accepted_counts)
    assert torch.equal(actual.used_reference, expected.used_reference)
    assert torch.equal(actual.target_fallback, expected.target_fallback)
    # The trusted vectorized path deliberately computes and discards correction
    # samples for full-accept rows. Their token IDs are outside the result
    # contract; compare correction IDs only where rejection used the reference.
    rejected = expected.used_reference
    assert torch.equal(
        actual.corrective_token_ids[rejected],
        expected.corrective_token_ids[rejected],
    )


def test_trusted_rejection_normalizes_fp32_row_mass_before_acceptance_ratio():
    target = torch.tensor(
        [[[0.1, 0.2, 0.7000001]]], dtype=torch.float32
    )
    draft = torch.tensor(
        [[[0.1, 0.2, 0.6999998]]], dtype=torch.float32
    )
    tokens = torch.tensor([[0]], dtype=torch.int64)
    # The largest representable FP32 value below one lies above normalized
    # p(d)/q(d), but below the old raw-weight ratio after it was clamped to one.
    uniform = torch.nextafter(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    noise = torch.ones(1, 3, dtype=torch.float32)
    sampler = ModifiedRejectionSampler()

    expected = sampler.accept(
        target,
        tokens,
        draft,
        uniforms=uniform,
        correction_noise=noise,
    )
    actual = sampler.accept_trusted(
        target,
        tokens,
        draft,
        uniforms=uniform,
        correction_noise=noise,
    )

    assert expected.accepted_counts.tolist() == [0]
    _assert_rejection_results_equal(actual, expected)


@pytest.mark.parametrize("batch_size,proposal_length,vocab_size", [(1, 1, 3), (3, 4, 11)])
def test_trusted_rejection_matches_validated_oracle_for_fixed_randomness(
    batch_size,
    proposal_length,
    vocab_size,
):
    generator = torch.Generator().manual_seed(
        20260908 + batch_size * 100 + proposal_length
    )
    target = torch.rand(
        batch_size, proposal_length, vocab_size, generator=generator
    ).add_(0.01)
    draft = torch.rand(
        batch_size, proposal_length, vocab_size, generator=generator
    ).add_(0.01)
    tokens = draft.argmax(dim=-1)
    uniforms = torch.rand(
        batch_size, proposal_length, generator=generator
    )
    correction_noise = torch.rand(
        batch_size, vocab_size, generator=generator
    ).add_(0.01)
    sampler = ModifiedRejectionSampler()

    expected = sampler.accept(
        target,
        tokens,
        draft,
        uniforms=uniforms,
        correction_noise=correction_noise,
    )
    actual = sampler.accept_trusted(
        target,
        tokens,
        draft,
        uniforms=uniforms,
        correction_noise=correction_noise,
    )
    _assert_rejection_results_equal(actual, expected)
