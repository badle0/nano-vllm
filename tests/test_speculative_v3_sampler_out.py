import math

import pytest
import torch

from nanovllm.layers.sampler import (
    Sampler,
    SpeculativeSamplingInvariantError,
)


def _independent_probability_row(
    logits: torch.Tensor,
    temperature: float,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Independent dense oracle for one exact-backend sampling row."""

    row = logits.to(dtype=torch.float32, copy=True)
    if temperature == 0.0:
        probabilities = torch.zeros_like(row)
        probabilities[int(row.argmax().item())] = 1.0
        return probabilities

    row.div_(temperature)
    if top_k is not None:
        threshold = row.topk(top_k, sorted=False).values.amin()
        row.masked_fill_(row < threshold, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_indices = row.sort(descending=False)
        remove_sorted = sorted_logits.softmax(dim=-1).cumsum(dim=-1) <= (
            1.0 - top_p
        )
        remove_sorted[-1] = False
        remove = torch.zeros_like(remove_sorted).scatter(
            0, sorted_indices, remove_sorted
        )
        row.masked_fill_(remove, float("-inf"))
    return row.softmax(dim=-1)


def _plans(mode: str, batch_size: int):
    temperatures = torch.tensor([0.0, 0.7, 1.3], dtype=torch.float32)
    top_k_buckets = ()
    top_p_plan = None
    row_top_ks = [None] * batch_size
    row_top_ps = [None] * batch_size

    if mode == "greedy":
        temperatures.zero_()
    elif mode == "top_k":
        temperatures[:] = torch.tensor([0.8, 1.1, 0.6])
        top_k_buckets = ((2, None),)
        row_top_ks = [2] * batch_size
    elif mode == "top_p":
        temperatures[:] = torch.tensor([0.8, 1.1, 0.6])
        top_ps = [0.55, 0.75, 0.9]
        top_p_plan = (
            None,
            torch.tensor([1.0 - value for value in top_ps]),
        )
        row_top_ps = top_ps
    elif mode == "combined":
        top_k_buckets = (
            (2, torch.tensor([1], dtype=torch.int64)),
            (4, torch.tensor([2], dtype=torch.int64)),
        )
        top_ps = [0.8, 0.65, 0.9]
        top_p_plan = (
            None,
            torch.tensor([1.0 - value for value in top_ps]),
        )
        row_top_ks = [None, 2, 4]
        row_top_ps = top_ps
    elif mode not in {"plain", "mixed"}:
        raise AssertionError(f"unknown mode: {mode}")

    return (
        temperatures,
        top_k_buckets,
        top_p_plan,
        row_top_ks,
        row_top_ps,
    )


@pytest.mark.parametrize(
    "mode", ("plain", "greedy", "mixed", "top_k", "top_p", "combined")
)
def test_caller_storage_preserves_exact_sampler_behavior(mode):
    logits = torch.tensor(
        [
            [1.0, 5.0, 5.0, -2.0, 0.0],
            [2.0, 1.0, 0.0, -1.0, -2.0],
            [0.2, 1.4, -0.3, 0.8, 0.1],
        ],
        dtype=torch.float32,
    )
    (
        temperatures,
        top_k_buckets,
        top_p_plan,
        row_top_ks,
        row_top_ps,
    ) = _plans(mode, logits.size(0))
    noise = torch.tensor(
        [
            [0.7, 1.1, 0.2, 3.0, 1.5],
            [1.3, 0.4, 2.0, 0.9, 1.7],
            [2.1, 0.8, 1.4, 0.3, 1.0],
        ],
        dtype=torch.float32,
    )
    original_logits = logits.clone()
    original_temperatures = temperatures.clone()
    original_noise = noise.clone()
    expected_probabilities = torch.stack(
        tuple(
            _independent_probability_row(
                logits[row],
                float(temperatures[row]),
                top_k=row_top_ks[row],
                top_p=row_top_ps[row],
            )
            for row in range(logits.size(0))
        )
    )
    sampled = (
        expected_probabilities.to(torch.float64) / noise.to(torch.float64)
    ).argmax(dim=-1)
    expected_tokens = torch.where(
        temperatures == 0,
        expected_probabilities.argmax(dim=-1),
        sampled,
    )

    baseline = Sampler().sample_exact_with_probabilities(
        logits,
        temperatures,
        top_k_buckets=top_k_buckets,
        top_p_plan=top_p_plan,
        race_noise=noise,
    )
    out = torch.full_like(logits, float("nan"))
    retained = Sampler().sample_exact_with_probabilities(
        logits,
        temperatures,
        top_k_buckets=top_k_buckets,
        top_p_plan=top_p_plan,
        race_noise=noise,
        probabilities_out=out,
    )

    assert retained.probabilities is out
    assert retained.probabilities.data_ptr() == out.data_ptr()
    assert torch.equal(retained.probabilities, baseline.probabilities)
    assert torch.equal(retained.probabilities, expected_probabilities)
    assert torch.equal(retained.token_ids, baseline.token_ids)
    assert torch.equal(retained.token_ids, expected_tokens)
    assert not bool(torch.isnan(out).any())
    assert torch.equal(logits, original_logits)
    assert torch.equal(temperatures, original_temperatures)
    assert torch.equal(noise, original_noise)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_prepare_softmax_writes_directly_into_one_kbv_step(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    logits = torch.tensor(
        [[1.0, 0.5, -0.5, 2.0], [0.1, 1.2, 0.7, -0.3]],
        dtype=torch.bfloat16,
        device=device,
    )
    original_logits = logits.clone()
    temperatures = torch.tensor([0.8, 1.1], device=device)
    original_temperatures = temperatures.clone()
    rows = torch.tensor([1], dtype=torch.int64, device=device)
    original_rows = rows.clone()
    cutoffs = torch.tensor([0.2], device=device)
    original_cutoffs = cutoffs.clone()
    expected = torch.stack(
        (
            _independent_probability_row(logits[0], float(temperatures[0])),
            _independent_probability_row(
                logits[1],
                float(temperatures[1]),
                top_k=3,
                top_p=0.8,
            ),
        )
    )

    storage = torch.full(
        (3, *logits.shape),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    out = storage[1]
    original_softmax = torch.softmax
    observed_out = []

    def _softmax_spy(input_tensor, dim, *args, **kwargs):
        observed_out.append(kwargs.get("out"))
        return original_softmax(input_tensor, dim, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", _softmax_spy)
    probabilities = Sampler().prepare_exact_probabilities(
        logits,
        temperatures,
        top_k_buckets=((3, rows),),
        top_p_plan=(rows, cutoffs),
        probabilities_out=out,
    )

    assert len(observed_out) == 1
    assert observed_out[0] is out
    assert probabilities is out
    assert probabilities.data_ptr() == storage[1].data_ptr()
    assert torch.equal(probabilities, expected)
    assert bool(torch.isnan(storage[0]).all())
    assert bool(torch.isnan(storage[2]).all())
    # The runner can expose [B, K, V] as a zero-copy view of the same storage.
    batch_major = storage.permute(1, 0, 2)
    assert (
        batch_major.untyped_storage().data_ptr()
        == storage.untyped_storage().data_ptr()
    )
    assert torch.equal(batch_major[:, 1], probabilities)
    assert torch.equal(logits, original_logits)
    assert torch.equal(temperatures, original_temperatures)
    assert torch.equal(rows, original_rows)
    assert torch.equal(cutoffs, original_cutoffs)


@pytest.mark.parametrize(
    ("out", "error", "message"),
    (
        (object(), TypeError, "must be a tensor or None"),
        (torch.empty(2, 3), ValueError, "must have shape"),
        (torch.empty(2, 4, dtype=torch.float64), TypeError, "torch.float32"),
        (torch.empty(4, 2).t(), ValueError, "must be contiguous"),
        (torch.empty(2, 4, device="meta"), ValueError, "logits device"),
    ),
)
def test_prepare_rejects_invalid_output_contract_without_mutating_logits(
    out, error, message
):
    logits = torch.tensor(
        [[1.0, 0.0, -1.0, 2.0], [0.5, 0.25, -0.5, 1.0]]
    )
    original = logits.clone()

    with pytest.raises(error, match=message):
        Sampler().prepare_exact_probabilities(
            logits,
            torch.ones(2),
            probabilities_out=out,
        )

    assert torch.equal(logits, original)


def test_prepare_rejects_output_that_aliases_logits():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.25, -0.5]])
    original = logits.clone()

    with pytest.raises(ValueError, match="share storage with logits"):
        Sampler().prepare_exact_probabilities(
            logits,
            torch.ones(2),
            probabilities_out=logits,
        )

    assert torch.equal(logits, original)


def test_prepare_rejects_output_that_aliases_temperatures():
    out = torch.full((2, 3), float("nan"))
    temperatures = out[:, 0]
    temperatures.fill_(1.0)
    before = out.clone()

    with pytest.raises(ValueError, match="share storage with temperatures"):
        Sampler().prepare_exact_probabilities(
            torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.25, -0.5]]),
            temperatures,
            probabilities_out=out,
        )

    torch.testing.assert_close(out, before, equal_nan=True)


def test_prepare_rejects_output_that_aliases_top_p_plan():
    out = torch.full((2, 3), float("nan"))
    cutoffs = out[:, 0]
    cutoffs[:] = torch.tensor([0.2, 0.1])
    before = out.clone()

    with pytest.raises(ValueError, match="top_p_plan.probability_cutoffs"):
        Sampler().prepare_exact_probabilities(
            torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.25, -0.5]]),
            torch.ones(2),
            top_p_plan=(None, cutoffs),
            probabilities_out=out,
        )

    torch.testing.assert_close(out, before, equal_nan=True)


def test_sample_rejects_output_that_aliases_race_noise_before_writing():
    shared = torch.tensor([[0.7, 1.1, 0.2], [1.3, 0.4, 2.0]])
    before = shared.clone()

    with pytest.raises(ValueError, match="share storage with race_noise"):
        Sampler().sample_exact_with_probabilities(
            torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.25, -0.5]]),
            torch.ones(2),
            race_noise=shared,
            probabilities_out=shared,
        )

    assert torch.equal(shared, before)


def test_invalid_race_noise_does_not_partially_fill_caller_storage():
    out = torch.full((2, 3), float("nan"))
    invalid_noise = torch.ones_like(out)
    invalid_noise[1, 2] = math.nan

    with pytest.raises(SpeculativeSamplingInvariantError, match="strictly positive"):
        Sampler().sample_exact_with_probabilities(
            torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.25, -0.5]]),
            torch.ones(2),
            race_noise=invalid_noise,
            probabilities_out=out,
        )

    assert bool(torch.isnan(out).all())
