from functools import lru_cache
from typing import NamedTuple

import torch
from torch import nn


class SpeculativeSamplingInvariantError(RuntimeError):
    """The retained target/proposal laws cannot support exact rejection."""


class ExactSample(NamedTuple):
    token_ids: torch.Tensor
    probabilities: torch.Tensor


class ResidualSample(NamedTuple):
    token_ids: torch.Tensor
    used_reference: torch.Tensor
    target_fallback: torch.Tensor


class ModifiedRejectionResult(NamedTuple):
    accepted_counts: torch.Tensor
    corrective_token_ids: torch.Tensor
    used_reference: torch.Tensor
    target_fallback: torch.Tensor


def _validate_row_indices(
    row_indices: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> int:
    if row_indices is None:
        return batch_size
    if not isinstance(row_indices, torch.Tensor):
        raise TypeError(f"{name} must be a tensor or None")
    if row_indices.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype torch.int64")
    if row_indices.device != device:
        raise ValueError(f"{name} must be on the logits device")
    if row_indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if row_indices.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if bool(((row_indices < 0) | (row_indices >= batch_size)).any()):
        raise ValueError(f"{name} contains an out-of-range row")
    if row_indices.unique().numel() != row_indices.numel():
        raise ValueError(f"{name} contains duplicate rows")
    return row_indices.numel()


def _validate_race_noise(
    noise: torch.Tensor,
    *,
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
    name: str,
):
    if not isinstance(noise, torch.Tensor):
        raise SpeculativeSamplingInvariantError(f"{name} must be a tensor")
    if tuple(noise.shape) != tuple(shape):
        raise SpeculativeSamplingInvariantError(
            f"{name} must have shape {tuple(shape)}"
        )
    if noise.device != device:
        raise SpeculativeSamplingInvariantError(
            f"{name} must be on the sampling device"
        )
    if noise.dtype != torch.float32:
        raise SpeculativeSamplingInvariantError(
            f"{name} must have dtype torch.float32"
        )
    if not bool(torch.isfinite(noise).all()) or not bool((noise > 0).all()):
        raise SpeculativeSamplingInvariantError(
            f"{name} must contain finite, strictly positive values"
        )


def _sample_exponential_race(
    weights: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Sample rows without mutating the retained categorical weights."""

    # Valid FP32 exponential draws can be subnormal. Dividing in FP32 may turn
    # several distinct race scores into ``inf`` and let argmax's tie rule choose
    # the wrong token. V1 is a reference path, so retain the supplied draw and
    # score it in FP64. The ordinary compiled sampler remains unchanged.
    return weights.to(torch.float64).div(noise.to(torch.float64)).argmax(dim=-1)


def _validate_canonical_weights(
    weights: torch.Tensor,
    *,
    name: str,
    ndim: int,
) -> torch.Tensor:
    if not isinstance(weights, torch.Tensor):
        raise SpeculativeSamplingInvariantError(f"{name} must be a tensor")
    if weights.dtype != torch.float32:
        raise SpeculativeSamplingInvariantError(
            f"{name} must retain the canonical torch.float32 dtype"
        )
    if weights.ndim != ndim:
        raise SpeculativeSamplingInvariantError(
            f"{name} must be {ndim}-dimensional"
        )
    if any(size == 0 for size in weights.shape):
        raise SpeculativeSamplingInvariantError(
            f"{name} must have no empty dimensions"
        )
    if not bool(torch.isfinite(weights).all()):
        raise SpeculativeSamplingInvariantError(f"{name} contains non-finite weights")
    if bool((weights < 0).any()):
        raise SpeculativeSamplingInvariantError(f"{name} contains negative weights")
    row_masses = weights.sum(dim=-1, dtype=torch.float64)
    if not bool(torch.isfinite(row_masses).all()) or not bool((row_masses > 0).all()):
        raise SpeculativeSamplingInvariantError(
            f"{name} rows must have finite, strictly positive mass"
        )
    return row_masses


@lru_cache(maxsize=1)
def _load_flashinfer_sampling():
    try:
        from flashinfer import sampling
    except ImportError as exc:
        raise RuntimeError(
            "top_p_backend='flashinfer' requires the optional fast-sampling "
            "dependencies; install nano-vllm[fast-sampling]"
        ) from exc
    return sampling


def require_flashinfer_sampling():
    """Fail before engine workers start if the optional backend is missing."""

    return _load_flashinfer_sampling()


class Sampler(nn.Module):

    TOP_P_CHUNK_SIZE = 64

    @torch.compile
    def greedy(self, logits: torch.Tensor):
        return logits.argmax(dim=-1)

    @torch.inference_mode()
    def filter_top_k(
        self,
        logits: torch.Tensor,
        row_indices: torch.Tensor | None,
        top_k: int,
    ):
        active_logits = logits if row_indices is None else logits.index_select(0, row_indices)
        top_k_values = torch.topk(active_logits, top_k, dim=-1, sorted=False).values
        threshold = top_k_values.amin(dim=-1, keepdim=True)
        active_logits.masked_fill_(active_logits < threshold, float("-inf"))
        if row_indices is not None:
            logits.index_copy_(0, row_indices, active_logits)
        return logits

    @torch.inference_mode()
    def filter_top_p(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        row_indices: torch.Tensor | None,
        probability_cutoffs: torch.Tensor,
    ):
        active_logits = logits if row_indices is None else logits.index_select(0, row_indices)
        active_temperatures = (
            temperatures
            if row_indices is None
            else temperatures.index_select(0, row_indices)
        )
        for start in range(0, active_logits.size(0), self.TOP_P_CHUNK_SIZE):
            end = min(start + self.TOP_P_CHUNK_SIZE, active_logits.size(0))
            chunk_logits = active_logits[start:end]
            chunk_temperatures = active_temperatures[start:end].clamp_min(1e-10)
            # Tensor.float() aliases FP32 input. Use a private workspace so
            # filtering does not pre-scale logits that forward() will scale.
            scaled_logits = chunk_logits.to(
                dtype=torch.float32,
                copy=True,
            ).div_(
                chunk_temperatures.unsqueeze(dim=1)
            )
            sorted_logits, sorted_indices = torch.sort(
                scaled_logits, dim=-1, descending=False
            )
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= (
                probability_cutoffs[start:end].unsqueeze(dim=1)
            )
            sorted_indices_to_remove[:, -1] = False
            indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter_(
                -1, sorted_indices, sorted_indices_to_remove
            )
            chunk_logits.masked_fill_(indices_to_remove, float("-inf"))
        if row_indices is not None:
            logits.index_copy_(0, row_indices, active_logits)
        return logits

    @torch.inference_mode()
    def prepare_exact_probabilities(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        *,
        top_k_buckets: tuple[
            tuple[int, torch.Tensor | None], ...
        ] = (),
        top_p_plan: tuple[torch.Tensor | None, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Build the exact sampler law without mutating caller-owned logits.

        This is an opt-in seam for speculative decoding.  The ordinary compiled
        ``forward`` path below intentionally remains unchanged.
        """

        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("logits must be a two-dimensional tensor")
        if not logits.is_floating_point():
            raise TypeError("logits must have a floating-point dtype")
        batch_size, vocab_size = logits.shape
        if batch_size == 0 or vocab_size == 0:
            raise ValueError("logits must have non-empty batch and vocabulary dimensions")
        if not isinstance(temperatures, torch.Tensor):
            raise TypeError("temperatures must be a tensor")
        if temperatures.shape != (batch_size,):
            raise ValueError("temperatures must have shape [batch_size]")
        if temperatures.device != logits.device:
            raise ValueError("temperatures must be on the logits device")
        if not temperatures.is_floating_point():
            raise TypeError("temperatures must have a floating-point dtype")
        if not bool(torch.isfinite(temperatures).all()) or bool(
            (temperatures < 0).any()
        ):
            raise ValueError("temperatures must be finite and non-negative")

        validated_top_k = []
        top_k_rows_seen = torch.zeros(
            batch_size, dtype=torch.bool, device=logits.device
        )
        for bucket_index, bucket in enumerate(top_k_buckets):
            if not isinstance(bucket, tuple) or len(bucket) != 2:
                raise TypeError("each top-k bucket must be a (top_k, row_indices) tuple")
            top_k, row_indices = bucket
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise TypeError("top_k must be an integer")
            if not 1 <= top_k <= vocab_size:
                raise ValueError("top_k must be in [1, vocab_size]")
            _validate_row_indices(
                row_indices,
                batch_size=batch_size,
                device=logits.device,
                name=f"top_k_buckets[{bucket_index}].row_indices",
            )
            active_rows = (
                torch.arange(batch_size, device=logits.device)
                if row_indices is None
                else row_indices
            )
            if bool(top_k_rows_seen.index_select(0, active_rows).any()):
                raise ValueError("top-k buckets must not overlap")
            top_k_rows_seen.index_fill_(0, active_rows, True)
            validated_top_k.append((top_k, row_indices))

        validated_top_p = None
        if top_p_plan is not None:
            if not isinstance(top_p_plan, tuple) or len(top_p_plan) != 2:
                raise TypeError(
                    "top_p_plan must be a (row_indices, probability_cutoffs) tuple"
                )
            row_indices, probability_cutoffs = top_p_plan
            active_count = _validate_row_indices(
                row_indices,
                batch_size=batch_size,
                device=logits.device,
                name="top_p_plan.row_indices",
            )
            if not isinstance(probability_cutoffs, torch.Tensor):
                raise TypeError("probability_cutoffs must be a tensor")
            if probability_cutoffs.shape != (active_count,):
                raise ValueError(
                    "probability_cutoffs must have one value per active row"
                )
            if probability_cutoffs.device != logits.device:
                raise ValueError("probability_cutoffs must be on the logits device")
            if not probability_cutoffs.is_floating_point():
                raise TypeError("probability_cutoffs must be floating point")
            if (
                not bool(torch.isfinite(probability_cutoffs).all())
                or bool((probability_cutoffs < 0).any())
                or bool((probability_cutoffs >= 1).any())
            ):
                raise ValueError("probability_cutoffs must be finite and in [0, 1)")
            validated_top_p = (row_indices, probability_cutoffs)

        # Both current filters mutate their argument.  Preserve their exact tie
        # and mixed-active-row rules on a private copy.
        filtered_logits = logits.clone()
        greedy_tokens = logits.argmax(dim=-1)
        for top_k, row_indices in validated_top_k:
            self.filter_top_k(filtered_logits, row_indices, top_k)
        if validated_top_p is not None:
            row_indices, probability_cutoffs = validated_top_p
            self.filter_top_p(
                filtered_logits,
                temperatures,
                row_indices,
                probability_cutoffs,
            )

        scaled_logits = filtered_logits.to(
            dtype=torch.float32, copy=True
        ).div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        probabilities = torch.softmax(scaled_logits, dim=-1)

        greedy_rows = torch.nonzero(temperatures == 0, as_tuple=False).flatten()
        if greedy_rows.numel():
            probabilities.index_fill_(0, greedy_rows, 0.0)
            probabilities[greedy_rows, greedy_tokens.index_select(0, greedy_rows)] = 1.0

        if not bool(torch.isfinite(probabilities).all()) or not bool(
            (probabilities.sum(dim=-1) > 0).all()
        ):
            raise SpeculativeSamplingInvariantError(
                "canonical probability rows must be finite with positive mass"
            )
        return probabilities

    @torch.inference_mode()
    def sample_exact_with_probabilities(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        *,
        top_k_buckets: tuple[
            tuple[int, torch.Tensor | None], ...
        ] = (),
        top_p_plan: tuple[torch.Tensor | None, torch.Tensor] | None = None,
        race_noise: torch.Tensor | None = None,
    ) -> ExactSample:
        """Draw from, and retain, the exact canonical FP32 distribution."""

        probabilities = self.prepare_exact_probabilities(
            logits,
            temperatures,
            top_k_buckets=top_k_buckets,
            top_p_plan=top_p_plan,
        )
        if race_noise is not None:
            _validate_race_noise(
                race_noise,
                shape=probabilities.shape,
                device=probabilities.device,
                name="race_noise",
            )

        greedy_tokens = probabilities.argmax(dim=-1)
        sampled_rows = temperatures != 0
        if not bool(sampled_rows.any()):
            return ExactSample(greedy_tokens, probabilities)

        if race_noise is None:
            race_noise = torch.empty_like(probabilities).exponential_(1)
            _validate_race_noise(
                race_noise,
                shape=probabilities.shape,
                device=probabilities.device,
                name="generated race_noise",
            )
        sample_tokens = _sample_exponential_race(probabilities, race_noise)
        token_ids = torch.where(sampled_rows, sample_tokens, greedy_tokens)
        return ExactSample(token_ids, probabilities)

    @torch.inference_mode()
    def sample_top_p_flashinfer(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
    ):
        """Sample with FlashInfer's sorting-free, statistical top-p contract.

        This is deliberately separate from :meth:`filter_top_p`: FlashInfer
        uses its own Philox draws and boundary-tie rule, so it cannot preserve
        the exact backend's fixed-seed token stream.
        """

        sampling = _load_flashinfer_sampling()
        greedy_tokens = logits.argmax(dim=-1)
        probabilities = sampling.softmax(
            logits,
            temperature=temperatures.clamp_min(1e-10),
        )
        sample_tokens = sampling.top_p_sampling_from_probs(
            probabilities,
            top_ps,
            deterministic=True,
        )
        return torch.where(
            temperatures == 0,
            greedy_tokens,
            sample_tokens.to(greedy_tokens.dtype),
        )

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_tokens = logits.argmax(dim=-1)
        logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)


class ModifiedRejectionSampler:
    """Exact modified rejection over retained canonical FP32 weight rows.

    The implementation is deliberately eager in V1.  Validation and the FP64
    recovery branch are easier to audit here; graph/compile integration belongs
    to the later execution-path milestone.
    """

    @staticmethod
    def _validate_pair(
        target_weights: torch.Tensor,
        draft_weights: torch.Tensor,
        *,
        ndim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_masses = _validate_canonical_weights(
            target_weights, name="target_weights", ndim=ndim
        )
        draft_masses = _validate_canonical_weights(
            draft_weights, name="draft_weights", ndim=ndim
        )
        if target_weights.shape != draft_weights.shape:
            raise SpeculativeSamplingInvariantError(
                "target_weights and draft_weights must have identical shapes"
            )
        if target_weights.device != draft_weights.device:
            raise SpeculativeSamplingInvariantError(
                "target_weights and draft_weights must be on the same device"
            )
        return target_masses, draft_masses

    @torch.inference_mode()
    def sample_correction(
        self,
        target_weights: torch.Tensor,
        draft_weights: torch.Tensor,
        *,
        correction_noise: torch.Tensor | None = None,
    ) -> ResidualSample:
        """Sample ``normalize(max(p-q, 0))`` with explicit recovery metadata."""

        target_masses, draft_masses = self._validate_pair(
            target_weights, draft_weights, ndim=2
        )
        row_count, vocab_size = target_weights.shape
        if correction_noise is not None:
            _validate_race_noise(
                correction_noise,
                shape=(row_count, vocab_size),
                device=target_weights.device,
                name="correction_noise",
            )

        # V1 is the correctness oracle, not the eventual optimized kernel.  A
        # merely positive FP32 residual is insufficient: normalization can lose
        # FP64-positive support even when its mass is far above ``tiny``.  Build
        # every corrective law by upcasting the original retained FP32 rows.
        reference_target = target_weights.to(torch.float64) / target_masses.unsqueeze(
            dim=-1
        )
        reference_draft = draft_weights.to(torch.float64) / draft_masses.unsqueeze(
            dim=-1
        )
        reference_residual = (reference_target - reference_draft).clamp_min(0.0)
        reference_masses = reference_residual.sum(dim=-1)
        if not bool(torch.isfinite(reference_masses).all()) or not bool(
            torch.isfinite(reference_residual).all()
        ):
            raise SpeculativeSamplingInvariantError(
                "the FP64 residual reference path produced non-finite values"
            )
        target_fallback = reference_masses == 0

        correction_laws = reference_residual.clone()
        positive_rows = ~target_fallback
        if bool(positive_rows.any()):
            correction_laws[positive_rows] = (
                correction_laws[positive_rows]
                / reference_masses[positive_rows].unsqueeze(dim=-1)
            )
        if bool(target_fallback.any()):
            # A numerical rejection with robust p == q is analytically
            # unreachable.  The declared availability policy samples p, never
            # an arbitrary/uniform token, and exposes the event to later metrics.
            correction_laws[target_fallback] = reference_target[target_fallback]

        # All caller data and all derived branch invariants have been validated
        # before this first possible random draw.
        if correction_noise is None:
            correction_noise = torch.empty_like(target_weights).exponential_(1)
            _validate_race_noise(
                correction_noise,
                shape=(row_count, vocab_size),
                device=target_weights.device,
                name="generated correction_noise",
            )
        token_ids = _sample_exponential_race(correction_laws, correction_noise)
        used_reference = torch.ones(
            row_count, dtype=torch.bool, device=target_weights.device
        )
        return ResidualSample(token_ids, used_reference, target_fallback)

    @torch.inference_mode()
    def accept(
        self,
        target_weights: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_weights: torch.Tensor,
        *,
        uniforms: torch.Tensor | None = None,
        correction_noise: torch.Tensor | None = None,
    ) -> ModifiedRejectionResult:
        """Accept the longest draft prefix and sample its first correction."""

        target_masses, draft_masses = self._validate_pair(
            target_weights, draft_weights, ndim=3
        )
        batch_size, proposal_length, vocab_size = target_weights.shape
        if not isinstance(draft_tokens, torch.Tensor):
            raise SpeculativeSamplingInvariantError("draft_tokens must be a tensor")
        if draft_tokens.dtype != torch.int64:
            raise SpeculativeSamplingInvariantError(
                "draft_tokens must have dtype torch.int64"
            )
        if draft_tokens.shape != (batch_size, proposal_length):
            raise SpeculativeSamplingInvariantError(
                "draft_tokens must have shape [batch_size, proposal_length]"
            )
        if draft_tokens.device != target_weights.device:
            raise SpeculativeSamplingInvariantError(
                "draft_tokens must be on the sampling device"
            )
        if bool(((draft_tokens < 0) | (draft_tokens >= vocab_size)).any()):
            raise SpeculativeSamplingInvariantError(
                "draft_tokens contains an out-of-range token ID"
            )

        if uniforms is not None:
            if not isinstance(uniforms, torch.Tensor):
                raise SpeculativeSamplingInvariantError("uniforms must be a tensor")
            if uniforms.shape != (batch_size, proposal_length):
                raise SpeculativeSamplingInvariantError(
                    "uniforms must have shape [batch_size, proposal_length]"
                )
            if uniforms.device != target_weights.device:
                raise SpeculativeSamplingInvariantError(
                    "uniforms must be on the sampling device"
                )
            if uniforms.dtype != torch.float32:
                raise SpeculativeSamplingInvariantError(
                    "uniforms must have dtype torch.float32"
                )
            if not bool(torch.isfinite(uniforms).all()) or bool(
                ((uniforms < 0) | (uniforms >= 1)).any()
            ):
                raise SpeculativeSamplingInvariantError(
                    "uniforms must be finite and in the half-open interval [0, 1)"
                )
        if correction_noise is not None:
            _validate_race_noise(
                correction_noise,
                shape=(batch_size, vocab_size),
                device=target_weights.device,
                name="correction_noise",
            )

        selected = draft_tokens.unsqueeze(dim=-1)
        selected_target_weights = target_weights.gather(-1, selected).squeeze(-1)
        selected_draft_weights = draft_weights.gather(-1, selected).squeeze(-1)
        if not bool((selected_draft_weights > 0).all()):
            raise SpeculativeSamplingInvariantError(
                "every selected draft token must have finite q(d) > 0"
            )
        selected_target = selected_target_weights.to(torch.float64) / target_masses
        selected_draft = selected_draft_weights.to(torch.float64) / draft_masses
        acceptance_probabilities = (selected_target / selected_draft).clamp(max=1.0)
        if not bool(torch.isfinite(acceptance_probabilities).all()):
            raise SpeculativeSamplingInvariantError(
                "acceptance probabilities must be finite"
            )

        # Validation is complete before either acceptance or correction RNG is
        # consumed.  Full-accept rows never enter residual classification.
        if uniforms is None:
            uniforms = torch.rand(
                (batch_size, proposal_length),
                dtype=torch.float32,
                device=target_weights.device,
            )
        accepts = uniforms.to(torch.float64) < acceptance_probabilities
        accepted_prefix = accepts.to(torch.int64).cumprod(dim=1).to(torch.bool)
        accepted_counts = accepted_prefix.sum(dim=1, dtype=torch.int64)

        corrective_token_ids = torch.full(
            (batch_size,), -1, dtype=torch.int64, device=target_weights.device
        )
        used_reference = torch.zeros(
            batch_size, dtype=torch.bool, device=target_weights.device
        )
        target_fallback = torch.zeros_like(used_reference)
        rejected_rows = torch.nonzero(
            accepted_counts < proposal_length, as_tuple=False
        ).flatten()
        if rejected_rows.numel():
            rejection_positions = accepted_counts.index_select(0, rejected_rows)
            rejected_target = target_weights[
                rejected_rows, rejection_positions
            ]
            rejected_draft = draft_weights[
                rejected_rows, rejection_positions
            ]
            rejected_noise = (
                None
                if correction_noise is None
                else correction_noise.index_select(0, rejected_rows)
            )
            correction = self.sample_correction(
                rejected_target,
                rejected_draft,
                correction_noise=rejected_noise,
            )
            corrective_token_ids.index_copy_(
                0, rejected_rows, correction.token_ids
            )
            used_reference.index_copy_(
                0, rejected_rows, correction.used_reference
            )
            target_fallback.index_copy_(
                0, rejected_rows, correction.target_fallback
            )

        return ModifiedRejectionResult(
            accepted_counts,
            corrective_token_ids,
            used_reference,
            target_fallback,
        )
