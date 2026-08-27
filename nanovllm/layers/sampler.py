from functools import lru_cache

import torch
from torch import nn


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
