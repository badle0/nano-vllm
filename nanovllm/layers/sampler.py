import torch
from torch import nn


class Sampler(nn.Module):

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

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_tokens = logits.argmax(dim=-1)
        logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
