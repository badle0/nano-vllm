import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_ks: torch.Tensor | None):
        greedy_tokens = logits.argmax(dim=-1)
        logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        if top_ks is not None:
            vocab = logits.size(-1)
            k = torch.where(top_ks > 0, top_ks, vocab).clamp(max=vocab)
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            kth = sorted_logits.gather(-1, (k - 1).unsqueeze(1))
            sorted_logits = sorted_logits.masked_fill(sorted_logits < kth, float("-inf"))
            logits = torch.empty_like(sorted_logits).scatter_(-1, sorted_idx, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
