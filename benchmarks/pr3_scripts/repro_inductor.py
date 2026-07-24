import torch

@torch.compile
def f(logits, top_ps):
    s, idx = logits.sort(dim=-1, descending=True)
    p = s.softmax(dim=-1)
    cum = p.cumsum(dim=-1)
    s = s.masked_fill((cum - p) > top_ps.unsqueeze(1), float("-inf"))
    return torch.empty_like(s).scatter_(-1, idx, s)

x = torch.randn(256, 151936, device="cuda")
tp = torch.full((256,), 0.9, device="cuda")
print(f(x, tp).shape)
x2 = torch.randn(64, 151936, device="cuda")
tp2 = torch.full((64,), 0.9, device="cuda")
print(f(x2, tp2).shape)   # second distinct bs -> dynamic-dim recompile
