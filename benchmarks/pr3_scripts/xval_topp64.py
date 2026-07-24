import torch
from transformers import TopPLogitsWarper

torch.manual_seed(0)
worst = 0.0
mismatches64 = 0
for trial in range(200):
    logits = torch.randn(4, 1000, dtype=torch.bfloat16).double()
    for p in (0.3, 0.8, 0.95):
        sp, si = torch.softmax(logits, -1).sort(-1, descending=True)
        keep_sorted = (sp.cumsum(-1) - sp) <= p
        mine = torch.zeros_like(keep_sorted).scatter_(-1, si, keep_sorted)
        hf = TopPLogitsWarper(top_p=p)(None, logits.clone()) > float("-inf")
        if not torch.equal(mine, hf):
            mismatches64 += 1
            d = (mine ^ hf).nonzero()
            excl = sp.cumsum(-1) - sp
            for r, c in d.tolist():
                sc = int((si[r] == c).nonzero())
                worst = max(worst, abs(float(excl[r, sc] - p)))
print("fp64 mismatches:", mismatches64, "/ 600 | worst boundary offset among diffs:", worst)
