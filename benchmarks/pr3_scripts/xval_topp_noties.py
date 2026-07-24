import torch
from transformers import TopPLogitsWarper

torch.manual_seed(0)
mismatches = 0
for trial in range(200):
    logits = torch.randn(4, 1000, dtype=torch.float32)
    for p in (0.3, 0.8, 0.95):
        sp, si = torch.softmax(logits, -1).sort(-1, descending=True)
        keep_sorted = (sp.cumsum(-1) - sp) <= p
        mine = torch.zeros_like(keep_sorted).scatter_(-1, si, keep_sorted)
        hf = TopPLogitsWarper(top_p=p)(None, logits.clone()) > float("-inf")
        if not torch.equal(mine, hf):
            mismatches += 1
            if mismatches <= 3:
                d = (mine ^ hf).nonzero()
                excl = sp.cumsum(-1) - sp
                r, c = d[0].tolist()
                sc = int((si[r] == c).nonzero())
                print(f"trial {trial} p={p}: {len(d)} diffs; first at row {r} tok {c}, excl-cum-minus-p={float(excl[r, sc]-p):+.2e}")
print("mismatches:", mismatches, "/ 600 comparisons")
