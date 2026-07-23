import torch
from transformers import TopKLogitsWarper

torch.manual_seed(0)
mismatches = 0
for trial in range(200):
    logits = torch.randn(4, 1000, dtype=torch.bfloat16).float()
    logits[0, 100:105] = logits[0, 200]              # forced ties in row 0
    for k in (1, 5, 50):
        kth = logits.sort(-1, descending=True).values[:, k-1:k]
        mine = logits >= kth
        hf = TopKLogitsWarper(top_k=k)(None, logits.clone()) > float("-inf")
        if not torch.equal(mine, hf):
            mismatches += 1
            if mismatches <= 3:
                d = (mine ^ hf).nonzero()
                print(f"trial {trial} k={k}: {len(d)} differing positions, e.g. {d[:3].tolist()}")
print("mismatches:", mismatches, "/ 600 comparisons")
