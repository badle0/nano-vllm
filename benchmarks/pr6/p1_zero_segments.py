import torch
from flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
nq, nkv, dh = 16, 8, 128
q = torch.randn(12, nq, dh, device="cuda", dtype=torch.bfloat16)
k = torch.randn(12, nkv, dh, device="cuda", dtype=torch.bfloat16)
v = torch.randn(12, nkv, dh, device="cuda", dtype=torch.bfloat16)

def run(cu):
    c = torch.tensor(cu, device="cuda", dtype=torch.int32)
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q=c, cu_seqlens_k=c,
                                  max_seqlen_q=7, max_seqlen_k=7,
                                  softmax_scale=dh**-0.5, causal=True)

ref = run([0, 5, 12])                                  # two real segments: 5, 7
for name, cu in (("trailing", [0, 5, 12, 12, 12]), ("middle", [0, 5, 5, 12])):
    try:
        out = run(cu)
        print(f"P1 {name}-empty: OK | bitwise={torch.equal(out, ref)} "
              f"| allclose={torch.allclose(out.float(), ref.float(), atol=1e-3)}")
    except Exception as e:
        print(f"P1 {name}-empty: FAIL — {type(e).__name__}: {e}")