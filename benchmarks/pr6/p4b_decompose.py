# benchmarks/pr6/p4b_decompose.py — locate the P4 divergence: kernel, plumbing, or pipeline
import os, random, torch
from flash_attn import flash_attn_varlen_func
from nanovllm import LLM, SamplingParams

# ---- P4b-2 first: pure kernel A/B, no engine, no nano plumbing ----
torch.manual_seed(0)
nq, nkv, dh, L, B = 16, 8, 128, 512, 256
q = torch.randn(L, nq, dh, device="cuda", dtype=torch.bfloat16)
k = torch.randn(L, nkv, dh, device="cuda", dtype=torch.bfloat16)
v = torch.randn(L, nkv, dh, device="cuda", dtype=torch.bfloat16)
cu = torch.tensor([0, L], device="cuda", dtype=torch.int32)
args = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=L, max_seqlen_k=L,
            softmax_scale=dh**-0.5, causal=True)
ref = flash_attn_varlen_func(q, k, v, **args)                       # contiguous

pk = torch.full((4, B, nkv, dh), 7.0, device="cuda", dtype=torch.bfloat16)  # loud garbage
pv = pk.clone()
pk[0], pk[1] = k[:B], k[B:]                                          # real data in pages 0,1
pv[0], pv[1] = v[:B], v[B:]
bt = torch.tensor([[0, 1, 2, 3]], device="cuda", dtype=torch.int32)  # extra pages listed on purpose
paged = flash_attn_varlen_func(q, pk, pv, block_table=bt, **args)
print("P4b-2a paged(kernel)==contig: bitwise", torch.equal(paged, ref),
      "| allclose", torch.allclose(paged.float(), ref.float(), atol=1e-3))
d = (paged - ref).float().abs()
print("P4c-kernel max", d.max().item(), "mean", d.mean().item(),
      "exact-equal fraction", (paged == ref).float().mean().item())
pk[2:] = -7.0; pv[2:] = -7.0                                         # change ONLY the garbage
paged2 = flash_attn_varlen_func(q, pk, pv, block_table=bt, **args)
print("P4b-2b insensitive to unused pages:", torch.equal(paged2, paged))
pk2 = torch.zeros_like(pk); pv2 = torch.zeros_like(pv)
pk2[2], pk2[3] = k[:B], k[B:]; pv2[2], pv2[3] = v[:B], v[B:]         # same data, pages 2,3
bt2 = torch.tensor([[2, 3, 0, 1]], device="cuda", dtype=torch.int32)
paged3 = flash_attn_varlen_func(q, pk2, pv2, block_table=bt2, **args)
print("P4b-2c block_table honored (pages 2,3):", torch.equal(paged3, paged))

# ---- P4b-1 + P4b-3: engine-level, one process ----
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False, max_model_len=4096)
random.seed(0)
prompt = [random.randint(1000, 150000) for _ in range(1500)]
gsp = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)   # greedy on dev
mono = llm.generate([prompt], gsp, use_tqdm=False)[0]["token_ids"]      # budget 16384: one prefill step
llm.scheduler.max_num_batched_tokens = 512
chunked = llm.generate([prompt], gsp, use_tqdm=False)[0]["token_ids"]   # chunks 512/512/476: paged path live
match = sum(a == b for a, b in zip(mono, chunked))
print(f"P4b-3 production chunked vs monolithic (greedy): {match}/16 tokens match")
print("       mono   :", mono)
print("       chunked:", chunked)