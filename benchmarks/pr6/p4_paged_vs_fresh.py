# benchmarks/pr6/p4_paged_vs_fresh.py
# HISTORICAL (pre-C2 era): the line-14 assert pins the fresh-branch premise this
# probe measured (block_tables None unless prefix-cached). C2's F3b unification
# made real steps always-paged, so on post-C2 trees the assert fires BY DESIGN —
# run this probe at its evidence-era checkout (<= 9f7bffe). Not a defect.
import os, random, torch
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context

llm = LLM(
    os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False,
    max_model_len=4096, max_num_batched_tokens=512,
)
random.seed(0)
llm.add_request([random.randint(1000, 150000) for _ in range(3968)],
                SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True))
seqs, _ = llm.scheduler.schedule()
ids, pos = llm.model_runner.prepare_ragged(seqs)
ctx = get_context()
assert ctx.block_tables is None                     # today's fresh branch (k == q)
with torch.inference_mode():
    fresh = llm.model_runner.model(ids, pos).clone()      # side effect: chunk KV now stored
    torch.cuda.synchronize()
    bt = llm.model_runner.prepare_block_tables(seqs)
    set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
                ctx.max_seqlen_k, ctx.slot_mapping, None, bt)
    paged = llm.model_runner.model(ids, pos)              # re-stores same values; paged read
    print("P4 paged==fresh bitwise:", torch.equal(paged, fresh),
          "| allclose:", torch.allclose(paged.float(), fresh.float(), atol=1e-3))
    d = (paged - fresh).float().abs()
    print("P4c-model max", d.max().item(), "mean", d.mean().item(),
      "exact-equal fraction", (paged == fresh).float().mean().item())
    rows = d.max(dim=-1).values                      # per-token worst drift
    print("P4d fresh |max|", fresh.abs().max().item(),
      "| row-max min/median/max", rows.min().item(), rows.median().item(), rows.max().item(),
      "| worst row", rows.argmax().item(), "of", rows.numel())
reset_context(); llm.scheduler.cancel_all()
