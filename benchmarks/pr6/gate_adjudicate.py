# benchmarks/pr6/gate_adjudicate.py — fp64 tie adjudication for greedy token-gate
# divergences (axis-2 discipline: cross-composition claims are judged at token level,
# ties adjudicated in fp64). For each divergence between BASE.json and CAND.json,
# rebuilds the shared prefix, computes last-position logits through BOTH prefill
# numerics on this tree — fresh (C1-equivalent kernel template) and paged (C2's
# always-paged template) — and reports the fp64 gap between the two candidate
# tokens on each side. Tie-class: pair is top-2 on both sides, |gap| at bf16-ULP
# scale (~value/256 of logit magnitude).
# usage: gate_adjudicate.py BASE.json CAND.json
import json, sys
if len(sys.argv) != 3:
    sys.exit("usage: gate_adjudicate.py BASE.json CAND.json")
base = json.load(open(sys.argv[1])); cand = json.load(open(sys.argv[2]))

import torch
from probe_common import make_llm
from nanovllm import SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context

PROMPTS = ["The history of the Roman Empire begins with",
           "In machine learning, gradient descent works by",
           "The recipe calls for two cups of flour and",
           "Photosynthesis converts sunlight into"]     # must match gate_generate.py

llm = make_llm()
mr = llm.model_runner
for i, (b, c) in enumerate(zip(base, cand)):
    if b == c:
        print(f"prompt {i}: match ({len(b)} tokens) — nothing to adjudicate")
        continue
    d = next(j for j in range(min(len(b), len(c))) if b[j] != c[j])
    prefix = llm.tokenizer.encode(PROMPTS[i]) + b[:d]
    llm.add_request(prefix, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    seqs, _ = llm.scheduler.schedule()
    ids, pos = mr.prepare_ragged(seqs)
    ctx = get_context()
    with torch.inference_mode():
        paged = mr.model.compute_logits(mr.model(ids, pos))[-1].double()
        set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
                    ctx.max_seqlen_k, ctx.slot_mapping, None, None)     # fresh branch
        fresh = mr.model.compute_logits(mr.model(ids, pos))[-1].double()
    reset_context(); llm.scheduler.cancel_all()
    tb, tc = b[d], c[d]
    for name, lg in (("fresh(C1-side)", fresh), ("paged(C2-side)", paged)):
        top2 = torch.topk(lg, 2)
        gap = (lg[tb] - lg[tc]).item()
        print(f"prompt {i} tok {d} [{name}]: gap(base {tb} - cand {tc}) = {gap:+.6f} | "
              f"top2 {top2.indices.tolist()} gaps {(top2.values[0]-top2.values[1]).item():.6f} | "
              f"pair-is-top2 {sorted(top2.indices.tolist()) == sorted([tb, tc])}")
