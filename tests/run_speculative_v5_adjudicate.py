"""Independent HF cold-prefix adjudication; reports numerical ties, not parity."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control")
    parser.add_argument("candidate")
    parser.add_argument("--model", default="/workspace/models/Qwen3-0.6B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    a, b = (json.loads(Path(path).read_text()) for path in (args.control, args.candidate))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                attn_implementation="eager").cuda().eval()
    results = []
    with torch.inference_mode():
        for x, y in zip(a["results"], b["results"], strict=True):
            assert (x["batch"], x["length"]) == (y["batch"], y["length"])
            for row, (base, spec) in enumerate(zip(x["tokens"], y["tokens"], strict=True)):
                first = next((i for i, pair in enumerate(zip(base, spec, strict=True)) if pair[0] != pair[1]), None)
                if first is None:
                    results.append(dict(length=x["length"], row=row, exact=True))
                    continue
                prefix = [42 + row] * (x["length"] - row) + base[:first]
                logits = model(torch.tensor([prefix], device="cuda"), use_cache=False).logits[0, -1].float()
                target_max = logits.max().item()
                u, v = base[first], spec[first]
                gap = abs(logits[u].item() - logits[v].item())
                # Registered absolute BF16 near-tie band, not a claim of exact ties.
                near_tie = max(target_max - logits[u].item(), target_max - logits[v].item()) <= 0.125
                results.append(dict(length=x["length"], row=row, exact=False, index=first,
                                    base=u, spec=v, hf_argmax=logits.argmax().item(),
                                    hf_base=logits[u].item(), hf_spec=logits[v].item(),
                                    hf_max=target_max, gap=gap, near_tie=near_tie))
    with Path(args.output).open("x") as handle:
        json.dump(dict(schema="spec-v5-hf-adjudication-v1", control=args.control,
                       candidate=args.candidate, absolute_band=0.125, results=results), handle, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    assert all(row.get("exact") or row["near_tie"] for row in results), "unexplained greedy divergence"


if __name__ == "__main__":
    main()
