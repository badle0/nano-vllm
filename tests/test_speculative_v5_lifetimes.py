"""Independent declarative tensor-owner ledger, not production-plan arithmetic."""
import pytest
import torch
from nanovllm.engine.speculative_memory import plan_speculative_workspace


def ledger(batch, k, vocab, draft_bytes, target_bytes):
    # Named independent owners with byte counts. An alias is represented by one
    # owner; mutually exclusive filtering families get separate phase sets.
    b, r = batch, batch * (k + 1)
    owners = {
        "q": 4 * b * k * vocab, "p": 4 * r * vocab,
        "metadata": b * k * 78 + (b + r) * 24 + b * 35,
        "race": b * vocab * 28, "residual": b * vocab * 68,
    }
    phases = {}
    for prefix, rows, itemsize in (("draft", b, draft_bytes), ("verify", r, target_bytes)):
        owners.update({
            prefix + "_logits": rows * vocab * itemsize,
            prefix + "_clone": rows * vocab * itemsize,
            prefix + "_scaled": rows * vocab * 4,
            prefix + "_topk": rows * vocab * (2 * itemsize + 8) + rows * itemsize,
            prefix + "_topp": rows * vocab * itemsize + min(rows, 64) * vocab * 38,
        })
        base = {"q", "metadata", prefix + "_logits", prefix + "_clone"}
        phases[prefix + "_topk"] = base | {prefix + "_topk"}
        phases[prefix + "_topp"] = base | {prefix + "_topp"}
        phases[prefix + "_softmax"] = base | {prefix + "_scaled"} | ({"p"} if prefix == "verify" else set())
    phases["draft_race"] = {"q", "metadata", "draft_logits", "race"}
    phases["accept"] = {"q", "p", "metadata", "residual"}
    phases["bonus"] = {"q", "p", "metadata", "race"}
    return owners, phases


@pytest.mark.parametrize("batch", [1, 3, 4, 17])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("vocab", [7, 128, 151936])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_lifetimes_and_alias_mutations(batch, k, vocab, dtype):
    itemsize = torch.empty((), dtype=dtype).element_size()
    owners, phases = ledger(batch, k, vocab, 2, itemsize)
    totals = {name: sum(owners[owner] for owner in live) for name, live in phases.items()}
    production = plan_speculative_workspace(vocab_size=vocab, configured_k=k,
                                            max_num_seqs=batch, max_num_batched_tokens=4096,
                                            max_model_len=512, target_logits_dtype=dtype,
                                            draft_logits_dtype=torch.float16)
    assert production.modeled_live_peak_bytes == max(totals.values())
    assert production.draft_phase_bytes == max(v for n, v in totals.items() if n.startswith("draft"))
    assert production.verifier_phase_bytes == max(v for n, v in totals.items() if n.startswith("verify"))
    assert production.rejection_phase_bytes == totals["accept"]
    assert production.bonus_phase_bytes == totals["bonus"]
    # Each omitted owner or accidental duplicate changes its phase certificate,
    # even when a different unchanged phase still dominates the global max.
    for phase, live in phases.items():
        for owner in live:
            assert totals[phase] - owners[owner] != totals[phase]
            assert totals[phase] + owners[owner] != totals[phase]
    # Sequential greedy lane retains preallocated p, but filters only B rows;
    # it must fit the already-reserved all-query envelope for every K >= 1.
    row_private = max(batch * vocab * (2 * itemsize + 8) + batch * itemsize,
                      batch * vocab * itemsize + min(batch, 64) * vocab * 38)
    sequential_peak = (owners["q"] + owners["p"] + owners["metadata"]
                       + 2 * batch * vocab * itemsize + max(row_private, 4 * batch * vocab))
    assert sequential_peak <= production.modeled_live_peak_bytes
