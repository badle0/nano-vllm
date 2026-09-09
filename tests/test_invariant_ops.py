import math

import pytest
import torch

# Skip before importing CUDA-only dependencies: a pytest marker is evaluated
# after collection and cannot protect a CPU-only environment from these imports.
if not torch.cuda.is_available():
    pytest.skip("invariant kernels require CUDA", allow_module_level=True)

from nanovllm.layers.attention import Attention
from nanovllm.layers.invariant_ops import invariant_linear, invariant_rms_norm
from nanovllm.utils.context import reset_context, set_context



def test_invariant_linear_is_bitwise_independent_of_live_row_geometry():
    torch.manual_seed(20260908)
    x = torch.randn(19, 96, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(130, 96, device="cuda", dtype=torch.bfloat16)

    batched = invariant_linear(x, weight)
    isolated = torch.cat(
        [invariant_linear(x[index:index + 1], weight) for index in range(19)]
    )

    assert torch.equal(batched, isolated)


def test_invariant_rmsnorm_is_bitwise_independent_of_live_row_geometry():
    torch.manual_seed(20260908)
    x = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)

    batched, batched_residual = invariant_rms_norm(
        x, weight, 1e-6, residual
    )
    isolated_pairs = [
        invariant_rms_norm(
            x[index:index + 1], weight, 1e-6, residual[index:index + 1]
        )
        for index in range(19)
    ]
    isolated = torch.cat([pair[0] for pair in isolated_pairs])
    isolated_residual = torch.cat([pair[1] for pair in isolated_pairs])

    assert torch.equal(batched, isolated)
    assert torch.equal(batched_residual, isolated_residual)



@pytest.mark.parametrize(
    "rows,input_width,output_width",
    [(1, 96, 130), (7, 1024, 1536)],
)
def test_invariant_linear_tracks_independent_fp64_reference(
    rows,
    input_width,
    output_width,
):
    torch.manual_seed(20260908 + rows)
    x = torch.randn(rows, input_width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        output_width, input_width, device="cuda", dtype=torch.bfloat16
    )

    actual = invariant_linear(x, weight).float()
    reference = (x.double() @ weight.double().T).float()
    relative_peak = (
        (actual - reference).abs().max()
        / reference.abs().max().clamp_min(1e-12)
    )

    assert relative_peak.item() < 0.005


@pytest.mark.parametrize("rows,width", [(1, 128), (7, 1024), (2, 2048)])
def test_invariant_rmsnorm_tracks_independent_fp64_reference(rows, width):
    torch.manual_seed(20261008 + rows)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)

    actual, actual_residual = invariant_rms_norm(
        x, weight, 1e-6, residual
    )
    values = x.double() + residual.double()
    reference = (
        values
        * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)
        * weight.double()
    )
    relative_peak = (
        (actual.float() - reference.float()).abs().max()
        / reference.abs().max().clamp_min(1e-12)
    )

    assert relative_peak.item() < 0.01
    torch.testing.assert_close(
        actual_residual.float(), values.float(), rtol=0.0, atol=0.016
    )


def test_invariant_paged_attention_tracks_fp64_through_context_4096():
    torch.manual_seed(20261108)
    block_size, num_blocks = 256, 32
    num_kv_heads, num_heads, head_dim = 2, 4, 32
    attention = Attention(
        num_heads, head_dim, head_dim**-0.5, num_kv_heads
    ).cuda()
    attention.k_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    attention.v_cache = torch.randn_like(attention.k_cache)
    tables = torch.stack((torch.arange(16), torch.arange(16, 32))).to(
        device="cuda", dtype=torch.int32
    )
    lengths = (1, 255, 256, 257, 4096)
    owners = (0, 0, 0, 1, 1)
    queries = torch.randn(
        len(lengths),
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    set_context(
        True,
        block_tables=tables,
        query_sequence_ids=owners,
        query_context_lengths=lengths,
    )
    try:
        actual = attention._invariant_paged_attention(queries)
        for index, (owner, length) in enumerate(
            zip(owners, lengths, strict=True)
        ):
            ids = tables[owner, :math.ceil(length / block_size)].long()
            keys = attention.k_cache.index_select(0, ids).flatten(0, 1)[:length]
            values = attention.v_cache.index_select(0, ids).flatten(0, 1)[:length]
            groups = num_heads // num_kv_heads
            keys = keys[:, :, None, :].expand(-1, -1, groups, -1).reshape(
                length, num_heads, head_dim
            )
            values = values[:, :, None, :].expand(-1, -1, groups, -1).reshape(
                length, num_heads, head_dim
            )
            scores = (
                queries[index].double()[:, None, :]
                * keys.double().permute(1, 0, 2)
            ).sum(dim=-1) * attention.scale
            probabilities = torch.softmax(scores, dim=-1)
            reference = (
                probabilities[:, :, None]
                * values.double().permute(1, 0, 2)
            ).sum(dim=1)
            torch.testing.assert_close(
                actual[index].float(),
                reference.float(),
                rtol=0.01,
                atol=0.001,
            )
    finally:
        reset_context()
