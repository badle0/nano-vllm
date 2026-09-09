import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.numerical_mode = "fast"
        self.k_cache = self.v_cache = torch.tensor([])

    def _invariant_paged_attention(self, q: torch.Tensor) -> torch.Tensor:
        context = get_context()
        owners = context.query_sequence_ids
        lengths = context.query_context_lengths
        if owners is None or lengths is None or len(owners) != q.size(0) or len(lengths) != q.size(0):
            raise RuntimeError("invariant attention requires complete query metadata")
        if context.block_tables is None:
            raise RuntimeError("invariant attention requires paged KV block tables")
        if self.num_heads % self.num_kv_heads:
            raise RuntimeError("invariant attention requires integral grouped-query heads")
        groups = self.num_heads // self.num_kv_heads
        outputs = []
        for query_index, (sequence_index, context_length) in enumerate(zip(owners, lengths)):
            if context_length < 1:
                raise RuntimeError("invariant attention context lengths must be positive")
            block_count = (context_length + self.k_cache.size(1) - 1) // self.k_cache.size(1)
            block_ids = context.block_tables[sequence_index, :block_count].to(torch.int64)
            keys = self.k_cache.index_select(0, block_ids).flatten(0, 1)[:context_length]
            values = self.v_cache.index_select(0, block_ids).flatten(0, 1)[:context_length]
            keys = keys[:, :, None, :].expand(-1, -1, groups, -1).reshape(
                context_length, self.num_heads, self.head_dim
            )
            values = values[:, :, None, :].expand(-1, -1, groups, -1).reshape(
                context_length, self.num_heads, self.head_dim
            )
            query = q[query_index].float()
            scores = (
                query[:, None, :] * keys.permute(1, 0, 2).float()
            ).sum(dim=-1) * self.scale
            probabilities = torch.softmax(scores, dim=-1)
            output = (
                probabilities[:, :, None] * values.permute(1, 0, 2).float()
            ).sum(dim=1)
            outputs.append(output.to(q.dtype))
        return torch.stack(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if self.numerical_mode == "invariant" and context.block_tables is not None:
            return self._invariant_paged_attention(q)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
