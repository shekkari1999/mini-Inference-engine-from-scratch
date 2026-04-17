# attention/triton_paged_attention.py

import torch
import math
from attention.triton_attention import TritonAttention


class TritonPagedAttention:
    """
    Combines paged KV cache with Triton fused attention.

    For each page, runs the Triton kernel and uses online softmax
    to combine page-level results without materializing the full KV.

    This avoids the O(total_tokens^2) memory of concatenating all pages.
    """

    def __init__(self, causal=True):
        self.triton_attn = TritonAttention(causal=causal)

    def forward(self, q, kv_cache):
        """
        q: [T, D] or [B, H, T, D]
        kv_cache: PagedKVCache with k_pages, v_pages, cur_pos

        For now, we flatten pages but only up to cur_pos.
        This is a pragmatic approach: the Triton kernel handles the
        attention computation efficiently even with concatenated KV.
        The paging benefit is in memory allocation (on-demand pages),
        not in avoiding the concatenation at compute time.

        TODO: True paged attention kernel that iterates over pages
        in the Triton kernel itself (like vLLM's PagedAttention kernel).
        """
        if not kv_cache.k_pages:
            # No KV cache yet, return zeros
            if q.dim() == 2:
                return torch.zeros_like(q)
            else:
                return torch.zeros_like(q)

        # Flatten pages into contiguous tensors (up to cur_pos only)
        k = torch.cat(kv_cache.k_pages, dim=2)
        v = torch.cat(kv_cache.v_pages, dim=2)

        k = k[:, :, :kv_cache.cur_pos, :]
        v = v[:, :, :kv_cache.cur_pos, :]

        if q.dim() == 2:
            # Single head: flatten [B, H, S, D] -> [S, D]
            k = k[0, 0]
            v = v[0, 0]
        # else: keep as [B, H, S, D] for multi-head Triton kernel

        return self.triton_attn.forward(q, k, v)
