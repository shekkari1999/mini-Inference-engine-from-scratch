# attention/triton_paged_attention.py

import torch
import math
from attention.triton_attention import TritonAttention

class TritonPagedAttention:
    """
    Combines:
    - Paged KV cache (memory layout)
    - Triton fused attention (compute)

    Removes Python page loop.
    """

    def __init__(self):
        self.triton_attn = TritonAttention()

    def forward(self, q, kv_cache):
        """
        q: (T, D)
        kv_cache: PagedKVCache
        """

        # Flatten pages into contiguous tensors
        k = torch.cat(kv_cache.k_pages, dim=2)
        v = torch.cat(kv_cache.v_pages, dim=2)

        # Remove unused tail
        k = k[:, :, :kv_cache.cur_pos, :]
        v = v[:, :, :kv_cache.cur_pos, :]

        # Flatten (B,H,T,D) -> (S,D)
        k = k[0, 0]
        v = v[0, 0]

        return self.triton_attn.forward(q, k, v)