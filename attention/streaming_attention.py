import torch
import torch.nn as nn
import math

"""
Streaming attention implementation using page-based KV storage.

- Avoids materializing full T x T attention matrix.
- Uses online softmax accumulation across pages.
- Compute complexity remains O(T^2) for prefill.
- Decode cost is O(T) per step.

This is algorithmically similar to FlashAttention,
but not kernel-fused or IO-optimized.
"""
# NOTE:
# This loop introduces multiple kernel launches (one per page).
# For production systems, a fused kernel (e.g., Triton) would eliminate this overhead.

class PageStreamingAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        scale = 1.0 / math.sqrt(self.head_dim)

        # Fallback to naive full attention
        if kv_cache is None:
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1,2).contiguous().view(B, T, C)
            return self.out_proj(out)

        # Update cache
        kv_cache.update(k, v)

        output = torch.zeros_like(q)

        # Online softmax state
        m_i = torch.full((B, self.num_heads, T),
                         -float("inf"), device=q.device)
        l_i = torch.zeros((B, self.num_heads, T),
                          device=q.device)

        for page_idx in range(len(kv_cache.k_pages)):
            k_page = kv_cache.k_pages[page_idx]
            v_page = kv_cache.v_pages[page_idx]

            if page_idx == len(kv_cache.k_pages) - 1:
                valid = kv_cache.cur_pos % kv_cache.page_size
                if valid != 0:
                    k_page = k_page[:, :, :valid, :]
                    v_page = v_page[:, :, :valid, :]

            scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale

            block_max = scores.max(dim=-1).values
            new_m = torch.maximum(m_i, block_max)

            exp_scores = torch.exp(scores - new_m.unsqueeze(-1))
            exp_m_diff = torch.exp(m_i - new_m)

            l_i = exp_m_diff * l_i + exp_scores.sum(dim=-1)
            output = exp_m_diff.unsqueeze(-1) * output + torch.matmul(exp_scores, v_page)

            m_i = new_m

        output = output / l_i.unsqueeze(-1)
        output = output.transpose(1,2).contiguous().view(B, T, C)

        return self.out_proj(output)