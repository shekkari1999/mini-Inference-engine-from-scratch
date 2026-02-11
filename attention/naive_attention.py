### Naive Attention with KV cache
import torch
import torch.nn as nn
import math

class NaiveAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj   =   nn.Linear(hidden_size, hidden_size)
        self.k_proj   =   nn.Linear(hidden_size, hidden_size)
        self.v_proj   =   nn.Linear(hidden_size, hidden_size)
        self.out_proj =   nn.Linear(hidden_size, hidden_size)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        ## this is where we update our KV cache
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask (lower triangular)
        causal_mask = torch.tril(
            torch.ones(T, T, device=q.device)
        ).unsqueeze(0).unsqueeze(0)
        
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)