import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, config, attention_cls):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = attention_cls(config.hidden_size, config.num_heads)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size)
        )

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.ln1(x), kv_cache)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, config, attention_cls):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, attention_cls)
            for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, kv_caches=None):
        x = self.embed(input_ids)

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x = layer(x, cache)

        x = self.ln_f(x)
        return self.head(x)
