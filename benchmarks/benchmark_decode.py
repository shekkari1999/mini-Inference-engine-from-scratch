"""
Decode Benchmark

Measures:
- Tokens per second
- Throughput degradation with context length

Important:
Decode cost scales linearly with context length.
"""

import torch

from config import Config
from model import MiniTransformer

from attention import PageStreamingAttention
from kv_cache import PagedKVCache

from utils.benchmark import benchmark_decode


# -------------------------------------------------
# Setup
# -------------------------------------------------

device = "cuda"
torch.set_grad_enabled(False)

config = Config()

model = MiniTransformer(config, PageStreamingAttention).to(device)
model.eval()

print("Streaming Attention Decode Benchmark")
print("=" * 60)


# -------------------------------------------------
# Scaling Loop
# -------------------------------------------------

for prefill_len in [512, 1024, 2048, 4096]:
    print(f"\nContext Length: {prefill_len}")

    benchmark_decode(
        model=model,
        config=config,
        device=device,
        prefill_len=prefill_len,
        steps=256,
        kv_cache_cls=PagedKVCache
    )

    print("-" * 60)