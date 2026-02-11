"""
Prefill Benchmark

Measures:
- End-to-end forward latency
- GPU memory usage
- Scaling with sequence length

Compares:
- Naive attention
- Streaming attention
- Different KV cache strategies
"""

import torch

from config import Config
from model import MiniTransformer

from attention import PageStreamingAttention
from kv_cache import PagedKVCache

from utils.memory import print_gpu_memory
from utils.benchmark import benchmark_prefill


# -------------------------------------------------
# Setup
# -------------------------------------------------

device = "cuda"
torch.set_grad_enabled(False)

config = Config()

model = MiniTransformer(config, PageStreamingAttention).to(device)
model.eval()

print("Streaming Attention Prefill Benchmark")
print("=" * 60)


# -------------------------------------------------
# Scaling Loop
# -------------------------------------------------

for seq_len in [4096, 8192, 16384]:
    print(f"\nSequence Length: {seq_len}")
    print_gpu_memory()

    benchmark_prefill(
        model=model,
        config=config,
        device=device,
        seq_len=seq_len,
        kv_cache_cls=PagedKVCache
    )

    print_gpu_memory()
    print("-" * 60)