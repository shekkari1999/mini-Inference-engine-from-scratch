import torch
import time
import math

from attention.streaming_attention import PageStreamingAttention
from attention.triton_paged_attention import TritonPagedAttention
from kv_cache.paged_cache import PagedKVCache

device = "cuda"


def measure(fn, warmup=5, runs=20):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(runs):
        fn()
    end.record()

    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / runs
    peak = torch.cuda.max_memory_allocated() / 1e9

    return ms, peak


def run_test(seq_len=4096, head_dim=64):

    print("=" * 60)
    print(f"Sequence Length: {seq_len}, Head Dim: {head_dim}")

    B = 1
    H = 1
    D = head_dim

    # ---------------------------------------------------
    # Create random Q, K, V
    # ---------------------------------------------------

    q = torch.randn(seq_len, D, device=device)
    k = torch.randn(seq_len, D, device=device)
    v = torch.randn(seq_len, D, device=device)

    # ---------------------------------------------------
    # Build paged KV cache
    # ---------------------------------------------------

    kv_cache = PagedKVCache(
        batch_size=B,
        num_heads=H,
        head_dim=D,
        page_size=512,
        device=device
    )

    # Insert K/V into cache
    kv_cache.update(
        k.unsqueeze(0).unsqueeze(0),
        v.unsqueeze(0).unsqueeze(0)
    )
  # ---------------------------------------------------
    # 1️⃣ Python Streaming Attention
    # ---------------------------------------------------

    streaming_attn = PageStreamingAttention(D, H).to(device)

    def streaming():
        return streaming_attn.forward(
            q.unsqueeze(0),  # (B, T, D)
            kv_cache
        )

    t_stream, mem_stream = measure(streaming)

    print(f"Streaming (Python): {t_stream:.3f} ms | Peak mem: {mem_stream:.2f} GB")


    # ---------------------------------------------------
    # 2️⃣ Triton Paged Attention
    # ---------------------------------------------------

    triton_paged = TritonPagedAttention()

    def triton():
        return triton_paged.forward(q, kv_cache)

    t_triton, mem_triton = measure(triton)

    print(f"Triton Paged:       {t_triton:.3f} ms | Peak mem: {mem_triton:.2f} GB")

    print("-" * 60)
    print(f"Speedup: {t_stream / t_triton:.2f}x")
  

if __name__ == "__main__":

    print("Streaming vs TritonPaged Attention Benchmark")
    print("=" * 60)

    for T in [2048, 4096, 8192, 16000]:
        run_test(seq_len=T, head_dim=64)