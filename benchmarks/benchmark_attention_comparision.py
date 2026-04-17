"""
Attention Benchmark: Naive vs Streaming vs Triton (with causal masking)

Measures latency and peak memory at various sequence lengths.
Saves results to results/ directory as JSON + plots.

Usage:
    python benchmarks/benchmark_attention_comparision.py
"""

import torch
import time
import math
import json
import os
from datetime import datetime

from attention.naive_attention import NaiveAttention
from attention.streaming_attention import PageStreamingAttention
from attention.triton_attention import TritonAttention
from kv_cache.paged_cache import PagedKVCache

device = "cuda"
RESULTS_DIR = "results"


def measure(fn, warmup=3, repeats=10):
    """Measure latency and peak memory with warmup and repeated runs."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # timed runs
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / repeats
    peak = torch.cuda.max_memory_allocated() / 1e9
    return avg_time, peak


def run_test(seq_len=4096, head_dim=64, num_heads=12, batch_size=1, causal=True):
    """Run benchmark for all attention implementations at given config."""
    B = batch_size
    H = num_heads
    D = head_dim
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float32)

    results = {"seq_len": seq_len, "head_dim": D, "num_heads": H, "batch_size": B, "causal": causal}

    print("=" * 70)
    print(f"Seq={seq_len}, Heads={H}, HeadDim={D}, Batch={B}, Causal={causal}")
    print("=" * 70)

    # 1. Naive (PyTorch)
    def naive():
        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = attn.masked_fill(mask == 0, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
        else:
            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        return torch.matmul(attn, v)

    t_naive, mem_naive = measure(naive)
    results["naive"] = {"latency_ms": t_naive * 1000, "peak_mem_gb": mem_naive}
    print(f"  Naive:     {t_naive*1000:8.3f} ms | Peak mem: {mem_naive:.3f} GB")

    # 2. Streaming (Python paged loop)
    kv_cache = PagedKVCache(
        batch_size=B, num_heads=H, head_dim=D, page_size=512, device=device
    )
    streaming_attn = PageStreamingAttention(D * H, H).to(device)

    def streaming():
        kv_cache.cur_pos = 0
        kv_cache.k_pages = []
        kv_cache.v_pages = []
        return streaming_attn.forward(
            q.transpose(1, 2).reshape(B, seq_len, D * H), kv_cache
        )

    t_stream, mem_stream = measure(streaming)
    results["streaming"] = {"latency_ms": t_stream * 1000, "peak_mem_gb": mem_stream}
    print(f"  Streaming: {t_stream*1000:8.3f} ms | Peak mem: {mem_stream:.3f} GB")

    # 3. Triton fused (multi-head)
    triton_attn = TritonAttention(causal=causal)

    def triton_mh():
        return triton_attn.forward(q, k, v)

    t_triton, mem_triton = measure(triton_mh)
    results["triton"] = {"latency_ms": t_triton * 1000, "peak_mem_gb": mem_triton}
    print(f"  Triton:    {t_triton*1000:8.3f} ms | Peak mem: {mem_triton:.3f} GB")

    # Speedups
    speedup_vs_naive = t_naive / t_triton
    speedup_vs_streaming = t_stream / t_triton
    mem_saving_vs_naive = (1 - mem_triton / mem_naive) * 100 if mem_naive > 0 else 0

    results["speedup_triton_vs_naive"] = speedup_vs_naive
    results["speedup_triton_vs_streaming"] = speedup_vs_streaming
    results["mem_saving_pct_vs_naive"] = mem_saving_vs_naive

    print(f"  ---")
    print(f"  Speedup (Triton vs Naive):     {speedup_vs_naive:.2f}x")
    print(f"  Speedup (Triton vs Streaming): {speedup_vs_streaming:.2f}x")
    print(f"  Memory saving vs Naive:        {mem_saving_vs_naive:.1f}%")
    print()

    return results


def run_kv_cache_comparison(seq_lengths, head_dim=64, num_heads=12):
    """Compare contiguous vs paged KV cache memory usage."""
    from kv_cache.contiguous_cache import ContiguousKVCache
    from kv_cache.paged_cache import PagedKVCache

    results = []
    print("=" * 70)
    print("KV Cache Memory Comparison: Contiguous vs Paged")
    print("=" * 70)

    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Contiguous
        cache_c = ContiguousKVCache(1, num_heads, seq_len, head_dim, device)
        mem_contiguous = torch.cuda.max_memory_allocated() / 1e9

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Paged
        cache_p = PagedKVCache(1, num_heads, head_dim, page_size=256, device=device)
        # Simulate filling to seq_len
        for t in range(seq_len):
            k_token = torch.randn(1, num_heads, 1, head_dim, device=device)
            v_token = torch.randn(1, num_heads, 1, head_dim, device=device)
            cache_p.append(k_token, v_token)
        mem_paged = torch.cuda.max_memory_allocated() / 1e9

        saving = (1 - mem_paged / mem_contiguous) * 100 if mem_contiguous > 0 else 0

        results.append({
            "seq_len": seq_len,
            "contiguous_gb": mem_contiguous,
            "paged_gb": mem_paged,
            "saving_pct": saving,
        })

        print(f"  Seq={seq_len:6d}: Contiguous={mem_contiguous:.4f} GB | Paged={mem_paged:.4f} GB | Saving={saving:.1f}%")

        del cache_c, cache_p

    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Attention benchmarks
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    all_results = []

    for T in seq_lengths:
        try:
            result = run_test(T, head_dim=64, num_heads=12, causal=True)
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED at seq_len={T}: {e}")

    # KV cache comparison
    try:
        kv_results = run_kv_cache_comparison([256, 512, 1024, 2048, 4096])
    except Exception as e:
        print(f"  KV cache comparison failed: {e}")
        kv_results = []

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "attention_benchmarks": all_results,
        "kv_cache_benchmarks": kv_results,
    }

    results_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Also save as latest
    latest_path = os.path.join(RESULTS_DIR, "benchmark_latest.json")
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Latest results at {latest_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Seq Len':>8} | {'Naive (ms)':>12} | {'Stream (ms)':>12} | {'Triton (ms)':>12} | {'Speedup':>8}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['seq_len']:>8} | "
            f"{r['naive']['latency_ms']:>12.2f} | "
            f"{r['streaming']['latency_ms']:>12.2f} | "
            f"{r['triton']['latency_ms']:>12.2f} | "
            f"{r['speedup_triton_vs_naive']:>7.2f}x"
        )
