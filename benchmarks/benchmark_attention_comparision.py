import torch
import time
import math

from attention.naive_attention import NaiveAttention
from attention.streaming_attention import PageStreamingAttention
from attention.triton_attention import TritonAttention
from kv_cache.paged_cache import PagedKVCache

device = "cuda"

def measure(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.time()
    fn()
    torch.cuda.synchronize()
    end = time.time()

    peak = torch.cuda.max_memory_allocated() / 1e9
    return end - start, peak

def run_test(seq_len=4096, head_dim=64):

    print("="*60)
    print(f"Sequence Length: {seq_len}, Head Dim: {head_dim}")

    B = 1
    H = 1
    D = head_dim

    q = torch.randn(B, H, seq_len, D, device=device)
    k = torch.randn(B, H, seq_len, D, device=device)
    v = torch.randn(B, H, seq_len, D, device=device)

    scale = 1.0 / math.sqrt(D)

    # ---------------------------
    # 1. Naive
    # ---------------------------
    def naive():
        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * scale,
            dim=-1
        )
        out = torch.matmul(attn, v)
        return out

    t_naive, mem_naive = measure(naive)
    print(f"Naive:    {t_naive*1000:.3f} ms | Peak mem: {mem_naive:.2f} GB")

    # ---------------------------
    # 2. Streaming (Python paged)
    # ---------------------------
    kv_cache = PagedKVCache(
        batch_size=B,
        num_heads=H,
        head_dim=D,
        page_size=512,
        device=device
    )

    streaming_attn = PageStreamingAttention(D*H, H).to(device)

    def streaming():
        kv_cache.cur_pos = 0
        kv_cache.k_pages = []
        kv_cache.v_pages = []
        return streaming_attn.forward(
            q.transpose(1,2).reshape(B, seq_len, D*H),
            kv_cache
        )

    t_stream, mem_stream = measure(streaming)
    print(f"Streaming:{t_stream*1000:.3f} ms | Peak mem: {mem_stream:.2f} GB")

    # ---------------------------
    # 3. Triton
    # ---------------------------
    triton_attn = TritonAttention()

    def triton():
        return triton_attn.forward(q[0, 0], k[0, 0], v[0, 0])

    t_triton, mem_triton = measure(triton)
    print(f"Triton:   {t_triton*1000:.3f} ms | Peak mem: {mem_triton:.2f} GB")

    print("-"*60)
    print(f"Speedup (Triton vs Streaming): {t_stream / t_triton:.2f}x")

if __name__ == "__main__":
    for T in [2048, 4096, 8192, 16000]:
        run_test(T)