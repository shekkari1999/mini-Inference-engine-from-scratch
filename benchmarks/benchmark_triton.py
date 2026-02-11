import torch
import math
import time
from attention import TritonAttention

device = "cuda"


def benchmark(T=1024, D=64, warmup=10, runs=50):
    print(f"\nSequence Length: {T}, Head Dim: {D}")

    q = torch.randn(T, D, device=device)
    k = torch.randn(T, D, device=device)
    v = torch.randn(T, D, device=device)

    scale = 1.0 / math.sqrt(D)

    # --------------------------------------------
    # Correctness check
    # --------------------------------------------
    ref = torch.softmax(q @ k.T * scale, dim=-1) @ v

    triton_attn = TritonAttention()
    out = triton_attn.forward(q, k, v)

    diff = (ref - out).abs().max()
    print("Max difference:", diff.item())

    # --------------------------------------------
    # PyTorch timing
    # --------------------------------------------
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = torch.softmax(q @ k.T * scale, dim=-1) @ v
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        _ = torch.softmax(q @ k.T * scale, dim=-1) @ v
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / runs * 1000
    print(f"PyTorch attention: {torch_time:.3f} ms")

    # --------------------------------------------
    # Triton timing
    # --------------------------------------------
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = triton_attn.forward(q, k, v)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        _ = triton_attn.forward(q, k, v)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / runs * 1000
    print(f"Triton fused attention: {triton_time:.3f} ms")

    print(f"Speedup: {torch_time / triton_time:.2f}x")


if __name__ == "__main__":
    print("Triton vs PyTorch Attention Benchmark")
    print("=" * 60)

    for T in [2048, 16000]:
        benchmark(T=T, D=64)