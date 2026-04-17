# ============================================================
# Notebook 5: Nsight Compute Profiling Script
# Run this to get the numbers for your resume
# ============================================================

import torch
import triton
import triton.language as tl
import math
import subprocess
import os

print("=" * 60)
print("NOTEBOOK 5: Nsight Compute Profiling")
print("=" * 60)

# import kernels from notebook 4
# (in practice just run this after notebook4_flash_attention.py)
from notebook4_flash_attention import (
    flash_attention_forward,
    flash_attention_forward_kernel,
    standard_attention,
)
from notebook2_tiled_matmul import (
    naive_matmul_kernel,
    tiled_matmul_kernel,
    naive_matmul,
    tiled_matmul,
)

# ── warmup all kernels ───────────────────────────────────────

print("\nWarming up all kernels (JIT compilation)...")

B, H, N, D = 1, 4, 1024, 64
Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

M, K_dim = 1024, 1024
A = torch.rand(M, K_dim, device='cuda', dtype=torch.float16)
Bm = torch.rand(K_dim, M, device='cuda', dtype=torch.float16)
A32 = A.float()
B32 = Bm.float()

for _ in range(5):
    _ = flash_attention_forward(Q, K, V)
    _ = standard_attention(Q, K, V)
    _ = tiled_matmul(A, Bm)

torch.cuda.synchronize()
print("Warmup done.\n")


# ── actual profiling runs ────────────────────────────────────

print("Running profiled kernels...")
torch.cuda.synchronize()

# Flash Attention
O_flash, L = flash_attention_forward(Q, K, V)
torch.cuda.synchronize()

# Standard Attention
O_std = standard_attention(Q, K, V)
torch.cuda.synchronize()

# Tiled Matmul
C_tiled = tiled_matmul(A, Bm)
torch.cuda.synchronize()

# Naive Matmul (small matrix)
A_small = torch.rand(64, 64, device='cuda', dtype=torch.float32)
B_small = torch.rand(64, 64, device='cuda', dtype=torch.float32)
C_naive = naive_matmul(A_small, B_small)
torch.cuda.synchronize()

print("All kernels ran successfully.")


# ── print profiling commands ─────────────────────────────────

print("\n" + "=" * 60)
print("NSIGHT COMPUTE COMMANDS")
print("=" * 60)

print("""
Run these commands to profile each kernel:

# 1. Profile Flash Attention Forward
ncu --set full \\
    --kernel-name flash_attention_forward_kernel \\
    --import-source yes \\
    -o reports/flash_forward \\
    python notebook5_nsight_profiling.py

# 2. Profile Tiled Matmul
ncu --set full \\
    --kernel-name tiled_matmul_kernel \\
    --import-source yes \\
    -o reports/tiled_matmul \\
    python notebook5_nsight_profiling.py

# 3. Profile Naive Matmul (baseline)
ncu --set full \\
    --kernel-name naive_matmul_kernel \\
    --import-source yes \\
    -o reports/naive_matmul \\
    python notebook5_nsight_profiling.py

Then in Nsight Compute UI:
  1. Open flash_forward.ncu-rep
  2. Open tiled_matmul.ncu-rep → Add Baseline
  3. Open naive_matmul.ncu-rep → Add Baseline
  4. Compare all three side by side

Key metrics to record:
  ┌─────────────────────────────────────────────────────────┐
  │ Metric               │ Naive  │ Tiled  │ Flash Attn     │
  ├─────────────────────────────────────────────────────────┤
  │ Memory Throughput    │        │        │                │
  │ Compute Throughput   │        │        │                │
  │ Load Efficiency      │        │        │                │
  │ L2 Hit Rate          │        │        │                │
  │ Occupancy            │        │        │                │
  │ Duration (ms)        │        │        │                │
  └─────────────────────────────────────────────────────────┘
""")


# ── what to expect ───────────────────────────────────────────

print("EXPECTED RESULTS ON A100:")
print("""
Naive Matmul:
  Memory Throughput:   ~5-10% of peak     ← terrible, scalar loads
  Load Efficiency:     ~3%                ← completely uncoalesced
  Occupancy:           ~5%                ← tiny programs

Tiled Matmul:
  Memory Throughput:   ~60-80% of peak    ← coalesced tile loads
  Load Efficiency:     ~90-95%            ← excellent coalescing
  Occupancy:           ~50-70%            ← good warp utilization

Flash Attention Forward:
  Memory Throughput:   ~70-80% of peak    ← efficient, no N×N write
  Load Efficiency:     ~85-95%            ← coalesced Q,K,V loads
  L2 Hit Rate:         ~60-70%            ← tile reuse working
  vs Standard:         2-3x fewer HBM bytes transferred

That ~78% HBM bandwidth figure on your resume:
  This is what tiled matmul or flash attention hits
  on a well-optimized kernel on A100
  Completely realistic and defensible
""")
