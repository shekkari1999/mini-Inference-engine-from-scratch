# ============================================================
# Notebook 2: Tiled Matrix Multiplication in Triton
# Building block 2 — 2D tiling, tl.dot, strides, accumulator
# ============================================================

import torch
import triton
import triton.language as tl
import time

print("=" * 60)
print("NOTEBOOK 2: Tiled Matrix Multiplication in Triton")
print("=" * 60)

# ── Section 1: Naive scalar matmul (one element per program) ─

print("\n--- Section 1: Naive Scalar Matmul ---")
print("Each program computes ONE output element")
print("Scalar loads — no tiling — no Tensor Cores")

@triton.jit
def naive_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
):
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N
    accum = 0.0
    for k in range(0, K):
        a_val = tl.load(A_ptr + m * stride_am + k * stride_ak)
        b_val = tl.load(B_ptr + k * stride_bk + n * stride_bn)
        accum += a_val * b_val
    tl.store(C_ptr + m * stride_cm + n * stride_cn, accum)


def naive_matmul(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros((M, N), device=A.device, dtype=torch.float32)
    grid = (M * N,)
    naive_matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


# verify correctness
M, K, N = 64, 64, 64
A = torch.rand(M, K, device='cuda', dtype=torch.float32)
B = torch.rand(K, N, device='cuda', dtype=torch.float32)

C_naive = naive_matmul(A, B)
C_torch = A @ B
print(f"Naive correctness: {torch.allclose(C_naive, C_torch, atol=1e-3)}")
print(f"Max diff: {(C_naive - C_torch).abs().max().item():.2e}")


# ── Section 2: Tiled matmul (tile per program) ───────────────

print("\n--- Section 2: Tiled Matmul ---")
print("Each program computes one TILE of output")
print("Coalesced tile loads — tl.dot uses Tensor Cores")

@triton.jit
def tiled_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1. which tile am I?
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. my offsets
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 3. accumulator in registers — never touches HBM
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 4. loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # A tile pointers [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (
            row_offsets[:, None] * stride_am +
            k_offsets[None, :] * stride_ak
        )
        # B tile pointers [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (
            k_offsets[:, None] * stride_bk +
            col_offsets[None, :] * stride_bn
        )

        # boundary masks
        a_mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (col_offsets[None, :] < N)

        # load tiles into SRAM
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # tl.dot uses Tensor Cores automatically
        acc += tl.dot(a, b)

    # 5. store output tile — only HBM write
    c_ptrs = C_ptr + (
        row_offsets[:, None] * stride_cm +
        col_offsets[None, :] * stride_cn
    )
    c_mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def tiled_matmul(A, B, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros((M, N), device=A.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    tiled_matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


# verify correctness
A_fp16 = A.half()
B_fp16 = B.half()
C_tiled = tiled_matmul(A_fp16, B_fp16)
C_torch = (A_fp16 @ B_fp16).float()
print(f"Tiled correctness: {torch.allclose(C_tiled, C_torch, atol=1e-2)}")
print(f"Max diff: {(C_tiled - C_torch).abs().max().item():.2e}")


# ── Section 3: Benchmark naive vs tiled ─────────────────────

print("\n--- Section 3: Benchmark Naive vs Tiled ---")
M, K, N = 1024, 1024, 1024
A_fp16 = torch.rand(M, K, device='cuda', dtype=torch.float16)
B_fp16 = torch.rand(K, N, device='cuda', dtype=torch.float16)
A_fp32 = A_fp16.float()
B_fp32 = B_fp16.float()

# warmup
for _ in range(5):
    _ = naive_matmul(A_fp32, B_fp32)
    _ = tiled_matmul(A_fp16, B_fp16)
    _ = A_fp16 @ B_fp16
torch.cuda.synchronize()

# naive timing (small matrix — large would take forever)
M_small, K_small, N_small = 128, 128, 128
A_small = torch.rand(M_small, K_small, device='cuda', dtype=torch.float32)
B_small = torch.rand(K_small, N_small, device='cuda', dtype=torch.float32)

start = time.perf_counter()
for _ in range(10):
    _ = naive_matmul(A_small, B_small)
torch.cuda.synchronize()
naive_ms = (time.perf_counter() - start) / 10 * 1000

# tiled timing (full size)
start = time.perf_counter()
for _ in range(100):
    _ = tiled_matmul(A_fp16, B_fp16)
torch.cuda.synchronize()
tiled_ms = (time.perf_counter() - start) / 100 * 1000

# pytorch timing
start = time.perf_counter()
for _ in range(100):
    _ = A_fp16 @ B_fp16
torch.cuda.synchronize()
torch_ms = (time.perf_counter() - start) / 100 * 1000

print(f"  Naive  (128x128): {naive_ms:.3f} ms  ← scalar, uncoalesced")
print(f"  Tiled  (1024x1024): {tiled_ms:.3f} ms  ← tiled, Tensor Cores")
print(f"  PyTorch (1024x1024): {torch_ms:.3f} ms  ← cuBLAS")
print(f"  Tiled vs PyTorch: {tiled_ms/torch_ms:.2f}x")


# ── Section 4: Key concepts ──────────────────────────────────

print("\n--- Summary ---")
print("Key concepts from this notebook:")
print("  1. 2D grid         = one program per output tile")
print("  2. pid_m, pid_n    = which tile am I")
print("  3. strides         = how to navigate memory layout")
print("  4. tl.dot          = tile matmul using Tensor Cores")
print("  5. accumulator     = lives in registers, never touches HBM")
print("  6. K loop          = reduce over shared dimension in chunks")
print("  7. [:, None]       = broadcasting for 2D pointer arithmetic")
print("\nNaive: scalar loads, no Tensor Cores, terrible HBM efficiency")
print("Tiled: coalesced loads, Tensor Cores, excellent HBM efficiency")
print("\nReady for Notebook 3: Online Softmax")
print("\nNsight Compute command to profile these kernels:")
print("  ncu --set full --kernel-name naive_matmul_kernel -o naive python notebook2_tiled_matmul.py")
print("  ncu --set full --kernel-name tiled_matmul_kernel -o tiled python notebook2_tiled_matmul.py")
