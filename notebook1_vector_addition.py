# ============================================================
# Notebook 1: Vector Addition in Triton
# Building block 1 — understanding kernels, pid, offsets, masks
# ============================================================

# !pip install triton torch --quiet

import torch
import triton
import triton.language as tl
import time

print("=" * 60)
print("NOTEBOOK 1: Vector Addition in Triton")
print("=" * 60)

# ── Section 1: The simplest possible kernel ──────────────────

@triton.jit
def add_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # which program am I?
    pid = tl.program_id(0)

    # which elements are mine?
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # boundary mask — last program may go out of bounds
    mask = offsets < N

    # load from HBM into SRAM
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # compute — happens in SRAM/registers
    c = a + b

    # write back to HBM
    tl.store(C_ptr + offsets, c, mask=mask)


def vector_add(A, B):
    N = A.shape[0]
    C = torch.empty(N, device=A.device, dtype=A.dtype)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    add_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
    return C


# ── Section 2: Verify correctness ───────────────────────────

print("\n--- Section 1: Correctness Check ---")
N = 100_000
A = torch.randn(N, device='cuda', dtype=torch.float32)
B = torch.randn(N, device='cuda', dtype=torch.float32)

C_triton = vector_add(A, B)
C_torch  = A + B

print(f"Max diff: {(C_triton - C_torch).abs().max().item():.2e}")
print(f"Correct:  {torch.allclose(C_triton, C_torch, atol=1e-5)}")


# ── Section 3: Understanding pid and offsets ─────────────────

print("\n--- Section 2: Understanding pid/offsets ---")
print("N=1000, BLOCK_SIZE=128")
print(f"  programs launched: {triton.cdiv(1000, 128)}")
print(f"  pid=0 handles indices: 0 to 127")
print(f"  pid=1 handles indices: 128 to 255")
print(f"  pid=7 handles indices: 896 to 999 (mask kills 1000-1023)")


# ── Section 4: Benchmark ─────────────────────────────────────

print("\n--- Section 3: Performance ---")
N = 10_000_000

A = torch.randn(N, device='cuda', dtype=torch.float32)
B = torch.randn(N, device='cuda', dtype=torch.float32)

# warmup
for _ in range(10):
    _ = vector_add(A, B)
    _ = A + B
torch.cuda.synchronize()

# triton timing
start = time.perf_counter()
for _ in range(100):
    _ = vector_add(A, B)
torch.cuda.synchronize()
triton_ms = (time.perf_counter() - start) / 100 * 1000

# pytorch timing
start = time.perf_counter()
for _ in range(100):
    _ = A + B
torch.cuda.synchronize()
torch_ms = (time.perf_counter() - start) / 100 * 1000

print(f"  Triton:  {triton_ms:.3f} ms")
print(f"  PyTorch: {torch_ms:.3f} ms")
print(f"  Ratio:   {triton_ms/torch_ms:.2f}x (should be ~1x, same op)")


# ── Section 5: What we learned ──────────────────────────────

print("\n--- Summary ---")
print("Key concepts from this notebook:")
print("  1. @triton.jit   = GPU kernel")
print("  2. grid          = how many programs launch")
print("  3. pid           = which program am I")
print("  4. offsets       = which elements are mine")
print("  5. mask          = guard against out-of-bounds")
print("  6. tl.load       = HBM → SRAM")
print("  7. tl.store      = SRAM → HBM")
print("  8. CPU launcher  = allocate output, define grid, launch kernel")
print("\nReady for Notebook 2: Tiled Matrix Multiplication")
