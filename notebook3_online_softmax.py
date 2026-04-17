# ============================================================
# Notebook 3: Online Softmax
# The algorithmic trick that makes Flash Attention possible
# ============================================================

import torch
import math

print("=" * 60)
print("NOTEBOOK 3: Online Softmax")
print("=" * 60)

# ── Section 1: Why naive softmax breaks with tiling ──────────

print("\n--- Section 1: Why Naive Softmax Breaks With Tiling ---")
print("""
Standard softmax for a row x = [x0, x1, x2, ..., xN]:

  step 1: m = max(x0, x1, ..., xN)          <- needs FULL row
  step 2: exp_i = exp(xi - m) for all i      <- needs m first
  step 3: l = sum(exp_i)                     <- needs FULL row
  step 4: output_i = exp_i / l               <- needs l first

Problem: steps 1 and 3 require seeing the ENTIRE row
         before you can compute anything.

If you only see a tile at a time:
  tile 0: [x0, x1, x2, x3]  <- you don't know the global max yet!
  tile 1: [x4, x5, x6, x7]  <- might have a larger value!

You can't normalize correctly without the full row.
This is why naive tiling doesn't work for attention softmax.
""")


# ── Section 2: The online softmax trick ─────────────────────

print("--- Section 2: Online Softmax Algorithm ---")
print("""
Solution: keep running statistics (m, l) and RESCALE as you go.

For each new tile:
  1. compute local max m_new = max(current tile)
  2. update global max: m = max(m_prev, m_new)
  3. rescale previous sum: l = l_prev * exp(m_prev - m) + sum(exp(tile - m))
  4. rescale previous output: O = O_prev * exp(m_prev - m) + exp(tile - m) @ V_tile

By the time you've seen ALL tiles:
  m = exact global max
  l = exact global sum of exponentials
  O = exact attention output

One pass through the data. No N×N matrix ever materialized.
""")


# ── Section 3: Implement naive softmax ───────────────────────

print("--- Section 3: Naive vs Online Softmax ---")

def naive_softmax(x):
    """Standard softmax — needs full row"""
    m = x.max()               # needs full row
    e = torch.exp(x - m)      # stable exp
    return e / e.sum()        # normalize


def online_softmax_tiled(x, tile_size=4):
    """
    Online softmax — processes one tile at a time
    Produces exact same result as naive softmax
    """
    N = x.shape[0]

    # running statistics — start at neutral values
    m = float('-inf')    # running max
    l = 0.0              # running sum of exp

    # first pass — compute m and l tile by tile
    for i in range(0, N, tile_size):
        tile = x[i : i + tile_size]

        m_new = max(m, tile.max().item())

        # rescale previous l for new max
        l = l * math.exp(m - m_new) + torch.exp(tile - m_new).sum().item()

        m = m_new

    # second pass — compute output using final m and l
    output = torch.zeros_like(x)
    for i in range(0, N, tile_size):
        tile = x[i : i + tile_size]
        output[i : i + tile_size] = torch.exp(tile - m) / l

    return output


def online_softmax_single_pass(x, tile_size=4):
    """
    True single pass online softmax
    Accumulates output as it goes — no second pass needed
    This is what Flash Attention uses
    """
    N = x.shape[0]

    m = float('-inf')    # running max
    l = 0.0              # running sum
    O = torch.zeros(N)   # running output (in Flash Attention this is the weighted V sum)

    for i in range(0, N, tile_size):
        tile = x[i : i + tile_size]

        # local max for this tile
        m_new = max(m, tile.max().item())

        # rescale factor
        alpha = math.exp(m - m_new)

        # update running sum
        l_new = l * alpha + torch.exp(tile - m_new).sum().item()

        # rescale previous output and add new contribution
        # (in attention this would be: O = O * alpha + softmax(tile) @ V_tile)
        O[:i] = O[:i] * alpha
        O[i : i + tile_size] = torch.exp(tile - m_new)

        m = m_new
        l = l_new

    # normalize
    O = O / l
    return O


# ── Section 4: Verify all three give same result ─────────────

torch.manual_seed(42)
x = torch.randn(16)

result_naive  = naive_softmax(x)
result_tiled  = online_softmax_tiled(x, tile_size=4)
result_single = online_softmax_single_pass(x, tile_size=4)
result_torch  = torch.softmax(x, dim=0)

print(f"Naive vs torch:         {torch.allclose(result_naive, result_torch, atol=1e-6)}")
print(f"Online tiled vs torch:  {torch.allclose(result_tiled, result_torch, atol=1e-6)}")
print(f"Single pass vs torch:   {torch.allclose(result_single, result_torch, atol=1e-6)}")
print(f"\nAll produce identical results despite never seeing full row at once!")


# ── Section 5: The math behind rescaling ─────────────────────

print("\n--- Section 4: Why Rescaling Works ---")
print("""
When you see a new tile with a larger max m_new > m_prev:

  Old sum:    l_old = sum(exp(xi - m_prev)) for previous tiles
  Need:       l_new = sum(exp(xi - m_new))  for same tiles

  Relationship:
    exp(xi - m_new) = exp(xi - m_prev) * exp(m_prev - m_new)
    l_new = l_old * exp(m_prev - m_new)

  So:  l = l_old * exp(m_prev - m_new) + sum(exp(new_tile - m_new))

This rescaling factor exp(m_prev - m_new) is always <= 1
  (since m_new >= m_prev)
So it's numerically stable — values shrink, never grow.

Same logic applies to the output accumulator O in Flash Attention.
""")


# ── Section 6: Connection to Flash Attention ─────────────────

print("--- Section 5: Connection to Flash Attention ---")
print("""
In Flash Attention, for each Q tile:
  - loop over K/V tiles
  - S_tile = Q_tile @ K_tile.T          <- attention scores for this tile
  - update running max m with S_tile
  - update running sum l with exp(S_tile - m)
  - update output O: O = O * rescale + exp(S_tile - m) @ V_tile

After all K/V tiles:
  - O = O / l                           <- final normalization

Result: exact softmax attention without ever materializing N×N matrix
        all intermediate values live in registers
        HBM only touched for K/V tile loads and final O write
""")


# ── Section 7: Batch/head attention with online softmax ──────

print("--- Section 6: Full Attention with Online Softmax ---")

def attention_online_softmax(Q, K, V, scale=None):
    """
    Exact attention using online softmax
    Mimics what Flash Attention does algorithmically
    Q: [N, d]
    K: [N, d]
    V: [N, d]
    """
    N, d = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    tile_size = 32
    O = torch.zeros(N, d, device=Q.device, dtype=torch.float32)
    m = torch.full((N,), float('-inf'), device=Q.device)
    l = torch.zeros(N, device=Q.device)

    for j in range(0, N, tile_size):
        K_tile = K[j : j + tile_size]   # [tile, d]
        V_tile = V[j : j + tile_size]   # [tile, d]

        # compute scores for this K tile
        S = Q @ K_tile.T * scale         # [N, tile]

        # online softmax update
        m_new = torch.maximum(m, S.max(dim=1).values)

        # rescale factor
        alpha = torch.exp(m - m_new)     # [N]

        # update l
        l_new = l * alpha + torch.exp(S - m_new[:, None]).sum(dim=1)

        # update O
        O = O * alpha[:, None] + torch.exp(S - m_new[:, None]) @ V_tile

        m = m_new
        l = l_new

    # final normalization
    O = O / l[:, None]
    return O


# verify against PyTorch standard attention
N, d = 128, 64
Q = torch.randn(N, d, device='cuda', dtype=torch.float32)
K = torch.randn(N, d, device='cuda', dtype=torch.float32)
V = torch.randn(N, d, device='cuda', dtype=torch.float32)

O_online = attention_online_softmax(Q, K, V)

# standard attention
scale = 1.0 / math.sqrt(d)
S = Q @ K.T * scale
P = torch.softmax(S, dim=-1)
O_standard = P @ V

print(f"Online softmax attention vs standard attention:")
print(f"  Max diff: {(O_online - O_standard).abs().max().item():.2e}")
print(f"  Correct:  {torch.allclose(O_online, O_standard, atol=1e-4)}")
print(f"\nIdentical results — online softmax is mathematically exact!")


print("\n--- Summary ---")
print("Key concepts from this notebook:")
print("  1. Naive softmax needs full row — breaks with tiling")
print("  2. Online softmax tracks running (m, l) statistics")
print("  3. Rescaling factor exp(m_prev - m_new) corrects previous estimates")
print("  4. Single pass — tiles processed once, result accumulated in registers")
print("  5. Mathematically identical to naive softmax")
print("  6. This is the core algorithmic innovation of Flash Attention")
print("\nReady for Notebook 4: Flash Attention Forward + Backward in Triton")
