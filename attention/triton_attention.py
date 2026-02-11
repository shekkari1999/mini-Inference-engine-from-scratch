import torch
import triton
import triton.language as tl
import math


# ------------------------------------------------------------
# Online Softmax Fused Attention (Single-head, 2D Q,K,V)
# Q: [T, D]
# K: [S, D]
# V: [S, D]
# O: [T, D]
# ------------------------------------------------------------

@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    T, S,
    stride_qt, stride_qd,
    stride_ks, stride_kd,
    stride_vs, stride_vd,
    stride_ot, stride_od,
    scale,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    t_start = pid * BLOCK_T

    offs_t = t_start + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile
    q_ptrs = Q_ptr + offs_t[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_t[:, None] < T, other=0.0)

    # Running values per query row
    m_i = tl.full((BLOCK_T,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)

        k_ptrs = K_ptr + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=offs_s[:, None] < S, other=0.0)
        v = tl.load(v_ptrs, mask=offs_s[:, None] < S, other=0.0)

        # Compute scores
        scores = tl.dot(q, tl.trans(k)) * scale

        # Block max
        block_max = tl.max(scores, axis=1)

        # New running max
        new_m = tl.maximum(m_i, block_max)

        # Rescale previous
        exp_m_diff = tl.exp(m_i - new_m)

        # Compute exp scores
        exp_scores = tl.exp(scores - new_m[:, None])

        # Update l_i
        l_i = exp_m_diff * l_i + tl.sum(exp_scores, axis=1)

        # Update accumulator
        acc = exp_m_diff[:, None] * acc + tl.dot(exp_scores, v)

        # Update running max
        m_i = new_m

    # Normalize
    acc = acc / l_i[:, None]

    # Store output
    o_ptrs = O_ptr + offs_t[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=offs_t[:, None] < T)


# ------------------------------------------------------------
# Python Wrapper
# ------------------------------------------------------------

class TritonAttention:
    def __init__(self, block_t=32, block_s=32):
        self.block_t = block_t
        self.block_s = block_s

    def forward(self, q, k, v):
        assert q.dim() == 2
        assert k.dim() == 2
        assert v.dim() == 2

        T, D = q.shape
        S = k.shape[0]

        o = torch.empty_like(q)

        grid = ((T + self.block_t - 1) // self.block_t,)

        _attention_kernel[grid](
            q, k, v, o,
            T, S,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            o.stride(0), o.stride(1),
            1.0 / math.sqrt(D),
            BLOCK_T=self.block_t,
            BLOCK_S=self.block_s,
            BLOCK_D=D,
        )

        return o