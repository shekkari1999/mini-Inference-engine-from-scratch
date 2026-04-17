import torch
import triton
import triton.language as tl
import math


# ------------------------------------------------------------
# Online Softmax Fused Attention with Causal Masking
# Supports both single-head (2D) and multi-head (4D) inputs
#
# Single-head: Q [T, D], K [S, D], V [S, D] -> O [T, D]
# Multi-head:  Q [B, H, T, D], K [B, H, S, D], V [B, H, S, D] -> O [B, H, T, D]
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
    IS_CAUSAL: tl.constexpr,
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

    # For causal: only iterate up to the relevant keys
    s_end = S
    if IS_CAUSAL:
        s_end = tl.minimum(S, t_start + BLOCK_T)

    for s_start in range(0, s_end, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)

        k_ptrs = K_ptr + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=offs_s[:, None] < S, other=0.0)
        v = tl.load(v_ptrs, mask=offs_s[:, None] < S, other=0.0)

        # Compute scores: [BLOCK_T, BLOCK_S]
        scores = tl.dot(q, tl.trans(k)) * scale

        # Apply causal mask: query at position t can only attend to keys at position s <= t
        if IS_CAUSAL:
            causal_mask = offs_t[:, None] >= offs_s[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        # Mask out-of-bounds keys
        key_mask = offs_s[None, :] < S
        scores = tl.where(key_mask, scores, float("-inf"))

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
# Multi-head kernel launcher (one program per (batch, head, block_t))
# ------------------------------------------------------------

@triton.jit
def _multihead_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    B, H, T, S,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Program IDs: (block_t_idx, batch_head_idx)
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    t_start = pid_t * BLOCK_T

    offs_t = t_start + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    # Offset pointers to this (batch, head)
    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    o_base = O_ptr + batch_idx * stride_ob + head_idx * stride_oh

    # Load Q tile
    q_ptrs = q_base + offs_t[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_t[:, None] < T, other=0.0)

    m_i = tl.full((BLOCK_T,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)

    s_end = S
    if IS_CAUSAL:
        s_end = tl.minimum(S, t_start + BLOCK_T)

    for s_start in range(0, s_end, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)

        k_ptrs = k_base + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=offs_s[:, None] < S, other=0.0)
        v = tl.load(v_ptrs, mask=offs_s[:, None] < S, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale

        if IS_CAUSAL:
            causal_mask = offs_t[:, None] >= offs_s[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        key_mask = offs_s[None, :] < S
        scores = tl.where(key_mask, scores, float("-inf"))

        block_max = tl.max(scores, axis=1)
        new_m = tl.maximum(m_i, block_max)
        exp_m_diff = tl.exp(m_i - new_m)
        exp_scores = tl.exp(scores - new_m[:, None])

        l_i = exp_m_diff * l_i + tl.sum(exp_scores, axis=1)
        acc = exp_m_diff[:, None] * acc + tl.dot(exp_scores, v)
        m_i = new_m

    acc = acc / l_i[:, None]

    o_ptrs = o_base + offs_t[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=offs_t[:, None] < T)


# ------------------------------------------------------------
# Python Wrapper
# ------------------------------------------------------------

class TritonAttention:
    """Triton fused attention supporting both single-head and multi-head inputs."""

    def __init__(self, block_t=32, block_s=32, causal=False):
        self.block_t = block_t
        self.block_s = block_s
        self.causal = causal

    def forward(self, q, k, v, causal=None):
        """
        Args:
            q: Query tensor, either [T, D] or [B, H, T, D]
            k: Key tensor, either [S, D] or [B, H, S, D]
            v: Value tensor, either [S, D] or [B, H, S, D]
            causal: Override default causal setting
        """
        is_causal = causal if causal is not None else self.causal

        if q.dim() == 2:
            return self._forward_2d(q, k, v, is_causal)
        elif q.dim() == 4:
            return self._forward_4d(q, k, v, is_causal)
        else:
            raise ValueError(f"Expected 2D or 4D input, got {q.dim()}D")

    def _forward_2d(self, q, k, v, is_causal):
        """Single-head attention: Q [T, D], K [S, D], V [S, D]."""
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
            IS_CAUSAL=is_causal,
            BLOCK_T=self.block_t,
            BLOCK_S=self.block_s,
            BLOCK_D=D,
        )

        return o

    def _forward_4d(self, q, k, v, is_causal):
        """Multi-head attention: Q [B, H, T, D], K [B, H, S, D], V [B, H, S, D]."""
        B, H, T, D = q.shape
        S = k.shape[2]

        o = torch.empty_like(q)

        grid_t = (T + self.block_t - 1) // self.block_t
        grid_bh = B * H
        grid = (grid_t, grid_bh)

        _multihead_attention_kernel[grid](
            q, k, v, o,
            B, H, T, S,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            1.0 / math.sqrt(D),
            IS_CAUSAL=is_causal,
            BLOCK_T=self.block_t,
            BLOCK_S=self.block_s,
            BLOCK_D=D,
        )

        return o
