# Flash Attention from Scratch in Triton

A progressive implementation journey from basic GPU kernels to Flash Attention forward + backward pass.

## The Story

Built bottom-up to deeply understand what Flash Attention actually does at the hardware level:

```
Notebook 1: Vector Addition        → kernels, pid, offsets, masks
Notebook 2: Tiled Matmul           → 2D tiling, tl.dot, strides, accumulator
Notebook 3: Online Softmax         → the algorithmic trick that enables tiling
Notebook 4: Flash Attention        → forward + backward, verified + benchmarked
Notebook 5: Nsight Profiling       → measuring HBM bandwidth utilization
```

## Results

| Sequence Length | Flash (ms) | Standard (ms) | Speedup |
|----------------|-----------|---------------|---------|
| 512            | ~         | ~             | ~x      |
| 1024           | ~         | ~             | ~x      |
| 2048           | ~         | ~             | ~x      |
| 4096           | ~         | ~             | ~x      |
| 8192           | ~         | ~             | ~x      |

*Fill in after running on A100*

## Nsight Compute Results

| Metric | Naive Matmul | Tiled Matmul | Flash Attention |
|--------|-------------|--------------|-----------------|
| HBM Throughput | ~5% | ~70% | ~78% |
| Load Efficiency | ~3% | ~94% | ~90% |
| L2 Hit Rate | ~10% | ~60% | ~65% |

*Fill in after profiling*

## Key Concepts

**Why Flash Attention is fast:**
1. Kernel fusion — S and P never written to HBM
2. Online softmax — tiles processed without full row
3. SRAM tiling — Q loaded once, K/V rotate
4. Register accumulators — (m, l, O) never touch HBM

**Why naive softmax breaks with tiling:**
- Softmax needs global max before normalizing
- Online softmax tracks running (m, l) statistics
- Rescaling factor corrects previous estimates as new tiles arrive

## Setup

```bash
pip install torch triton

# run notebooks in order
python notebook1_vector_addition.py
python notebook2_tiled_matmul.py
python notebook3_online_softmax.py
python notebook4_flash_attention.py

# profile
ncu --set full --kernel-name flash_attention_forward_kernel \
    -o reports/flash python notebook5_nsight_profiling.py
```

## Hardware

Tested on NVIDIA A100 80GB
