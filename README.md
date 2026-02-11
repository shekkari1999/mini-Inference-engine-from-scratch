# Mini Inference Engine from Scratch

A minimal, modular transformer inference engine built to study:

- KV cache design
- Paged KV memory management
- Streaming (blocked) attention
- Prefill vs Decode scaling behavior
- Memory fragmentation vs compute bottlenecks

This project focuses on **inference-time systems behavior**, not training.


---

## Goals

This repository was built to understand how modern LLM inference engines work under the hood.

Specifically:

- Why naive attention explodes in memory
- Why decode latency scales linearly with context length
- How paged KV caches avoid fragmentation
- How streaming attention avoids materializing full T×T matrices
- Why FlashAttention requires kernel fusion for real performance gains


---

## Architecture

attention/
naive_attention.py
streaming_attention.py

kv_cache/
contiguous_cache.py
paged_cache.py

utils/
memory.py
benchmark.py

benchmarks/
benchmark_prefill.py
benchmark_decode.py

config.py
model.py


### Design Principles

- Clean separation of concerns
- Swappable attention implementations
- Swappable KV cache implementations
- No global state
- Reusable benchmark utilities
- Experiment drivers separate from core logic


---

## Implemented Components

### Naive Attention

- Full QKᵀ materialization
- O(T²) memory
- O(T²) compute
- Used as baseline

---

### Streaming (Blocked) Attention

- Reads KV pages sequentially
- Uses online softmax accumulation
- Avoids materializing full T×T matrix
- O(T) memory
- Still O(T²) compute during prefill

Algorithmically similar to FlashAttention,
but not kernel-fused or IO-optimized.

---

### Contiguous KV Cache

- Pre-allocates `[B, H, max_seq_len, D]`
- Simple but can cause fragmentation
- Used as baseline

---

### Paged KV Cache

- Allocates fixed-size pages on demand
- Avoids large contiguous allocations
- Reduces fragmentation
- Enables streaming attention

---

## Benchmarks

### Prefill

Measures:

- End-to-end latency
- Allocated vs reserved GPU memory
- Scaling with sequence length

Run: python benchmarks/benchmark_prefill.py

Expected behavior:

- Latency grows ~quadratically
- Memory grows linearly with streaming attention
- Naive attention exhibits T² memory behavior

---

### Decode

Measures:

- Tokens per second
- Throughput degradation with context length

Run: python benchmarks/benchmark_decode.py

Expected behavior:

- Decode throughput decreases roughly linearly
- Per-step cost grows with context length

---

## Key Observations

- Streaming attention removes T² memory explosion but not T² compute.
- Decode is memory-bound and scales linearly with context length.
- Paging prevents large memory reservation spikes.
- Python-level page loops reduce throughput.
- Kernel fusion (e.g., Triton) would be required for production performance.

---

## Limitations

- No causal masking (for simplicity)
- No fused kernels
- No mixed precision optimization
- No tensor parallelism
- Single GPU only

This project focuses strictly on core inference mechanics.


---

## Future Work

- Add proper causal masking
- Implement Triton-fused streaming attention
- Compare against FlashAttention
- Add multi-request batching
- Simulate production KV allocator behavior
- Add profiling (nsys / torch.profiler)


---

## Why This Exists

Modern LLM inference systems (vLLM, FlashAttention, etc.)
optimize two things:

1. Memory efficiency
2. Kernel efficiency

This project explores the first step:
understanding memory layout and attention scaling
before moving to kernel fusion.


---

## Requirements

- PyTorch
- CUDA GPU
- Python 3.10+
