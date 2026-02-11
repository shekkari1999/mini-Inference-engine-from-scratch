import time
import torch


def benchmark_prefill(model, config, device,
                      seq_len=1024,
                      batch_size=1,
                      kv_cache_cls=None):
    """
    Measures prefill latency.

    Arguments:
        model: instantiated transformer
        config: model config
        device: cuda device
        seq_len: sequence length
        kv_cache_cls: optional KV cache class
    """

    input_ids = torch.randint(
        0, config.vocab_size,
        (batch_size, seq_len),
        device=device
    )

    if kv_cache_cls is not None:
        kv_caches = [
            kv_cache_cls(
                batch_size,
                config.num_heads,
                config.hidden_size // config.num_heads,
                device=device
            )
            for _ in range(config.num_layers)
        ]
    else:
        kv_caches = None

    torch.cuda.synchronize()
    start = time.time()

    _ = model(input_ids, kv_caches)

    torch.cuda.synchronize()
    end = time.time()

    latency = end - start
    print(f"Prefill latency: {latency:.4f}s")
    return latency


def benchmark_decode(model, config, device,
                     prefill_len=1024,
                     steps=128,
                     batch_size=1,
                     kv_cache_cls=None):
    """
    Measures decode throughput (tokens/sec).
    """

    # Prefill phase
    input_ids = torch.randint(
        0, config.vocab_size,
        (batch_size, prefill_len),
        device=device
    )

    if kv_cache_cls is not None:
        kv_caches = [
            kv_cache_cls(
                batch_size,
                config.num_heads,
                config.hidden_size // config.num_heads,
                device=device
            )
            for _ in range(config.num_layers)
        ]
    else:
        kv_caches = None

    _ = model(input_ids, kv_caches)

    # Decode phase
    input_ids = torch.randint(
        0, config.vocab_size,
        (batch_size, 1),
        device=device
    )

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(steps):
        logits = model(input_ids, kv_caches)
        input_ids = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )

    torch.cuda.synchronize()
    end = time.time()

    tok_per_sec = steps / (end - start)
    print(f"Decode tokens/sec: {tok_per_sec:.2f}")
    return tok_per_sec