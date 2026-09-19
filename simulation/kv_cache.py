def estimate_kv_cache_memory(seq_len, layers=12, num_heads=8, head_dim=64, precision_bytes=2, batch_size=1):
    """
    Returns an analytical estimate of the KV cache memory footprint in Bytes.
    Formula: seq_len * layers * 2 (K & V) * num_heads * head_dim * precision_bytes * batch_size
    """
    bytes_used = seq_len * layers * 2 * num_heads * head_dim * precision_bytes * batch_size
    return bytes_used
