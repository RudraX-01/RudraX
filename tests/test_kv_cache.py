from simulation.kv_cache import estimate_kv_cache_memory

def test_kv_cache_scaling():
    mem_100 = estimate_kv_cache_memory(100, layers=1, num_heads=2, head_dim=64)
    mem_200 = estimate_kv_cache_memory(200, layers=1, num_heads=2, head_dim=64)
    
    # KV cache should scale linearly with sequence length
    assert mem_200 == mem_100 * 2
