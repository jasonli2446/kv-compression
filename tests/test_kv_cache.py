from cache.kv_cache import KVCache


def test_kv_cache_basic():
    cache = KVCache()
    cache.set("a", 123)
    assert cache.get("a") == 123
    cache.clear()
    assert cache.get("a") is None


def test_kv_cache_compression():
    cache = KVCache(compress_fn=lambda x: x + 1, decompress_fn=lambda x: x - 1)
    cache.set("b", 10)
    assert cache.get("b") == 10
