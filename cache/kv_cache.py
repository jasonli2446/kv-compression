class KVCache:
    def __init__(self, compress_fn=None, decompress_fn=None):
        self.cache = {}
        self.compress_fn = compress_fn
        self.decompress_fn = decompress_fn

    def set(self, key, value):
        if self.compress_fn:
            value = self.compress_fn(value)
        self.cache[key] = value

    def get(self, key):
        value = self.cache.get(key, None)
        if value is not None and self.decompress_fn:
            value = self.decompress_fn(value)
        return value

    def clear(self):
        self.cache.clear()
