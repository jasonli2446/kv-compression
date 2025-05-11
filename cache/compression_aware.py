class CompressionAwareCache:
    def __init__(self, cache, compression_policy=None):
        self.cache = cache
        self.compression_policy = compression_policy or (lambda k, v: None)

    def set(self, key, value):
        method = self.compression_policy(key, value)
        if method:
            value = method["compress"](value)
        self.cache.set(key, value)

    def get(self, key):
        return self.cache.get(key)

    def clear(self):
        self.cache.clear()
