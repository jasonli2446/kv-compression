import torch
from compression.svd import svd_compress, svd_decompress


def test_svd_compression():
    x = torch.randn(4, 8)
    rank = 3
    compressed = svd_compress(x, rank)
    x_hat = svd_decompress(compressed, x.shape)
    assert x_hat.shape == x.shape
