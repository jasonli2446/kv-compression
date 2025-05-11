import torch
import pytest

try:
    import tensorly as tl
    from compression.tucker import tucker_compress, tucker_decompress

    HAS_TL = True
except ImportError:
    HAS_TL = False


def test_tucker_compression():
    if not HAS_TL:
        pytest.skip("tensorly not installed")
    x = torch.randn(4, 8, 8)
    ranks = [2, 4, 4]
    core, factors = tucker_compress(x, ranks)
    x_hat = tucker_decompress(core, factors)
    assert x_hat.shape == x.shape
