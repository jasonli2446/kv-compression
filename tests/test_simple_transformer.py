import torch
from models.simple_transformer import SimpleTransformer


def test_forward():
    model = SimpleTransformer()
    input_ids = torch.randint(0, 1000, (2, 10))
    logits, kv_cache = model(input_ids)
    assert logits.shape == (2, 10, 1000)
    assert (
        kv_cache is None
        or isinstance(kv_cache, dict)
        or isinstance(kv_cache, type(None))
    )
