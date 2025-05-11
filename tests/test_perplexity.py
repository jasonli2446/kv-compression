import torch
from evaluation.perplexity import compute_perplexity


class DummyModel:
    def eval(self):
        pass

    def to(self, device):
        return self

    def __call__(self, input_ids):
        logits = torch.randn(*input_ids.shape, 1000)
        return logits, None


def test_compute_perplexity():
    class DummyLoader:
        def __iter__(self):
            for _ in range(2):
                yield {
                    "input_ids": torch.randint(0, 1000, (2, 10)),
                    "labels": torch.randint(0, 1000, (2, 10)),
                }

    model = DummyModel()
    data_loader = DummyLoader()
    ppl = compute_perplexity(model, data_loader)
    assert ppl > 0
