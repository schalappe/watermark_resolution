import pytest
import torch

from watermark.attacks import TRAINING_ATTACKS
from watermark.data import random_marks
from watermark.evaluate import ber
from watermark.model import Embedder, Extractor


@pytest.mark.parametrize("host_size", [64, 128, 256, 512])
def test_table5_resolutions(host_size):
    host = torch.rand(2, 1, host_size, host_size) * 2 - 1
    mark = random_marks(2, host_size // 16)
    watermarked = Embedder()(host, mark)
    assert watermarked.shape == host.shape
    assert Extractor()(watermarked).shape == mark.shape


@pytest.mark.parametrize("name", list(TRAINING_ATTACKS))
def test_extraction_loss_reaches_embedder(name):
    # L2 alone: a total-loss gradient could be non-zero through L1 only, which hid the 2023 bug.
    torch.manual_seed(0)
    embedder, extractor = Embedder(), Extractor()
    host = torch.rand(4, 1, 128, 128) * 2 - 1
    mark = random_marks(4, 8)
    watermarked = embedder(host, mark)
    l2 = (extractor(TRAINING_ATTACKS[name](watermarked, host)) - mark).abs().mean()
    grads = torch.autograd.grad(l2, list(embedder.mark.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_ber_bounds():
    mark = random_marks(3, 8)
    assert ber(mark, mark * 0.3).tolist() == [0, 0, 0]
    assert ber(mark, -mark).tolist() == [100, 100, 100]
