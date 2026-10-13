import torch

from src.dataset import get_batch, split_data


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(1000)  # token i has id i, so "next token" is easy to check
    x, y = get_batch(data, batch_size=8, context_length=16)
    assert x.shape == y.shape == (8, 16)
    assert torch.equal(y, x + 1)  # target at t is the token right after x at t


def test_windows_stay_inside_the_data():
    data = torch.arange(50)
    for _ in range(200):
        x, y = get_batch(data, batch_size=4, context_length=16)
        assert y.max() < 50


def test_split_is_contiguous():
    data = torch.arange(100)
    train, val = split_data(data, 0.9)
    assert len(train) == 90 and len(val) == 10
    assert torch.equal(torch.cat([train, val]), data)


def test_generator_makes_batches_reproducible():
    data = torch.arange(1000)
    a = get_batch(data, 4, 8, generator=torch.Generator().manual_seed(1))
    b = get_batch(data, 4, 8, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a[0], b[0])
