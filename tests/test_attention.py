import torch

from experiments.hand_calculation import K, Q, V, compute
from src.attention import scaled_dot_product_attention


def test_output_and_weight_shapes():
    q = k = v = torch.randn(2, 5, 8)  # (B, T, d_k)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 5, 8)
    assert weights.shape == (2, 5, 5)


def test_weight_rows_sum_to_one():
    q = k = v = torch.randn(2, 4, 6, 8)  # extra leading dim = heads
    _, weights = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 6), atol=1e-5)


def test_matches_hand_calculation():
    expected = compute()
    q, k, v = (torch.tensor(m) for m in (Q, K, V))
    out, weights = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(weights, torch.tensor(expected["weights"]), atol=1e-6)
    assert torch.allclose(out, torch.tensor(expected["output"]), atol=1e-6)
