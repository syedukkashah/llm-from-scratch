import torch

from experiments.hand_calculation import K, Q, V, compute
from src.attention import causal_mask, scaled_dot_product_attention


def test_causal_mask_is_lower_triangular():
    expected = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    assert torch.equal(causal_mask(3), expected)


def test_future_positions_get_zero_weight():
    q = k = v = torch.randn(2, 6, 4)
    _, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask(6))
    future = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    assert (weights[:, future] == 0).all()
    assert torch.allclose(weights.sum(-1), torch.ones(2, 6), atol=1e-5)


def test_first_token_only_sees_itself():
    q, k, v = (torch.tensor(m) for m in (Q, K, V))
    out, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask(3))
    assert torch.allclose(weights[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(out[0], v[0])  # output = its own value


def test_masked_matches_hand_calculation():
    expected = compute()
    q, k, v = (torch.tensor(m) for m in (Q, K, V))
    out, _ = scaled_dot_product_attention(q, k, v, mask=causal_mask(3))
    assert torch.allclose(out, torch.tensor(expected["masked_output"]), atol=1e-6)


def test_changing_a_future_value_does_not_change_earlier_outputs():
    q = k = torch.randn(1, 5, 4)
    v = torch.randn(1, 5, 4)
    mask = causal_mask(5)
    out_before, _ = scaled_dot_product_attention(q, k, v, mask)
    v2 = v.clone()
    v2[:, 4] += 10.0  # edit only the last token's value
    out_after, _ = scaled_dot_product_attention(q, k, v2, mask)
    assert torch.allclose(out_before[:, :4], out_after[:, :4])  # earlier positions unaffected
