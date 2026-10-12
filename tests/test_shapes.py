import pytest
import torch

from src.attention import MultiHeadAttention, causal_mask


def test_split_and_merge_shapes():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 10, 32)
    heads = mha.split_heads(x)
    assert heads.shape == (2, 4, 10, 8)  # (B, H, T, d_k)
    assert mha.merge_heads(heads).shape == (2, 10, 32)


def test_merge_undoes_split():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 10, 32)
    assert torch.equal(mha.merge_heads(mha.split_heads(x)), x)


def test_mha_output_shape_and_attention_shape():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 10, 32)
    out = mha(x, causal_mask(10))
    assert out.shape == (2, 10, 32)
    assert mha.last_attn.shape == (2, 4, 10, 10)  # one T x T map per head


def test_d_model_must_divide_by_heads():
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model=30, n_heads=4)


def test_mha_is_causal_in_every_head():
    mha = MultiHeadAttention(d_model=32, n_heads=4).eval()
    x = torch.randn(1, 8, 32)
    mha(x, causal_mask(8))
    future = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert (mha.last_attn[:, :, future] == 0).all()
