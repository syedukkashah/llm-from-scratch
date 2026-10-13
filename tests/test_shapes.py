import pytest
import torch

from src.attention import MultiHeadAttention, causal_mask
from src.positional_encoding import LearnedPositionalEncoding, SinusoidalPositionalEncoding
from src.transformer_block import FeedForward, TransformerBlock


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


def test_positional_encoding_shape_and_first_row():
    pe = SinusoidalPositionalEncoding(d_model=16, max_len=50)
    out = pe(torch.zeros(2, 10, 16))
    assert out.shape == (2, 10, 16)
    # position 0: sin(0)=0 in even dims, cos(0)=1 in odd dims
    assert torch.allclose(out[0, 0, 0::2], torch.zeros(8))
    assert torch.allclose(out[0, 0, 1::2], torch.ones(8))


def test_positional_encoding_differs_by_position_and_is_bounded():
    pe = SinusoidalPositionalEncoding(d_model=16, max_len=50).pe
    assert pe.abs().max() <= 1.0
    assert not torch.allclose(pe[3], pe[4])  # every position has its own vector


def test_feed_forward_shape_and_hidden_size():
    ffn = FeedForward(d_model=32, ffn_dim=128)
    assert ffn.net[0].out_features == 128  # expands to ffn_dim
    assert ffn(torch.randn(2, 10, 32)).shape == (2, 10, 32)


def test_block_keeps_shape():
    block = TransformerBlock(d_model=32, n_heads=4, ffn_dim=128)
    assert block(torch.randn(2, 10, 32), causal_mask(10)).shape == (2, 10, 32)


def test_block_is_identity_when_sublayers_output_zero():
    # Residual paths: if attention and FFN add nothing, the block returns x unchanged.
    block = TransformerBlock(d_model=32, n_heads=4, ffn_dim=128).eval()
    for layer in (block.attn.w_o, block.ffn.net[2]):
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    x = torch.randn(2, 10, 32)
    assert torch.allclose(block(x, causal_mask(10)), x)


def test_learned_positional_encoding_shape_and_parameters():
    pe = LearnedPositionalEncoding(d_model=16, max_len=50)
    assert pe(torch.zeros(2, 10, 16)).shape == (2, 10, 16)
    assert sum(p.numel() for p in pe.parameters()) == 50 * 16  # one trainable vector per position
