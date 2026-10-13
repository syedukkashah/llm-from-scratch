"""Decoder-style Transformer block (Milestone 3).

    x ──┬─► LayerNorm ─► MultiHeadAttention ─►(+)──┬─► LayerNorm ─► FeedForward ─►(+)─► out
        └────────────── residual ────────────────►┘ └──────────── residual ──────────►┘
"""
import torch.nn as nn

from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Applied to each token on its own: D -> ffn_dim -> D."""

    def __init__(self, d_model, ffn_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),  # expand (usually 4 * d_model)
            nn.GELU(),                    # nonlinearity
            nn.Linear(ffn_dim, d_model),  # project back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """(B, T, D) -> (B, T, D)"""
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x, mask=None):
        """x: (B, T, D), mask: (T, T) bool -> (B, T, D)"""
        x = x + self.attn(self.ln1(x), mask)  # residual around attention
        x = x + self.ffn(self.ln2(x))         # residual around feed-forward
        return x
