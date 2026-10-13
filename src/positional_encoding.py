"""Sinusoidal positional encoding (Milestone 2).

Attention ignores token order, so we add a fixed vector per position:
    PE[pos, 2i]   = sin(pos / 10000^(2i/D))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/D))
"""
import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even (sin/cos pairs)"
        pos = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)  positions 0..max_len-1
        i = torch.arange(0, d_model, 2)           # (D/2,)        even dimensions
        freq = torch.exp(-math.log(10000.0) * i / d_model)  # 1 / 10000^(2i/D)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * freq)  # even dims -> sin
        pe[:, 1::2] = torch.cos(pos * freq)  # odd dims  -> cos
        self.register_buffer("pe", pe)       # saved with the model but not trained

    def forward(self, x):
        """x: (B, T, D) token embeddings -> same shape with position added."""
        T = x.size(1)
        return x + self.pe[:T]  # (T, D) broadcasts over the batch


class LearnedPositionalEncoding(nn.Module):
    """One trainable vector per position (used for the optional position ablation)."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """x: (B, T, D) -> same shape with the learned position vectors added."""
        positions = torch.arange(x.size(1), device=x.device)  # (T,)
        return x + self.embed(positions)                      # (T, D) broadcasts over batch
