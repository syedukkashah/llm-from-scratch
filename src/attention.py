"""Self-attention building blocks (Milestones 1 and 2).

Notation used in every shape comment:
    B = batch size, T = sequence length (tokens), D = d_model,
    H = number of heads, d_k = D // H (size of one head)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T, device=None):
    """Lower-triangular boolean matrix of shape (T, T).

    mask[i, j] is True when token i is ALLOWED to look at token j (j <= i).
    Row i = the token asking, column j = the token being looked at.
    For T=3:
        [[ True, False, False],     token 0 sees only itself
         [ True,  True, False],     token 1 sees tokens 0, 1
         [ True,  True,  True]]     token 2 sees tokens 0, 1, 2
    """
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def scaled_dot_product_attention(q, k, v, mask=None):
    """softmax(Q K^T / sqrt(d_k)) V

    q, k, v : (..., T, d_k)   -- any leading dims (batch, heads) are allowed
    mask    : (T, T) bool, True = allowed to attend. None = bidirectional.
    returns : output (..., T, d_k) and attention weights (..., T, T)
    """
    d_k = q.size(-1)

    # (..., T, d_k) @ (..., d_k, T) -> (..., T, T)   score[i, j] = q_i . k_j
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        # Forbidden positions get -inf BEFORE softmax -> weight exactly 0.
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)  # each row sums to 1

    # (..., T, T) @ (..., T, d_k) -> (..., T, d_k)   weighted average of values
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """H attention heads run in parallel, then concatenated and projected."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # final projection after concatenation
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None  # (B, H, T, T) weights, kept for visualization

    def split_heads(self, x):
        """(B, T, D) -> (B, H, T, d_k)"""
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_k)  # cut D into H pieces of d_k
        return x.transpose(1, 2)                  # move heads next to batch

    def merge_heads(self, x):
        """(B, H, T, d_k) -> (B, T, D)   (this is the concatenation)"""
        B, _, T, _ = x.shape
        x = x.transpose(1, 2).contiguous()        # (B, T, H, d_k)
        return x.view(B, T, self.n_heads * self.d_k)

    def forward(self, x, mask=None):
        """x: (B, T, D), mask: (T, T) bool or None -> (B, T, D)"""
        q = self.split_heads(self.w_q(x))  # (B, H, T, d_k)
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        out, weights = scaled_dot_product_attention(q, k, v, mask)  # each head separately
        self.last_attn = weights.detach()

        out = self.merge_heads(out)        # (B, T, D)
        return self.dropout(self.w_o(out))  # (B, T, D)
