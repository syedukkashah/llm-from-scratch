"""Self-attention building blocks (Milestones 1 and 2).

Notation used in every shape comment:
    B = batch size, T = sequence length (tokens), D = d_model,
    H = number of heads, d_k = D // H (size of one head)
"""
import math

import torch
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
