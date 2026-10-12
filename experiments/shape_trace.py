"""Trace tensor shapes through multi-head attention (Fig 3).

Run:  python -m experiments.shape_trace
"""
import math

import torch

from src.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention

B, T, D, H = 2, 6, 32, 4  # batch, tokens, d_model, heads  ->  d_k = 8


def step(name, t, note=""):
    print(f"{name:<34} {str(tuple(t.shape)):<18} {note}")


def main():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=D, n_heads=H).eval()
    x = torch.randn(B, T, D)
    print(f"B={B} T={T} D={D} H={H} d_k={D // H}\n")
    print(f"{'step':<34} {'shape':<18} meaning")
    print("-" * 78)

    step("input x", x, "(B, T, D)")
    q_full, k_full, v_full = mha.w_q(x), mha.w_k(x), mha.w_v(x)
    step("Q = x W_q", q_full, "linear projection, same shape")
    step("K = x W_k", k_full)
    step("V = x W_v", v_full)

    q, k, v = mha.split_heads(q_full), mha.split_heads(k_full), mha.split_heads(v_full)
    step("split heads -> Q", q, "(B, H, T, d_k): D cut into H pieces")
    step("split heads -> K", k)
    step("split heads -> V", v)

    scores = q @ k.transpose(-2, -1) / math.sqrt(D // H)
    step("scores = Q K^T / sqrt(d_k)", scores, "one T x T table per head")
    out, weights = scaled_dot_product_attention(q, k, v, causal_mask(T))
    step("weights = softmax(masked scores)", weights, "rows sum to 1, future = 0")
    step("per-head output = weights @ V", out, "each head has its own d_k output")

    merged = mha.merge_heads(out)
    step("merge heads (concatenate)", merged, "(B, T, D): heads side by side")
    final = mha.w_o(merged)
    step("output = merged W_o", final, "final projection, back to (B, T, D)")

    assert torch.allclose(final, mha(x, causal_mask(T)), atol=1e-6)
    print("\nstep-by-step result matches mha(x): True")

    print("\nAttention weights, batch 0, token 4 (row 4) for each head:")
    for h in range(H):
        row = ", ".join(f"{w:.2f}" for w in weights[0, h, 4])
        print(f"  head {h}: [{row}]   (last position 5 is masked)")


if __name__ == "__main__":
    main()
