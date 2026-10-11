"""Why Transformers: sequential vs parallel depth (lesson 01).

An RNN must process token t before token t+1 -> sequential depth grows with T.
Attention looks at all tokens in one matrix multiply -> sequential depth is constant,
but the score matrix is T x T -> memory grows with T^2.

Run:  python -m experiments.why_transformers
"""
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.attention import causal_mask, scaled_dot_product_attention

D = 128  # feature size for both models
LENGTHS = [64, 128, 256, 512, 1024]


def time_rnn(T):
    """One RNN cell applied step by step: T sequential steps."""
    cell = nn.RNNCell(D, D)
    x = torch.randn(T, 1, D)
    h = torch.zeros(1, D)
    start = time.perf_counter()
    with torch.no_grad():
        for t in range(T):  # step t needs h from step t-1, so no parallelism
            h = cell(x[t], h)
    return time.perf_counter() - start


def time_attention(T):
    """Self-attention over the whole sequence in one call."""
    x = torch.randn(1, T, D)
    mask = causal_mask(T)
    start = time.perf_counter()
    with torch.no_grad():
        scaled_dot_product_attention(x, x, x, mask)
    return time.perf_counter() - start


def best_of(fn, T, repeats=5):
    fn(T)  # warm-up
    return min(fn(T) for _ in range(repeats))


def main():
    rows = []
    print(f"{'T':>6} {'RNN time (ms)':>14} {'Attn time (ms)':>15} {'RNN seq steps':>14} {'Attn seq steps':>15} {'Attn scores (KB)':>17}")
    for T in LENGTHS:
        rnn_ms = best_of(time_rnn, T) * 1000
        att_ms = best_of(time_attention, T) * 1000
        score_kb = T * T * 4 / 1024  # T x T float32 score matrix
        rows.append((T, rnn_ms, att_ms, score_kb))
        print(f"{T:>6} {rnn_ms:>14.2f} {att_ms:>15.2f} {T:>14} {1:>15} {score_kb:>17.0f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(LENGTHS, [r[1] for r in rows], "o-", label="RNN (T sequential steps)")
    ax1.plot(LENGTHS, [r[2] for r in rows], "s-", label="Attention (1 parallel step)")
    ax1.set_xlabel("sequence length T")
    ax1.set_ylabel("time (ms, CPU)")
    ax1.set_title("Compute time")
    ax1.legend()
    ax2.plot(LENGTHS, [r[3] for r in rows], "o-", color="tab:red")
    ax2.set_xlabel("sequence length T")
    ax2.set_ylabel("score matrix size (KB)")
    ax2.set_title("Attention memory grows as T^2")
    fig.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig("outputs/figures/why_transformers.png", dpi=150)
    print("saved outputs/figures/why_transformers.png")


if __name__ == "__main__":
    main()
