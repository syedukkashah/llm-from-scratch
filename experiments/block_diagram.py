"""Draw the Transformer block with tensor shapes, and check them (lesson 05).

Run:  python -m experiments.block_diagram
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.attention import causal_mask
from src.transformer_block import TransformerBlock

B, T, D, H, FFN = 2, 10, 128, 4, 512


def check_shapes():
    block = TransformerBlock(D, H, FFN)
    x = torch.randn(B, T, D)
    h = block.ln1(x)
    print("input                 ", tuple(x.shape))
    print("after LayerNorm       ", tuple(h.shape))
    a = block.attn(h, causal_mask(T))
    print("attention output      ", tuple(a.shape))
    x = x + a
    print("after residual 1      ", tuple(x.shape))
    f = block.ffn(block.ln2(x))
    print("feed-forward output   ", tuple(f.shape))
    x = x + f
    print("after residual 2      ", tuple(x.shape))
    assert x.shape == (B, T, D)
    print("all shapes (B, T, D) preserved: True")


def draw(path):
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)

    def box(y, text, color="#cfe2f3"):
        ax.add_patch(plt.Rectangle((1.8, y), 6.4, 1.2, fc=color, ec="black"))
        ax.text(5, y + 0.6, text, ha="center", va="center", fontsize=10)

    def arrow(y0, y1):
        ax.annotate("", xy=(5, y1), xytext=(5, y0), arrowprops=dict(arrowstyle="->"))

    def plus(y):
        ax.add_patch(plt.Circle((5, y), 0.4, fc="white", ec="black"))
        ax.text(5, y, "+", ha="center", va="center", fontsize=14)

    def skip(y0, y1):  # residual path drawn on the side
        ax.plot([5, 8.6, 8.6, 5.4], [y0, y0, y1, y1], color="tab:red", lw=1.5)
        ax.annotate("", xy=(5.4, y1), xytext=(6, y1), arrowprops=dict(arrowstyle="->", color="tab:red"))
        ax.text(8.8, (y0 + y1) / 2, "residual", rotation=90, va="center", color="tab:red", fontsize=9)

    ax.text(5, 15.5, f"input x   (B, T, {D})", ha="center")
    arrow(15.2, 14.4)
    box(13.2, "LayerNorm")
    arrow(13.2, 12.4)
    box(11.2, f"Multi-Head Attention\n{H} heads, causal mask", "#fce5cd")
    arrow(11.2, 10.4)
    plus(10)
    skip(14.6, 10)
    arrow(9.6, 8.8)
    box(7.6, "LayerNorm")
    arrow(7.6, 6.8)
    box(5.4, f"Feed-Forward\n{D} -> {FFN} -> GELU -> {D}", "#d9ead3")
    arrow(5.4, 4.6)
    plus(4.2)
    skip(9.0, 4.2)
    arrow(3.8, 3.0)
    ax.text(5, 2.6, f"output   (B, T, {D})", ha="center")
    ax.set_title("Decoder-style Transformer block (pre-LayerNorm)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    check_shapes()
    draw("outputs/figures/transformer_block.png")
    print("saved outputs/figures/transformer_block.png")
