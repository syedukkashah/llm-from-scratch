"""Visualize sinusoidal positional encoding (Fig 4).

Run:  python -m experiments.positional_viz
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.positional_encoding import SinusoidalPositionalEncoding

D, MAX_LEN = 128, 100


def main():
    pe = SinusoidalPositionalEncoding(D, MAX_LEN).pe  # (MAX_LEN, D)
    sim = pe @ pe.T  # sim[i, j] = dot product of position i and position j

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im = axes[0].imshow(pe, aspect="auto", cmap="RdBu")
    axes[0].set_title("PE matrix (rows = position, cols = dimension)")
    axes[0].set_xlabel("dimension")
    axes[0].set_ylabel("position")
    fig.colorbar(im, ax=axes[0])

    for dim in [0, 8, 32, 64]:  # fast to slow oscillation
        axes[1].plot(pe[:, dim], label=f"dim {dim}")
    axes[1].set_title("Low dims oscillate fast, high dims slowly")
    axes[1].set_xlabel("position")
    axes[1].legend()

    axes[2].plot(sim[0], label="position 0")
    axes[2].plot(sim[50], label="position 50")
    axes[2].set_title("Similarity (dot product) vs position")
    axes[2].set_xlabel("other position")
    axes[2].legend()

    fig.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig("outputs/figures/fig4_positional_encoding.png", dpi=150)
    print("saved outputs/figures/fig4_positional_encoding.png")
    print("PE[0, :6] =", [round(v, 3) for v in pe[0, :6].tolist()])
    print("PE[1, :6] =", [round(v, 3) for v in pe[1, :6].tolist()])
    print("similarity(0, k) for k=0,1,5,20,50:",
          [round(sim[0, k].item(), 1) for k in (0, 1, 5, 20, 50)])


if __name__ == "__main__":
    main()
