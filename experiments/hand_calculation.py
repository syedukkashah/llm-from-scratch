"""Worked attention example: 3 tokens, d_k = d_v = 2, computed with plain Python.

Run:  python -m experiments.hand_calculation
Produces Fig 1 (printout + figure) and Fig 2 (causal mask) for the evidence list.
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.attention import causal_mask, scaled_dot_product_attention

# ---- Chosen so the arithmetic can be checked by hand ------------------------
Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
V = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
D_K = 2
NEG_INF = float("-inf")


# ---- Tiny pure-Python linear algebra (no torch, nothing hidden) -------------
def transpose(m):
    return [list(col) for col in zip(*m)]


def matmul(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def softmax_row(row):
    m = max(row)  # subtract max for numerical safety; exp(-inf) = 0
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]


def apply_causal_mask(scores):
    """Token i may not see token j > i -> put -inf there."""
    n = len(scores)
    return [[scores[i][j] if j <= i else NEG_INF for j in range(n)] for i in range(n)]


def show(name, m):
    print(f"{name} =")
    for row in m:
        print("   [" + ", ".join(f"{x:7.4f}" for x in row) + "]")
    print()


def compute():
    """Return every intermediate matrix so tests can check them."""
    scores = matmul(Q, transpose(K))                                   # step 2: Q K^T
    scaled = [[x / math.sqrt(D_K) for x in r] for r in scores]         # step 3: / sqrt(d_k)
    weights = [softmax_row(r) for r in scaled]                         # step 4: softmax per row
    output = matmul(weights, V)                                        # step 5: weights @ V
    masked = apply_causal_mask(scaled)                                 # step 6: causal mask
    masked_weights = [softmax_row(r) for r in masked]
    masked_output = matmul(masked_weights, V)
    return dict(scores=scores, scaled=scaled, weights=weights, output=output,
                masked=masked, masked_weights=masked_weights, masked_output=masked_output)


def save_figure(r, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [("Scaled scores  QK^T/sqrt(d_k)", r["scaled"], None),
              ("Attention weights (no mask)", r["weights"], "Blues"),
              ("Attention weights (causal mask)", r["masked_weights"], "Blues")]
    for ax, (title, m, cmap) in zip(axes, panels):
        shown = [[0.0 if x == NEG_INF else x for x in row] for row in m]
        ax.imshow(shown, cmap=cmap or "viridis", vmin=0 if cmap else None, vmax=1 if cmap else None)
        for i in range(3):
            for j in range(3):
                label = "-inf" if m[i][j] == NEG_INF else f"{m[i][j]:.2f}"
                dark = cmap and m[i][j] > 0.6  # white text on dark cells
                ax.text(j, i, label, ha="center", va="center",
                        color="white" if dark else "black", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(3), ["tok1", "tok2", "tok3"])
        ax.set_yticks(range(3), ["tok1", "tok2", "tok3"])
        ax.set_xlabel("key (looked at)")
        ax.set_ylabel("query (looking)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    r = compute()
    print("Step 1: inputs (3 tokens, d_k = d_v = 2)")
    show("Q", Q); show("K", K); show("V", V)
    print("Step 2: scores = Q K^T"); show("QK^T", r["scores"])
    print(f"Step 3: divide by sqrt(d_k) = sqrt({D_K}) = {math.sqrt(D_K):.4f}"); show("scaled", r["scaled"])
    print("Step 4: row-wise softmax (each row sums to 1)"); show("weights", r["weights"])
    print("Step 5: output = weights @ V"); show("output", r["output"])
    print("Step 6: causal mask on the scaled scores (-inf = forbidden)")
    for row in r["masked"]:
        print("   [" + ", ".join("   -inf" if x == NEG_INF else f"{x:7.4f}" for x in row) + "]")
    print()
    show("masked weights", r["masked_weights"])
    show("masked output", r["masked_output"])

    print("Step 7: what changed (information available to each position)")
    print("  token 1: unmasked it blends all 3 values; masked it sees ONLY itself (weight 1.0), output = V1.")
    print("  token 2: unmasked it sees tokens 1,2,3; masked it sees tokens 1 and 2 only.")
    print("  token 3: sees everything in both cases, so its row is unchanged.")
    print("  => with the mask no token receives information from a later token.\n")

    # Cross-check: our torch implementation must reproduce the hand numbers.
    q, k, v = (torch.tensor(m) for m in (Q, K, V))
    out_free, _ = scaled_dot_product_attention(q, k, v)
    out_mask, _ = scaled_dot_product_attention(q, k, v, mask=causal_mask(3))
    ok = (torch.allclose(out_free, torch.tensor(r["output"]), atol=1e-6)
          and torch.allclose(out_mask, torch.tensor(r["masked_output"]), atol=1e-6))
    print("torch implementation matches hand calculation:", ok)

    save_figure(r, "outputs/figures/fig1_2_hand_attention.png")
    print("saved outputs/figures/fig1_2_hand_attention.png")


if __name__ == "__main__":
    main()
