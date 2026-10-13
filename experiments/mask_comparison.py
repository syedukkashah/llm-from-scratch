"""BERT-style vs GPT-style attention masks (lesson 06).

  BERT (masked LM):   every token sees every token. Some tokens are hidden with
                      [MASK] and predicted from BOTH sides.
  GPT  (causal LM):   token i sees only tokens <= i. Predicts the NEXT token.

Run:  python -m experiments.mask_comparison
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention import causal_mask
from src.positional_encoding import SinusoidalPositionalEncoding
from src.transformer_block import TransformerBlock


class TinyLM(nn.Module):
    """Small stack of blocks. causal=False removes the mask (BERT-like attention)."""

    def __init__(self, vocab, d_model=64, n_heads=4, n_layers=2, causal=True):
        super().__init__()
        self.causal = causal
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, 4 * d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx):
        """idx: (B, T) token ids -> logits (B, T, vocab)"""
        mask = causal_mask(idx.size(1)) if self.causal else None
        x = self.pos(self.embed(idx))
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))


# ---- (a) the two masks -------------------------------------------------------
def plot_masks(path):
    words = ["the", "cat", "sat", "on", "the", "mat"]
    n = len(words)
    bert = torch.ones(n, n)          # everyone sees everyone
    gpt = causal_mask(n).float()     # lower triangle only
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    titles = ["BERT / MLM: bidirectional\ninput: the cat [MASK] on the mat -> predict 'sat'",
              "GPT / causal LM: lower-triangular\nevery position predicts the NEXT token"]
    for ax, m, title in zip(axes, (bert, gpt), titles):
        ax.imshow(m, cmap="Greens", vmin=0, vmax=1.3)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, "1" if m[i, j] else "0", ha="center", va="center")
        ax.set_xticks(range(n), words)
        ax.set_yticks(range(n), words)
        ax.set_xlabel("key (token being looked at)")
        ax.set_ylabel("query (token looking)")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


# ---- (b) does changing a LATER token change an EARLIER output? ----------------
def leakage_test():
    torch.manual_seed(0)
    vocab, T, t = 16, 10, 4
    idx = torch.randint(0, vocab, (1, T))
    changed = idx.clone()
    changed[0, t + 1] = (idx[0, t + 1] + 1) % vocab  # edit the token AFTER position t
    print(f"\nChange token {t + 1}; how much do the logits at position {t} move?")
    for causal in (True, False):
        model = TinyLM(vocab, causal=causal).eval()
        with torch.no_grad():
            diff = (model(idx)[0, t] - model(changed)[0, t]).abs().max().item()
        name = "causal (GPT)      " if causal else "bidirectional(BERT)"
        print(f"  {name}: max logit change = {diff:.6f}")


# ---- (c) next-token prediction on RANDOM tokens: who can cheat? ---------------
def cheating_experiment(steps=1500, vocab=16, T=16, batch=64):
    print(f"\nNext-token loss on random tokens (guessing = ln({vocab}) = {math.log(vocab):.3f})")
    results = {}
    for causal in (True, False):
        torch.manual_seed(0)
        model = TinyLM(vocab, d_model=64, n_layers=2, causal=causal)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        curve = []
        for step in range(steps):
            seq = torch.randint(0, vocab, (batch, T + 1))
            x, y = seq[:, :-1], seq[:, 1:]  # y[t] is the token right after x[t]
            loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            curve.append(loss.item())
            if step % 300 == 0 or step == steps - 1:
                print(f"  {'causal' if causal else 'bidirectional':<14} step {step:>4}  loss {loss.item():.3f}")
        results[causal] = curve
    return results


def plot_cheating(results, path):
    plt.figure(figsize=(6, 4))
    plt.plot(results[True], label="causal (GPT mask)")
    plt.plot(results[False], label="bidirectional (no mask)")
    plt.axhline(math.log(16), color="gray", ls="--", label="guessing: ln(16)")
    plt.xlabel("step")
    plt.ylabel("next-token loss")
    plt.title("Without the causal mask the model cheats")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("saved", path)


if __name__ == "__main__":
    plot_masks("outputs/figures/fig2_mask_comparison.png")
    leakage_test()
    res = cheating_experiment()
    plot_cheating(res, "outputs/figures/mask_cheating_loss.png")
