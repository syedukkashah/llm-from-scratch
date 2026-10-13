"""Look inside a trained model: attention maps for every layer and head (Fig 7).

    python -m experiments.attention_visualization --checkpoint checkpoints/tiny_gpt.pt
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils import ensure_checkpoint, get_device, load_checkpoint

PROMPT = "ROMEO:\nO, wherefore art thou"


def pretty(ch):
    return {"\n": "\\n", " ": "_"}.get(ch, ch)  # make newline/space visible on the axis


@torch.no_grad()
def get_attention(model, tokenizer, prompt):
    """returns a list (one per layer) of (H, T, T) attention weights for the prompt."""
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    model(idx)  # each MultiHeadAttention stores its weights in .last_attn
    return [block.attn.last_attn[0].cpu() for block in model.blocks]


def plot_attention(maps, labels, path):
    n_layers, n_heads = len(maps), maps[0].shape[0]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(3.2 * n_heads, 3.2 * n_layers))
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l][h]
            ax.imshow(maps[l][h], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"layer {l}, head {h}", fontsize=9)
            ax.set_xticks(range(len(labels)), labels, fontsize=5)
            ax.set_yticks(range(len(labels)), labels, fontsize=5)
    fig.suptitle("Attention weights (row = token looking, column = token looked at)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def head_statistics(maps):
    """Average attention each head puts on itself, the previous token, and the first token."""
    print(f"\n{'head':<14}{'self':>8}{'prev token':>12}{'first token':>13}")
    for l, m in enumerate(maps):
        T = m.shape[-1]
        for h in range(m.shape[0]):
            a = m[h]
            self_w = a.diagonal().mean().item()
            prev_w = a.diagonal(offset=-1).mean().item()  # token t looking at t-1
            first_w = a[1:, 0].mean().item()              # tokens looking at position 0
            print(f"layer {l} head {h:<2}{self_w:>8.2f}{prev_w:>12.2f}{first_w:>13.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--out", default="outputs/figures/fig7_attention.png")
    args = p.parse_args()

    ensure_checkpoint(args.checkpoint)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, get_device())
    maps = get_attention(model, tokenizer, args.prompt)
    labels = [pretty(c) for c in args.prompt]
    plot_attention(maps, labels, args.out)
    print(f"saved {args.out}  ({len(maps)} layers x {maps[0].shape[0]} heads, {len(labels)} tokens)")
    head_statistics(maps)


if __name__ == "__main__":
    main()
