"""Evaluate a trained model: loss, perplexity, fixed-prompt samples, curves, summary.

    python -m experiments.evaluate --checkpoint checkpoints/tiny_gpt.pt --name tiny_gpt
Writes outputs/samples/samples.md, outputs/logs/model_summary.txt,
outputs/figures/fig6_loss_curves.png
"""
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.dataset import load_text, split_data
from src.generate import generate_text
from src.utils import get_device, load_checkpoint, set_seed

# Fixed prompts, same for every evaluation, so samples are comparable.
PROMPTS = ["ROMEO:", "To be, or not to be", "KING HENRY:\n", "First Citizen:\n", "O, my love"]


@torch.no_grad()
def evaluate_split(model, data, context_length, batch_size=64, max_windows=None):
    """Mean next-token loss over consecutive, non-overlapping windows of `data`."""
    device = next(model.parameters()).device
    n = (len(data) - 1) // context_length  # number of full windows
    if max_windows:
        n = min(n, max_windows)
    x = torch.stack([data[i * context_length : (i + 1) * context_length] for i in range(n)])
    y = torch.stack([data[i * context_length + 1 : (i + 1) * context_length + 1] for i in range(n)])
    total, count = 0.0, 0
    for i in range(0, n, batch_size):
        xb, yb = x[i : i + batch_size].to(device), y[i : i + batch_size].to(device)
        _, loss = model(xb, yb)
        total += loss.item() * len(xb)
        count += len(xb)
    return total / count


def model_summary(model):
    """Parameters per component (what the architecture looks like in numbers)."""
    lines = [f"{'component':<28}{'parameters':>12}", "-" * 40]
    groups = [("token embedding", model.embed), ("final LayerNorm", model.ln_f), ("lm_head", model.lm_head)]
    for i, block in enumerate(model.blocks):
        groups.insert(1 + i, (f"block {i} (attn+ffn+2 LN)", block))
    for name, module in groups:
        lines.append(f"{name:<28}{sum(p.numel() for p in module.parameters()):>12,}")
    lines += ["-" * 40, f"{'total':<28}{model.count_parameters():>12,}",
              ("(sinusoidal position table: %s, a buffer, not trained)" % (tuple(model.pos.pe.shape),))
              if hasattr(model.pos, "pe") else "(learned position embedding is included in the total)"]
    return "\n".join(lines)


def plot_curves(log_path, out_path):
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, [float(r["train_loss"]) for r in rows], label="train")
    plt.plot(steps, [float(r["val_loss"]) for r in rows], label="validation")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title("Training vs validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    p.add_argument("--name", default="tiny_gpt", help="run name, used to find outputs/logs/<name>_log.csv")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = get_device()
    model, tokenizer, cfg = load_checkpoint(args.checkpoint, device)
    text = load_text(cfg["data_path"])
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = split_data(data, cfg["train_frac"])
    ctx = cfg["context_length"]

    val_loss = evaluate_split(model, val_data, ctx)
    n_val = (len(val_data) - 1) // ctx
    train_loss = evaluate_split(model, train_data, ctx, max_windows=n_val)  # same number of windows
    print(f"validation loss {val_loss:.4f}  perplexity {math.exp(val_loss):.2f}")
    print(f"train loss      {train_loss:.4f}  perplexity {math.exp(train_loss):.2f}")
    print(f"(guessing would be loss {math.log(tokenizer.vocab_size):.4f}, perplexity {tokenizer.vocab_size})")

    os.makedirs("outputs/samples", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    summary = model_summary(model)
    print("\n" + summary)
    with open("outputs/logs/model_summary.txt", "w") as f:
        f.write(summary + "\n")

    set_seed(args.seed)  # fixed seed -> the same samples every run
    with open("outputs/samples/samples.md", "w", encoding="utf-8") as f:
        f.write(f"# Generated samples\n\ncheckpoint: `{args.checkpoint}` | temperature {args.temperature} "
                f"| seed {args.seed} | parameters {model.count_parameters():,}\n\n"
                f"validation loss {val_loss:.4f} (perplexity {math.exp(val_loss):.2f})\n")
        for prompt in PROMPTS:
            sample = generate_text(model, tokenizer, prompt, args.max_new_tokens, args.temperature)
            f.write(f"\n## Prompt: `{prompt!r}`\n\n```\n{sample}\n```\n")
            print(f"\n--- prompt {prompt!r} ---\n{sample}")
    print("\nsaved outputs/samples/samples.md and outputs/logs/model_summary.txt")

    log_path = f"outputs/logs/{args.name}_log.csv"
    if os.path.exists(log_path):
        plot_curves(log_path, "outputs/figures/fig6_loss_curves.png")
        print("saved outputs/figures/fig6_loss_curves.png")


if __name__ == "__main__":
    main()
