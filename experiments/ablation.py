"""Controlled ablation experiments: change ONE factor, keep everything else fixed.

    python -m experiments.ablation --experiment depth       # 2 vs 4 vs 6 layers
    python -m experiments.ablation --experiment heads       # 2 vs 4 vs 8 heads
    python -m experiments.ablation --experiment context     # 64 vs 128 vs 256
    python -m experiments.ablation --experiment position    # sinusoidal vs learned
    python -m experiments.ablation --experiment sampling    # greedy vs temperature (no training)
    python -m experiments.ablation --experiment all

Every run uses the same seed, steps, data and evaluation batches. The baseline run
(the config value of each factor) is trained once and reused. Results go to
outputs/logs/ablation_<name>.csv, outputs/logs/ablation_table.md and outputs/figures/.
"""
import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import load_text
from src.generate import generate_text
from src.train import train
from src.utils import get_device, load_checkpoint, load_config, set_seed

EXPERIMENTS = {  # name -> (config key that changes, values to compare)
    "depth": ("n_layers", [2, 4, 6]),
    "heads": ("n_heads", [2, 4, 8]),
    "context": ("context_length", [64, 128, 256]),
    "position": ("position", ["sinusoidal", "learned"]),
}
LOG_DIR = "outputs/logs/ablation"
FIELDS = ["factor", "value", "params", "train_loss", "val_loss", "perplexity",
          "train_time_s", "tokens_per_sec", "peak_memory_mb"]


# ---- training ablations ------------------------------------------------------
def run_one(base_cfg, name, overrides, text, steps, eval_interval, eval_iters):
    """Train one configuration and return its metrics."""
    cfg = dict(base_cfg, **overrides, max_steps=steps, eval_interval=eval_interval,
               eval_iters=eval_iters, sample_tokens=50)
    s = train(cfg, name=name, text=text, save=False, verbose=False, log_dir=LOG_DIR)
    return {"params": s["parameters"], "train_loss": s["final_train_loss"],
            "val_loss": s["final_val_loss"], "perplexity": s["val_perplexity"],
            "train_time_s": s["train_time_s"], "tokens_per_sec": s["tokens_per_sec"],
            "peak_memory_mb": s["peak_memory_mb"]}


def run_experiment(exp, base_cfg, text, args, cache):
    factor, values = EXPERIMENTS[exp]
    rows = []
    for value in values:
        overrides = {} if value == base_cfg[factor] else {factor: value}
        key = repr(sorted(overrides.items()))  # the baseline is shared between experiments
        if key not in cache:
            name = "baseline" if not overrides else f"{factor}_{value}"
            print(f"training {name} ({args.steps} steps) ...", flush=True)
            cache[key] = run_one(base_cfg, name, overrides, text, args.steps,
                                 args.eval_interval, args.eval_iters)
            print("  ", cache[key], flush=True)
        rows.append({"factor": factor, "value": value, **cache[key]})
    write_csv(f"outputs/logs/ablation_{exp}.csv", rows)
    plot_rows(rows, f"{exp}: {factor}", f"outputs/figures/ablation_{exp}.png")
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, title, path):
    panels = [("val_loss", "validation loss"), ("train_time_s", "training time (s)")]
    if rows[0]["peak_memory_mb"] is not None:
        panels.append(("peak_memory_mb", "peak GPU memory (MB)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.8))
    labels = [str(r["value"]) for r in rows]
    for ax, (key, label) in zip(axes, panels):
        bars = ax.bar(labels, [r[key] for r in rows], color="tab:blue")
        ax.bar_label(bars, fmt="%.3g")
        ax.set_title(label)
        ax.set_xlabel(rows[0]["factor"])
    if panels[0][0] == "val_loss":  # zoom in so small differences are visible
        low = min(r["val_loss"] for r in rows)
        high = max(r["val_loss"] for r in rows)
        axes[0].set_ylim(max(0, low - 0.1), high + 0.05)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_table():
    """Combine every ablation_<exp>.csv that exists into one markdown table."""
    lines = []
    for exp in EXPERIMENTS:
        path = f"outputs/logs/ablation_{exp}.csv"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        lines += [f"### {exp} ({rows[0]['factor']})", "",
                  "| value | params | train loss | val loss | perplexity | train time (s) | tokens/s | peak mem (MB) |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            mem = r["peak_memory_mb"] or "n/a"
            lines.append(f"| {r['value']} | {int(r['params']):,} | {r['train_loss']} | {r['val_loss']} | "
                         f"{r['perplexity']} | {r['train_time_s']} | {r['tokens_per_sec']} | {mem} |")
        lines.append("")
    if os.path.exists("outputs/logs/ablation_sampling.csv"):
        with open("outputs/logs/ablation_sampling.csv") as f:
            rows = list(csv.DictReader(f))
        lines += ["### sampling (no retraining)", "",
                  "| setting | real-word rate | repetition | distinct samples (of 5) |", "|---|---|---|---|"]
        lines += [f"| {r['value']} | {r['real_word_rate']} | {r['repetition']} | {r['distinct_samples']} |" for r in rows]
        lines.append("")
    with open("outputs/logs/ablation_table.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# ---- sampling ablation (no training) -----------------------------------------
def word_metrics(continuations, train_words):
    """Simple text-quality numbers for a list of generated continuations."""
    words = [re.findall(r"[a-z']+", c.lower()) for c in continuations]
    all_words = [w for ws in words for w in ws]
    real = sum(w in train_words for w in all_words) / max(1, len(all_words))
    repeats = []
    for ws in words:
        pairs = list(zip(ws, ws[1:]))
        repeats.append(1 - len(set(pairs)) / max(1, len(pairs)))  # share of repeated word pairs
    return {"real_word_rate": round(real, 3),
            "repetition": round(sum(repeats) / len(repeats), 3),
            "distinct_samples": len(set(continuations))}


def sampling_experiment(checkpoint, n=5, length=300, prompt="ROMEO:\n"):
    device = get_device()
    model, tokenizer, cfg = load_checkpoint(checkpoint, device)
    train_words = set(re.findall(r"[a-z']+", load_text(cfg["data_path"]).lower()))
    settings = [("greedy (temp 0)", dict(temperature=0)),
                ("temp 0.5", dict(temperature=0.5)),
                ("temp 0.8", dict(temperature=0.8)),
                ("temp 1.2", dict(temperature=1.2)),
                ("temp 0.8 + top-k 10", dict(temperature=0.8, top_k=10))]
    rows, examples = [], []
    for name, kw in settings:
        set_seed(0)  # same random stream for every setting
        texts = [generate_text(model, tokenizer, prompt, length, **kw)[len(prompt):] for _ in range(n)]
        rows.append({"factor": "sampling", "value": name, **word_metrics(texts, train_words)})
        examples.append((name, texts[0][:250]))
    write_csv("outputs/logs/ablation_sampling.csv", rows)
    os.makedirs("outputs/samples", exist_ok=True)
    with open("outputs/samples/sampling_comparison.md", "w", encoding="utf-8") as f:
        f.write(f"# Sampling comparison\n\nprompt `{prompt!r}`, {n} samples of {length} characters per setting\n\n")
        for name, text in examples:
            f.write(f"## {name}\n\n```\n{text}\n```\n\n")
    print(f"{'setting':<22}{'real words':>12}{'repetition':>12}{'distinct':>10}")
    for r in rows:
        print(f"{r['value']:<22}{r['real_word_rate']:>12}{r['repetition']:>12}{r['distinct_samples']:>8}/{n}")
    plot_sampling(rows, "outputs/figures/ablation_sampling.png")
    return rows


def plot_sampling(rows, path):
    labels = [r["value"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, title in zip(axes, ["real_word_rate", "repetition"],
                              ["share of generated words found in the training text", "share of repeated word pairs"]):
        bars = ax.bar(labels, [r[key] for r in rows], color="tab:green")
        ax.bar_label(bars, fmt="%.2f")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="all", choices=list(EXPERIMENTS) + ["sampling", "all"])
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt", help="used by the sampling experiment")
    args = p.parse_args()

    base_cfg = load_config(args.config)
    text = load_text(base_cfg["data_path"])
    names = list(EXPERIMENTS) + ["sampling"] if args.experiment == "all" else [args.experiment]
    cache = {}
    for name in names:
        print(f"\n=== {name} ===")
        if name == "sampling":
            sampling_experiment(args.checkpoint)
        else:
            run_experiment(name, base_cfg, text, args, cache)
    write_table()


if __name__ == "__main__":
    main()
