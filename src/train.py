"""Training loop.

    python -m src.train                          # full run with configs/tiny_gpt.yaml
    python -m src.train --n_layers 6             # override any config value
    python -m src.train --overfit                # debugging gate: memorize one tiny batch
"""
import argparse
import csv
import json
import math
import os
import platform
import time

import torch

from src.dataset import get_batch, load_text, split_data
from src.gpt import build_model
from src.tokenizer import CharTokenizer
from src.utils import get_device, load_config, save_checkpoint, set_seed


def get_lr(step, cfg):
    """Linear warm-up, then cosine decay from lr down to min_lr."""
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))  # 1 -> 0
    return cfg["min_lr"] + cosine * (cfg["lr"] - cfg["min_lr"])


@torch.no_grad()
def estimate_loss(model, data, cfg, device):
    """Average loss over eval_iters batches. The batches are the same every call."""
    model.eval()
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(cfg["eval_iters"]):
        x, y = get_batch(data, cfg["batch_size"], cfg["context_length"], device, gen)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def sample_text(model, tokenizer, cfg, device):
    start = torch.tensor([tokenizer.encode(cfg["sample_prompt"])], device=device)
    out = model.generate(start, cfg["sample_tokens"], cfg["temperature"])
    return tokenizer.decode(out[0].tolist())


def hardware_name(device):
    return torch.cuda.get_device_name(0) if device == "cuda" else f"CPU ({platform.processor() or platform.machine()})"


def train(cfg, name="tiny_gpt", text=None, save=True, verbose=True,
          ckpt_dir="checkpoints", log_dir="outputs/logs"):
    """Train a model, log to CSV, keep the best checkpoint. Returns a summary dict."""
    set_seed(cfg["seed"])
    device = get_device()
    text = text if text is not None else load_text(cfg["data_path"])
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = split_data(data, cfg["train_frac"])

    model = build_model(cfg, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n_params = model.count_parameters()
    if verbose:
        print(f"device: {hardware_name(device)} | vocab: {tokenizer.vocab_size} | parameters: {n_params:,}")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}_log.csv")
    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])
    samples_path = os.path.join(log_dir, f"{name}_samples.txt")
    open(samples_path, "w").close()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_val, start = float("inf"), time.time()
    train_loss = val_loss = float("nan")
    train_time = 0.0  # time spent training only (evaluation excluded) for throughput

    for step in range(cfg["max_steps"] + 1):
        # evaluate, log and sample every eval_interval steps (and at the end)
        if step % cfg["eval_interval"] == 0 or step == cfg["max_steps"]:
            train_loss = estimate_loss(model, train_data, cfg, device)
            val_loss = estimate_loss(model, val_data, cfg, device)
            lr = get_lr(min(step, cfg["max_steps"] - 1), cfg)
            elapsed = time.time() - start
            log.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{lr:.6f}", f"{elapsed:.1f}"])
            log_file.flush()
            sample = sample_text(model, tokenizer, cfg, device)
            with open(samples_path, "a", encoding="utf-8") as f:
                f.write(f"--- step {step} (val loss {val_loss:.3f}) ---\n{sample}\n\n")
            if verbose:
                print(f"step {step:>5} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.5f} | {elapsed:.0f}s")
                print("  sample:", repr(sample[:120]))
            if val_loss < best_val:
                best_val = val_loss
                if save:
                    save_checkpoint(os.path.join(ckpt_dir, f"{name}.pt"), model, cfg, tokenizer.chars)
        if step == cfg["max_steps"]:
            break

        t0 = time.time()
        for group in optimizer.param_groups:
            group["lr"] = get_lr(step, cfg)
        x, y = get_batch(train_data, cfg["batch_size"], cfg["context_length"], device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        train_time += time.time() - t0

    log_file.close()
    tokens = cfg["max_steps"] * cfg["batch_size"] * cfg["context_length"]
    summary = {
        "name": name,
        "parameters": n_params,
        "hardware": hardware_name(device),
        "wall_clock_s": round(time.time() - start, 1),
        "train_time_s": round(train_time, 1),
        "tokens_per_sec": round(tokens / max(train_time, 1e-9)),
        "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
        "final_train_loss": round(train_loss, 4),
        "final_val_loss": round(val_loss, 4),
        "best_val_loss": round(best_val, 4),
        "val_perplexity": round(math.exp(val_loss), 3),
        "config": cfg,
    }
    if save:
        with open(os.path.join(log_dir, f"{name}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    if verbose:
        print({k: v for k, v in summary.items() if k != "config"})
    return summary


def overfit(cfg, text, steps=200):
    """Debugging gate: train on ONE small batch repeatedly. Loss must go way down."""
    set_seed(cfg["seed"])
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    cfg = dict(cfg, dropout=0.0)  # dropout would stop it from memorizing exactly
    model = build_model(cfg, tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    x, y = get_batch(data, 4, cfg["context_length"])  # the one fixed batch
    losses = []
    for _ in range(steps):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def parse_args(cfg):
    """--config/--name/--overfit plus one optional flag per config key."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--name", default="tiny_gpt")
    p.add_argument("--overfit", action="store_true", help="memorize one tiny batch (debug gate)")
    for key, value in cfg.items():
        kind = (lambda s: s.lower() == "true") if isinstance(value, bool) else type(value)
        p.add_argument(f"--{key}", type=kind, default=None)
    return p.parse_args()


def main():
    first = argparse.ArgumentParser(add_help=False)  # read --config first to know the keys
    first.add_argument("--config", default="configs/tiny_gpt.yaml")
    cfg = load_config(first.parse_known_args()[0].config)
    args = parse_args(cfg)
    for key in cfg:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    if args.overfit:
        losses = overfit(cfg, load_text(cfg["data_path"]))
        print(f"overfit one batch: loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    else:
        train(cfg, name=args.name)


if __name__ == "__main__":
    main()
