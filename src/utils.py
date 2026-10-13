"""Small helpers: seeding, device, config, checkpoints."""
import random

import numpy as np
import torch
import yaml

from src.gpt import build_model
from src.tokenizer import CharTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path="configs/tiny_gpt.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path, model, cfg, chars):
    """One file holds everything needed for inference: weights, config and vocabulary."""
    torch.save({"model": model.state_dict(), "config": cfg, "chars": chars}, path)


def load_checkpoint(path, device="cpu"):
    """returns (model in eval mode, tokenizer, config)"""
    ckpt = torch.load(path, map_location=device)
    tokenizer = CharTokenizer(ckpt["chars"])
    model = build_model(ckpt["config"], tokenizer.vocab_size)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), tokenizer, ckpt["config"]
