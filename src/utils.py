"""Small helpers: seeding, device, config, checkpoints."""
import os
import random
import urllib.request

import numpy as np
import torch
import yaml

from src.gpt import build_model
from src.tokenizer import CharTokenizer


DEFAULT_CHECKPOINT_URL = ("https://github.com/syedukkashah/llm-from-scratch/"
                          "releases/download/v1.0/tiny_gpt.pt")


def ensure_checkpoint(path, url=None):
    """Return path; download the released checkpoint first if the file is missing."""
    path = str(path)
    if not os.path.exists(path):
        url = url or DEFAULT_CHECKPOINT_URL
        print(f"{path} not found, downloading {url} ...")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)  # only keep the file once it fully downloaded
    return path


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
