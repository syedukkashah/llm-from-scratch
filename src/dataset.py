"""Tiny Shakespeare data pipeline: download, split, and next-token batches."""
import os
import urllib.request

import torch

# Same text as the Hugging Face "tiny_shakespeare" dataset (Karpathy's char-rnn copy).
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path="data/input.txt"):
    """Read the corpus, downloading it first if it is not on disk."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_data(data, train_frac=0.9):
    """data: 1-D tensor of token ids -> (train, val). Val is the contiguous tail."""
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(data, batch_size, context_length, device="cpu", generator=None):
    """Random windows of the corpus and the same windows shifted by one.

    returns x, y of shape (batch_size, context_length)
        x = data[i     : i + T]
        y = data[i + 1 : i + T + 1]     y[:, t] is the token that FOLLOWS x[:, t]
    """
    starts = torch.randint(len(data) - context_length - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + context_length] for i in starts])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in starts])  # the shift
    return x.to(device), y.to(device)
