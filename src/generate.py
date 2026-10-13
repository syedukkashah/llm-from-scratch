"""Generate text from a trained checkpoint.

    python -m src.generate --prompt "ROMEO:" --max-new-tokens 80
"""
import argparse
import sys

import torch

from src.utils import get_device, load_checkpoint, set_seed


def generate_text(model, tokenizer, prompt, max_new_tokens=80, temperature=0.8, top_k=None):
    """prompt string -> prompt + generated continuation (string)."""
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)  # (1, T)
    out = model.generate(idx, max_new_tokens, temperature, top_k)
    return tokenizer.decode(out[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="set for reproducible output")
    args = p.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
    model, tokenizer, cfg = load_checkpoint(args.checkpoint, get_device())
    try:
        text = generate_text(model, tokenizer, args.prompt, args.max_new_tokens,
                             args.temperature, args.top_k)
    except ValueError as e:  # prompt contains a character the tokenizer does not know
        sys.exit(f"error: {e}")

    print(f"Prompt: {args.prompt}")
    print(f"Generated: {text}\n")
    print(f"model parameters: {model.count_parameters() / 1e6:.1f}M")
    print(f"context length: {cfg['context_length']}")
    print(f"temperature: {args.temperature}")


if __name__ == "__main__":
    main()
