"""MiniGPT: embedding + position + causal Transformer blocks + LM head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention import causal_mask
from src.positional_encoding import SinusoidalPositionalEncoding
from src.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, n_heads, n_layers,
                 ffn_dim, dropout=0.0, causal=True, tie_weights=False):
        super().__init__()
        self.context_length = context_length
        self.causal = causal  # False = no mask (BERT-like attention), only for experiments
        self.embed = nn.Embedding(vocab_size, d_model)             # token id -> vector
        self.pos = SinusoidalPositionalEncoding(d_model, context_length)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)              # vector -> a score per token
        if tie_weights:
            self.lm_head.weight = self.embed.weight                # share embedding and output weights

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids, targets: (B, T) next-token ids or None.

        returns logits (B, T, vocab) and the loss (None when no targets are given).
        """
        B, T = idx.shape
        assert T <= self.context_length, "sequence longer than context_length"
        mask = causal_mask(T, idx.device) if self.causal else None

        x = self.drop(self.pos(self.embed(idx)))  # (B, T, D)
        for block in self.blocks:
            x = block(x, mask)                    # (B, T, D)
        logits = self.lm_head(self.ln_f(x))       # (B, T, vocab)

        loss = None
        if targets is not None:
            # flatten so every position is one classification over the vocabulary
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())  # shared (tied) weights counted once

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """Autoregressive sampling. idx: (B, T) starting tokens -> (B, T + max_new_tokens)."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.context_length:]    # keep only the last context_length tokens
            logits, _ = self(context)
            logits = logits[:, -1, :]                  # (B, vocab): scores for the NEXT token
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)  # greedy
            else:
                logits = logits / temperature          # <1 sharper, >1 flatter
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx


def build_model(cfg, vocab_size):
    """Create a GPT from a config dict (see configs/tiny_gpt.yaml)."""
    return GPT(
        vocab_size=vocab_size,
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        ffn_dim=cfg["ffn_dim"],
        dropout=cfg["dropout"],
        tie_weights=cfg.get("tie_weights", False),
    )
