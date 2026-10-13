"""Character-level tokenizer (deterministic).

Vocabulary = the sorted unique characters of the training text.
Special-token policy: none. A character outside the vocabulary raises an error.
"""
import json


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)  # index in this list = token id
        self.stoi = {c: i for i, c in enumerate(self.chars)}  # char -> id
        self.itos = {i: c for i, c in enumerate(self.chars)}  # id -> char

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))  # sorted -> same vocab every run

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        """str -> list of token ids"""
        unknown = sorted({c for c in text if c not in self.stoi})
        if unknown:
            raise ValueError(f"characters not in vocabulary: {unknown}")
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        """list of token ids -> str"""
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chars, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
