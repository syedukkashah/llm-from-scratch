import pytest

from src.tokenizer import CharTokenizer

TEXT = "hello world\nhello transformer"


def test_round_trip():
    tok = CharTokenizer.from_text(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_vocab_is_sorted_and_deterministic():
    a = CharTokenizer.from_text(TEXT)
    b = CharTokenizer.from_text(TEXT[::-1])  # same characters, different order
    assert a.chars == b.chars == sorted(set(TEXT))
    assert a.vocab_size == len(set(TEXT))


def test_unknown_character_raises():
    tok = CharTokenizer.from_text(TEXT)
    with pytest.raises(ValueError):
        tok.encode("hello {")


def test_save_and_load(tmp_path):
    tok = CharTokenizer.from_text(TEXT)
    tok.save(tmp_path / "vocab.json")
    assert CharTokenizer.load(tmp_path / "vocab.json").chars == tok.chars
