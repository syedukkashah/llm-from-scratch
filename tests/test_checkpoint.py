import torch

from src.gpt import build_model
from src.tokenizer import CharTokenizer
from src.utils import load_checkpoint, save_checkpoint

CFG = dict(context_length=16, d_model=32, n_heads=4, n_layers=2, ffn_dim=128,
           dropout=0.0, tie_weights=False)


def test_reloaded_model_gives_identical_logits(tmp_path):
    tok = CharTokenizer.from_text("hello world")
    model = build_model(CFG, tok.vocab_size).eval()
    save_checkpoint(tmp_path / "m.pt", model, CFG, tok.chars)

    loaded, tok2, cfg2 = load_checkpoint(tmp_path / "m.pt")
    idx = torch.tensor([tok.encode("hello")])
    with torch.no_grad():
        assert torch.equal(model(idx)[0], loaded(idx)[0])
    assert tok2.chars == tok.chars and cfg2 == CFG


def test_reloaded_model_can_generate(tmp_path):
    tok = CharTokenizer.from_text("hello world")
    save_checkpoint(tmp_path / "m.pt", build_model(CFG, tok.vocab_size), CFG, tok.chars)
    loaded, tok2, _ = load_checkpoint(tmp_path / "m.pt")
    out = loaded.generate(torch.tensor([tok2.encode("he")]), max_new_tokens=5, temperature=0.8)
    assert len(tok2.decode(out[0].tolist())) == 7
