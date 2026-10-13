import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.gpt import build_model
from src.tokenizer import CharTokenizer
from src.utils import save_checkpoint

CFG = dict(context_length=16, d_model=32, n_heads=4, n_layers=2, ffn_dim=128,
           dropout=0.0, tie_weights=False)


@pytest.fixture
def client(tmp_path, monkeypatch):
    tok = CharTokenizer.from_text("hello world, to be or not")
    path = tmp_path / "m.pt"
    save_checkpoint(path, build_model(CFG, tok.vocab_size), CFG, tok.chars)
    monkeypatch.setenv("CHECKPOINT_PATH", str(path))
    with TestClient(app) as c:  # runs startup, which loads the model
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["context_length"] == 16


def test_generate_returns_prompt_plus_new_tokens(client):
    r = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 12})
    assert r.status_code == 200
    body = r.json()
    assert body["generated"].startswith("hello")
    assert len(body["generated"]) == len("hello") + 12
    assert body["continuation"] == body["generated"][5:]


def test_same_seed_gives_same_text_and_greedy_is_deterministic(client):
    req = {"prompt": "to be", "max_new_tokens": 20, "seed": 3}
    assert client.post("/generate", json=req).json()["generated"] == client.post("/generate", json=req).json()["generated"]
    greedy = {"prompt": "to be", "max_new_tokens": 20, "temperature": 0}
    assert client.post("/generate", json=greedy).json()["generated"] == client.post("/generate", json=greedy).json()["generated"]


def test_unknown_character_is_a_400_with_help(client):
    r = client.post("/generate", json={"prompt": "hello {", "max_new_tokens": 5})
    assert r.status_code == 400
    assert "Allowed characters" in r.json()["detail"]


def test_limits_are_enforced(client):
    assert client.post("/generate", json={"prompt": "hello", "max_new_tokens": 100000}).status_code == 422
    assert client.post("/generate", json={"prompt": "hello", "temperature": 5}).status_code == 422
    assert client.post("/generate", json={"prompt": ""}).status_code == 422
