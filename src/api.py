"""REST API for the trained model.

    uvicorn src.api:app --host 0.0.0.0 --port 8000
    curl -X POST localhost:8000/generate -H "Content-Type: application/json" \
         -d '{"prompt": "ROMEO:", "max_new_tokens": 80}'

Environment variables (both optional):
    CHECKPOINT_PATH  where the model file lives   (default checkpoints/tiny_gpt.pt)
    CHECKPOINT_URL   where to download it from if the file is missing
                     (default: the GitHub Release of this repository)
"""
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generate import generate_text
from src.utils import ensure_checkpoint, load_checkpoint, set_seed

state = {}                  # holds the loaded model, filled at startup
lock = threading.Lock()     # one generation at a time (seeding is global in PyTorch)


def load_model():
    path = os.environ.get("CHECKPOINT_PATH", "checkpoints/tiny_gpt.pt")
    ensure_checkpoint(path, os.environ.get("CHECKPOINT_URL"))
    model, tokenizer, cfg = load_checkpoint(path, "cpu")  # loaded once, reused by every request
    state.update(model=model, tokenizer=tokenizer, cfg=cfg)


@asynccontextmanager
async def lifespan(app):
    load_model()
    yield
    state.clear()


app = FastAPI(title="MiniGPT", lifespan=lifespan,
              description="A small GPT-style character-level language model trained from scratch "
                          "on Tiny Shakespeare. Generated text is not factually reliable.")


class GenerateRequest(BaseModel):
    prompt: str = Field("ROMEO:", min_length=1, max_length=500)
    max_new_tokens: int = Field(100, ge=1, le=500)
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="0 = greedy")
    top_k: Optional[int] = Field(None, ge=1)
    seed: Optional[int] = Field(None, description="set for reproducible output")


@app.get("/")
def root():
    return {"service": "MiniGPT", "docs": "/docs", "health": "/health", "generate": "POST /generate"}


@app.get("/health")
def health():
    model, tokenizer, cfg = state["model"], state["tokenizer"], state["cfg"]
    return {"status": "ok", "parameters": model.count_parameters(),
            "context_length": cfg["context_length"], "vocab_size": tokenizer.vocab_size}


@app.post("/generate")
def generate(req: GenerateRequest):
    model, tokenizer = state["model"], state["tokenizer"]
    with lock:
        if req.seed is not None:
            set_seed(req.seed)
        try:
            text = generate_text(model, tokenizer, req.prompt, req.max_new_tokens,
                                 req.temperature, req.top_k)
        except ValueError as e:  # a character the tokenizer does not know
            raise HTTPException(status_code=400,
                                detail=f"{e}. Allowed characters: {''.join(tokenizer.chars)!r}")
    return {"prompt": req.prompt, "generated": text, "continuation": text[len(req.prompt):],
            "parameters": model.count_parameters(), "temperature": req.temperature,
            "top_k": req.top_k, "seed": req.seed}
