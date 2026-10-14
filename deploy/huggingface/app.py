"""Hugging Face Space entry point (Gradio SDK).

Serves the FastAPI service from src/api.py (GET /health, POST /generate, /docs) and a small
Gradio page at "/" that calls the same generate function.
"""
import gradio as gr
import uvicorn
from fastapi import HTTPException
from pydantic import ValidationError

from src.api import GenerateRequest, app, generate


# ZeroGPU hardware refuses to start an app without a @spaces.GPU function.
# The model runs on CPU, so this placeholder is never called.
try:
    import spaces

    @spaces.GPU
    def gpu_placeholder():
        return 0
except ImportError:   # CPU hardware or running locally
    pass


def run(prompt, max_new_tokens, temperature, top_k, seed):
    try:
        req = GenerateRequest(prompt=prompt, max_new_tokens=int(max_new_tokens),
                              temperature=temperature,
                              top_k=int(top_k) if top_k else None,
                              seed=int(seed) if seed is not None else None)
        return generate(req)["generated"]          # same code path as POST /generate
    except HTTPException as e:                     # a character the tokenizer does not know
        raise gr.Error(str(e.detail))
    except ValidationError as e:                   # a value outside the allowed range
        raise gr.Error(e.errors()[0]["msg"])


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox("ROMEO:", label="Prompt", lines=2),
        gr.Slider(1, 500, value=150, step=1, label="New characters"),
        gr.Slider(0, 2, value=0.8, step=0.1, label="Temperature (0 = greedy)"),
        gr.Number(label="Top-k (empty = off)", precision=0),
        gr.Number(1, label="Seed (empty = random)", precision=0),
    ],
    outputs=gr.Textbox(label="Generated text", lines=12),
    title="llm-from-scratch: a GPT built from scratch",
    description="3.2M-parameter character-level model trained on Tiny Shakespeare. "
                "The text is an experiment and not meaningful. REST API docs: /docs",
    flagging_mode="never",
)

# The Gradio page takes over "/", so drop the JSON route that src/api.py defines there
app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/"]
# ssr_mode=False: Spaces turn on server-side rendering, which starts a Node server on port 7860
# and would block uvicorn from using it
app = gr.mount_gradio_app(app, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
