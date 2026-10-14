"""Hugging Face Space entry point (Gradio SDK).

Gradio serves a small web page at "/" and the FastAPI service from src/api.py is mounted
under /api:  GET /api/health, POST /api/generate, docs at /api/docs.
"""
import gradio as gr
from fastapi import HTTPException
from pydantic import ValidationError

from src.api import GenerateRequest, app as api_app, generate, load_model


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
                "The text is an experiment and not meaningful. REST API docs: /api/docs",
    flagging_mode="never",
)

load_model()   # the mounted API does not run its startup code, so load the model here

# launch() must be called: on ZeroGPU it reports the @spaces.GPU function to Hugging Face.
# ssr_mode=False: Spaces turn on server-side rendering, which would start a Node server.
fastapi_app, _, _ = demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False,
                                prevent_thread_lock=True)
fastapi_app.mount("/api", api_app)
demo.block_thread()
