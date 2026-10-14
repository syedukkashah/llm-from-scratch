---
title: llm-from-scratch
emoji: 🎭
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---

# llm-from-scratch

A 3.2M-parameter GPT trained from scratch on Tiny Shakespeare. Source code:
https://github.com/syedukkashah/llm-from-scratch

- Web page: the form on this page
- `GET /health`
- `POST /generate` with `{"prompt": "ROMEO:", "max_new_tokens": 80, "temperature": 0.8}`
- Interactive API docs at `/docs`
