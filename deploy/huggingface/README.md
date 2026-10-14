---
title: llm-from-scratch
emoji: 🎭
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# llm-from-scratch API

A 3.2M-parameter GPT trained from scratch on Tiny Shakespeare. Source code:
https://github.com/syedukkashah/llm-from-scratch

- `GET /health`
- `POST /generate` with `{"prompt": "ROMEO:", "max_new_tokens": 80, "temperature": 0.8}`
- Interactive docs at `/docs`
