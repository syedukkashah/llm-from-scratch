# Hugging Face Space (Docker). Clones the project from GitHub, so the Space stays a two-file repo.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

RUN git clone --depth 1 https://github.com/syedukkashah/llm-from-scratch .

# CPU-only PyTorch, then the API requirements
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt

# Download the trained checkpoint at build time so startup is fast
RUN python -c "from src.utils import ensure_checkpoint; ensure_checkpoint('checkpoints/tiny_gpt.pt')"

# Spaces expect the app on port 7860
EXPOSE 7860
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
