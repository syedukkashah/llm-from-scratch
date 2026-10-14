# llm-from-scratch

A small GPT-style **causal language model built from scratch in PyTorch**: tokenizer, scaled dot-product
attention, multi-head attention, sinusoidal positions, Transformer blocks, training loop, text generation,
tests, ablation experiments, a CLI and a REST API. No pretrained weights and no `transformers` library.

Trained on Tiny Shakespeare. Final model: **3.19M parameters, validation loss 1.51 (perplexity 4.5)**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/syedukkashah/llm-from-scratch/blob/main/notebooks/colab_demo.ipynb)

Live API: _not deployed yet, see [Deploy to Render](#deploy-to-render)_

## Quick start

```bash
git clone https://github.com/syedukkashah/llm-from-scratch
cd llm-from-scratch
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                            # all tests should pass
python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80
```

The first generation downloads the trained checkpoint (12.6 MB) from the
[GitHub Release](https://github.com/syedukkashah/llm-from-scratch/releases/tag/v1.0) automatically.

Expected output (exact text differs unless you pass `--seed 1`):

```
Prompt: The future of artificial intelligence
Generated: The future of artificial intelligence,
And that many fair with her, as false as
not cry him; but one that hath been r

model parameters: 3.2M
context length: 128
temperature: 0.8
```

## Where things live

| What | Where |
|---|---|
| Attention (scaled dot-product, causal mask, multi-head) | `src/attention.py` |
| Positional encoding (sinusoidal, learned) | `src/positional_encoding.py` |
| Feed-forward network and Transformer block | `src/transformer_block.py` |
| GPT model and text generation | `src/gpt.py` |
| Tokenizer (character level) | `src/tokenizer.py` |
| Data pipeline (download, split, shifted batches) | `src/dataset.py` |
| Training loop | `src/train.py` |
| CLI generation | `src/generate.py` |
| REST API | `src/api.py` |
| All hyperparameters | `configs/tiny_gpt.yaml` |
| Tests | `tests/` |
| Experiments, hand calculation, ablations | `experiments/` |
| Figures, logs, samples | `outputs/figures`, `outputs/logs`, `outputs/samples` |
| Colab notebook | `notebooks/colab_demo.ipynb` |

## Dataset

[Tiny Shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare): 1,115,394 characters,
40,000 lines. Downloaded automatically to `data/input.txt` (not committed). Character-level tokenizer with a
vocabulary of 65 (sorted unique characters, no special tokens). Split 90% train / 10% validation, contiguous.
See `data/README.md`.

## Model and training

| | |
|---|---|
| Architecture | 4 pre-LayerNorm Transformer blocks, 4 heads, d_model 256, FFN 1024 (GELU) |
| Context length | 128 characters |
| Position | sinusoidal (learned is available: `--position learned`) |
| Parameters | 3,192,897 |
| Optimizer | AdamW, lr 1e-3 with 100 warm-up steps then cosine decay to 1e-4, weight decay 0.01, gradient clip 1.0 |
| Batch / steps | 64 sequences x 128 tokens, 5000 steps |
| Dropout | 0.1 |
| Hardware / time | Tesla T4 (Google Colab), 440 s |
| Checkpoint kept | lowest validation loss (step 3250) |

Train it yourself (a GPU is recommended; the CPU speed is roughly 6k tokens/s):

```bash
python -m src.train                          # uses configs/tiny_gpt.yaml
python -m src.train --n_layers 6 --name deep # override any config value
python -m src.train --overfit                # debugging gate: memorize one tiny batch
```

Outputs: `checkpoints/<name>.pt`, `outputs/logs/<name>_log.csv` (step, train loss, validation loss,
learning rate, time), `outputs/logs/<name>_summary.json`.

## Results

| Metric | Value |
|---|---|
| Validation loss / perplexity (all validation windows, saved checkpoint) | 1.5076 / 4.52 |
| Train loss / perplexity (same number of windows) | 1.1762 / 3.24 |
| Guessing baseline | 4.1744 / 65 |
| Loss at the last step (5000) | train 1.0862, validation 1.5225 |

Validation loss stops improving around step 3000 while training loss keeps falling, so the model starts to
memorise the 1 MB corpus (see `outputs/figures/fig6_loss_curves.png`).

```bash
python -m experiments.evaluate                 # loss, perplexity, 5 fixed-prompt samples, loss curves
python -m experiments.attention_visualization  # attention maps for every layer and head
```

Generated samples are in `outputs/samples/samples.md`. Sampling comparison (no retraining, 5 samples of 300
characters each, `python -m experiments.ablation --experiment sampling`):

| setting | real-word rate | repetition | distinct samples (of 5) |
|---|---|---|---|
| greedy (temp 0) | 1.0 | 0.515 | 1 |
| temp 0.5 | 0.983 | 0.033 | 5 |
| temp 0.8 | 0.97 | 0.007 | 5 |
| temp 1.2 | 0.859 | 0.0 | 5 |
| temp 0.8 + top-k 10 | 0.964 | 0.01 | 5 |

Greedy decoding falls into loops ("the world of the world of the world"); a high temperature invents words.

## Experiments

```bash
python -m experiments.hand_calculation   # 3-token attention worked by hand (Figures 1 and 2)
python -m experiments.why_transformers   # sequential (RNN) vs parallel (attention) cost
python -m experiments.shape_trace        # tensor shapes through multi-head attention
python -m experiments.positional_viz     # positional encoding
python -m experiments.block_diagram      # Transformer block diagram and shape check
python -m experiments.mask_comparison    # BERT-style vs GPT-style masks, and why GPT needs the causal mask
python -m experiments.ablation --experiment all   # depth, heads, context length, position, sampling
```

### Ablation results

Each run trains for 2000 steps with the same seed, data and evaluation batches, changing one factor at a time
(T4 GPU, about 23 minutes of training for all 8 models). The baseline (4 layers, 4 heads, context 128, sinusoidal) is trained
once and shared. Full table: `outputs/logs/ablation_table.md`, plots: `outputs/figures/ablation_*.png`.

| Factor | Value | Parameters | Val loss | Perplexity | Train time (s) | Peak memory (MB) |
|---|---|---|---|---|---|---|
| layers | 2 | 1.61M | 1.622 | 5.06 | 69 | 399 |
| layers | **4** | 3.19M | 1.550 | 4.71 | 157 | 714 |
| layers | 6 | 4.77M | 1.522 | 4.58 | 236 | 1028 |
| heads | 2 | 3.19M | 1.547 | 4.70 | 152 | 682 |
| heads | **4** | 3.19M | 1.550 | 4.71 | 157 | 714 |
| heads | 8 | 3.19M | 1.549 | 4.71 | 173 | 778 |
| context | 64 | 3.19M | 1.589 | 4.90 | 74 | 368 |
| context | **128** | 3.19M | 1.550 | 4.71 | 157 | 714 |
| context | 256 | 3.19M | 1.523 | 4.59 | 349 | 1489 |
| position | **sinusoidal** | 3.19M | 1.550 | 4.71 | 157 | 714 |
| position | learned | 3.23M | 1.542 | 4.67 | 158 | 714 |

What the numbers say:

- **Depth** helps, with diminishing returns: 2 to 4 layers gains 0.07 in loss, 4 to 6 gains 0.03, while time and
  memory grow about linearly (6 layers takes 1.5x as long as 4).
- **Heads** make no measurable difference (spread 0.003). The parameter count is identical, because the heads
  split the same `d_model`. More heads only cost a little extra time.
- **Context** helps too (64 to 128 gains 0.04, 128 to 256 gains 0.03), but the cost is steep: 256 takes 2.2x as
  long and 2.1x the memory of 128, because attention scales with the square of the context.
- **Position**: learned is 0.008 better than sinusoidal for 32,768 extra parameters. That gap is within what a
  single seed can produce, so the fixed sinusoidal encoding is enough here.
- Every run is a single seed, so differences below about 0.01 to 0.02 are not significant. The depth and context
  effects are larger than that; the heads and position effects are not.

## Command-line and API

```bash
python -m src.generate --prompt "ROMEO:" --max-new-tokens 80 --temperature 0.8 --top-k 40 --seed 1
uvicorn src.api:app --port 8000
curl -X POST localhost:8000/generate -H "Content-Type: application/json" \
     -d '{"prompt": "ROMEO:", "max_new_tokens": 80, "temperature": 0.8, "seed": 1}'
```

Endpoints: `GET /health`, `POST /generate`, interactive docs at `/docs`. Limits: prompt up to 500 characters,
up to 500 new tokens, temperature 0 to 2. A prompt containing a character the tokenizer does not know returns
HTTP 400 with the list of allowed characters.

## Deploy to Render

`render.yaml` describes the service (CPU-only PyTorch, `requirements-api.txt`, health check at `/health`).

1. Push this repository to GitHub.
2. On [render.com](https://render.com): **New + > Blueprint**, pick this repository, apply.
3. On first start the service downloads the checkpoint from the GitHub Release (override with the
   `CHECKPOINT_URL` environment variable).
4. Test: `curl https://<your-service>.onrender.com/health`.

The free tier sleeps when idle, so the first request after a pause is slow.

## Acceptance checks

| Check | Command |
|---|---|
| Install and unit tests | `pip install -r requirements-dev.txt && pytest -q` |
| Tiny overfit | `python -m src.train --overfit` (loss falls from about 4.4 to about 0) |
| Checkpoint reload and generation in a fresh process | `python -m src.generate --prompt "ROMEO:"` |
| No attention to future tokens | `pytest tests/test_mask.py tests/test_causality.py` |

## Known limitations

- Character-level model on a 1 MB corpus: it learns the format and vocabulary of the plays, but the text is
  not meaningful and sometimes invents words. Generated text is an experiment, not a reliable source.
- It overfits after about 3000 steps; dropout 0.1 is the only regulariser.
- Greedy decoding loops and high temperatures produce non-words (see the sampling table).
- Only the 65 training characters are accepted in prompts (for example digits other than `3` and symbols like
  `{` are rejected).
- Ablations use single runs with one seed and fewer steps than the final model, so small differences are not
  statistically meaningful.
- Context is limited to 128 characters; the attention cost grows with the square of the context.

## References

- Vaswani et al., "Attention Is All You Need", 2017 (Transformer, scaled dot-product and multi-head attention,
  sinusoidal positional encoding).
- Radford et al., "Language Models are Unsupervised Multitask Learners", 2019 (GPT-2, decoder-only causal LM).
- Devlin et al., "BERT", 2018 (masked language modelling, used for the mask comparison).
- Tiny Shakespeare dataset: Andrej Karpathy, [char-rnn](https://github.com/karpathy/char-rnn).
- Andrej Karpathy's "Let's build GPT" video and nanoGPT were consulted as learning resources. The code in this
  repository was written for this assignment and does not copy theirs.
