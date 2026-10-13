# Data

Tiny Shakespeare (about 1.1 MB, 40,000 lines of Shakespeare plays).

- Hugging Face dataset card: https://huggingface.co/datasets/karpathy/tiny_shakespeare
- Raw file used by the code: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

`src/dataset.py` downloads it to `data/input.txt` on first use. The file is not committed (see `.gitignore`).

| Item | Value |
|---|---|
| Characters | 1,115,394 |
| Vocabulary (unique characters) | 65 |
| Train / validation split | 90% / 10%, contiguous (validation is the last part of the text) |
