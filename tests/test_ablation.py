from experiments.ablation import run_one, word_metrics

TEXT = ("To be, or not to be, that is the question: "
        "Whether 'tis nobler in the mind to suffer ") * 30

CFG = dict(context_length=16, d_model=32, n_heads=4, n_layers=2, ffn_dim=128, dropout=0.0,
           tie_weights=False, position="sinusoidal", data_path="unused", train_frac=0.9,
           batch_size=8, lr=1e-3, min_lr=1e-4, warmup_steps=2, weight_decay=0.01,
           grad_clip=1.0, seed=0, sample_prompt="T", temperature=0.8)


def test_word_metrics_detects_repetition_and_unknown_words():
    vocab = {"the", "world", "of"}
    looping = word_metrics(["the world of the world of the world"] * 2, vocab)
    assert looping["real_word_rate"] == 1.0
    assert looping["repetition"] > 0.5     # the same word pairs keep coming back
    assert looping["distinct_samples"] == 1
    invented = word_metrics(["the zzzz of qqqq"], vocab)
    assert invented["real_word_rate"] == 0.5


def test_run_one_returns_metrics_and_params_change_with_depth(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.ablation.LOG_DIR", str(tmp_path))
    small = run_one(CFG, "t_small", {}, TEXT, steps=4, eval_interval=2, eval_iters=2)
    deeper = run_one(CFG, "t_deep", {"n_layers": 4}, TEXT, steps=4, eval_interval=2, eval_iters=2)
    assert {"params", "val_loss", "perplexity", "train_time_s", "tokens_per_sec"} <= set(small)
    assert deeper["params"] > small["params"]
