from src.train import get_lr, overfit

TEXT = ("To be, or not to be, that is the question: "
        "Whether 'tis nobler in the mind to suffer ") * 20

CFG = dict(context_length=32, d_model=64, n_heads=4, n_layers=2, ffn_dim=256,
           dropout=0.0, tie_weights=False, lr=3e-3, seed=0)


def test_tiny_overfit_loss_falls_a_lot():
    # If gradients, the target shift or the causal mask were broken, this would stall.
    losses = overfit(CFG, TEXT, steps=150)
    assert losses[-1] < 0.1 * losses[0]


def test_lr_schedule_warmup_then_decay():
    cfg = dict(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    assert get_lr(0, cfg) < get_lr(9, cfg) <= cfg["lr"]       # warming up
    assert get_lr(10, cfg) > get_lr(50, cfg) > get_lr(99, cfg)  # decaying
    assert get_lr(99, cfg) >= cfg["min_lr"]
