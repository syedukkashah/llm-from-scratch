import torch

from src.gpt import GPT


def make_model(causal=True, **kw):
    args = dict(vocab_size=20, context_length=16, d_model=32, n_heads=4,
                n_layers=2, ffn_dim=128, dropout=0.0, causal=causal)
    args.update(kw)
    return GPT(**args).eval()


def test_logits_have_expected_shape_and_loss():
    model = make_model()
    idx = torch.randint(0, 20, (3, 10))
    logits, loss = model(idx, targets=torch.randint(0, 20, (3, 10)))
    assert logits.shape == (3, 10, 20)
    assert loss.dim() == 0


def test_initial_loss_is_near_uniform_guess():
    # an untrained model should be about as bad as guessing: loss ~ ln(vocab)
    model = make_model()
    idx = torch.randint(0, 20, (8, 16))
    _, loss = model(idx, targets=torch.randint(0, 20, (8, 16)))
    assert abs(loss.item() - torch.log(torch.tensor(20.0)).item()) < 1.5


def test_future_tokens_cannot_change_earlier_logits():
    model = make_model(causal=True)
    idx = torch.randint(0, 20, (1, 12))
    changed = idx.clone()
    changed[0, 7:] = (changed[0, 7:] + 1) % 20  # edit everything from position 7 on
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-6)  # positions 0..6 unchanged
    assert not torch.allclose(a[:, 7:], b[:, 7:])         # positions 7+ do change


def test_without_causal_mask_the_future_leaks():
    model = make_model(causal=False)
    idx = torch.randint(0, 20, (1, 12))
    changed = idx.clone()
    changed[0, 7:] = (changed[0, 7:] + 1) % 20
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert not torch.allclose(a[:, :7], b[:, :7], atol=1e-6)


def test_generate_length_and_greedy_is_deterministic():
    model = make_model()
    start = torch.zeros(1, 3, dtype=torch.long)
    out1 = model.generate(start, max_new_tokens=10, temperature=0)
    out2 = model.generate(start, max_new_tokens=10, temperature=0)
    assert out1.shape == (1, 13)
    assert torch.equal(out1, out2)


def test_generate_works_past_the_context_length():
    model = make_model(context_length=8)
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=30)
    assert out.shape == (1, 31)  # context is cropped internally, no error


def test_top_k_one_is_greedy():
    model = make_model()
    start = torch.zeros(1, 3, dtype=torch.long)
    greedy = model.generate(start, 10, temperature=0)
    top1 = model.generate(start, 10, temperature=1.0, top_k=1)
    assert torch.equal(greedy, top1)


def test_learned_positions_are_causal_and_add_parameters():
    sinus, learned = make_model(), make_model(position="learned")
    assert learned.count_parameters() == sinus.count_parameters() + 16 * 32  # context 16 x d_model 32
    idx = torch.randint(0, 20, (1, 12))
    changed = idx.clone()
    changed[0, 7:] = (changed[0, 7:] + 1) % 20
    with torch.no_grad():
        a, _ = learned(idx)
        b, _ = learned(changed)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-6)
