"""Generation-path equivalence: hooks and zero steering must not change outputs."""

from _fake_models import FakeGpt2Model, FakeGpt2Tokenizer
from equivalence_utils import assert_generation_equal

from biointerp import gen


def test_capture_layers_does_not_change_generation():
    model = FakeGpt2Model(hidden=12, intermediate=20, n_layers=2, seed=0)
    tokenizer = FakeGpt2Tokenizer()

    baseline = gen.generate_with_cett(
        model, tokenizer, n_steps=4, seed=11, greedy=True, device="cpu"
    )
    with_capture = gen.generate_with_cett(
        model,
        tokenizer,
        n_steps=4,
        seed=11,
        greedy=True,
        device="cpu",
        capture_layers=[0, 1],
    )
    assert_generation_equal(baseline, with_capture[:5])
    assert len(with_capture[5][0]) == len(with_capture[0])


def test_zero_steering_does_not_change_generation():
    model = FakeGpt2Model(hidden=12, intermediate=20, n_layers=2, seed=0)
    tokenizer = FakeGpt2Tokenizer()

    baseline = gen.generate_with_cett(
        model, tokenizer, n_steps=4, seed=12, greedy=True, device="cpu"
    )
    steered = gen.generate_with_cett(
        model,
        tokenizer,
        n_steps=4,
        seed=12,
        greedy=True,
        device="cpu",
        steer_map={0: ([0.0] * model.config.n_embd, 0.0)},
    )
    assert_generation_equal(baseline, steered)
