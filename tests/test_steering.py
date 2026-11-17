import numpy as np

from biointerp.steering import (
    Condition,
    build_steer_map,
    negate_direction,
    parse_conditions,
    summarize_conditions,
    unit_random_direction,
)


def test_parse_conditions():
    payload = {
        "conditions": [
            {"name": "baseline", "direction": "none"},
            {"name": "positive", "direction": "true", "layer": 24, "alpha": 0.3},
        ]
    }
    conditions = parse_conditions(payload)
    assert conditions[0].direction == "none"
    assert conditions[1].layer == 24
    assert conditions[1].alpha == 0.3


def test_parse_conditions_rejects_unknown_direction():
    try:
        parse_conditions({"conditions": [{"name": "x", "direction": "sideways"}]})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_negate_direction():
    np.testing.assert_array_equal(negate_direction([1.0, -2.0]), [-1.0, 2.0])


def test_random_direction_is_unit_norm():
    vector = unit_random_direction(64, seed=0)
    assert np.linalg.norm(vector) == 1.0


def test_build_steer_map_none_returns_none():
    assert build_steer_map(Condition("baseline"), {}, {}, hidden_dim=8) is None


def test_build_steer_map_negative_flips_direction():
    direction = np.array([3.0, 4.0])
    positive = build_steer_map(
        Condition("p", "true", layer=0, alpha=0.5), {0: direction}, {0: 10.0}, 2
    )
    negative = build_steer_map(
        Condition("n", "negative", layer=0, alpha=0.5), {0: direction}, {0: 10.0}, 2
    )
    assert positive is not None and negative is not None
    np.testing.assert_allclose(positive[0][0], direction)
    np.testing.assert_allclose(negative[0][0], -direction)
    assert positive[0][1] == negative[0][1]


def test_summarize_conditions_reports_deltas():
    summary = summarize_conditions(
        [
            {"condition": "baseline", "mean_naturalness": 1.0},
            {"condition": "top", "mean_naturalness": 1.2},
        ]
    )
    assert summary["delta_vs_baseline"]["top"] == 0.2
