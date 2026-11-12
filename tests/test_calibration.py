import numpy as np
import pytest

from biointerp.calibration import (
    brier_score,
    calibration_summary,
    enrichment_at_fraction,
    expected_calibration_error,
    precision_at_fraction,
    reliability_curve,
)


def test_reliability_curve_counts_sum_to_total():
    y = [0, 0, 1, 1, 1, 0]
    p = [0.05, 0.15, 0.85, 0.95, 0.55, 0.45]
    curve = reliability_curve(y, p, n_bins=5)
    assert sum(item["count"] for item in curve) == len(y)


def test_reliability_curve_reports_observed_rate():
    y = [0, 0, 1, 1]
    p = [0.1, 0.1, 0.9, 0.9]
    curve = reliability_curve(y, p, n_bins=2)
    assert curve[0]["observed_rate"] == 0.0
    assert curve[1]["observed_rate"] == 1.0


def test_ece_zero_for_perfectly_calibrated_bins():
    y = [0] * 50 + [1] * 50
    p = [0.0] * 50 + [1.0] * 50
    assert expected_calibration_error(y, p, n_bins=10) == 0.0


def test_ece_positive_for_miscalibrated_scores():
    y = [0] * 100
    p = [0.9] * 100
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.9)


def test_brier_score_known_value():
    y = [0, 1]
    p = [0.0, 1.0]
    assert brier_score(y, p) == 0.0


def test_brier_score_worst_case():
    y = [0, 1]
    p = [1.0, 0.0]
    assert brier_score(y, p) == 1.0


def test_precision_at_fraction_selects_top_scores():
    y = [0, 0, 1, 1, 1, 0]
    scores = [0.1, 0.2, 0.9, 0.8, 0.7, 0.3]
    assert precision_at_fraction(y, scores, fraction=0.5) == pytest.approx(1.0)


def test_enrichment_is_one_for_random_ranking():
    y = [0, 1] * 50
    scores = np.linspace(0, 1, 100)
    assert enrichment_at_fraction(y, scores, fraction=0.2) == pytest.approx(1.0, abs=0.15)


def test_calibration_summary_has_expected_keys():
    summary = calibration_summary([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])
    assert set(summary) == {
        "ece",
        "brier",
        "precision_at_fraction",
        "enrichment",
        "base_rate",
        "fraction",
    }
