import math

import numpy as np
import pytest

from kalshi_lab.metrics import (
    benjamini_hochberg,
    brier,
    directional_accuracy,
    ece,
    log_loss,
    murphy_decomposition,
    reliability_table,
)


def test_brier_perfect_is_zero():
    assert brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0


def test_brier_always_half_is_quarter():
    assert brier(np.full(10, 0.5), np.array([1, 0] * 5)) == pytest.approx(0.25)


def test_brier_worked_example():
    assert brier(np.array([0.7]), np.array([1])) == pytest.approx(0.09)
    assert brier(np.array([0.7]), np.array([0])) == pytest.approx(0.49)


def test_log_loss_clips_extremes():
    v = log_loss(np.array([0.0, 1.0]), np.array([1, 0]))
    assert math.isfinite(v) and v > 10


def test_log_loss_known_value():
    assert log_loss(np.array([0.7]), np.array([1])) == pytest.approx(-math.log(0.7))


def test_reliability_table_counts_and_gap():
    p = np.array([0.05, 0.15, 0.15, 0.55, 0.55, 0.55, 0.85, 0.85, 0.95, 0.95])
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 1])
    t = reliability_table(p, y)
    assert t["n"].sum() == 10
    row = t[t["bucket"] == "0.5-0.6"].iloc[0]
    assert row["n"] == 3
    assert row["actual"] == pytest.approx(2 / 3)
    assert row["gap"] == pytest.approx(2 / 3 - 0.55)


def test_reliability_empty_bin_has_n0_and_nan():
    t = reliability_table(np.array([0.05, 0.06]), np.array([0, 1]))
    empty = t[t["bucket"] == "0.5-0.6"].iloc[0]
    assert empty["n"] == 0 and np.isnan(empty["actual"])


def test_reliability_ci_nan_when_n_below_2():
    t = reliability_table(np.array([0.05, 0.95, 0.96]), np.array([0, 1, 1]))
    assert np.isnan(t[t["bucket"] == "0.0-0.1"].iloc[0]["ci95"])
    assert not np.isnan(t[t["bucket"] == "0.9-1.0"].iloc[0]["ci95"])


def test_reliability_p_equal_one_lands_in_last_bucket():
    t = reliability_table(np.array([1.0]), np.array([1]))
    assert t[t["bucket"] == "0.9-1.0"].iloc[0]["n"] == 1


def test_ece_zero_when_calibrated():
    p = np.array([0.5, 0.5, 0.5, 0.5])
    y = np.array([1, 0, 1, 0])
    assert ece(p, y) == pytest.approx(0.0)


def test_ece_matches_weighted_gap():
    # bucket 0.1-0.2: 2 preds of 0.15, actual 0 -> gap 0.15 ; bucket 0.8-0.9: 2 preds 0.85, actual 1 -> gap 0.15
    p = np.array([0.15, 0.15, 0.85, 0.85])
    y = np.array([0, 0, 1, 1])
    assert ece(p, y) == pytest.approx(0.15)


def test_murphy_identity_holds_when_p_constant_within_bins():
    p = np.array([0.15] * 4 + [0.85] * 4)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    d = murphy_decomposition(p, y)
    assert d["brier_binned"] == pytest.approx(brier(p, y))
    assert d["reliability"] - d["resolution"] + d["uncertainty"] == pytest.approx(brier(p, y))


def test_murphy_perfect_forecaster_resolution_equals_uncertainty():
    p = np.array([0.05, 0.05, 0.95, 0.95])
    y = np.array([0, 0, 1, 1])
    d = murphy_decomposition(p, y)
    assert d["uncertainty"] == pytest.approx(0.25)
    assert d["resolution"] == pytest.approx(0.25)


def test_directional_accuracy_tie_is_incorrect():
    p = np.array([0.7, 0.3, 0.5, 0.5])
    y = np.array([1, 0, 1, 0])
    assert directional_accuracy(p, y) == pytest.approx(0.5)


def test_bh_known_example():
    pv = np.array([0.001, 0.008, 0.039, 0.041, 0.6])
    # thresholds 0.05*k/5: .01,.02,.03,.04,.05 -> largest k with p<=thr is k=2
    assert benjamini_hochberg(pv, 0.05).tolist() == [True, True, False, False, False]


def test_bh_ignores_nan():
    pv = np.array([0.001, np.nan, 0.9])
    out = benjamini_hochberg(pv, 0.05)
    assert out.tolist() == [True, False, False]
