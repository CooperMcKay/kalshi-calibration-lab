import numpy as np
import pandas as pd
import pytest

from kalshi_lab.metrics import brier
from kalshi_lab.stats_tests import (
    bootstrap_ci, bootstrap_scores, favorite_longshot, main, volume_closeness)


def synth(n=4000, seed=0, slope=1.0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    logit = np.log(p / (1 - p))
    true = 1 / (1 + np.exp(-slope * logit))
    return p, (rng.uniform(size=n) < true).astype(int)


def frame(p, y, vol=None, h=24, series="S"):
    n = len(p)
    return pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "event_ticker": [f"E{i}" for i in range(n)],
                         "series_ticker": series, "horizon_h": h, "prob": p, "outcome": y,
                         "volume_fp": np.full(n, 500.0) if vol is None else vol})


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    p, y = synth(500)
    point, lo, hi = bootstrap_ci(p, y, brier, n_boot=300, seed=1)
    assert lo <= point <= hi and point == pytest.approx(brier(p, y))
    assert bootstrap_ci(p, y, brier, n_boot=300, seed=1) == (point, lo, hi)


def test_bootstrap_ci_narrows_with_more_data():
    w = lambda n: (lambda t: t[2] - t[1])(bootstrap_ci(*synth(n), brier, n_boot=300, seed=1))
    assert w(4000) < w(300)


def test_bootstrap_ci_perfect_forecast_is_degenerate():
    p = np.array([0.0, 1.0] * 50); y = np.array([0, 1] * 50)
    assert bootstrap_ci(p, y, brier, n_boot=100) == (0.0, 0.0, 0.0)


def test_bootstrap_scores_has_ci_columns_per_horizon():
    p, y = synth(300)
    out = bootstrap_scores(pd.concat([frame(p, y, h=24), frame(p, y, h=1)]), n_boot=100)
    assert set(out["horizon_h"]) == {24, 1}
    assert {"brier", "brier_lo", "brier_hi", "ece", "ece_lo", "ece_hi", "n"} <= set(out.columns)


def test_favorite_longshot_detects_stretched_probabilities():
    p, y = synth(6000, slope=1.5)
    r = favorite_longshot(frame(p, y), 24, n_boot=200)
    assert r["slope_lo"] > 1.0 and r["verdict"] == "favorite-longshot bias"


def test_favorite_longshot_calibrated_data_includes_one():
    p, y = synth(6000, slope=1.0)
    r = favorite_longshot(frame(p, y), 24, n_boot=200)
    assert r["slope_lo"] <= 1.0 <= r["slope_hi"] and r["verdict"] == "no evidence of bias"


def test_favorite_longshot_too_few_rows_is_inconclusive():
    p, y = synth(40)
    assert favorite_longshot(frame(p, y), 24, n_boot=50)["verdict"] == "inconclusive"


def test_volume_closeness_separates_confounding_from_direct_effect():
    rng = np.random.default_rng(3)
    n = 4000
    closeness = rng.uniform(0, 0.45, n)             # |p - 0.5|
    p = 0.5 + closeness * rng.choice([-1, 1], n)
    vol = 1000 * (0.5 - closeness) + rng.normal(0, 20, n) + 600  # closer game -> more volume
    y = (rng.uniform(size=n) < p).astype(int)        # error depends on p only, never on volume
    r = volume_closeness(frame(p, y, vol=vol), 24)
    assert r["rho_volume_vs_distance"] < -0.5        # busy games are closer
    assert abs(r["partial_rho_volume_error"]) < 0.08  # nothing left once closeness is controlled


def test_main_writes_outputs_and_fails_cleanly(tmp_path):
    p, y = synth(400)
    csv = tmp_path / "d.csv"
    frame(p, y).to_csv(csv, index=False)
    assert main(["--data", str(csv), "--out", str(tmp_path / "o"), "--n-boot", "50"]) == 0
    for name in ("stats_bootstrap.csv", "stats_favorite_longshot.csv", "stats_volume_closeness.csv"):
        assert (tmp_path / "o" / name).exists(), name
    empty = tmp_path / "e.csv"; pd.DataFrame().to_csv(empty, index=False)
    assert main(["--data", str(empty), "--out", str(tmp_path / "o2")]) == 1


def test_volume_closeness_uses_pre_horizon_volume_when_asked():
    rng = np.random.default_rng(3)
    n = 400
    p = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(size=n) < p).astype(int)
    df = frame(p, y)
    df["volume_pre"] = np.arange(n, dtype=float)
    default = volume_closeness(df, 24)
    pre = volume_closeness(df, 24, volume_col="volume_pre")
    assert default["rho_volume_error"] != pre["rho_volume_error"]
