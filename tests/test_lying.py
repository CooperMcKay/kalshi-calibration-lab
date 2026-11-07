import numpy as np
import pandas as pd
import pytest

from kalshi_lab.lying import BANDS, band_table, category_table, cluster_bootstrap, main


def frame(p, y, cats=None, horizon=24, events=None):
    n = len(p)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "event_ticker": events if events is not None else [f"E{i}" for i in range(n)],
        "category": cats if cats is not None else ["Sports"] * n,
        "series_ticker": ["S"] * n, "horizon_h": horizon, "prob": p, "outcome": y,
        "volume_fp": 500.0, "volume_pre": 100.0,
        "close_time": "2026-11-01T00:00:00Z",
    })


def calibrated_frame(n, seed=0, **kw):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    return frame(p, (rng.uniform(size=n) < p).astype(int), **kw)


def test_cluster_bootstrap_is_wider_when_rows_share_an_outcome():
    rng = np.random.default_rng(1)
    y_cluster = rng.integers(0, 2, 60)
    y = np.repeat(y_cluster, 10)                 # 60 events, 10 identical markets each
    p = np.full(600, 0.5)
    groups = np.repeat(np.arange(60), 10)
    stat = lambda pp, yy: float(np.mean(yy) - np.mean(pp))  # noqa: E731
    _, lo_c, hi_c = cluster_bootstrap(p, y, groups, stat, n_boot=400)
    _, lo_i, hi_i = cluster_bootstrap(p, y, np.arange(600), stat, n_boot=400)
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)


def test_band_table_reports_gap_and_event_counts():
    p = np.array([0.72] * 40 + [0.30] * 40)
    y = np.array([1] * 20 + [0] * 20 + [0] * 30 + [1] * 10)   # 70% band: 50% actual; 30% band: 25% actual
    t = band_table(frame(p, y), n_boot=200)
    row = t[np.isclose(t["band_lo"], 0.65)].iloc[0]
    assert row["n"] == 40 and row["n_events"] == 40
    assert row["actual"] == pytest.approx(0.5) and row["gap"] == pytest.approx(0.5 - 0.72)
    assert row["gap_lo"] < row["gap"] < row["gap_hi"]


def test_band_table_covers_every_band_and_skips_empty_ones_gracefully():
    t = band_table(frame(np.array([0.5] * 30), np.array([1, 0] * 15)), n_boot=100)
    assert len(t) == len(BANDS)
    assert t.loc[t["n"] == 0, "gap"].isna().all()


def test_band_table_calibrated_data_flags_nothing():
    t = band_table(calibrated_frame(4000), n_boot=300)
    assert not t["bh_significant"].any()


def test_band_table_flags_a_clearly_wrong_band():
    p = np.array([0.8] * 400)
    y = np.array([1] * 200 + [0] * 200)                          # says 80%, happens 50%
    t = band_table(frame(p, y), n_boot=300)
    row = t[np.isclose(t["band_lo"], 0.75)].iloc[0]
    assert bool(row["bh_significant"]) and row["verdict"] == "price too high"


def test_category_table_one_row_per_category_and_horizon():
    df = pd.concat([calibrated_frame(300, 1, cats=["Sports"] * 300),
                    calibrated_frame(300, 2, cats=["Economics"] * 300)], ignore_index=True)
    df["event_ticker"] = [f"E{i}" for i in range(len(df))]
    t = category_table(df, n_boot=100)
    assert set(t["category"]) == {"Sports", "Economics"} and len(t) == 2
    assert {"brier", "brier_lo", "brier_hi", "ece", "slope", "slope_lo", "slope_hi", "n_events"} <= set(t.columns)


def test_category_table_marks_small_categories_inconclusive():
    t = category_table(calibrated_frame(40), n_boot=100)
    assert t.iloc[0]["verdict"] == "too few events"


def test_main_writes_outputs(tmp_path):
    calibrated_frame(600).to_csv(tmp_path / "d.csv", index=False)
    assert main(["--data", str(tmp_path / "d.csv"), "--out", str(tmp_path), "--n-boot", "50"]) == 0
    for name in ("lying_by_band.csv", "lying_by_category.csv", "lying_category_band_gaps.csv",
                 "lying_bands_24h.png", "lying_category_gaps_24h.png"):
        assert (tmp_path / name).exists()
