import numpy as np
import pandas as pd
import pytest

from kalshi_lab.report_plots import (
    main,
    plot_calibration_gap,
    plot_horizon_comparison,
    plot_prob_histogram,
)


def synthetic(n=200, horizons=(24, 1), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = float(rng.uniform(0.05, 0.95))
        y = int(rng.uniform() < p)
        for h in horizons:
            rows.append((f"T{i}", f"E{i}", "Sports", h, p, y, 500.0))
    return pd.DataFrame(rows, columns=["ticker", "event_ticker", "category", "horizon_h", "prob", "outcome", "volume_fp"])


def assert_png(path):
    assert path.exists() and path.stat().st_size > 1000


def test_plot_horizon_comparison_writes_png(tmp_path):
    scores = pd.DataFrame({"horizon_h": [24, 1], "brier": [0.2, 0.08], "log_loss": [0.57, 0.26], "ece": [0.09, 0.09]})
    plot_horizon_comparison(scores, tmp_path / "h.png")
    assert_png(tmp_path / "h.png")


def test_plot_calibration_gap_writes_png(tmp_path):
    plot_calibration_gap(synthetic(), 24, tmp_path / "g.png")
    assert_png(tmp_path / "g.png")


def test_plot_calibration_gap_handles_empty_bins(tmp_path):
    df = synthetic(n=30)
    df = df[df["prob"] < 0.5]  # upper buckets are empty
    plot_calibration_gap(df, 24, tmp_path / "g.png")
    assert_png(tmp_path / "g.png")


def test_plot_prob_histogram_writes_png(tmp_path):
    plot_prob_histogram(synthetic(), 24, tmp_path / "p.png")
    assert_png(tmp_path / "p.png")


def test_plot_prob_histogram_single_outcome_class(tmp_path):
    df = synthetic(n=40).assign(outcome=1)
    plot_prob_histogram(df, 24, tmp_path / "p.png")
    assert_png(tmp_path / "p.png")


def test_main_writes_all_figures(tmp_path):
    csv = tmp_path / "d.csv"
    synthetic().to_csv(csv, index=False)
    out = tmp_path / "o"
    assert main(["--data", str(csv), "--out", str(out)]) == 0
    for name in ("horizon_comparison.png", "calibration_gap_24h.png", "prob_histogram_24h.png",
                 "calibration_gap_1h.png", "prob_histogram_1h.png"):
        assert_png(out / name)


def test_main_fails_cleanly_on_empty_dataset(tmp_path):
    csv = tmp_path / "d.csv"
    synthetic(n=0).to_csv(csv, index=False)
    assert main(["--data", str(csv), "--out", str(tmp_path / "o")]) == 1
