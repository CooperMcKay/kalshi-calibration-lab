import numpy as np
import pandas as pd
import pytest

from kalshi_lab.calibration import (
    bucket_significance,
    fixed_cohort,
    main,
    plot_reliability,
    score_by,
    tail_truncate,
    volume_band,
)


def frame(rows):
    cols = ["ticker", "event_ticker", "category", "horizon_h", "prob", "outcome", "volume_fp"]
    return pd.DataFrame(rows, columns=cols)


def test_score_by_horizon_known_values():
    df = frame([
        ("A", "E1", "Sports", 24, 0.8, 1, 500.0),
        ("B", "E2", "Sports", 24, 0.6, 0, 500.0),
        ("A", "E1", "Sports", 1, 0.9, 1, 500.0),
    ])
    out = score_by(df, "horizon_h").set_index("horizon_h")
    assert out.loc[24, "n"] == 2
    assert out.loc[24, "brier"] == pytest.approx(0.2)
    assert out.loc[24, "base_rate"] == pytest.approx(0.5)
    assert out.loc[24, "brier_always_half"] == pytest.approx(0.25)
    assert out.loc[1, "brier"] == pytest.approx(0.01)


def test_score_by_multiple_keys():
    df = frame([
        ("A", "E1", "Sports", 24, 0.8, 1, 500.0),
        ("B", "E2", "Economics", 24, 0.6, 0, 500.0),
    ])
    out = score_by(df, ["horizon_h", "category"])
    assert len(out) == 2 and {"horizon_h", "category", "n", "brier", "log_loss", "ece"} <= set(out.columns)


def test_volume_band_boundaries():
    s = volume_band(pd.DataFrame({"volume_fp": [999, 1000, 9999, 10000]}))
    assert s.tolist() == ["<1k", "1k-10k", "1k-10k", ">=10k"]


def test_fixed_cohort_drops_tickers_missing_base_horizon():
    df = frame([
        ("A", "E1", "S", 24, 0.5, 1, 1.0), ("A", "E1", "S", 1, 0.6, 1, 1.0),
        ("B", "E2", "S", 1, 0.7, 0, 1.0),  # no 24h row
    ])
    out = fixed_cohort(df, base_horizon_h=24)
    assert set(out["ticker"]) == {"A"} and len(out) == 2


def test_tail_truncate_excludes_boundaries():
    df = frame([("A", "E1", "S", 24, p, 1, 1.0) for p in (0.01, 0.02, 0.5, 0.98, 0.99)])
    assert tail_truncate(df)["prob"].tolist() == [0.5]


def test_bucket_significance_false_discovery_rate_is_controlled():
    # On truly calibrated data BH at 5% should flag something in only ~5% of runs, not in most of them.
    runs, hits = 40, 0
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.05, 0.95, 2000)
        y = (rng.uniform(size=2000) < p).astype(int)
        hits += int(bucket_significance(p, y)["bh_significant"].any())
    assert hits <= 6


def test_bucket_significance_flags_gross_miscalibration():
    p = np.full(200, 0.9)
    y = np.zeros(200, dtype=int)
    out = bucket_significance(p, y)
    row = out[out["bucket"] == "0.9-1.0"].iloc[0]
    assert row["bh_significant"] and row["p_value"] < 1e-6
    assert np.isnan(out[out["bucket"] == "0.0-0.1"].iloc[0]["p_value"])


def test_plot_reliability_writes_png(tmp_path):
    rng = np.random.default_rng(1)
    p = rng.uniform(size=150)
    df = frame([(f"T{i}", f"E{i}", "S", 24, p[i], int(rng.uniform() < p[i]), 500.0) for i in range(150)])
    out = tmp_path / "rel.png"
    plot_reliability(df, 24, out)
    assert out.exists() and out.stat().st_size > 1000


def test_main_writes_outputs_and_warns_on_small_slice(tmp_path, capsys):
    rng = np.random.default_rng(2)
    rows = []
    for i in range(60):
        p = float(rng.uniform(0.1, 0.9))
        for h in (24, 1):
            rows.append((f"T{i}", f"E{i}", "Sports", h, p, int(rng.uniform() < p), 500.0))
    csv = tmp_path / "dataset.csv"
    frame(rows).to_csv(csv, index=False)
    out = tmp_path / "out"
    assert main(["--data", str(csv), "--out", str(out)]) == 0
    for name in ("scores_by_horizon.csv", "scores_by_category.csv", "scores_by_volume_band.csv",
                 "scores_fixed_cohort.csv", "scores_tail_truncated.csv", "murphy_by_horizon.csv",
                 "reliability_24h.csv", "reliability_24h.png"):
        assert (out / name).exists(), name
    assert "n < 100" in capsys.readouterr().out


def test_main_fails_cleanly_on_empty_dataset(tmp_path):
    csv = tmp_path / "dataset.csv"
    frame([]).to_csv(csv, index=False)
    assert main(["--data", str(csv), "--out", str(tmp_path / "out")]) == 1
