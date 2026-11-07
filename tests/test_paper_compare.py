import numpy as np
import pandas as pd
import pytest

from kalshi_lab.paper_compare import (
    PAPER_SPORTS,
    accuracy_by_horizon,
    build_comparison,
    main,
    volume_error_spearman,
)


def frame(n=300, seed=0, horizons=(168, 24, 1), noise=0.2):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = float(rng.uniform(0.05, 0.95))
        y = int(rng.uniform() < p)
        vol = float(rng.uniform(100, 100000))
        for h in horizons:
            rows.append((f"T{i}", f"E{i}", "Sports", h, p, y, vol))
    return pd.DataFrame(rows, columns=["ticker", "event_ticker", "category", "horizon_h", "prob", "outcome", "volume_fp"])


def test_paper_constants_match_table_1_and_4_sports_rows():
    assert PAPER_SPORTS[168] == {"brier": 0.107, "accuracy": 0.846}
    assert PAPER_SPORTS[24] == {"brier": 0.155, "accuracy": 0.762}
    assert PAPER_SPORTS[1] == {"brier": 0.104, "accuracy": 0.855}


def test_accuracy_by_horizon_known_values():
    df = pd.DataFrame({"horizon_h": [24] * 4, "prob": [0.8, 0.6, 0.5, 0.3], "outcome": [1, 0, 1, 0]})
    out = accuracy_by_horizon(df).set_index("horizon_h")
    # 0.8->1 correct, 0.6->0 wrong, 0.5 counts as wrong (paper rule), 0.3->0 correct
    assert out.loc[24, "accuracy"] == pytest.approx(0.5) and out.loc[24, "n"] == 4


def test_volume_error_spearman_detects_negative_relationship():
    rng = np.random.default_rng(1)
    vol = rng.uniform(100, 1e5, 400)
    # squared error shrinks as volume grows
    p = np.where(rng.uniform(size=400) < vol / 1e5, 0.95, 0.5)
    y = np.ones(400, dtype=int)
    df = pd.DataFrame({"horizon_h": 24, "prob": p, "outcome": y, "volume_fp": vol})
    rho, pval = volume_error_spearman(df, 24)
    assert rho < -0.3 and pval < 0.01


def test_volume_error_spearman_small_sample_returns_nan():
    df = pd.DataFrame({"horizon_h": [24] * 5, "prob": [.5] * 5, "outcome": [1] * 5, "volume_fp": [1.0] * 5})
    rho, pval = volume_error_spearman(df, 24)
    assert np.isnan(rho) and np.isnan(pval)


def test_build_comparison_has_expected_columns_and_findings():
    df = frame()
    out = build_comparison(df, df)
    assert list(out.columns) == ["finding", "paper", "ours", "n", "verdict"]
    assert out["verdict"].isin(["consistent", "inconsistent", "inconclusive"]).all()
    findings = " ".join(out["finding"])
    for key in ("Brier", "accuracy", "tail", "volume", "anchor"):
        assert key.lower() in findings.lower()


def test_build_comparison_marks_small_samples_inconclusive():
    out = build_comparison(frame(n=20), frame(n=20))
    assert (out["verdict"] == "inconclusive").any()


def test_main_writes_csv(tmp_path):
    csv = tmp_path / "d.csv"
    frame().to_csv(csv, index=False)
    out = tmp_path / "o"
    assert main(["--data", str(csv), "--data-occurrence", str(csv), "--out", str(out)]) == 0
    assert (out / "paper_comparison.csv").exists()


def test_main_fails_cleanly_when_no_sports_rows(tmp_path):
    csv = tmp_path / "d.csv"
    frame().assign(category="Economics").to_csv(csv, index=False)
    assert main(["--data", str(csv), "--out", str(tmp_path / "o")]) == 1
