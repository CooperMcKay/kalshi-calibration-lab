import numpy as np
import pandas as pd
import pytest

from kalshi_lab.model import MIN_ROWS, evaluate, main, make_features, time_split


def synthetic(n=600, seed=0, slope=2.0, horizon=24):
    """Prices are under-confident: the true log-odds are `slope` times the market log-odds."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, n)
    true_p = 1 / (1 + np.exp(-slope * np.log(p / (1 - p))))
    y = (rng.uniform(size=n) < true_p).astype(int)
    times = pd.date_range("2026-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)], "event_ticker": [f"E{i // 2}" for i in range(n)],
        "close_time": times, "horizon_h": horizon, "prob": p, "outcome": y,
        "volume_fp": rng.uniform(100, 5000, n), "open_interest_fp": rng.uniform(10, 500, n),
    })


def test_time_split_orders_by_time_and_keeps_events_together():
    train, test = time_split(synthetic(), test_frac=0.3)
    assert train["close_time"].max() <= test["close_time"].min()
    assert not set(train["event_ticker"]) & set(test["event_ticker"])
    assert 0.25 < len(test) / (len(train) + len(test)) < 0.35


def test_time_split_does_not_mutate_input():
    df = synthetic()
    before = df.copy()
    time_split(df)
    pd.testing.assert_frame_equal(df, before)


def test_make_features_columns_and_finite_at_extremes():
    df = synthetic(n=10).assign(prob=[0.0, 1.0] + [0.5] * 8)
    X = make_features(df)
    assert list(X.columns) == ["logit_price"]  # final volume/open interest include post-horizon trading: lookahead
    assert np.isfinite(X.to_numpy()).all()


def test_evaluate_returns_three_models_with_scores():
    out = evaluate(synthetic(), 24).set_index("model")
    assert set(out.index) == {"market", "logistic_platt", "isotonic"}
    assert {"n_test", "brier", "log_loss", "ece"} <= set(out.columns)
    assert out["n_test"].nunique() == 1


def test_evaluate_platt_beats_market_on_miscalibrated_prices(capsys):
    out = evaluate(synthetic(n=3000), 24).set_index("model")
    assert out.loc["logistic_platt", "brier"] < out.loc["market", "brier"]
    assert "slope" in capsys.readouterr().out


def test_evaluate_rejects_small_sample():
    with pytest.raises(ValueError, match=str(MIN_ROWS)):
        evaluate(synthetic(n=MIN_ROWS - 1), 24)


def test_evaluate_rejects_single_class():
    with pytest.raises(ValueError):
        evaluate(synthetic().assign(outcome=1), 24)


def test_evaluate_filters_to_requested_horizon():
    df = pd.concat([synthetic(n=300), synthetic(n=50, horizon=1)])
    assert evaluate(df, 24)["n_test"].iloc[0] <= 90


def test_main_writes_csv_and_fails_cleanly_on_small_data(tmp_path):
    csv = tmp_path / "d.csv"
    synthetic().to_csv(csv, index=False)
    assert main(["--data", str(csv), "--horizon", "24", "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "model_24h.csv").exists()
    synthetic(n=50).to_csv(csv, index=False)
    assert main(["--data", str(csv), "--horizon", "24", "--out", str(tmp_path / "o2")]) == 1
