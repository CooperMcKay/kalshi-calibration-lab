"""Can a small model recalibrate Kalshi prices? Time-ordered train/test comparison of simple recalibrators."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from kalshi_lab.metrics import brier, ece, log_loss

MIN_ROWS = 200
TEST_FRAC = 0.3
PRICE_EPS = 1e-3


def time_split(df: pd.DataFrame, test_frac: float = TEST_FRAC) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oldest events train, newest test. Cut falls on an event boundary so no event spans both sets."""
    ordered = df.sort_values(["close_time", "event_ticker"]).reset_index(drop=True)
    cut = int(round(len(ordered) * (1 - test_frac)))
    while 0 < cut < len(ordered) and ordered.loc[cut, "event_ticker"] == ordered.loc[cut - 1, "event_ticker"]:
        cut += 1
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    p = df["prob"].clip(PRICE_EPS, 1 - PRICE_EPS)
    return pd.DataFrame({
        "logit_price": np.log(p / (1 - p)),
        "log_volume": np.log1p(df["volume_fp"]),
        "log_open_interest": np.log1p(df["open_interest_fp"]),
    }, index=df.index)


def _score(name: str, p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 0.0, 1.0)
    return {"model": name, "n_test": len(y), "brier": brier(p, y), "log_loss": log_loss(p, y), "ece": ece(p, y)}


def evaluate(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Fit on the early events, score on the late ones: market price vs three recalibrators."""
    sub = df[df["horizon_h"] == horizon_h]
    if len(sub) < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} rows at {horizon_h}h, got {len(sub)}")
    train, test = time_split(sub)
    y_tr, y_te = train["outcome"].to_numpy(), test["outcome"].to_numpy()
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2 or len(test) == 0:
        raise ValueError("train and test sets must each contain both outcomes")

    X_tr, X_te = make_features(train), make_features(test)
    platt = LogisticRegression(C=1e6).fit(X_tr[["logit_price"]], y_tr)
    print(f"Platt fit at {horizon_h}h: slope={platt.coef_[0][0]:.3f} intercept={platt.intercept_[0]:.3f} "
          "(slope 1, intercept 0 = already calibrated)")
    full = LogisticRegression(C=1.0, max_iter=1000).fit(X_tr, y_tr)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(train["prob"], y_tr)

    rows = [
        _score("market", test["prob"].to_numpy(), y_te),
        _score("logistic_platt", platt.predict_proba(X_te[["logit_price"]])[:, 1], y_te),
        _score("logistic_full", full.predict_proba(X_te)[:, 1], y_te),
        _score("isotonic", iso.predict(test["prob"]), y_te),
    ]
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recalibrate Kalshi prices and compare on a time-ordered test set")
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)
    try:
        result = evaluate(pd.read_csv(args.data), args.horizon)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / f"model_{args.horizon}h.csv", index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
