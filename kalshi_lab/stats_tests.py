"""Statistical follow-ups: bootstrap intervals, favorite-longshot bias, and the volume/closeness confound."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

from kalshi_lab.metrics import brier, ece

MIN_ROWS = 100
LONGSHOT, FAVORITE = 0.2, 0.8
CLIP = 0.01
N_BOOT = 1000


def bootstrap_ci(p, y, stat, n_boot: int = N_BOOT, seed: int = 0, level: float = 0.95):
    """(point, lo, hi) of stat(p, y) by resampling rows. Rows are one per event, so they are independent."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(p), size=(n_boot, len(p)))
    draws = np.array([stat(p[i], y[i]) for i in idx])
    a = (1 - level) / 2
    lo, hi = np.quantile(draws, [a, 1 - a])
    return float(stat(p, y)), float(lo), float(hi)


def bootstrap_scores(df: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for h, g in df.groupby("horizon_h"):
        p, y = g["prob"].to_numpy(), g["outcome"].to_numpy()
        b, bl, bh = bootstrap_ci(p, y, brier, n_boot)
        e, el, eh = bootstrap_ci(p, y, ece, n_boot)
        rows.append({"horizon_h": h, "n": len(g), "brier": b, "brier_lo": bl, "brier_hi": bh,
                     "ece": e, "ece_lo": el, "ece_hi": eh})
    return pd.DataFrame(rows)


def _slope(p: np.ndarray, y: np.ndarray) -> float:
    x = np.log(np.clip(p, CLIP, 1 - CLIP) / (1 - np.clip(p, CLIP, 1 - CLIP))).reshape(-1, 1)
    return float(LogisticRegression(C=1e6, max_iter=500).fit(x, y).coef_[0, 0])


def _gap(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return float(y[mask].mean() - p[mask].mean()) if mask.sum() else float("nan")


def favorite_longshot(df: pd.DataFrame, horizon_h: int, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Slope of logit(outcome rate) on logit(price). Slope > 1: favorites underpriced, longshots overpriced."""
    g = df[df["horizon_h"] == horizon_h]
    p, y = g["prob"].to_numpy(float), g["outcome"].to_numpy(int)
    out = {"horizon_h": horizon_h, "n": len(g), "slope": np.nan, "slope_lo": np.nan, "slope_hi": np.nan,
           "longshot_gap": np.nan, "favorite_gap": np.nan, "verdict": "inconclusive"}
    if len(g) < MIN_ROWS or len(np.unique(y)) < 2:
        return out
    slope, lo, hi = bootstrap_ci(p, y, _slope, n_boot, seed)
    verdict = ("favorite-longshot bias" if lo > 1 else
               "reverse bias" if hi < 1 else "no evidence of bias")
    return {**out, "slope": slope, "slope_lo": lo, "slope_hi": hi, "verdict": verdict,
            "longshot_gap": _gap(p, y, p < LONGSHOT), "favorite_gap": _gap(p, y, p > FAVORITE)}


def _residual(a: np.ndarray, control: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(control, a, 1)
    return a - (slope * control + intercept)


def volume_closeness(df: pd.DataFrame, horizon_h: int, volume_col: str = "volume_fp") -> dict:
    """Do busy markets have bigger errors directly, or only because they are closer (more uncertain) games?"""
    g = df[(df["horizon_h"] == horizon_h) & df[volume_col].notna()]
    out = {"horizon_h": horizon_h, "n": len(g), "rho_volume_vs_distance": np.nan,
           "rho_volume_error": np.nan, "partial_rho_volume_error": np.nan}
    if len(g) < 50:
        return out
    p, y = g["prob"].to_numpy(float), g["outcome"].to_numpy(float)
    vol = rankdata(g[volume_col].to_numpy(float))
    dist = rankdata(np.abs(p - 0.5))
    err = rankdata((p - y) ** 2)
    partial = float(np.corrcoef(_residual(vol, dist), _residual(err, dist))[0, 1])
    return {**out, "rho_volume_vs_distance": float(spearmanr(vol, dist)[0]),
            "rho_volume_error": float(spearmanr(vol, err)[0]), "partial_rho_volume_error": partial}


def _volume_table(df: pd.DataFrame) -> pd.DataFrame:
    """Volume vs error with both final volume (has lookahead) and pre-horizon volume (clean)."""
    rows = []
    cols = [c for c in ("volume_fp", "volume_pre") if c in df.columns]
    scopes = [("ALL", df)] + [(s, g) for s, g in df.groupby("series_ticker") if len(g) >= MIN_ROWS]
    for name, d in scopes:
        for h in sorted(d["horizon_h"].unique()):
            for col in cols:
                rows.append({"scope": name, "volume_col": col, **volume_closeness(d, int(h), col)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bootstrap CIs, favorite-longshot bias, volume/closeness")
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    try:
        df = pd.read_csv(args.data)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if df.empty:
        print("error: dataset is empty; run kalshi_lab.collect first")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bootstrap_scores(df, args.n_boot).to_csv(out / "stats_bootstrap.csv", index=False)
    pd.DataFrame([favorite_longshot(df, int(h), args.n_boot) for h in sorted(df["horizon_h"].unique())]).to_csv(
        out / "stats_favorite_longshot.csv", index=False)
    _volume_table(df).to_csv(out / "stats_volume_closeness.csv", index=False)
    print(f"wrote stats results to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
