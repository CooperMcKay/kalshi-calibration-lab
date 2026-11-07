"""Scoring metrics for probability forecasts (pure functions, no I/O).

p is an array of forecast probabilities in [0, 1]; y is an array of 0/1 outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_arrays(p, y) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(p, dtype=float), np.asarray(y, dtype=float)


def brier(p, y) -> float:
    """Mean squared error between probability and outcome. 0 is perfect, 0.25 is 'always 50%'."""
    p, y = _as_arrays(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps: float = 1e-6) -> float:
    """Negative mean log-likelihood. Probabilities are clipped to [eps, 1-eps] so it stays finite."""
    p, y = _as_arrays(p, y)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin index; p == 1.0 falls in the last bin."""
    return np.minimum((p * n_bins).astype(int), n_bins - 1)


def _bucket_label(k: int, n_bins: int) -> str:
    digits = 1 if n_bins <= 10 else 2
    return f"{k / n_bins:.{digits}f}-{(k + 1) / n_bins:.{digits}f}"


def reliability_table(p, y, n_bins: int = 10) -> pd.DataFrame:
    """Per-bucket predicted vs. actual frequency with a binomial 95% CI on the actual rate.

    Empty buckets are kept (n=0, NaN elsewhere). ci95 is NaN when n < 2.
    """
    p, y = _as_arrays(p, y)
    idx = _bin_index(p, n_bins)
    rows = []
    for k in range(n_bins):
        mask = idx == k
        n = int(mask.sum())
        if n == 0:
            rows.append((_bucket_label(k, n_bins), 0, np.nan, np.nan, np.nan, np.nan))
            continue
        mean_pred = float(p[mask].mean())
        actual = float(y[mask].mean())
        ci = 1.96 * np.sqrt(actual * (1 - actual) / n) if n >= 2 else np.nan
        rows.append((_bucket_label(k, n_bins), n, mean_pred, actual, actual - mean_pred, ci))
    return pd.DataFrame(rows, columns=["bucket", "n", "mean_pred", "actual", "gap", "ci95"])


def ece(p, y, n_bins: int = 10) -> float:
    """Expected calibration error: n-weighted mean of |actual - predicted| over non-empty buckets."""
    t = reliability_table(p, y, n_bins)
    t = t[t["n"] > 0]
    return float((t["n"] * t["gap"].abs()).sum() / t["n"].sum())


def murphy_decomposition(p, y, n_bins: int = 10) -> dict[str, float]:
    """Murphy (1973): Brier = reliability - resolution + uncertainty (exact when p is constant within bins)."""
    p, y = _as_arrays(p, y)
    idx = _bin_index(p, n_bins)
    n_total = len(p)
    base = float(y.mean())
    reliability = resolution = 0.0
    for k in range(n_bins):
        mask = idx == k
        n = int(mask.sum())
        if n == 0:
            continue
        pbar, obar = float(p[mask].mean()), float(y[mask].mean())
        reliability += n * (pbar - obar) ** 2
        resolution += n * (obar - base) ** 2
    reliability /= n_total
    resolution /= n_total
    uncertainty = base * (1 - base)
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_binned": reliability - resolution + uncertainty,
    }


def directional_accuracy(p, y) -> float:
    """Share of forecasts where p > 0.5 matched the outcome. A p of exactly 0.5 counts as incorrect."""
    p, y = _as_arrays(p, y)
    correct = ((p > 0.5) & (y == 1)) | ((p < 0.5) & (y == 0))
    return float(correct.mean())


def benjamini_hochberg(pvals, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of nulls rejected at false-discovery rate alpha. NaN p-values are never rejected."""
    pv = np.asarray(pvals, dtype=float)
    out = np.zeros(pv.shape, dtype=bool)
    valid = np.flatnonzero(~np.isnan(pv))
    m = len(valid)
    if m == 0:
        return out
    order = valid[np.argsort(pv[valid])]
    thresholds = alpha * np.arange(1, m + 1) / m
    passing = np.flatnonzero(pv[order] <= thresholds)
    if len(passing) == 0:
        return out
    out[order[: passing.max() + 1]] = True
    return out
