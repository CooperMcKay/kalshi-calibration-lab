"""Calibration analysis: scores by slice, fixed cohorts, tail truncation, bucket significance, reliability plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from kalshi_lab.metrics import (  # noqa: E402
    benjamini_hochberg,
    brier,
    ece,
    log_loss,
    murphy_decomposition,
    reliability_table,
)

MIN_SLICE_N = 100
TAIL_LO, TAIL_HI = 0.02, 0.98
DEFAULT_BASE_HORIZON_H = 24


def score_by(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Brier / log loss / ECE per group, with the 'always say 50%' Brier for reference."""
    keys = [by] if isinstance(by, str) else list(by)
    rows = []
    for group, g in df.groupby(keys, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        p, y = g["prob"].to_numpy(), g["outcome"].to_numpy()
        rows.append({
            **dict(zip(keys, group)), "n": len(g), "base_rate": float(y.mean()),
            "brier": brier(p, y), "log_loss": log_loss(p, y), "ece": ece(p, y),
            "brier_always_half": brier(np.full(len(g), 0.5), y),
        })
    return pd.DataFrame(rows)


def volume_band(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["volume_fp"], bins=[-np.inf, 1000, 10000, np.inf], right=False,
                  labels=["<1k", "1k-10k", ">=10k"]).astype(str)


def fixed_cohort(df: pd.DataFrame, base_horizon_h: int) -> pd.DataFrame:
    """Only markets that have a price at base_horizon_h, so horizons compare the same markets."""
    tickers = df.loc[df["horizon_h"] == base_horizon_h, "ticker"].unique()
    return df[df["ticker"].isin(tickers)]


def tail_truncate(df: pd.DataFrame, lo: float = TAIL_LO, hi: float = TAIL_HI) -> pd.DataFrame:
    """Drop near-certain prices (<= lo or >= hi) to look only at genuinely uncertain markets."""
    return df[(df["prob"] > lo) & (df["prob"] < hi)]


def bucket_significance(p, y, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table plus a binomial p-value per bucket and Benjamini-Hochberg significance."""
    t = reliability_table(p, y, n_bins)
    pvals = []
    for _, r in t.iterrows():
        if r["n"] == 0:
            pvals.append(np.nan)
            continue
        k = int(round(r["actual"] * r["n"]))
        pvals.append(binomtest(k, int(r["n"]), float(np.clip(r["mean_pred"], 0, 1))).pvalue)
    t["p_value"] = pvals
    t["bh_significant"] = benjamini_hochberg(np.array(pvals))
    return t


def plot_reliability(df: pd.DataFrame, horizon_h: int, path: Path) -> None:
    sub = df[df["horizon_h"] == horizon_h]
    t = reliability_table(sub["prob"].to_numpy(), sub["outcome"].to_numpy())
    t = t[t["n"] > 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.errorbar(t["mean_pred"], t["actual"], yerr=t["ci95"].fillna(0), fmt="o-", capsize=3, label="Kalshi")
    for _, r in t.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_pred"], r["actual"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    ax.set(xlim=(0, 1), ylim=(-0.05, 1.05), xlabel="Market price (predicted)", ylabel="Actual frequency",
           title=f"Reliability, {horizon_h}h before close (n={len(sub)})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _warn_small(name: str, table: pd.DataFrame, keys: list[str]) -> None:
    for _, r in table[table["n"] < MIN_SLICE_N].iterrows():
        label = ", ".join(f"{k}={r[k]}" for k in keys)
        print(f"warning: {name} [{label}] n < {MIN_SLICE_N} ({int(r['n'])}): treat as anecdote")


def _murphy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in df.groupby("horizon_h"):
        d = murphy_decomposition(g["prob"].to_numpy(), g["outcome"].to_numpy())
        rows.append({"horizon_h": h, "n": len(g), "brier": brier(g["prob"], g["outcome"]), **d})
    return pd.DataFrame(rows)


def _anchor_comparison(close_df: pd.DataFrame, occ_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for name, d in (("close", close_df), ("occurrence", occ_df)):
        sports = d[d["category"] == "Sports"]
        if len(sports):
            parts.append(score_by(sports, "horizon_h").assign(anchor=name))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score calibration of Kalshi prices by slice")
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--data-occurrence", default="data/dataset_occurrence.csv",
                    help="optional dataset built with --anchor occurrence")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--base-horizon", type=int, default=DEFAULT_BASE_HORIZON_H)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.data)
    if df.empty:
        print("error: dataset is empty; run kalshi_lab.collect first")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = df.assign(volume_band=volume_band(df))

    tables = {
        "scores_by_horizon": (score_by(df, "horizon_h"), ["horizon_h"]),
        "scores_by_category": (score_by(df, ["category", "horizon_h"]), ["category", "horizon_h"]),
        "scores_by_volume_band": (score_by(df, ["volume_band", "horizon_h"]), ["volume_band", "horizon_h"]),
        "scores_fixed_cohort": (score_by(fixed_cohort(df, args.base_horizon), "horizon_h"), ["horizon_h"]),
        "scores_tail_truncated": (score_by(tail_truncate(df), "horizon_h"), ["horizon_h"]),
    }
    for name, (table, keys) in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
        _warn_small(name, table, keys)
    _murphy_table(df).to_csv(out / "murphy_by_horizon.csv", index=False)

    for h in sorted(df["horizon_h"].unique()):
        sub = df[df["horizon_h"] == h]
        bucket_significance(sub["prob"].to_numpy(), sub["outcome"].to_numpy()).to_csv(
            out / f"reliability_{h}h.csv", index=False)
        plot_reliability(df, int(h), out / f"reliability_{h}h.png")

    occ_path = Path(args.data_occurrence)
    if occ_path.exists():
        cmp = _anchor_comparison(df, pd.read_csv(occ_path))
        if len(cmp):
            cmp.to_csv(out / "scores_by_anchor.csv", index=False)
    print(f"wrote results to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
