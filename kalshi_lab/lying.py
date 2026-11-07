"""'Is Kalshi lying?': when a price says X%, how often does it happen, overall and by category.

Every interval resamples whole clusters (default: events), because markets inside one event
(a rate ladder, a temperature bracket) move together and are not independent observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kalshi_lab.calibration import tail_truncate  # noqa: E402
from kalshi_lab.metrics import benjamini_hochberg, brier, ece  # noqa: E402
from kalshi_lab.stats_tests import LONGSHOT, FAVORITE, _slope  # noqa: E402

# Bands are centred on multiples of 10 so "a 70% price" means the 65-75% band.
BANDS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55),
         (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.95), (0.95, 1.0)]
N_BOOT = 1000
MIN_BAND_N = 20
MIN_EVENTS = 100
ALPHA = 0.05
KEY_HORIZON_H = 24
POOLED_NON_SPORTS = "All non-sports"


def _cluster_draws(p, y, groups, stat, n_boot: int, seed: int) -> np.ndarray:
    p, y, groups = np.asarray(p, float), np.asarray(y, float), np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    members = np.split(order, starts[1:])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        draws[b] = stat(p[idx], y[idx])
    return draws


def cluster_bootstrap(p, y, groups, stat, n_boot: int = N_BOOT, seed: int = 0, level: float = 0.95):
    """(point, lo, hi) of stat(p, y), resampling whole clusters so correlated rows count once."""
    draws = _cluster_draws(p, y, groups, stat, n_boot, seed)
    a = (1 - level) / 2
    lo, hi = np.nanquantile(draws, [a, 1 - a])
    return float(stat(np.asarray(p, float), np.asarray(y, float))), float(lo), float(hi)


def _gap(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(y) - np.mean(p))


def _band_label(lo: float, hi: float) -> str:
    return f"{lo:.0%}-{hi:.0%}"


def band_table(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0, cluster_col: str = "event_ticker",
               bands: list[tuple[float, float]] = BANDS) -> pd.DataFrame:
    """Per price band: what the market said, what happened, the gap, a cluster-bootstrap CI and BH flag."""
    p_all, y_all = df["prob"].to_numpy(float), df["outcome"].to_numpy(float)
    groups = df[cluster_col].to_numpy()
    rows, pvals = [], []
    for k, (lo, hi) in enumerate(bands):
        mask = (p_all >= lo) & ((p_all < hi) if k < len(bands) - 1 else (p_all <= hi))
        n = int(mask.sum())
        row = {"band": _band_label(lo, hi), "band_lo": lo, "band_hi": hi, "n": n,
               "n_events": int(len(np.unique(groups[mask]))) if n else 0,
               "mean_price": np.nan, "actual": np.nan, "gap": np.nan, "gap_lo": np.nan, "gap_hi": np.nan}
        pv = np.nan
        if n:
            p, y = p_all[mask], y_all[mask]
            row.update(mean_price=float(p.mean()), actual=float(y.mean()), gap=_gap(p, y))
            if n >= MIN_BAND_N:
                draws = _cluster_draws(p, y, groups[mask], _gap, n_boot, seed + k)
                row["gap_lo"], row["gap_hi"] = (float(v) for v in np.nanquantile(draws, [0.025, 0.975]))
                pv = min(1.0, 2 * max(min(np.mean(draws <= 0), np.mean(draws >= 0)), 1 / (n_boot + 1)))
        rows.append(row)
        pvals.append(pv)
    t = pd.DataFrame(rows)
    t["p_value"] = pvals
    t["bh_significant"] = benjamini_hochberg(np.array(pvals), ALPHA)
    t["verdict"] = [_band_verdict(r) for _, r in t.iterrows()]
    return t


def _band_verdict(r: pd.Series) -> str:
    if r["n"] < MIN_BAND_N:
        return "too few"
    if r["bh_significant"]:
        return "price too low" if r["gap"] > 0 else "price too high"
    return "consistent with calibrated"


def _category_row(g: pd.DataFrame, n_boot: int, seed: int, cluster_col: str) -> dict:
    p, y, groups = g["prob"].to_numpy(float), g["outcome"].to_numpy(float), g[cluster_col].to_numpy()
    n_events = int(len(np.unique(groups)))
    row = {"n": len(g), "n_events": n_events, "base_rate": float(y.mean())}
    keys = ("brier", "brier_lo", "brier_hi", "ece", "slope", "slope_lo", "slope_hi",
            "mean_gap", "mean_gap_lo", "mean_gap_hi", "longshot_gap", "favorite_gap")
    row.update({k: np.nan for k in keys})
    row["ece"] = ece(p, y)
    row["longshot_gap"] = _gap(p[p < LONGSHOT], y[p < LONGSHOT]) if (p < LONGSHOT).any() else np.nan
    row["favorite_gap"] = _gap(p[p > FAVORITE], y[p > FAVORITE]) if (p > FAVORITE).any() else np.nan
    if n_events < MIN_EVENTS:
        return {**row, "verdict": "too few events"}
    row["brier"], row["brier_lo"], row["brier_hi"] = cluster_bootstrap(p, y, groups, brier, n_boot, seed)
    row["mean_gap"], row["mean_gap_lo"], row["mean_gap_hi"] = cluster_bootstrap(p, y, groups, _gap, n_boot, seed)
    if len(np.unique(y)) == 2:
        row["slope"], row["slope_lo"], row["slope_hi"] = cluster_bootstrap(p, y, groups, _slope, n_boot, seed)
    flags = []
    if row["mean_gap_lo"] > 0 or row["mean_gap_hi"] < 0:
        flags.append("average bias")
    if row["slope_lo"] > 1 or row["slope_hi"] < 1:  # NaN comparisons are False
        flags.append("tail bias (slope != 1)")
    row["verdict"] = " + ".join(flags) if flags else "consistent with calibrated"
    return row


def category_table(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0,
                   cluster_col: str = "event_ticker") -> pd.DataFrame:
    """Headline scores and a calibration verdict for each (category, horizon)."""
    rows = []
    for (cat, h), g in df.groupby(["category", "horizon_h"], sort=True):
        rows.append({"category": cat, "horizon_h": int(h), **_category_row(g, n_boot, seed, cluster_col)})
    return pd.DataFrame(rows)


def category_band_gaps(df: pd.DataFrame, n_boot: int = N_BOOT, cluster_col: str = "event_ticker") -> pd.DataFrame:
    parts = []
    for (cat, h), g in df.groupby(["category", "horizon_h"], sort=True):
        parts.append(band_table(g, n_boot, cluster_col=cluster_col).assign(category=cat, horizon_h=int(h)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def with_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Append a copy of every non-sports row labelled as one pooled group (small categories alone are too thin)."""
    non_sports = df[df["category"] != "Sports"]
    return pd.concat([df, non_sports.assign(category=POOLED_NON_SPORTS)], ignore_index=True)


def plot_bands(t: pd.DataFrame, horizon_h: int, path: Path) -> None:
    t = t[t["n"] >= MIN_BAND_N]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="gray", ls="--")
    x = np.arange(len(t))
    err = np.vstack([t["gap"] - t["gap_lo"], t["gap_hi"] - t["gap"]])
    colors = np.where(t["bh_significant"], "tab:red", "tab:blue")
    ax.errorbar(x, t["gap"], yerr=err, fmt="none", ecolor="gray", capsize=3)
    ax.scatter(x, t["gap"], c=colors, zorder=3)
    ax.set_xticks(x, t["band"], rotation=45)
    for xi, n in zip(x, t["n"]):
        ax.annotate(f"n={n}", (xi, 0), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7)
    ax.set(xlabel="What Kalshi said", ylabel="Happened minus said (95% CI)",
           title=f"When Kalshi says X%, how far off is it? ({horizon_h}h before close)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_category_gaps(t: pd.DataFrame, horizon_h: int, path: Path) -> None:
    t = t[(t["horizon_h"] == horizon_h) & t["mean_gap"].notna()].sort_values("mean_gap")
    fig, ax = plt.subplots(figsize=(7, 0.6 * max(len(t), 3) + 1.5))
    y = np.arange(len(t))
    err = np.vstack([t["mean_gap"] - t["mean_gap_lo"], t["mean_gap_hi"] - t["mean_gap"]])
    ax.axvline(0, color="gray", ls="--")
    ax.errorbar(t["mean_gap"], y, xerr=err, fmt="o", capsize=3)
    ax.set_yticks(y, [f"{c} ({e} events)" for c, e in zip(t["category"], t["n_events"])])
    ax.set(xlabel="Average happened minus said (95% CI)", title=f"Average price bias by category ({horizon_h}h)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Is Kalshi lying? Gaps by price band and category")
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--cluster", default="event_ticker", help="column whose groups are resampled together")
    args = ap.parse_args(argv)
    try:
        df = pd.read_csv(args.data)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if df.empty:
        print("error: dataset is empty; run kalshi_lab.collect first")
        return 1
    df = df.assign(close_date=df["close_time"].astype(str).str[:10])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pooled = pd.concat([band_table(g, args.n_boot, cluster_col=args.cluster).assign(horizon_h=int(h))
                        for h, g in df.groupby("horizon_h")], ignore_index=True)
    pooled.to_csv(out / "lying_by_band.csv", index=False)
    grouped = with_pooled(df)
    cats = category_table(grouped, args.n_boot, cluster_col=args.cluster)
    cats.to_csv(out / "lying_by_category.csv", index=False)
    category_table(tail_truncate(grouped), args.n_boot, cluster_col=args.cluster).to_csv(
        out / "lying_by_category_truncated.csv", index=False)
    category_band_gaps(grouped, args.n_boot, args.cluster).to_csv(out / "lying_category_band_gaps.csv", index=False)
    key = KEY_HORIZON_H if (df["horizon_h"] == KEY_HORIZON_H).any() else int(df["horizon_h"].max())
    plot_bands(pooled[pooled["horizon_h"] == key], key, out / f"lying_bands_{KEY_HORIZON_H}h.png")
    plot_category_gaps(cats, key, out / f"lying_category_gaps_{KEY_HORIZON_H}h.png")
    print(f"wrote lying tables and plots to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
