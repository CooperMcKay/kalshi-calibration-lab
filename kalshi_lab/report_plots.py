"""Report figures: horizon comparison, calibration-gap bars, and probability histograms."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kalshi_lab.calibration import score_by  # noqa: E402
from kalshi_lab.metrics import reliability_table  # noqa: E402

SIGNIFICANCE_SE = 2.0
GAP_OK, GAP_FLAG = "#4c78a8", "#d62728"
METRICS = ("brier", "log_loss", "ece")


def plot_horizon_comparison(scores: pd.DataFrame, path: Path) -> None:
    """One panel per metric, one bar per horizon (lower is better for all three)."""
    scores = scores.sort_values("horizon_h", ascending=False)
    labels = [f"{int(h)}h" for h in scores["horizon_h"]]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 3.8))
    for ax, metric in zip(axes, METRICS):
        ax.bar(labels, scores[metric], color=GAP_OK)
        ax.set(title=metric.replace("_", " "), xlabel="Horizon before close")
    axes[0].set_ylabel("Score (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration_gap(df: pd.DataFrame, horizon_h: int, path: Path) -> None:
    """Actual minus predicted per bucket; red only when the gap exceeds 2 standard errors."""
    sub = df[df["horizon_h"] == horizon_h]
    t = reliability_table(sub["prob"].to_numpy(), sub["outcome"].to_numpy())
    t = t[t["n"] > 0]
    se = np.sqrt(t["mean_pred"] * (1 - t["mean_pred"]) / t["n"])
    flagged = t["gap"].abs() > SIGNIFICANCE_SE * se
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t["bucket"], t["gap"], [GAP_FLAG if f else GAP_OK for f in flagged])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Market price bucket", ylabel="Actual - predicted",
           title=f"Calibration gap, {horizon_h}h before close (red: |gap| > 2 SE)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prob_histogram(df: pd.DataFrame, horizon_h: int, path: Path) -> None:
    """Where prices sit, split by how the market resolved."""
    sub = df[df["horizon_h"] == horizon_h]
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(7, 4))
    for outcome, label in ((1, "resolved YES"), (0, "resolved NO")):
        vals = sub.loc[sub["outcome"] == outcome, "prob"]
        if len(vals):
            ax.hist(vals, bins=bins, alpha=0.6, label=label)
    ax.set(xlabel="Market price", ylabel="Markets", title=f"Price distribution, {horizon_h}h before close")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write report figures to outputs/")
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.data)
    if df.empty:
        print("error: dataset is empty; run kalshi_lab.collect first")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_horizon_comparison(score_by(df, "horizon_h"), out / "horizon_comparison.png")
    for h in sorted(df["horizon_h"].unique()):
        plot_calibration_gap(df, int(h), out / f"calibration_gap_{int(h)}h.png")
        plot_prob_histogram(df, int(h), out / f"prob_histogram_{int(h)}h.png")
    print(f"wrote figures to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
