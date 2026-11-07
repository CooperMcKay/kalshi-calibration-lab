"""Put our results next to the numbers published in Kagan & Baiocchi, 'Calibration in Prediction Markets'.

Paper figures are for ALL Kalshi sports markets (1.13M markets, close-anchored); ours are one game-winner
market per event, so levels differ by construction. Directional claims are the fairer comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kalshi_lab.metrics import brier, directional_accuracy

# Table 1 (Brier) and Table 4 (naive accuracy), Sports row. 1-Wk = 168h, 1-Day = 24h, 1-Hr = 1h.
PAPER_SPORTS = {
    168: {"brier": 0.107, "accuracy": 0.846},
    24: {"brier": 0.155, "accuracy": 0.762},
    1: {"brier": 0.104, "accuracy": 0.855},
}
MIN_N = 100
LEVEL_TOLERANCE = {"brier": 0.03, "accuracy": 0.05}
TAIL_LO, TAIL_HI = 0.02, 0.98
ALPHA = 0.05
COLUMNS = ["finding", "paper", "ours", "n", "verdict"]


def accuracy_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"horizon_h": h, "n": len(g), "accuracy": directional_accuracy(g["prob"].to_numpy(), g["outcome"].to_numpy())}
        for h, g in df.groupby("horizon_h")
    ]
    return pd.DataFrame(rows, columns=["horizon_h", "n", "accuracy"])


def volume_error_spearman(df: pd.DataFrame, horizon_h: int, volume_col: str = "volume_fp") -> tuple[float, float]:
    """Spearman correlation between volume and squared error. Negative = busier markets are more accurate."""
    g = df[df["horizon_h"] == horizon_h]
    if len(g) < MIN_N // 2 or g[volume_col].nunique() < 2:
        return float("nan"), float("nan")
    err = (g["prob"] - g["outcome"]) ** 2
    rho, pval = spearmanr(g[volume_col], err)
    return float(rho), float(pval)


def _row(finding: str, paper, ours, n: int, verdict: str) -> dict:
    return {"finding": finding, "paper": paper, "ours": ours, "n": n, "verdict": verdict}


def _level_verdict(ours: float, paper: float, n: int, metric: str) -> str:
    if n < MIN_N:
        return "inconclusive"
    return "consistent" if abs(ours - paper) <= LEVEL_TOLERANCE[metric] else "inconsistent"


def _sports(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["category"] == "Sports"]


def _level_rows(sports: pd.DataFrame) -> list[dict]:
    rows = []
    acc = accuracy_by_horizon(sports).set_index("horizon_h")
    for h, paper in PAPER_SPORTS.items():
        g = sports[sports["horizon_h"] == h]
        if g.empty:
            continue
        b = brier(g["prob"].to_numpy(), g["outcome"].to_numpy())
        rows.append(_row(f"Sports Brier at {h}h", paper["brier"], round(b, 3), len(g),
                         _level_verdict(b, paper["brier"], len(g), "brier")))
        a = float(acc.loc[h, "accuracy"])
        rows.append(_row(f"Sports naive accuracy at {h}h", paper["accuracy"], round(a, 3), len(g),
                         _level_verdict(a, paper["accuracy"], len(g), "accuracy")))
    return rows


def _shape_rows(sports: pd.DataFrame, occ_sports: pd.DataFrame) -> list[dict]:
    rows = []
    by_h = {h: g for h, g in sports.groupby("horizon_h")}
    if 1 in by_h and len(by_h) > 1:
        scores = {h: brier(g["prob"].to_numpy(), g["outcome"].to_numpy()) for h, g in by_h.items()}
        n = len(by_h[1])
        better = scores[1] < max(v for h, v in scores.items() if h != 1)
        rows.append(_row("Brier improves toward resolution", "declines to ~0.02 at close (all markets)",
                         " / ".join(f"{h}h {scores[h]:.3f}" for h in sorted(scores, reverse=True)), n,
                         "inconclusive" if n < MIN_N else ("consistent" if better else "inconsistent")))
    g1 = by_h.get(1)
    if g1 is not None and len(g1):
        kept = g1[(g1["prob"] > TAIL_LO) & (g1["prob"] < TAIL_HI)]
        full, trunc = brier(g1["prob"], g1["outcome"]), brier(kept["prob"], kept["outcome"])
        verdict = "inconclusive" if len(kept) == len(g1) or len(g1) < MIN_N else (
            "consistent" if trunc >= full else "inconsistent")
        rows.append(_row("Tail truncation (2-98c) raises Brier at 1h", "0.078 vs 0.030 at close",
                         f"{trunc:.3f} vs {full:.3f} ({len(g1) - len(kept)} rows dropped)", len(g1), verdict))
    vol_col = "volume_pre" if "volume_pre" in sports.columns else "volume_fp"  # volume_fp includes later trading
    for h in (24, 1):
        rho, pval = volume_error_spearman(sports, h, vol_col)
        if np.isnan(rho):
            continue
        verdict = "inconclusive" if pval >= ALPHA else ("consistent" if rho < 0 else "inconsistent")
        rows.append(_row(f"More volume, lower error at {h}h (Spearman, volume before the horizon)", "Brier falls monotonically with volume",
                         f"rho={rho:.3f}, p={pval:.3f}", int((sports['horizon_h'] == h).sum()), verdict))
    g1o = occ_sports[occ_sports["horizon_h"] == 1]
    if g1 is not None and len(g1) and len(g1o):
        bc, bo = brier(g1["prob"], g1["outcome"]), brier(g1o["prob"], g1o["outcome"])
        rows.append(_row("Event-time anchor changes Sports Brier at 1h", "re-anchoring improves Sports calibration",
                         f"close {bc:.3f} -> event {bo:.3f}", len(g1o),
                         "inconclusive" if len(g1o) < MIN_N else ("consistent" if bo < bc else "inconsistent")))
    return rows


def build_comparison(df: pd.DataFrame, occ_df: pd.DataFrame | None = None) -> pd.DataFrame:
    sports = _sports(df)
    occ_sports = _sports(occ_df) if occ_df is not None else sports.iloc[0:0]
    rows = _level_rows(sports) + _shape_rows(sports, occ_sports)
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare our results with the published Kalshi calibration paper")
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--data-occurrence", default="data/dataset_occurrence.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.data)
    if _sports(df).empty:
        print("error: no Sports rows in dataset; paper comparison needs them")
        return 1
    occ_path = Path(args.data_occurrence)
    occ = pd.read_csv(occ_path) if occ_path.exists() else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = build_comparison(df, occ)
    table.to_csv(out / "paper_comparison.csv", index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
