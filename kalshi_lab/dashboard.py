"""Build a self-contained interactive HTML dashboard from the local dataset (raw data never leaves your machine)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kalshi_lab.collect import SECONDS_PER_HOUR, to_epoch
from kalshi_lab.store import Store

PLACEHOLDER = "__KALSHI_DATA__"
MAX_PATH_HOURS = 200
KEY = ["ticker", "horizon_h"]


def _clean(v):
    return None if v is None or (isinstance(v, float) and v != v) else v


def _read_csv(path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _merge_rows(close_df: pd.DataFrame, occ_df: pd.DataFrame) -> list[dict]:
    a = close_df.rename(columns={"prob": "p"}) if len(close_df) else pd.DataFrame(columns=KEY + ["p"])
    b = occ_df[KEY + ["prob"]].rename(columns={"prob": "po"}) if len(occ_df) else pd.DataFrame(columns=KEY + ["po"])
    meta_src = pd.concat([x for x in (close_df, occ_df) if len(x)], ignore_index=True)
    meta = meta_src.drop_duplicates(KEY)[KEY + ["event_ticker", "category", "title", "close_time", "outcome",
                                                "volume_fp", "open_interest_fp"]]
    merged = meta.merge(a[KEY + ["p"]], on=KEY, how="left").merge(b, on=KEY, how="left")
    merged = merged.astype(object).where(merged.notna(), None)
    return [{
        "t": r["ticker"], "e": r["event_ticker"], "title": r["title"], "cat": r["category"],
        "h": int(r["horizon_h"]), "p": _clean(r["p"]), "po": _clean(r["po"]), "y": int(r["outcome"]),
        "v": _clean(r["volume_fp"]), "oi": _clean(r["open_interest_fp"]), "close": r["close_time"],
    } for _, r in merged.iterrows()]


def _price_paths(store: Store, rows: list[dict]) -> dict[str, list[list[float]]]:
    closes = {r["t"]: to_epoch(r["close"]) for r in rows if r["close"]}
    candles = store.candles_df()
    candles = candles[candles["ticker"].isin(closes) & candles["price_close"].notna()]
    paths = {}
    for ticker, g in candles.groupby("ticker"):
        pts = [[round((closes[ticker] - ts) / SECONDS_PER_HOUR, 3), round(float(p), 3)]
               for ts, p in zip(g["end_period_ts"], g["price_close"])]
        pts = sorted((x for x in pts if 0 <= x[0] <= MAX_PATH_HOURS), reverse=True)
        if pts:
            paths[ticker] = pts
    return paths


def build_payload(store: Store, close_df: pd.DataFrame, occ_df: pd.DataFrame,
                  model: dict | None = None, paper: list | None = None) -> dict:
    rows = _merge_rows(close_df, occ_df)
    return {
        "rows": rows, "paths": _price_paths(store, rows), "model": model or {}, "paper": paper or [],
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def render(template: str, payload: dict) -> str:
    if PLACEHOLDER not in template:
        raise ValueError(f"template has no {PLACEHOLDER} placeholder")
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return template.replace(PLACEHOLDER, blob)


def _load_extras(outputs: Path) -> tuple[dict, list]:
    model = {p.stem.split("_")[1].rstrip("h"): _read_csv(p).to_dict("records") for p in sorted(outputs.glob("model_*h.csv"))}
    paper = _read_csv(outputs / "paper_comparison.csv")
    return model, (paper.astype(object).where(paper.notna(), None).to_dict("records") if len(paper) else [])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the interactive dashboard from your local data")
    ap.add_argument("--db", default="data/kalshi.db")
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--data-occurrence", default="data/dataset_occurrence.csv")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--template", default="dashboard/template.html")
    ap.add_argument("--out", default="dashboard/dashboard.html")
    args = ap.parse_args(argv)

    close_df = _read_csv(args.data)
    if close_df.empty:
        print("error: dataset is empty; run kalshi_lab.collect first")
        return 1
    model, paper = _load_extras(Path(args.outputs))
    payload = build_payload(Store(args.db), close_df, _read_csv(args.data_occurrence), model, paper)
    html = render(Path(args.template).read_text(), payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(payload['rows'])} rows, {len(html) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
