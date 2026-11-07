"""Collect settled Kalshi markets into SQLite and build the analysis dataset (data/dataset.csv)."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kalshi_lab.client import KalshiClient, KalshiError
from kalshi_lab.store import Store

HORIZONS_H = (168, 24, 1)
MIN_VOLUME = 100.0
LOOKBACK_H = 12
CANDLE_WINDOW_DAYS = 8
SECONDS_PER_HOUR = 3600
ANCHORS = ("close", "occurrence")

DATASET_COLUMNS = [
    "ticker", "event_ticker", "series_ticker", "category", "title", "close_time", "occurrence_time",
    "horizon_h", "prob", "outcome", "volume_fp", "volume_pre", "open_interest_fp",
]


def to_epoch(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp())


def keep_market(m: dict, min_volume: float = MIN_VOLUME) -> bool:
    """Binary, cleanly settled, not a multivariate parlay, and above the volume floor."""
    if m.get("market_type") != "binary" or m.get("result") not in ("yes", "no"):
        return False
    if m.get("mve_collection_ticker"):
        return False
    try:
        return float(m.get("volume_fp") or 0) >= min_volume
    except (TypeError, ValueError):
        return False


def sample_price(
    candles: list[tuple[int, float | None]], close_ts: int, horizon_h: int, lookback_h: int = LOOKBACK_H
) -> float | None:
    """Last traded price at or before (close_ts - horizon), searching back at most lookback_h. Never uses the future."""
    cutoff = close_ts - horizon_h * SECONDS_PER_HOUR
    earliest = cutoff - lookback_h * SECONDS_PER_HOUR
    usable = [(t, p) for t, p in candles if p is not None and p == p and earliest < t <= cutoff]  # p == p drops NaN
    if not usable:
        return None
    return max(usable)[1]


def sample_volume(volumes: list[tuple[int, float | None]], close_ts: int, horizon_h: int) -> float:
    """Contracts traded up to (close_ts - horizon), from per-candle volume. Never counts post-horizon trading."""
    cutoff = close_ts - horizon_h * SECONDS_PER_HOUR
    return float(sum(v for t, v in volumes if t <= cutoff and v is not None and v == v))


def _anchor_ts(row: pd.Series, anchor: str) -> int:
    if anchor == "occurrence" and isinstance(row["occurrence_datetime"], str) and row["occurrence_datetime"]:
        return to_epoch(row["occurrence_datetime"])
    return to_epoch(row["close_time"])


def _one_per_event(markets: pd.DataFrame) -> pd.DataFrame:
    """Keep the alphabetically-first ticker of each event, so rows are independent events."""
    return markets.sort_values("ticker").drop_duplicates("event_ticker", keep="first")


def _choose_markets(markets: pd.DataFrame, categories: dict[str, str | None]) -> pd.DataFrame:
    """Sports events are two-sided (both markets carry the same information): keep one per event.
    Other events (rate ladders, temperature brackets, price thresholds) are distinct questions: keep all.
    Event-level resampling later accounts for those markets being correlated."""
    if markets.empty:
        return markets
    is_sports = markets["event_ticker"].map(categories).eq("Sports")
    return pd.concat([_one_per_event(markets[is_sports]), markets[~is_sports]]).sort_values("ticker")


def build_dataset(
    store: Store,
    horizons: tuple[int, ...] = HORIZONS_H,
    min_volume: float = MIN_VOLUME,
    anchor: str = "close",
) -> pd.DataFrame:
    """One row per (event's chosen market, horizon) that has a traded price at that horizon."""
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
    markets = store.markets_df()
    markets = markets[markets["result"].isin(["yes", "no"]) & (markets["volume_fp"] >= min_volume)]
    markets = markets[markets["close_time"].notna()]
    events = store.events_df()[["event_ticker", "series_ticker", "category"]]
    categories = dict(zip(events["event_ticker"], events["category"]))
    chosen = _choose_markets(markets, categories).drop(columns=["series_ticker"]).merge(
        events, on="event_ticker", how="left")

    candles = store.candles_df()
    by_ticker = {
        t: list(zip(g["end_period_ts"], g["price_close"].where(g["price_close"].notna(), None)))
        for t, g in candles.groupby("ticker")
    }
    vol_by_ticker = {
        t: list(zip(g["end_period_ts"], g["volume"].where(g["volume"].notna(), None)))
        for t, g in candles.groupby("ticker")
    }
    rows = []
    for _, m in chosen.iterrows():
        anchor_ts = _anchor_ts(m, anchor)
        for h in horizons:
            price = sample_price(by_ticker.get(m["ticker"], []), anchor_ts, h)
            if price is None:
                continue
            rows.append({
                "ticker": m["ticker"], "event_ticker": m["event_ticker"], "series_ticker": m["series_ticker"],
                "category": m["category"], "title": m["title"], "close_time": m["close_time"],
                "occurrence_time": m["occurrence_datetime"], "horizon_h": h, "prob": float(price),
                "outcome": int(m["result"] == "yes"), "volume_fp": m["volume_fp"],
                "volume_pre": sample_volume(vol_by_ticker.get(m["ticker"], []), anchor_ts, h),
                "open_interest_fp": m["open_interest_fp"],
            })
    return pd.DataFrame(rows, columns=DATASET_COLUMNS)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def collect_series(
    client: KalshiClient, store: Store, series: str, pages: int | None, min_volume: float = MIN_VOLUME
) -> dict[str, int]:
    """Backfill one series. Idempotent: events and candles already stored are not fetched again."""
    errors = 0
    seen = 0
    kept: list[dict] = []
    for m in client.settled_markets(series, max_pages=pages):
        seen += 1
        if keep_market(m, min_volume=0.0):
            kept.append({**m, "series_ticker": series})
    store.upsert_markets(kept)

    for event_ticker in sorted({m["event_ticker"] for m in kept}):
        if store.has_event(event_ticker):
            continue
        try:
            store.upsert_event({**client.event(event_ticker), "event_ticker": event_ticker})
        except KalshiError as exc:
            errors += 1
            _log(f"[{series}] event {event_ticker} failed: {exc}")

    eligible = pd.DataFrame([m for m in kept if keep_market(m, min_volume)])
    candle_fetches = 0
    if not eligible.empty:
        events = store.events_df()
        categories = dict(zip(events["event_ticker"], events["category"]))
        for _, m in _choose_markets(eligible, categories).iterrows():
            if store.has_candles(m["ticker"]) or not m.get("close_time"):
                continue
            anchors = [to_epoch(m["close_time"])]
            if isinstance(m.get("occurrence_datetime"), str) and m["occurrence_datetime"]:
                anchors.append(to_epoch(m["occurrence_datetime"]))
            start, end = min(anchors) - CANDLE_WINDOW_DAYS * 24 * SECONDS_PER_HOUR, max(anchors)
            try:
                store.upsert_candles(m["ticker"], client.candlesticks(series, m["ticker"], start, end))
                candle_fetches += 1
            except KalshiError as exc:
                errors += 1
                _log(f"[{series}] candles {m['ticker']} failed: {exc}")
    _log(f"[{series}] seen={seen} stored={len(kept)} candle_fetches={candle_fetches} errors={errors}")
    return {"markets_seen": seen, "markets_stored": len(kept), "candle_fetches": candle_fetches, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill settled Kalshi markets and build data/dataset.csv")
    p.add_argument("--series", required=True, help="comma-separated series tickers, e.g. KXNBAGAME,KXFED")
    p.add_argument("--pages", type=int, default=3, help="pages of 1000 markets per series")
    p.add_argument("--db", default="data/kalshi.db")
    p.add_argument("--out", default="data/dataset.csv")
    p.add_argument("--anchor", choices=ANCHORS, default="close")
    args = p.parse_args(argv)

    client, store = KalshiClient(), Store(args.db)
    total_errors = 0
    for series in [s.strip() for s in args.series.split(",") if s.strip()]:
        total_errors += collect_series(client, store, series, args.pages)["errors"]
    df = build_dataset(store, anchor=args.anchor)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    _log(f"wrote {len(df)} rows ({df['event_ticker'].nunique() if len(df) else 0} events) to {args.out}")
    if df.empty:
        _log("warning: dataset is empty")
    return 1 if total_errors and df.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
