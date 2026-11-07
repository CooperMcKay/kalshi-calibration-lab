from datetime import datetime, timezone

import pandas as pd
import pytest

from kalshi_lab.collect import DATASET_COLUMNS, build_dataset, keep_market, sample_price
from kalshi_lab.store import Store

HOUR = 3600


def ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp())


def market(**kw):
    base = {"market_type": "binary", "result": "yes", "volume_fp": "500.0"}
    base.update(kw)
    return base


# ---- keep_market -----------------------------------------------------------

def test_keep_market_drops_mve():
    assert not keep_market(market(mve_collection_ticker="KXMVE-1"))


def test_keep_market_drops_non_binary_and_unsettled_result():
    assert not keep_market(market(market_type="scalar"))
    assert not keep_market(market(result=""))
    assert not keep_market(market(result="void"))


def test_keep_market_volume_floor():
    assert not keep_market(market(volume_fp="99.9"))
    assert keep_market(market(volume_fp="100"))
    assert keep_market(market(volume_fp="99.9"), min_volume=0.0)


# ---- sample_price ----------------------------------------------------------

CLOSE = 1_000_000_000


def test_sample_price_uses_last_candle_before_cutoff():
    cutoff = CLOSE - 24 * HOUR
    candles = [(cutoff - 2 * HOUR, 0.4), (cutoff - HOUR, 0.5), (cutoff + HOUR, 0.9)]
    assert sample_price(candles, CLOSE, 24) == 0.5


def test_sample_price_includes_candle_ending_exactly_at_cutoff():
    cutoff = CLOSE - HOUR
    assert sample_price([(cutoff, 0.7)], CLOSE, 1) == 0.7


def test_sample_price_never_uses_future_candle():
    cutoff = CLOSE - HOUR
    assert sample_price([(cutoff + 1, 0.99)], CLOSE, 1) is None


def test_sample_price_none_when_outside_lookback():
    cutoff = CLOSE - 24 * HOUR
    assert sample_price([(cutoff - 13 * HOUR, 0.5)], CLOSE, 24) is None
    assert sample_price([(cutoff - 11 * HOUR, 0.5)], CLOSE, 24) == 0.5


def test_sample_price_skips_null_price_candles():
    cutoff = CLOSE - HOUR
    candles = [(cutoff - 3 * HOUR, 0.4), (cutoff - HOUR, None)]
    assert sample_price(candles, CLOSE, 1) == 0.4


def test_sample_price_empty_candles_returns_none():
    assert sample_price([], CLOSE, 24) is None


# ---- build_dataset ---------------------------------------------------------

def raw_market(ticker, event="E1", **kw):
    base = {
        "ticker": ticker, "event_ticker": event, "series_ticker": "S", "title": f"{ticker} wins",
        "market_type": "binary", "result": "yes", "close_time": "2026-11-05T12:00:00Z",
        "occurrence_datetime": "2026-11-05T12:00:00Z", "volume_fp": "500.00", "open_interest_fp": "20.00",
    }
    base.update(kw)
    return base


def candle(end_ts, price):
    return {"end_period_ts": end_ts, "price": {"close_dollars": price}, "volume_fp": "1"}


def seed(store, markets, candles_by_ticker):
    store.upsert_markets(markets)
    for m in markets:
        if not store.has_event(m["event_ticker"]):
            store.upsert_event({"event_ticker": m["event_ticker"], "series_ticker": "S",
                                "category": "Sports", "title": "t", "mutually_exclusive": True})
    for t, cs in candles_by_ticker.items():
        store.upsert_candles(t, cs)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "k.db")


def test_build_dataset_columns_and_dtypes(store):
    close = ts("2026-11-05T12:00:00Z")
    seed(store, [raw_market("E1-A")], {"E1-A": [candle(close - 24 * HOUR, "0.6")]})
    df = build_dataset(store, horizons=(24,))
    assert list(df.columns) == DATASET_COLUMNS and len(DATASET_COLUMNS) == 12
    assert df.iloc[0]["prob"] == 0.6 and df.iloc[0]["outcome"] == 1 and df.iloc[0]["horizon_h"] == 24


def test_build_dataset_one_market_per_event(store):
    close = ts("2026-11-05T12:00:00Z")
    cs = [candle(close - 24 * HOUR, "0.6")]
    seed(store, [raw_market("E1-B"), raw_market("E1-A")], {"E1-A": cs, "E1-B": cs})
    df = build_dataset(store, horizons=(24,))
    assert df["ticker"].tolist() == ["E1-A"]


def test_build_dataset_outcome_from_result_not_price(store):
    close = ts("2026-11-05T12:00:00Z")
    seed(store, [raw_market("E1-A", result="no")], {"E1-A": [candle(close - HOUR, "0.99")]})
    row = build_dataset(store, horizons=(1,)).iloc[0]
    assert row["outcome"] == 0 and row["prob"] == 0.99


def test_build_dataset_skips_rows_without_price(store):
    seed(store, [raw_market("E1-A")], {"E1-A": []})
    df = build_dataset(store, horizons=(24, 1))
    assert df.empty and list(df.columns) == DATASET_COLUMNS


def test_build_dataset_respects_volume_floor(store):
    close = ts("2026-11-05T12:00:00Z")
    seed(store, [raw_market("E1-A", volume_fp="50")], {"E1-A": [candle(close - HOUR, "0.5")]})
    assert build_dataset(store, horizons=(1,)).empty
    assert len(build_dataset(store, horizons=(1,), min_volume=0.0)) == 1


def test_build_dataset_occurrence_anchor_shifts_sampling(store):
    close = ts("2026-11-05T12:00:00Z")
    occ = "2026-11-05T15:00:00Z"
    seed(store, [raw_market("E1-A", occurrence_datetime=occ)],
         {"E1-A": [candle(close - HOUR, "0.3"), candle(close + 2 * HOUR, "0.9")]})
    by_close = build_dataset(store, horizons=(1,), anchor="close").iloc[0]["prob"]
    by_occ = build_dataset(store, horizons=(1,), anchor="occurrence").iloc[0]["prob"]
    assert (by_close, by_occ) == (0.3, 0.9)


def test_build_dataset_occurrence_anchor_falls_back_to_close(store):
    close = ts("2026-11-05T12:00:00Z")
    seed(store, [raw_market("E1-A", occurrence_datetime=None)], {"E1-A": [candle(close - HOUR, "0.3")]})
    assert build_dataset(store, horizons=(1,), anchor="occurrence").iloc[0]["prob"] == 0.3


def test_build_dataset_rejects_unknown_anchor(store):
    with pytest.raises(ValueError):
        build_dataset(store, anchor="bogus")


def test_build_dataset_returns_dataframe(store):
    assert isinstance(build_dataset(store), pd.DataFrame)


# ---- collect_series (orchestration, fake client) ---------------------------

from kalshi_lab.client import KalshiError  # noqa: E402
from kalshi_lab.collect import collect_series  # noqa: E402


class FakeClient:
    def __init__(self, markets, fail_candles_for=()):
        self.markets = markets
        self.fail_candles_for = set(fail_candles_for)
        self.event_calls, self.candle_calls = [], []

    def settled_markets(self, series, max_pages=None):
        yield from self.markets

    def event(self, event_ticker):
        self.event_calls.append(event_ticker)
        return {"event_ticker": event_ticker, "series_ticker": "S", "category": "Sports",
                "title": "t", "mutually_exclusive": True}

    def candlesticks(self, series, ticker, start_ts, end_ts, period_interval=60):
        self.candle_calls.append((series, ticker, start_ts, end_ts))
        if ticker in self.fail_candles_for:
            raise KalshiError("boom")
        return [candle(end_ts - HOUR, "0.5")]


def test_collect_series_fetches_candles_only_for_chosen_market(store):
    markets = [
        raw_market("E1-B"), raw_market("E1-A"),
        raw_market("E1-MVE", mve_collection_ticker="X"),
        raw_market("E2-A", event="E2", volume_fp="10"),  # below volume floor: stored, no candles
    ]
    fc = FakeClient(markets)
    summary = collect_series(fc, store, "S", pages=1)
    assert [c[1] for c in fc.candle_calls] == ["E1-A"]
    assert sorted(fc.event_calls) == ["E1", "E2"]
    assert set(store.markets_df()["ticker"]) == {"E1-A", "E1-B", "E2-A"}
    assert summary["markets_seen"] == 4 and summary["errors"] == 0


def test_collect_series_is_idempotent(store):
    fc = FakeClient([raw_market("E1-A")])
    collect_series(fc, store, "S", pages=1)
    collect_series(fc, store, "S", pages=1)
    assert len(fc.candle_calls) == 1 and fc.event_calls == ["E1"]


def test_collect_series_counts_errors_and_continues(store):
    fc = FakeClient([raw_market("E1-A"), raw_market("E2-A", event="E2")], fail_candles_for={"E1-A"})
    summary = collect_series(fc, store, "S", pages=1)
    assert summary["errors"] == 1
    assert store.has_candles("E2-A") and not store.has_candles("E1-A")


def test_sample_price_skips_nan_price_candles():
    cutoff = CLOSE - HOUR
    candles = [(cutoff - 3 * HOUR, 0.4), (cutoff - HOUR, float("nan"))]
    assert sample_price(candles, CLOSE, 1) == 0.4


def test_build_dataset_never_emits_nan_prob(store):
    close = ts("2026-11-05T12:00:00Z")
    seed(store, [raw_market("E1-A")],
         {"E1-A": [candle(close - 2 * HOUR, "0.4"), candle(close - HOUR, None)]})
    df = build_dataset(store, horizons=(1,))
    assert df["prob"].notna().all() and df.iloc[0]["prob"] == 0.4
