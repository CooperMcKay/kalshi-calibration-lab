from kalshi_lab.store import Store


def raw_market(ticker="E1-A", event="E1", **kw):
    base = {
        "ticker": ticker,
        "event_ticker": event,
        "series_ticker": "S",
        "title": "A wins",
        "market_type": "binary",
        "result": "yes",
        "close_time": "2026-11-05T12:00:00Z",
        "occurrence_datetime": "2026-11-05T12:30:00Z",
        "volume_fp": "500.00",
        "open_interest_fp": "20.00",
    }
    base.update(kw)
    return base


def test_store_upsert_markets_idempotent(tmp_path):
    s = Store(tmp_path / "k.db")
    s.upsert_markets([raw_market()])
    s.upsert_markets([raw_market(volume_fp="900.00")])
    df = s.markets_df()
    assert len(df) == 1
    assert df.iloc[0]["volume_fp"] == 900.0
    assert df.iloc[0]["occurrence_datetime"] == "2026-11-05T12:30:00Z"


def test_store_event_roundtrip(tmp_path):
    s = Store(tmp_path / "k.db")
    assert not s.has_event("E1")
    s.upsert_event({"event_ticker": "E1", "series_ticker": "S", "category": "Sports",
                    "title": "Game", "mutually_exclusive": True})
    assert s.has_event("E1")
    row = s.events_df().iloc[0]
    assert row["category"] == "Sports" and row["series_ticker"] == "S"


def test_store_candles_parse_prices_and_nulls(tmp_path):
    s = Store(tmp_path / "k.db")
    s.upsert_candles("E1-A", [
        {"end_period_ts": 100, "price": {"close_dollars": "0.6700"}, "volume_fp": "12.5"},
        {"end_period_ts": 200, "price": {"close_dollars": None}, "volume_fp": "0"},
        {"end_period_ts": 300, "price": {}, "volume_fp": "0"},
    ])
    df = s.candles_df().sort_values("end_period_ts")
    assert df["price_close"].tolist()[0] == 0.67
    assert df["price_close"].isna().tolist() == [False, True, True]


def test_store_candles_upsert_idempotent_and_fetch_marker(tmp_path):
    s = Store(tmp_path / "k.db")
    assert not s.has_candles("E1-A")
    s.upsert_candles("E1-A", [])
    assert s.has_candles("E1-A")  # an empty fetch still counts as fetched
    c = {"end_period_ts": 100, "price": {"close_dollars": "0.5"}, "volume_fp": "1"}
    s.upsert_candles("E1-A", [c])
    s.upsert_candles("E1-A", [c])
    assert len(s.candles_df()) == 1
