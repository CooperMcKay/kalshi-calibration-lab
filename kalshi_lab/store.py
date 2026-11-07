"""SQLite store for raw Kalshi payloads. Raw first, derived second: the dataset is built from these tables."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    ticker TEXT PRIMARY KEY, event_ticker TEXT, series_ticker TEXT, title TEXT, result TEXT,
    close_time TEXT, occurrence_datetime TEXT, volume_fp REAL, open_interest_fp REAL
);
CREATE TABLE IF NOT EXISTS events (
    event_ticker TEXT PRIMARY KEY, series_ticker TEXT, category TEXT, title TEXT, mutually_exclusive INTEGER
);
CREATE TABLE IF NOT EXISTS candles (
    ticker TEXT, end_period_ts INTEGER, price_close REAL, volume REAL,
    PRIMARY KEY (ticker, end_period_ts)
);
CREATE TABLE IF NOT EXISTS candle_fetches (ticker TEXT PRIMARY KEY);
"""


def _to_float(value) -> float | None:
    """Kalshi sends numbers as strings; missing or empty values become None."""
    if value is None or value == "":
        return None
    return float(value)


class Store:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def upsert_markets(self, rows: list[dict]) -> None:
        records = [
            (
                m["ticker"], m.get("event_ticker"), m.get("series_ticker"), m.get("title"), m.get("result"),
                m.get("close_time"), m.get("occurrence_datetime"),
                _to_float(m.get("volume_fp")), _to_float(m.get("open_interest_fp")),
            )
            for m in rows
        ]
        with self._connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO markets VALUES (?,?,?,?,?,?,?,?,?)", records)

    def upsert_event(self, row: dict) -> None:
        record = (
            row["event_ticker"], row.get("series_ticker"), row.get("category"), row.get("title"),
            None if row.get("mutually_exclusive") is None else int(bool(row["mutually_exclusive"])),
        )
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?)", record)

    def upsert_candles(self, ticker: str, rows: list[dict]) -> None:
        records = [
            (ticker, int(c["end_period_ts"]), _to_float((c.get("price") or {}).get("close_dollars")),
             _to_float(c.get("volume_fp")))
            for c in rows
        ]
        with self._connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO candles VALUES (?,?,?,?)", records)
            conn.execute("INSERT OR IGNORE INTO candle_fetches VALUES (?)", (ticker,))

    def has_event(self, event_ticker: str) -> bool:
        return self._exists("events", "event_ticker", event_ticker)

    def has_candles(self, ticker: str) -> bool:
        return self._exists("candle_fetches", "ticker", ticker)

    def _exists(self, table: str, column: str, value: str) -> bool:
        with self._connect() as conn:
            return conn.execute(f"SELECT 1 FROM {table} WHERE {column} = ?", (value,)).fetchone() is not None

    def _df(self, table: str) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)

    def markets_df(self) -> pd.DataFrame:
        return self._df("markets")

    def events_df(self) -> pd.DataFrame:
        return self._df("events")

    def candles_df(self) -> pd.DataFrame:
        return self._df("candles")
