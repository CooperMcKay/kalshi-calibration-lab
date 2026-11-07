"""Polite client for Kalshi's public, keyless market-data API."""
from __future__ import annotations

import time
from collections.abc import Iterator

import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
USER_AGENT = "kalshi-calibration-lab/0.1 (personal research; github.com/CooperMcKay/kalshi-calibration-lab)"
REQUEST_TIMEOUT_S = 30
RETRYABLE_STATUS = frozenset({429}) | frozenset(range(500, 600))


class KalshiError(Exception):
    """Raised on a non-retryable HTTP error or when retries are exhausted."""


class KalshiClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        min_interval: float = 0.2,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request = float("-inf")

    def _pace(self) -> None:
        wait = self._last_request + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        last_status = None
        for attempt in range(self.max_retries):
            self._pace()
            resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
            if resp.status_code == 200:
                return resp.json()
            last_status = resp.status_code
            if resp.status_code not in RETRYABLE_STATUS:
                raise KalshiError(f"GET {path} failed with HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(0.5 * 2**attempt)
        raise KalshiError(f"GET {path} still failing after {self.max_retries} attempts (last HTTP {last_status})")

    def paginate(self, path: str, params: dict, key: str, max_pages: int | None = None) -> Iterator[dict]:
        cursor = None
        pages = 0
        while True:
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            data = self.get(path, page_params)
            yield from data.get(key, [])
            pages += 1
            cursor = data.get("cursor")
            if not cursor or (max_pages is not None and pages >= max_pages):
                return

    def settled_markets(self, series_ticker: str, max_pages: int | None = None) -> Iterator[dict]:
        params = {"status": "settled", "series_ticker": series_ticker, "limit": 1000}
        return self.paginate("/markets", params, "markets", max_pages)

    def event(self, event_ticker: str) -> dict:
        data = self.get(f"/events/{event_ticker}")
        return data.get("event", data)

    def candlesticks(
        self, series_ticker: str, ticker: str, start_ts: int, end_ts: int, period_interval: int = 60
    ) -> list[dict]:
        path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval}
        return self.get(path, params).get("candlesticks", [])
