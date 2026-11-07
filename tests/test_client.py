import pytest

from kalshi_lab import client as client_mod
from kalshi_lab.client import BASE_URL, KalshiClient, KalshiError


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = str(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    """Returns queued responses in order; the last one repeats forever."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)


def make(responses, **kw):
    s = FakeSession(responses)
    return KalshiClient(session=s, **kw), s


def test_paginate_follows_cursor_until_empty():
    pages = [
        FakeResponse(200, {"markets": [{"t": 1}, {"t": 2}], "cursor": "c1"}),
        FakeResponse(200, {"markets": [{"t": 3}], "cursor": "c2"}),
        FakeResponse(200, {"markets": [{"t": 4}], "cursor": ""}),
    ]
    c, s = make(pages)
    out = list(c.paginate("/markets", {"status": "settled"}, "markets"))
    assert [m["t"] for m in out] == [1, 2, 3, 4]
    assert len(s.calls) == 3
    assert "cursor" not in s.calls[0][1]
    assert s.calls[1][1]["cursor"] == "c1"
    assert s.calls[2][1]["cursor"] == "c2"


def test_paginate_stops_when_cursor_missing():
    c, s = make([FakeResponse(200, {"markets": [{"t": 1}]})])
    assert len(list(c.paginate("/markets", {}, "markets"))) == 1
    assert len(s.calls) == 1


def test_paginate_respects_max_pages():
    page = FakeResponse(200, {"markets": [{"t": 1}], "cursor": "more"})
    c, s = make([page])
    assert len(list(c.paginate("/markets", {}, "markets", max_pages=2))) == 2
    assert len(s.calls) == 2


def test_get_retries_429_then_succeeds():
    c, s = make([FakeResponse(429), FakeResponse(200, {"ok": True})])
    assert c.get("/x") == {"ok": True}
    assert len(s.calls) == 2


def test_get_raises_after_max_retries_on_500():
    c, s = make([FakeResponse(500)], max_retries=3)
    with pytest.raises(KalshiError):
        c.get("/x")
    assert len(s.calls) == 3


def test_get_raises_immediately_on_404():
    c, s = make([FakeResponse(404)])
    with pytest.raises(KalshiError):
        c.get("/x")
    assert len(s.calls) == 1


def test_candlesticks_uses_series_path():
    c, s = make([FakeResponse(200, {"candlesticks": [{"end_period_ts": 1}]})])
    out = c.candlesticks("KXNBAGAME", "KXNBAGAME-26NOV04MIALAC-MIA", 100, 200)
    assert out == [{"end_period_ts": 1}]
    url, params = s.calls[0]
    assert url == f"{BASE_URL}/series/KXNBAGAME/markets/KXNBAGAME-26NOV04MIALAC-MIA/candlesticks"
    assert params == {"start_ts": 100, "end_ts": 200, "period_interval": 60}


def test_settled_markets_params():
    c, s = make([FakeResponse(200, {"markets": [], "cursor": ""})])
    list(c.settled_markets("KXNBAGAME"))
    assert s.calls[0][1] == {"status": "settled", "series_ticker": "KXNBAGAME", "limit": 1000}


def test_event_unwraps_event_key():
    c, _ = make([FakeResponse(200, {"event": {"category": "Sports"}, "markets": []})])
    assert c.event("KXNBAGAME-26NOV04MIALAC") == {"category": "Sports"}


def test_user_agent_header_set():
    assert "kalshi-calibration-lab" in KalshiClient().session.headers["User-Agent"]


def test_pacing_sleeps_between_requests(monkeypatch):
    sleeps = []
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: sleeps.append(s))
    clock = iter([0.0, 0.0, 0.05, 0.05, 0.3, 0.3])
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: next(clock))
    c, _ = make([FakeResponse(200, {})], min_interval=0.2)
    c.get("/a")
    c.get("/b")
    assert any(s > 0 for s in sleeps)
