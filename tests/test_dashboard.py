import json

import pandas as pd
import pytest

from kalshi_lab.dashboard import PLACEHOLDER, build_payload, main, render
from kalshi_lab.store import Store

HOUR = 3600
CLOSE_ISO = "2026-11-05T12:00:00Z"
CLOSE_TS = 1793880000  # 2026-11-05T12:00:00Z


def seeded_store(tmp_path):
    store = Store(tmp_path / "k.db")
    store.upsert_markets([{
        "ticker": "E1-A", "event_ticker": "E1", "series_ticker": "S", "title": "A wins", "result": "yes",
        "close_time": CLOSE_ISO, "occurrence_datetime": CLOSE_ISO, "volume_fp": "500", "open_interest_fp": "9"}])
    store.upsert_candles("E1-A", [
        {"end_period_ts": CLOSE_TS - 24 * HOUR, "price": {"close_dollars": "0.6123"}, "volume_fp": "1"},
        {"end_period_ts": CLOSE_TS - HOUR, "price": {"close_dollars": "0.9"}, "volume_fp": "1"}])
    return store


def dataset(prob_by_h, ticker="E1-A"):
    return pd.DataFrame([{
        "ticker": ticker, "event_ticker": "E1", "series_ticker": "S", "category": "Sports", "title": "A wins",
        "close_time": CLOSE_ISO, "occurrence_time": CLOSE_ISO, "horizon_h": h, "prob": p, "outcome": 1,
        "volume_fp": 500.0, "open_interest_fp": 9.0} for h, p in prob_by_h.items()])


def test_build_payload_rows_merge_close_and_occurrence(tmp_path):
    store = seeded_store(tmp_path)
    payload = build_payload(store, dataset({24: 0.61, 1: 0.9}), dataset({24: 0.62}))
    rows = {r["h"]: r for r in payload["rows"]}
    assert rows[24]["p"] == 0.61 and rows[24]["po"] == 0.62
    assert rows[1]["p"] == 0.9 and rows[1]["po"] is None


def test_build_payload_keeps_rows_that_only_exist_under_event_anchor(tmp_path):
    store = seeded_store(tmp_path)
    payload = build_payload(store, dataset({24: 0.61}), dataset({24: 0.62, 1: 0.95}))
    r1 = [r for r in payload["rows"] if r["h"] == 1][0]
    assert r1["p"] is None and r1["po"] == 0.95


def test_build_payload_paths_are_hours_before_close(tmp_path):
    payload = build_payload(seeded_store(tmp_path), dataset({24: 0.61}), dataset({}))
    assert payload["paths"]["E1-A"] == [[24.0, 0.612], [1.0, 0.9]]


def test_build_payload_is_json_serialisable_and_has_sections(tmp_path):
    payload = build_payload(seeded_store(tmp_path), dataset({24: 0.61}), dataset({}),
                            model={"24": [{"model": "market", "brier": 0.2}]}, paper=[{"finding": "x"}])
    json.dumps(payload)
    assert {"rows", "paths", "model", "paper", "generated"} <= set(payload)


def test_render_replaces_placeholder_and_escapes_script_end():
    html = f"<script>const D = {PLACEHOLDER};</script>"
    out = render(html, {"note": "</script><b>"})
    assert PLACEHOLDER not in out and "</script><b>" not in out


def test_render_fails_without_placeholder():
    with pytest.raises(ValueError):
        render("<html></html>", {})


def test_main_writes_dashboard(tmp_path):
    seeded_store(tmp_path)
    tpl = tmp_path / "t.html"
    tpl.write_text(f"<script>const D={PLACEHOLDER};</script>")
    dataset({24: 0.61}).to_csv(tmp_path / "d.csv", index=False)
    out = tmp_path / "dash.html"
    code = main(["--db", str(tmp_path / "k.db"), "--data", str(tmp_path / "d.csv"),
                 "--data-occurrence", str(tmp_path / "none.csv"), "--outputs", str(tmp_path / "o"),
                 "--template", str(tpl), "--out", str(out)])
    assert code == 0 and "E1-A" in out.read_text()


def test_main_fails_cleanly_on_empty_dataset(tmp_path):
    seeded_store(tmp_path)
    dataset({}).to_csv(tmp_path / "d.csv", index=False)
    assert main(["--db", str(tmp_path / "k.db"), "--data", str(tmp_path / "d.csv"),
                 "--template", str(tmp_path / "t.html"), "--out", str(tmp_path / "x.html")]) == 1
