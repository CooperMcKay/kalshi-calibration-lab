# Kalshi Calibration Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested Python pipeline that collects settled Kalshi markets, measures calibration by horizon/category/volume, tests logistic and isotonic recalibration, and ships as a public GitHub repo with results but no Kalshi data.

**Architecture:** `client.py` (HTTP, pacing, pagination) -> `store.py` (SQLite raw tables) -> `collect.py` (filters, price sampling, dataset.csv) -> `metrics.py` (pure scoring) -> `calibration.py` / `report_plots.py` (tables and figures) -> `model.py` (recalibration). Pure functions are unit-tested with fixtures; no test touches the network.

**Tech Stack:** Python 3.14, requests, numpy, pandas, scikit-learn, matplotlib, pytest, SQLite (stdlib).

**Spec:** `docs/superpowers/specs/2026-11-06-kalshi-calibration-lab-design.md` (see its Research basis section: methods follow the Kagan & Baiocchi papers)

## Global Constraints

- Base URL `https://external-api.kalshi.com/trade-api/v2`; no API key; ~5 req/s (`min_interval=0.2`), exponential backoff on 429/5xx, descriptive User-Agent.
- Horizons `(168, 24, 1)` hours before `close_time`; price = `price.close_dollars` of last traded candle ending at or before `close_time - h`, searching back at most 12h; otherwise skip the row (never forward-fill).
- Keep only `market_type == "binary"`, `result in {"yes","no"}`, `mve_collection_ticker` empty, `volume_fp >= 100`.
- One market per event in the dataset (lowest ticker alphabetically among eligible).
- Prices arrive as strings; times are UTC ISO-8601.
- Candlesticks path is `/series/{series}/markets/{ticker}/candlesticks` (not `/markets/...`).
- `data/` is gitignored and must never be committed. `model.py` refuses to run below 200 rows at a horizon.
- Tail truncation keeps `0.02 < prob < 0.98`. Fixed cohort = markets with a valid price at the longest horizon being compared. Multiple-comparison control = Benjamini-Hochberg at alpha 0.05.
- Research PDFs stay local (gitignored); never commit them.
- Commit messages: `<type>: <description>`; end with the attribution line from the session.

## Review Focus

- Candle list empty or all-null prices for a market: row skipped, no crash (Task 3).
- Pagination cursor missing/empty string on the last page: iteration stops (Task 2).
- HTTP 429 then 200: client retries and returns the 200 body; persistent 500 raises after max retries (Task 2).
- Probabilities exactly 0 or 1: log loss clipped, finite (Task 1).
- Bucket with n=0 or n=1 or actual rate 0/1: table row present, CI marked NaN or flagged, no divide-by-zero (Task 1).
- Test slice contains a single outcome class, or fewer than 200 rows: `model.py` reports/raises clearly instead of emitting garbage (Task 6).

---

### Task 1: Scaffold and metrics

**Files:**
- Create: `requirements.txt`, `LICENSE` (MIT, Cooper McKay, 2026), `README.md` (stub: title, disclaimer, "work in progress"), `kalshi_lab/__init__.py`, `kalshi_lab/metrics.py`, `tests/test_metrics.py`
- Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt pytest`

**Interfaces:**
- Produces (all take `p: np.ndarray` of probabilities in [0,1] and `y: np.ndarray` of 0/1):
  - `brier(p, y) -> float`
  - `log_loss(p, y, eps: float = 1e-6) -> float` (clips p to [eps, 1-eps])
  - `reliability_table(p, y, n_bins: int = 10) -> pd.DataFrame` with columns `bucket` (str like `"0.0-0.1"`), `n` (int), `mean_pred`, `actual`, `gap` (actual - mean_pred), `ci95` (1.96*sqrt(actual*(1-actual)/n)); empty bins appear with `n=0` and NaN elsewhere; `ci95` is NaN when `n < 2`
  - `ece(p, y, n_bins: int = 10) -> float` (n-weighted mean of |gap| over non-empty bins)
  - `murphy_decomposition(p, y, n_bins: int = 10) -> dict[str, float]` keys `reliability`, `resolution`, `uncertainty`, `brier_binned` where reliability = sum n_k (pbar_k - obar_k)^2 / N, resolution = sum n_k (obar_k - obar)^2 / N, uncertainty = obar(1-obar), and brier_binned = reliability - resolution + uncertainty
  - `directional_accuracy(p, y) -> float` (p > 0.5 predicts yes; p == 0.5 counts as incorrect)
  - `benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray` boolean mask of rejected nulls; NaN p-values are never rejected

- [ ] **Step 1: Write failing tests** in `tests/test_metrics.py`: `test_brier_perfect_is_zero`, `test_brier_always_half_is_quarter` (0.25), `test_brier_worked_example` (p=0.7,y=1 -> 0.09; y=0 -> 0.49), `test_log_loss_clips_extremes` (p=[0,1], y=[1,0] is finite and > 10), `test_log_loss_known_value` (p=0.7,y=1 -> `-ln 0.7`), `test_reliability_table_counts_and_gap` (known 10-sample input; n sums to 10, gap = actual - mean_pred), `test_reliability_empty_bin_has_n0_and_nan`, `test_reliability_ci_nan_when_n_below_2`, `test_ece_zero_when_calibrated`, `test_ece_matches_weighted_gap`, `test_murphy_identity_holds_when_p_constant_within_bins` (brier_binned == brier), `test_murphy_perfect_forecaster_resolution_equals_uncertainty`, `test_directional_accuracy_tie_is_incorrect`, `test_bh_known_example` (pvals [0.001,0.008,0.039,0.041,0.6] at 0.05 -> first two True only... compute by hand in the test), `test_bh_ignores_nan`.
- [ ] **Step 2: Run** `pytest tests/test_metrics.py -v`. Expected: collection error / FAIL (module missing).
- [ ] **Step 3: Implement** the four functions in `kalshi_lab/metrics.py`; use `np.digitize`/`pd.cut` with right-inclusive final bin so p=1.0 lands in the last bucket.
- [ ] **Step 4: Run** `pytest tests/test_metrics.py -v`. Expected: all PASS.
- [ ] **Step 5: Verify install works on Python 3.14**: `python -c "import numpy, pandas, sklearn, matplotlib, requests"`. If a wheel is missing, pin the newest working versions in `requirements.txt` and note it in the commit.
- [ ] **Step 6: Commit** `feat: scaffold project and add scoring metrics`.

### Task 2: Kalshi client

**Files:**
- Create: `kalshi_lab/client.py`, `tests/test_client.py`

**Interfaces:**
- Produces: `class KalshiClient(base_url: str = BASE_URL, min_interval: float = 0.2, max_retries: int = 5, session: requests.Session | None = None)` with:
  - `get(path: str, params: dict | None = None) -> dict` (paced, retries 429/5xx with exponential backoff, raises `KalshiError` after `max_retries` or on other 4xx)
  - `paginate(path: str, params: dict, key: str, max_pages: int | None = None) -> Iterator[dict]` (follows `cursor`; stops when cursor is falsy or `max_pages` reached)
  - `settled_markets(series_ticker: str, max_pages: int | None = None) -> Iterator[dict]` (`status=settled`, `limit=1000`)
  - `event(event_ticker: str) -> dict`
  - `candlesticks(series_ticker: str, ticker: str, start_ts: int, end_ts: int, period_interval: int = 60) -> list[dict]`
  - `KalshiError(Exception)`; module constant `BASE_URL`.
- Tests use a fake session object (a small class with `.get` returning canned responses) and monkeypatch `time.sleep`.

- [ ] **Step 1: Write failing tests**: `test_paginate_follows_cursor_until_empty` (3 pages, last cursor `""`), `test_paginate_respects_max_pages`, `test_get_retries_429_then_succeeds`, `test_get_raises_after_max_retries_on_500`, `test_get_raises_immediately_on_404`, `test_candlesticks_uses_series_path` (asserts requested path is `/series/S/markets/T/candlesticks`), `test_user_agent_header_set`.
- [ ] **Step 2: Run** `pytest tests/test_client.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `KalshiClient`; enforce `min_interval` between requests with `time.monotonic`; backoff `0.5 * 2**attempt`.
- [ ] **Step 4: Run** tests. Expected: PASS.
- [ ] **Step 5: Live smoke (manual, not a test)**: `python -c "from kalshi_lab.client import KalshiClient as C; print(next(C().settled_markets('KXNBAGAME', max_pages=1))['ticker'])"` prints a ticker.
- [ ] **Step 6: Commit** `feat: add Kalshi API client with pacing and retries`.

### Task 3: Store, filters, price sampling, dataset

**Files:**
- Create: `kalshi_lab/store.py`, `kalshi_lab/collect.py`, `tests/test_collect.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `KalshiClient` from Task 2.
- Produces in `store.py`: `class Store(path: str | Path)` creating tables `markets` (PK `ticker`), `events` (PK `event_ticker`; columns series_ticker, category, title, mutually_exclusive); `markets` also stores `occurrence_datetime` (real-world event time from the API), `candles` (PK `ticker, end_period_ts`; columns price_close, volume); methods `upsert_markets(rows: list[dict])`, `upsert_event(row: dict)`, `upsert_candles(ticker: str, rows: list[dict])`, `has_event(event_ticker) -> bool`, `has_candles(ticker) -> bool`, `markets_df() -> pd.DataFrame`, `events_df()`, `candles_df()`. Upserts are idempotent.
- Produces in `collect.py`:
  - `keep_market(m: dict, min_volume: float = 100.0) -> bool` (Global Constraints filters; converts `volume_fp` string)
  - `sample_price(candles: list[tuple[int, float | None]], close_ts: int, horizon_h: int, lookback_h: int = 12) -> float | None` where candles are `(end_period_ts, price_close)`; picks the latest candle with `end_period_ts <= close_ts - horizon_h*3600` and non-null price within lookback
  - `build_dataset(store: Store, horizons: tuple[int, ...] = (168, 24, 1), min_volume: float = 100.0) -> pd.DataFrame` with columns exactly: `ticker, event_ticker, series_ticker, category, title, close_time, occurrence_time, horizon_h, prob, outcome, volume_fp, open_interest_fp` (12 columns); `anchor: str = "close"` parameter, `"occurrence"` measures horizons back from `occurrence_datetime` (falls back to close_time when missing); applies one-market-per-event (lowest eligible ticker) before sampling
  - `main(argv: list[str] | None = None) -> int` CLI: `--series A,B --pages N --db data/kalshi.db --out data/dataset.csv`; fetches settled markets, filters, fetches each new event once, fetches hourly candles for `[close-8d, close]` once per market, writes the CSV.

- [ ] **Step 1: Write failing tests**: `test_keep_market_drops_mve`, `test_keep_market_drops_non_binary_and_unsettled_result`, `test_keep_market_volume_floor` (99.9 dropped, 100 kept, string input), `test_sample_price_uses_last_candle_before_cutoff`, `test_sample_price_never_uses_future_candle`, `test_sample_price_none_when_outside_lookback`, `test_sample_price_skips_null_price_candles`, `test_sample_price_empty_candles_returns_none`, `test_build_dataset_one_market_per_event` (two-sided event yields one ticker, the alphabetically lowest), `test_build_dataset_outcome_from_result_not_price`, `test_build_dataset_skips_rows_without_price` (empty candles -> no row, no crash), `test_build_dataset_columns_and_dtypes` (12 columns), `test_build_dataset_occurrence_anchor_shifts_sampling`, `test_store_upsert_idempotent` (insert twice, count 1).
- [ ] **Step 2: Run** `pytest tests/test_store.py tests/test_collect.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `store.py` with `sqlite3`; **Step 4: implement** `collect.py` per the signatures. Parse candle `price.close_dollars` (may be null/absent) to float; `close_time` to epoch via `datetime.fromisoformat(s.replace("Z","+00:00"))`.
- [ ] **Step 5: Run** tests. Expected: PASS.
- [ ] **Step 6: Live collect (data stays private)**: `python -m kalshi_lab.collect --series KXNBAGAME --pages 1`; confirm `data/dataset.csv` exists, has all 12 columns, and `git status` shows `data/` ignored.
- [ ] **Step 7: Commit** `feat: add SQLite store, collector and dataset builder`.

### Task 4: Calibration analysis

**Files:**
- Create: `kalshi_lab/calibration.py`, `tests/test_calibration.py`

**Interfaces:**
- Consumes: `metrics.*`; dataset columns from Task 3.
- Produces: `score_by(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame` (columns: group keys, `n`, `base_rate`, `brier`, `log_loss`, `ece`, `brier_always_half`); `fixed_cohort(df: pd.DataFrame, base_horizon_h: int) -> pd.DataFrame` (only tickers that have a row at `base_horizon_h`, all horizons kept); `tail_truncate(df, lo: float = 0.02, hi: float = 0.98) -> pd.DataFrame` (strict inequalities on `prob`); `bucket_significance(p, y, n_bins: int = 10) -> pd.DataFrame` (reliability_table plus `p_value` from `scipy.stats.binomtest(k, n, mean_pred)` two-sided, and `bh_significant` from `benjamini_hochberg`; n=0 buckets get NaN); `volume_band(df) -> pd.Series` (labels `"<1k"`, `"1k-10k"`, `">=10k"` from `volume_fp`); `plot_reliability(df: pd.DataFrame, horizon_h: int, path: Path) -> None` (diagonal, bucket points, CI whiskers, n labels); `main(argv)` CLI `--data data/dataset.csv --out outputs` writing `scores_by_horizon.csv`, `scores_by_category.csv`, `scores_by_volume_band.csv`, `scores_fixed_cohort.csv`, `scores_tail_truncated.csv`, `scores_by_anchor.csv` (close vs occurrence, Sports only), `murphy_by_horizon.csv`, `reliability_<h>h.csv`, `reliability_<h>h.png`, and printing a warning for any slice with fewer than 100 events.

- [ ] **Step 1: Write failing tests**: `test_score_by_horizon_known_values` (tiny frame, hand-computed Brier), `test_score_by_multiple_keys`, `test_volume_band_boundaries` (999, 1000, 9999, 10000), `test_fixed_cohort_drops_tickers_missing_base_horizon`, `test_tail_truncate_excludes_boundaries` (0.02 and 0.98 dropped), `test_bucket_significance_flags_gross_miscalibration_only` (calibrated synthetic data: no bh_significant; p=0.9 but y always 0 for n=200: flagged), `test_plot_reliability_writes_png` (tmp_path, file non-empty), `test_main_warns_on_small_slice` (capsys contains "n < 100").
- [ ] **Step 2: Run** tests. Expected: FAIL.
- [ ] **Step 3: Implement** with matplotlib `Agg` backend; **Step 4: Run** tests. Expected: PASS.
- [ ] **Step 5: Live run** `python -m kalshi_lab.calibration`; open one PNG to confirm it renders.
- [ ] **Step 6: Commit** `feat: add calibration scoring, slices and reliability plots`.

### Task 5: Report plots

**Files:**
- Create: `kalshi_lab/report_plots.py`, `tests/test_report_plots.py`

**Interfaces:**
- Produces: `plot_horizon_comparison(scores: pd.DataFrame, path: Path)` (Brier/log loss/ECE bars by horizon), `plot_calibration_gap(df, horizon_h, path)` (gap per bucket with 95% CI whiskers; red only when |gap| > 2*SE), `plot_prob_histogram(df, horizon_h, path)` (split by outcome), `main(argv)` writing `horizon_comparison.png`, `calibration_gap_<h>h.png`, `prob_histogram_<h>h.png` into `outputs/`. Price-path plot is dropped (needs raw candles; YAGNI for v1).

- [ ] **Step 1: Write failing tests**: one per plot function asserting the PNG is created and non-empty with a small synthetic frame, plus `test_plot_calibration_gap_handles_empty_bins`.
- [ ] **Step 2: Run** tests. Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: add report figures`.

### Task 6: Recalibration model

**Files:**
- Create: `kalshi_lab/model.py`, `tests/test_model.py`

**Interfaces:**
- Produces: `MIN_ROWS = 200`; `time_split(df: pd.DataFrame, frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]` (sort by `close_time`, oldest `frac` train; never splits by random); `make_features(df: pd.DataFrame, categories: list[str]) -> np.ndarray` (`logit(prob)` clipped to [0.01,0.99], `log1p(volume_fp)`, category one-hots in the given order); `evaluate(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame` returning rows `market`, `logistic_platt` (logit(p) only), `logistic_full`, `isotonic`, with columns `model, n_test, brier, log_loss, ece`; `main(argv)` CLI `--data --horizon 24 --out outputs/model_<h>h.csv`. Raises `ValueError("need >= 200 rows ...")` below `MIN_ROWS`, and `ValueError` if train or test has a single outcome class. Also prints the Platt slope/intercept (slope ~1, intercept ~0 means calibrated).

- [ ] **Step 1: Write failing tests**: `test_time_split_is_chronological` (all train close_time <= all test close_time), `test_time_split_no_event_straddles` (event_ticker never in both), `test_make_features_shape_and_order`, `test_evaluate_raises_below_min_rows`, `test_evaluate_raises_on_single_class_test`, `test_evaluate_returns_all_four_models` (synthetic 400-row frame where truth = prob, so market is calibrated), `test_evaluate_isotonic_not_fit_on_test` (fit only on train: mutate test labels, train predictions unchanged).
- [ ] **Step 2: Run** tests. Expected: FAIL. **Step 3: Implement** with `LogisticRegression` and `IsotonicRegression(out_of_bounds="clip")`; `time_split` should keep each event's rows together (split on event boundaries).
- [ ] **Step 4: Run** tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add recalibration models with time-based split`.

### Task 7: Real run, README, and public repo

**Files:**
- Modify: `README.md`; Create: `outputs/*` (generated plots/tables only), `.github` not needed.

- [ ] **Step 1: Full collect** `python -m kalshi_lab.collect --series KXNBAGAME,KXNFLGAME,KXNHLGAME,KXFED --pages 3` (idempotent; may take 30-45 min, run in background). Report how many events and rows result; if under 1,000 markets, say so rather than padding.
- [ ] **Step 2: Run** `python -m kalshi_lab.calibration`, `python -m kalshi_lab.report_plots`, `python -m kalshi_lab.model --horizon 24` (and 168 if >= 200 rows). Write down findings with n and CIs; apply the leakage checklist before calling any model result a win.
- [ ] **Step 3: Write `docs/papers.md`** (citation + one-paragraph takeaway + replicable? for each of the four papers) **and README**: question, plain-English explanation of each method (this is a learning project), the paper-finding -> our-result -> consistent/inconclusive table, method, data traps, results (embed PNGs, quote tables with n and CI), honest limitations (single time window, correlated markets, survivorship), how to reproduce, and the disclaimer (independent of Kalshi, no data included, not financial advice).
- [ ] **Step 4: Safety check before publishing**: `git ls-files | grep -E '^data/|\.db$|dataset\.csv'` must print nothing; confirm `outputs/` contains only plots and aggregate score tables (the `reliability_*.csv`/`scores_*.csv` hold bucket aggregates, not per-market rows; delete any file that has per-market rows).
- [ ] **Step 5: Run full suite** `pytest -v`. Expected: all PASS.
- [ ] **Step 6: Commit** `docs: add README with results`, then create and push: `gh repo create CooperMcKay/kalshi-calibration-lab --public --source . --push --description "Is Kalshi calibrated? Reliability analysis and recalibration of prediction-market prices"`.
- [ ] **Step 7: Verify** `gh repo view CooperMcKay/kalshi-calibration-lab --json visibility,url` shows PUBLIC.
