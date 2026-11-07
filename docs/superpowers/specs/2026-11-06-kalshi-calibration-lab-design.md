# Kalshi Calibration Lab: Design

Date: 2026-11-06. Owner: Cooper. Status: approved in chat; method detail lives in the private `Brainstorm.Context.md` (gitignored), summarised here.

## Intent
Answer, with honest statistics: **is the Kalshi prediction market calibrated, where is it not, and can a small model recalibrate it?** Arc: measure -> slice -> model. The deliverable is understanding and a public portfolio project. "The market is hard to beat" is a valid result.

## Goals
- Reproducible pipeline: collect settled markets, sample prices at fixed horizons (168h, 24h, 1h before close), score them.
- Metrics: Brier, log loss, ECE, reliability tables with binomial 95% CIs.
- Slices: category, horizon, volume band. No claim from a slice with n < 100 events.
- Model: logistic (Platt-style) and isotonic recalibration, time-ordered 70/30 split, scored on test only vs. raw market.
- Public repo with code, tests, method write-up, result plots and score tables.

## Research basis (added 2026-11-06)
Four Kalshi Research papers inspired the project (PDFs kept local, gitignored; cited in README and `docs/papers.md`). Personal goal: learn how prediction markets work and build skills, so each method is explained in the README, not just run.
- Kagan & Baiocchi, *Calibration in Prediction Markets* -> replicate in miniature: Brier by horizon, per-horizon vs fixed-cohort, tail truncation (2-98 cents), volume slices, naive accuracy, close-time vs event-time anchoring (via the market's `occurrence_datetime`).
- Kagan & Baiocchi, *2026 Midterms* -> Murphy decomposition (reliability / resolution / uncertainty), directional accuracy.
- Kagan & Baiocchi, *Who Calibrates Markets?* -> not replicable (needs trader IDs). Adapted idea: Benjamini-Hochberg false-discovery control across bucket tests, so we only believe buckets that survive multiple comparisons.
- Kalshi Research, *Who Trades Prediction Markets?* -> context only (survey data). Not replicated.
- Stretch (not v1): compare Kalshi to external forecasts (polls/models) as in the Midterms paper.
- Final README includes a table: paper finding -> our small-sample result -> consistent / inconclusive. Our n is tiny next to 2.3M markets, so "inconclusive" is an expected, honest answer.

## Non-goals / anti-goals
- No trading, no trading advice, no Kalshi account.
- No raw Kalshi data in the repo or in any published artifact. `data/` is gitignored.
- Anti-goal: a story told from noise. Any result must show its n and CI; a model "win" must pass the leakage checklist.

## Constraints
- Public keyless endpoints only (verified live 2026-11-06). Base `https://external-api.kalshi.com/trade-api/v2`. Polite client: ~5 req/s, backoff on 429/5xx, descriptive User-Agent.
- Personal/non-commercial use; Kalshi data stays local. README states: independent, not affiliated with Kalshi, no data included, not financial advice.

## Decisions (Cooper's)
- Code + results public, raw data private.
- Repo `CooperMcKay/kalshi-calibration-lab`, public, MIT.

## Decisions (my calls, flagged)
- Python 3.14, venv, pytest; layout `kalshi_lab/{client,collect,metrics,calibration,report_plots,model}.py`, `tests/`, `outputs/`.
- One market per event in the analysis dataset (not both sides), so rows are independent and base rate is not mechanically 0.5.
- SQLite raw store (markets/events/candles) -> derived `dataset.csv`; candles cached so changing horizons costs no API calls.
- Filter out MVE parlays, require `volume_fp >= 100`.
- Series: KXNBAGAME, KXNFLGAME, KXNHLGAME, plus KXFED (to populate the 168h horizon).
- `model.py` refuses to run below 200 rows at a horizon.

## Known data traps to handle (from recon)
MVE parlays; `last_price` collapses at settlement (sample by horizon); sports close after the game so the 1h horizon is in-game; threshold ladders are correlated; stale quotes on dead markets; 168h empty for short-lived markets; candles use last trade, not 0/1; sides needn't sum to 1; prices are strings, times UTC.

## Left to the builder
Exact CLI flags, plot styling, table formats, how to choose which side of an event to keep (default: first market ticker alphabetically, documented), README wording.

## Done when
Tests pass; a real run on 1,000+ markets produces plots and tables; README reports findings with CIs; repo is public with no data committed.
