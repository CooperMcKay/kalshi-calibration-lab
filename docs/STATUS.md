# Project status (handoff)

Repo: https://github.com/CooperMcKay/kalshi-calibration-lab (public, MIT). Raw data in `data/` is gitignored and stays private.
Goal: learn prediction markets, math and stats; answer "Is Kalshi lying? When it says 70%, does it happen 70% of the time?" Published as code and aggregate results only.

## Done (2026-11-07)
- Pipeline, metrics, calibration, recalibration, paper comparison, dashboard, stats tests, and `kalshi_lab/lying.py` (price-band and category gaps with event-clustered bootstrap, Benjamini-Hochberg flags). 123 tests, built test-first.
- Data: 22,019 rows, 3,695 events, 31 Aug - 7 Nov 2026. Sports (2,588 events) plus economics, weather, elections, mentions, commodities, financials, politics, entertainment. Non-sports keep every market per event; sports keep one.
- Pre-horizon volume (`volume_pre`) fixed the lookahead: the volume-error effect is gone (partial rho ~0.01-0.02).
- Findings: calibrated in the middle of the range and in sports; small favorite-longshot tail effect outside sports (slope 1.11, 1.04-1.19 by day); recalibration ties the market.
- Research note drafted: `docs/research-note.md`.

## Data terms (decided)
Kalshi Developer Agreement v1.1 section 3.1 bars collecting/caching/storing API data except to facilitate your own trading, and 4.1.1 requires Membership. Cooper is in Canada and cannot be a Member. Cooper's call (2026-11-08): publishing aggregate analysis is fine; raw data stays private. Ten weeks of data is enough; no need to extend history.

## Next
1. Push and share the note.
2. Add confidence intervals to the recalibration test scores.
3. Forecasting game (next, now that the note is done).
4. Later: sportsbook benchmark, scheduled refresh.

## Open caveats
- ECE bootstrap intervals are biased upward.
- 168h is a different (smaller) population.
- `occurrence_datetime` semantics unverified (event-time anchor lowers 1h sports Brier 0.134 -> 0.113, cause unknown).
- 1 cent price floor compresses the lowest prices; 21% of 24h non-sports prices are <= 2 cents.
- Most non-sports categories have fewer than 100 events and get no verdict.
