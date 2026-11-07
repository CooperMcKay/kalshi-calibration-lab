# Project status (handoff)

Repo: https://github.com/CooperMcKay/kalshi-calibration-lab (public, MIT). Raw data in `data/` is gitignored and stays private.
Goal: learn prediction markets, math and stats; publish code and results. Question: is Kalshi calibrated, where not, can a small model fix it.

## Done
- Collector, SQLite store, metrics (Brier, ECE, log loss, Murphy, BH), calibration analysis, plots, recalibration models, paper comparison, dashboard generator, stats tests. 105 tests passing, built test-first.
- Data: 2,588 sports events (NBA, NFL, NHL, MLB, NCAAF, NCAAMB, WNBA, UFC + 2 Fed). ATP/WTA tickers returned nothing.
- Findings: well calibrated (no BH-significant bucket); Brier 0.167 / 0.187 / 0.134 at 168h / 24h / 1h; no recalibrator beats the market price; no clear favorite-longshot bias (1h slope 1.08, CI 1.00-1.17); volume vs error is positive (opposite the paper), partly because busy games are closer games, partial rho +0.12 to +0.17 remains.
- Bug fixed: the old `logistic_full` model used final volume/open interest (lookahead). Removed.
- `make analyze` reruns all analyses; `make dashboard` builds `dashboard/dashboard.html` (gitignored). Private hosted copy: https://claude.ai/artifact/UoceBwLDviCtu67brEW7WR

## Next (in this order)
1. Pre-horizon volume: compute volume as of the prediction time from candle data (store has candle `volume`), then retest whether the volume-error effect survives. Do it test-first.
2. Research note outline (follow-up to Kalshi Research's calibration work), using `docs/papers.md` and `outputs/`.
3. Forecasting game: user guesses a probability before seeing Kalshi's price, both scored on resolution.
4. Later: sportsbook odds benchmark (hard to source), more series, scheduled refresh.

## Open caveats
- ECE bootstrap intervals are biased upward.
- 168h is a different (smaller) population.
- `occurrence_datetime` semantics unverified.
- Check Kalshi data terms before publishing any dataset.
