# Kalshi Calibration Lab

**Is Kalshi calibrated? Where is it not? Can a small model fix it?**

A small, from-scratch study of prediction-market prices. When a market says 70%, does the event happen about 70% of the time? The project measures that, slices it by time horizon and market type, then tests whether simple recalibration models beat the raw price.

> Independent project. Not affiliated with Kalshi. No Kalshi data is included in this repo (you regenerate it from the public API). Nothing here is financial advice or trading advice.

## Headline results

Data: 633 settled Kalshi events (NBA, NFL, NHL game winners plus Fed decisions), one market per event, volume >= 100. Almost everything is sports; the Fed series contributed only 2 usable events, so **these results are about sports markets**.

| Horizon before close | n | Brier (lower is better) | ECE | "Always say 50%" Brier |
|---|---|---|---|---|
| 168h | 153 | 0.230 | 0.035 | 0.25 |
| 24h | 627 | 0.225 | 0.032 | 0.25 |
| 1h | 633 | 0.177 | 0.033 | 0.25 |

- **Prices look well calibrated.** ECE is about 0.03 at every horizon, and after Benjamini-Hochberg false-discovery control **no reliability bucket is significantly off** at any horizon.
- **Prices get sharper as the event approaches** (Brier falls from 0.230 to 0.177), mostly through *resolution* (the Murphy term that rewards separating winners from losers: 0.019 at 168h, 0.025 at 24h, 0.074 at 1h). Reliability error stays tiny (0.002-0.003).
- **Recalibration does not clearly help.** On a time-ordered test set of about 190 events, a Platt (logistic) fit on price alone gave a slope of 0.96 (1.0 means already calibrated) and scored slightly *worse* than the raw price. Isotonic regression was no better. A logistic model that also sees volume and open interest improved Brier (24h: 0.194 vs 0.204; 1h: 0.159 vs 0.170) but worsened ECE, on a small test set with no confidence intervals, so treat that as a lead rather than a result. The 168h horizon has only 153 rows, below the 200-row minimum, so `model.py` refuses to run there.
- **The anchor choice matters.** Sampling relative to the market's close time vs the event's scheduled start changes the 1h Brier from 0.178 to 0.137 in this sample. I have not investigated why, so I report it as a data trap, not a finding.

Full tables and plots are in [`outputs/`](outputs/). Reliability diagrams show binomial 95% intervals and the sample size in every bucket.

## How it compares to the papers

| Paper finding | Our small-sample result | Verdict |
|---|---|---|
| Kalshi prices are well calibrated overall | ECE about 0.03, no BH-significant bucket | Consistent |
| Accuracy improves closer to resolution | Brier 0.230 -> 0.225 -> 0.177 | Consistent |
| Calibration depends on how you cut the data (horizon, tails, volume) | Fixed-cohort and tail-truncated scores match the headline closely; volume and category slices are too small to read | Inconclusive |
| Systematic miscalibration at extreme prices (favorite-longshot style) | The 0.0-0.1 and 0.9-1.0 buckets have 0-55 rows and show no significant gap | Inconclusive |

See [`docs/papers.md`](docs/papers.md) for the papers and what could and could not be replicated.

## Methods in plain English

- **Calibration**: group markets by their price (0-10%, 10-20%, ...). In each group, the share that resolved YES should be near the price. The gap between them is the calibration error.
- **Brier score**: the average of (price - outcome)^2, where outcome is 1 or 0. Lower is better; always saying 50% scores 0.25.
- **Log loss**: like Brier but punishes confident wrong answers much harder.
- **ECE (expected calibration error)**: the average gap between price and reality across buckets, weighted by bucket size.
- **Murphy decomposition**: Brier = reliability - resolution + uncertainty. *Reliability* is calibration error (want small), *resolution* is how well prices separate events that happen from ones that do not (want large), *uncertainty* is how unpredictable the events are in the first place.
- **Benjamini-Hochberg**: testing 10 buckets means some will look off by chance. BH controls the share of false alarms among the buckets we flag.
- **Sampling a price at a horizon**: take the last traded price at or before (close time - horizon), looking back at most 12 hours. Never a future price, never forward-filled. Settlement prices collapse to 0 or 1, which is why a naive "last price" would be pure leakage.
- **Fixed cohort**: compare horizons on the *same* markets, so a score change reflects time, not a different set of markets.
- **Tail truncation**: drop prices at or beyond 2 and 98 cents to look only at genuinely uncertain markets.
- **Recalibration**: fit on the oldest 70% of events, score on the newest 30%, with the cut on an event boundary so no event appears in both sets.

## Data cleaning rules

Binary, cleanly settled markets only; multivariate parlays removed; volume floor of 100; one market per event (alphabetically first ticker) so rows are independent; prices parsed from Kalshi's string format; all times UTC.

## Reproduce it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m kalshi_lab.collect --series KXNBAGAME,KXNFLGAME,KXNHLGAME,KXFED --pages 3
python -m kalshi_lab.collect --series KXNBAGAME,KXNFLGAME,KXNHLGAME,KXFED --pages 3 \
    --anchor occurrence --out data/dataset_occurrence.csv
python -m kalshi_lab.calibration
python -m kalshi_lab.report_plots
python -m kalshi_lab.model --horizon 24
pytest
```

No API key is needed; the public endpoints are used with a polite request rate. Raw data lands in a git-ignored `data/` folder.

## Limitations

- 633 events, nearly all sports. Not enough to say anything about politics, economics, or extreme prices.
- Results are point estimates in a small sample; model comparisons have no confidence intervals.
- The 168h horizon only exists for markets that opened a week out, so it is a different (and smaller) population.
- Candle prices are last traded prices, which can be stale on thin markets.
- This describes how prices behaved; it says nothing about whether any strategy would have made money.

## License

MIT. See [LICENSE](LICENSE).
