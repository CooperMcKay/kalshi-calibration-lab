# Kalshi Calibration Lab

**Is Kalshi lying? When a market says 70%, does it happen 70% of the time?**

A small, from-scratch study of prediction-market prices. It measures how far prices are from the frequencies they imply, slices that by price level, time horizon and category (sports, weather, elections, economics and more), then tests whether simple recalibration models beat the raw price. A short write-up is in [`docs/research-note.md`](docs/research-note.md).

> Independent project. Not affiliated with Kalshi. No Kalshi data is included in this repo. Nothing here is financial advice or trading advice. **Read [Data terms](#data-terms) before collecting data yourself.**

## The short answer

**Mostly no, with one small exception at the extremes.** Across 22,019 price observations from 3,695 settled events (ten weeks of data, 31 Aug to 7 Nov 2026), when Kalshi says 70% the thing happens about 70% of the time.

| Kalshi said (24h before close) | Happened | n events | Verdict |
|---|---|---|---|
| 65-75% (all categories) | 67.3% (said 69.3%) | 371 | Consistent: gap -2.0 pts, 95% interval -6.4 to +2.9 |
| 45-55% | 50.5% (said 49.4%) | 724 | Consistent |
| 85-95% | 90.6% (said 90.0%) | 305 | Consistent |
| 0-5% | **0.7% (said 1.9%)** | 960 | **Longshots happen less often than priced** (significant after Benjamini-Hochberg false-discovery control) |
| 95-100% | 98.8% (said 97.9%) | 436 | Slightly under-confident, not significant at 24h; significant at 1h |

The exception is the classic favorite-longshot pattern: very cheap contracts win about 1 to 2 points less often than their price, and near-certain ones win about 1 point more often. It is small, it shows up in non-sports markets (sports has too few extreme prices to tell), and part of it is the 1 cent minimum price: a contract that is really 0.5% likely still cannot trade below 1 cent. Treat it as "prices are rounded toward the middle at the extremes", not as Kalshi misleading anyone.

All intervals resample whole events, because markets inside one event (a rate ladder, a temperature bracket) move together. Plots: [`outputs/lying_bands_24h.png`](outputs/lying_bands_24h.png), [`outputs/lying_category_gaps_24h.png`](outputs/lying_category_gaps_24h.png). Tables: `outputs/lying_*.csv`.

## By category (24h before close)

| Category | Events | Brier | Verdict |
|---|---|---|---|
| Sports (game winners) | 2,498 | 0.187 | Consistent with calibrated |
| Weather (daily highs) | 476 | 0.095 | Small average bias (-0.8 pts) and tail bias |
| Elections (House races) | 296 | 0.012 | Tail bias; but one election night, so far fewer independent observations than events |
| All non-sports pooled | 981 | 0.098 | Tail bias (slope 1.11, interval 1.03-1.20; 1.04-1.19 when resampling whole days) |
| Economics, Politics, Entertainment, Financials, Mentions, Commodities | 14-85 each | n/a | **Too few events to say anything** (floor is 100) |

Read the sports vs non-sports contrast carefully: they are different kinds of question. Sports games are near coin flips; most non-sports contracts are strikes far from the money (a 1% price on a temperature bracket), so their Brier scores are not comparable and the extremes dominate the non-sports result.

## Other results

- **Recalibration does not help.** On a time-ordered test set (all categories), the raw market scored Brier 0.1080 at 24h against 0.1083 (Platt) and 0.1082 (isotonic); 0.0437 vs 0.0436 and 0.0439 at 1h; 0.0953 vs 0.0954 and 0.0979 at 168h. These are ties. The only visible gain is log loss at 1h (0.1455 to 0.1388 with Platt), which is where mispriced tails cost the most. No confidence intervals on these test scores yet.
- **Volume does not explain error.** An earlier version of this README reported that busier markets had *larger* errors (the opposite of Kalshi's paper). That was lookahead: it used each market's final volume, which includes trading after the prediction time. Using volume traded *before* the horizon, the partial correlation with error (controlling for how close to 50/50 the price is) falls from +0.12 to +0.17 down to about +0.01 to +0.02. There is no volume effect left to explain.
- **The 1h horizon is not forecasting for most non-sports markets.** A daily high temperature is effectively known hours before the market closes (weather Brier at 1h is about 0.000). Use 24h and 168h to judge pricing; 1h measures how fast prices converge.
- **Horizons differ in population.** 168h only exists for markets that opened a week out, so it is a smaller, different set, not a trend.

## How it compares to the papers

I read all four Kalshi Research papers and compared concrete numbers (`outputs/paper_comparison.csv`, detail in [`docs/papers.md`](docs/papers.md)). Sports only:

| Finding | Paper | Ours | Verdict |
|---|---|---|---|
| Tail truncation (2-98c) raises Brier | yes | 0.166 vs 0.134 at 1h | Consistent |
| Event-time anchor changes Sports Brier | improves | 0.134 -> 0.113 at 1h | Consistent in direction (cause not investigated) |
| Sports Brier / accuracy levels | 0.104-0.155 / 76-86% | 0.134-0.187 / 69-80% | Not comparable: the paper pools all sports markets including near-certain props; we use near-coin-flip game winners |
| More volume, lower error | yes | rho -0.11 at 24h (consistent), +0.14 at 1h (not), none after controlling for closeness | Inconclusive |

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

Binary, cleanly settled markets only; multivariate parlays removed; volume floor of 100. Sports events are two-sided, so one market per event (alphabetically first ticker). All other events keep every market, since a ladder or bracket is a set of distinct questions; confidence intervals then resample whole events (or whole days) so correlated markets are not counted as independent. Prices parsed from Kalshi's string format; all times UTC.

## Reproduce it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
SPORTS=KXNBAGAME,KXNFLGAME,KXNHLGAME,KXMLBGAME,KXNCAAFGAME,KXNCAAMBGAME,KXWNBAGAME,KXUFCFIGHT
OTHER=KXFED,KXFEDDECISION,KXCPI,KXCPIYOY,KXGDP,KXPAYROLLS,KXU3,KXAAAGASW,KXHIGHNY,KXHIGHCHI,KXHIGHMIA,KXHIGHLAX,KXHIGHAUS,KXHIGHDEN,KXHIGHPHIL,KXINX,KXWTI,KXGOLDD,KXBTCD,KXEURUSD,KXHOUSERACE,KXEOWEEK,KXAPRPOTUS,KXTRUMPMENTION,KXTRUMPSAY,KXNETFLIXRANKMOVIE,KXTOPSONG
SERIES=$SPORTS,$OTHER
python -m kalshi_lab.collect --series $SERIES --pages 10
python -m kalshi_lab.collect --series $SERIES --pages 10 \
    --anchor occurrence --out data/dataset_occurrence.csv
python -m kalshi_lab.calibration
python -m kalshi_lab.report_plots
python -m kalshi_lab.model --horizon 24
python -m kalshi_lab.paper_compare
python -m kalshi_lab.stats_tests
python -m kalshi_lab.lying            # the 'is Kalshi lying' tables and plots
pytest
```

No API key is needed; the public endpoints are used with a polite request rate. Raw data lands in a git-ignored `data/` folder.

## Interactive dashboard

```bash
make dashboard   # builds dashboard/dashboard.html from your local data and opens it
```

One self-contained HTML file: a reliability explorer (change horizon, bucket count, tail cutoff and volume floor and watch the flags appear and vanish), a market browser with price paths, a Brier-score playground, the recalibration results, and the paper comparison. The generated file embeds per-market data, so it is gitignored; only the generator is public.

## Data terms

Kalshi's [Developer Agreement](https://kalshi-public-docs.s3.amazonaws.com/Kalshi-Developer-Agreement.pdf) (v1.1) says API use is limited to facilitating a member's own trading (section 3), prohibits "collecting, caching, aggregating, or storing data or content accessed via the API except for purposes of facilitating your own trading" and sharing it with third parties without written authorization (3.1), and requires API users to be Members (4.1.1). This project reads public, keyless endpoints, and whether those fall under the agreement is not something I can settle. The author is in Canada and cannot hold a Kalshi account. So: **this repo ships code and aggregate results only, never raw data, and you should read the agreement and decide for yourself before running the collector.** Publishing the write-up widely should wait for written permission from Kalshi.

## Limitations

- Ten weeks of settled markets (31 Aug to 7 Nov 2026), 69 distinct close dates. Older markets sit behind Kalshi's historical endpoints and are not included. Sports is 2,588 events; most other categories are too small to judge alone.
- Election results come mostly from one night, so they carry far fewer independent observations than their event count suggests.
- The 1 cent minimum price compresses the lowest prices; 21% of 24h non-sports prices are at or below 2 cents.
- Model comparisons have no confidence intervals on the test-set scores, and the test set is time-ordered, so it reflects one period.
- The 168h horizon only exists for markets that opened a week out, so it is a different (and smaller) population.
- Candle prices are last traded prices, which can be stale on thin markets.
- This describes how prices behaved; it says nothing about whether any strategy would have made money.

## License

MIT. See [LICENSE](LICENSE).
