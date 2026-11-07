# Kalshi Calibration Lab

**Is Kalshi calibrated? Where is it not? Can a small model fix it?**

A small, from-scratch study of prediction-market prices. When a market says 70%, does the event happen about 70% of the time? The project measures that, slices it by time horizon and market type, then tests whether simple recalibration models beat the raw price.

> Independent project. Not affiliated with Kalshi. No Kalshi data is included in this repo (you regenerate it from the public API). Nothing here is financial advice or trading advice.

## Headline results

Data: 2,588 settled Kalshi events across nine sports series (NBA, NFL, NHL, MLB, college football, college basketball, WNBA, UFC, plus 2 Fed events), one market per event, volume >= 100. **These results are about sports game-winner markets.** Brier and ECE are shown with bootstrap 95% intervals (1,000 resamples).

| Horizon before close | n | Brier (lower is better) | ECE | "Always say 50%" Brier |
|---|---|---|---|---|
| 168h | 866 | 0.167 (0.153-0.180) | 0.036 | 0.25 |
| 24h | 2,498 | 0.187 (0.180-0.193) | 0.012 | 0.25 |
| 1h | 2,587 | 0.134 (0.127-0.141) | 0.017 | 0.25 |

- **Prices look well calibrated.** ECE is 0.01-0.04 and after Benjamini-Hochberg false-discovery control **no reliability bucket is significantly off** at any horizon. (ECE intervals are biased upward under resampling, so read them as rough.)
- **Prices are sharpest at 1h.** The 168h row is a different, smaller population (markets that opened a week out), so it should not be read as a trend with the other two.
- **No evidence of favorite-longshot bias at 24h or 168h.** At 1h the logit slope is 1.08 (95% CI 1.00-1.17), only just above the calibrated value of 1, with price gaps around one percentage point. I would not call that a finding.
- **Recalibration does not help.** On a time-ordered test set (749 events at 24h), the raw market scored Brier 0.168, Platt 0.169, isotonic 0.171. Same pattern at 1h and 168h. An earlier version of this project also fed final volume and open interest into a model and saw it "win"; that was lookahead (those totals include trading after the prediction time), so the model was removed.
- **The anchor choice matters.** Measuring from the event's scheduled time instead of the market's close time changes the 1h Brier from 0.134 to 0.113 in this sample. I have not investigated why, so I report it as a data trap, not a finding.
- **Volume and error: not just "busy games are closer games".** Higher-volume markets are closer to 50/50 (Spearman -0.25 to -0.36), which explains part of the volume-error link. A positive partial correlation of +0.12 to +0.17 remains after controlling for closeness, in nearly every series. Caveat: final volume includes post-horizon trading, so this may still be partly lookahead.

Full tables and plots are in [`outputs/`](outputs/). Reliability diagrams show binomial 95% intervals and the sample size in every bucket.

## How it compares to the papers

I read all four papers and compared concrete numbers (`outputs/paper_comparison.csv`, full detail in [`docs/papers.md`](docs/papers.md)). Short version, for Sports:

| Finding | Paper | Ours | Verdict |
|---|---|---|---|
| Tail truncation (2-98c) raises Brier | yes | 0.166 vs 0.134 at 1h | Consistent |
| Event-time anchor changes Sports Brier | improves | 0.134 -> 0.113 at 1h | Consistent in direction |
| Sports Brier / accuracy levels | 0.104-0.155 / 76-86% | 0.134-0.187 / 69-80% | Not comparable: the paper pools all sports markets including near-certain props; we use near-coin-flip game winners |
| More volume, lower error | yes | **opposite** (rho +0.29 at 24h, +0.35 at 1h) | Disagrees; part of it is busy games being closer games, a smaller effect remains |

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
SERIES=KXNBAGAME,KXNFLGAME,KXNHLGAME,KXMLBGAME,KXNCAAFGAME,KXNCAAMBGAME,KXWNBAGAME,KXUFCFIGHT
python -m kalshi_lab.collect --series $SERIES --pages 10
python -m kalshi_lab.collect --series $SERIES --pages 10 \
    --anchor occurrence --out data/dataset_occurrence.csv
python -m kalshi_lab.calibration
python -m kalshi_lab.report_plots
python -m kalshi_lab.model --horizon 24
python -m kalshi_lab.paper_compare
python -m kalshi_lab.stats_tests
pytest
```

No API key is needed; the public endpoints are used with a polite request rate. Raw data lands in a git-ignored `data/` folder.

## Interactive dashboard

```bash
make dashboard   # builds dashboard/dashboard.html from your local data and opens it
```

One self-contained HTML file: a reliability explorer (change horizon, bucket count, tail cutoff and volume floor and watch the flags appear and vanish), a market browser with price paths, a Brier-score playground, the recalibration results, and the paper comparison. The generated file embeds per-market data, so it is gitignored; only the generator is public.

## Limitations

- 2,588 events, all sports. Nothing here speaks to politics, economics, or extreme prices.
- Model comparisons have no confidence intervals on the test-set scores, and the test set is time-ordered, so it reflects one period.
- The 168h horizon only exists for markets that opened a week out, so it is a different (and smaller) population.
- Candle prices are last traded prices, which can be stale on thin markets.
- This describes how prices behaved; it says nothing about whether any strategy would have made money.

## License

MIT. See [LICENSE](LICENSE).
