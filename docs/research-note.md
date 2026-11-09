# Is Kalshi lying? A small calibration check

*Draft, 7 Nov 2026. Independent project, not affiliated with Kalshi. Not trading or financial advice.*

## The question

When a prediction market says 70%, does the event happen about 70% of the time? If it does, the price can be read as a probability. If it does not, someone is paying for a story. Kalshi Research has published large-scale versions of this check; this note repeats it in miniature, adds the categories its sports-heavy sample hides, and is candid about what ten weeks of data can and cannot show.

## Data and method

22,019 observations from 3,695 settled events closing between 31 Aug and 7 Nov 2026: sports game winners (2,588 events) plus economics, weather, elections, mentions, commodities, financials, politics and entertainment. For each market I take the last traded price at 168, 24 and 1 hours before close, never a later one, and compare it with how the market settled. Sports events keep one market per event. Other events keep every market, and every interval resamples whole events so correlated markets (a rate ladder, a temperature bracket) are not counted as independent. Buckets are 10 points wide; a bucket is flagged only if it survives Benjamini-Hochberg false-discovery control across the buckets tested.

## What I found

1. **When Kalshi says 70%, it is about right.** At 24h the 65-75% band said 69.3% and happened 67.3% (371 events, gap -2.0 points, interval -6.4 to +2.9). The middle of the price range shows no significant gap anywhere.
2. **Sports are calibrated at 24h.** No sports band is flagged there. Two extreme sports bands are flagged at other horizons (under 5% at 1h, over 95% at 168h), in the same direction as the non-sports tail effect. The intervals are wide (3 to 10 points per band), so this means "no detectable lie", not "provably honest".
3. **The one exception is at the extremes, and only outside sports.** Contracts priced under 5% happened 0.7% of the time against an average price of 1.9%; contracts priced over 95% happened slightly more often than priced. This favorite-longshot pattern survives resampling by day, not just by event (calibration slope 1.11, interval 1.04-1.19). It is worth about 1 to 2 points, and part of it is the 1 cent price floor: a contract that is really 0.5% likely cannot trade below 1 cent.
4. **A small model cannot beat the price.** Platt and isotonic recalibration trained on the oldest 70% of events tie the raw market on the newest 30% (Brier 0.1080 vs 0.1083 and 0.1082 at 24h). The only visible gain is log loss at 1h, where tails matter.

## A result that did not survive

An early version of this project found that busier markets had *larger* errors, the opposite of Kalshi's paper, and a model using volume appeared to beat the market. Both came from using each market's final volume, which includes trading after the prediction time. With volume measured as of the prediction time, the effect vanishes (partial correlation between -0.04 and +0.02 across horizons, down from +0.03 to +0.13). The model was removed. I mention it because it is the most instructive thing in the project: a plausible, even publishable-looking, edge that was future information in disguise.

## What this cannot say

- Ten weeks, 69 distinct close dates. Nothing here says how Kalshi behaved in other seasons.
- Economics (24 events), politics (14), entertainment (17) and others are too small to judge alone. The cross-category comparison is mostly sports vs weather vs elections.
- Election markets are one night of results, so their 296 events are far fewer independent observations.
- The 1h horizon is not forecasting for most non-sports markets (the outcome is nearly known), so conclusions rest on 24h and 168h.
- Last-traded prices can be stale on thin markets. Calibration says nothing about whether any strategy would have made money.

## Data terms

Kalshi's Developer Agreement (v1.1) limits API use to facilitating a member's own trading, prohibits collecting, caching, aggregating or storing API data for other purposes (3.1), bars sharing it without written authorization, and requires API users to be Members (4.1.1). I read public, keyless endpoints from Canada, where I cannot hold an account, so it is unclear whether and how the agreement applies. The public repo contains code and aggregate results only. This note shares aggregate results only, never raw data.

## Reproduce

See the README. `make analyze` and `python -m kalshi_lab.lying` regenerate every table and figure from a local database.
