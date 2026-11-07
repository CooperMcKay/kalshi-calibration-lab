# Research papers behind this project

Four Kalshi Research papers inspired the lab. The PDFs are not redistributed here. This file records what each one claims, what we replicated, and how our numbers line up (`python -m kalshi_lab.paper_compare` regenerates `outputs/paper_comparison.csv`).

## 1. Kagan & Baiocchi, *Calibration in Prediction Markets: Theory and Evidence* (Aug 2026, rev. Oct 2026)
**Data:** 2,324,833 resolved Kalshi markets, 2021 to mid-2026. **Claims we tested:**
- Brier falls as resolution nears (about 0.087 at 3 months to about 0.02 at close, all markets). Sports is *not* monotone in the paper: 0.107 (1-week), 0.155 (1-day), 0.104 (1-hour), 0.023 (close), because in-game prices are hard.
- Tail truncation (keep only 2-98 cents) *raises* Brier (0.030 -> 0.078 at close on the filtered subsample): confident prices carry real information.
- Brier falls with event volume, within every horizon.
- Re-anchoring Sports to the real game-end time (a proprietary to-the-second dataset) keeps calibration excellent. Sports market close times lag game end by minutes.
- Naive accuracy (price > 50c predicts YES; 50c counts as wrong) for Sports: 84.6% (1-week), 76.2% (1-day), 85.5% (1-hour).
- The *fixed cohort* trick: compare horizons on the same markets so changing composition does not fake a trend.

## 2. Kagan & Baiocchi, *Kalshi and the 2026 Midterms: Calibration and Accuracy* (Nov 2026)
Races called by 6:26am ET Nov 4: directional accuracy 93.8% (3 months) to 98.2% (poll close); Brier 0.040 to 0.016; Kalshi beat polling averages at every horizon and had the lowest Brier of all compared models on Election Day. Introduces the Murphy decomposition (reliability - resolution + uncertainty), which we implement. **Not replicated:** the poll and model comparisons need election data and external forecasts.

## 3. Kagan & Baiocchi, *Who Calibrates Markets? Evidence from Trader Micro-Data* (Oct 2026)
Uses over 1 billion trades. Under a strict false-discovery threshold, 0.15% of takers beat the markets they trade against (Brier skill score), and prices move toward the outcome after their trades. **Not replicable:** trader IDs are not public. **Adapted idea:** when you test many things, control the false-discovery rate. We apply Benjamini-Hochberg across the reliability buckets.

## 4. Kalshi Research, *Who Trades Prediction Markets?* (Sep 2026)
Survey and ZIP-code study: traders are middle-income; 82% of top-PnL traders never worked at a finance firm; they score high on numeracy tests. Context only, nothing to replicate with market data.

## How our numbers line up (Sports, one game-winner market per event)

| Finding | Paper | Ours | n | Verdict |
|---|---|---|---|---|
| Sports Brier, 168h / 24h / 1h | 0.107 / 0.155 / 0.104 | 0.230 / 0.225 / 0.178 | 153 / 627 / 631 | Level differs (see below) |
| Sports naive accuracy, 168h / 24h / 1h | 84.6% / 76.2% / 85.5% | 61.4% / 59.0% / 73.1% | 153 / 627 / 631 | Level differs (see below) |
| Brier improves toward resolution | yes | 0.230 -> 0.225 -> 0.178 | 631 | Consistent |
| Tail truncation raises Brier | yes | 0.186 vs 0.178 at 1h (28 rows dropped) | 631 | Consistent, tiny effect |
| Event-time anchor changes Sports Brier | improves calibration | 0.178 -> 0.137 at 1h | 631 | Consistent in direction |
| Volume lowers error | yes | 24h: rho -0.03 (p=0.48); 1h: rho +0.12 (p=0.003) | 627 / 631 | Inconclusive at 24h, **opposite** at 1h |

**Why the levels differ.** The paper pools every sports market, including spreads, totals and props whose prices sit near 0 or 100 cents, and those easy markets pull Brier down. We keep one game-winner market per event, which sits near a coin flip for most of its life. A Brier of 0.18 to 0.23 on near-coin-flip games is the *expected* result, not a failure of calibration (our ECE is about 0.03). Levels are not comparable across the two samples. Directions are.

**The volume result.** The paper notes that for Sports, volume mostly measures how much people care, not how easy a market is to price. In our sample the busiest games are probably the closest ones. We did not test that. It is a hypothesis, and a good thing to explore in the dashboard.

**The anchor result.** Kalshi's `occurrence_datetime` sits a median of about 12 minutes *after* the market close, so it is not the paper's to-the-second game-end timestamp. Moving the anchor later puts the "1 hour before" price closer to game end, and 14.6% of those prices are within 2 cents of 0 or 100, versus 4.4% for close-anchored. That fits the paper's story that timing relative to the true event matters, but we have not verified what `occurrence_datetime` represents.
