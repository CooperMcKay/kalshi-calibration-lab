# Research papers behind this project

Four Kalshi Research papers inspired the lab. The PDFs are not redistributed here; this file records what each one contributed and how far this project could replicate it with public data.

| Paper | What we took from it | Replicated? |
|---|---|---|
| Kagan & Baiocchi, *Calibration in Prediction Markets* | Brier score by time horizon, fixed-cohort comparison, tail truncation (2-98 cents), volume slices, close-time vs event-time anchoring | Yes, in miniature (`calibration.py`) |
| Kagan & Baiocchi, *2026 Midterms: Calibration and Accuracy* | Murphy decomposition (reliability, resolution, uncertainty), directional accuracy | Murphy decomposition yes; no external-forecast comparison |
| Kagan & Baiocchi, *Who Calibrates Markets?* | Idea: do not trust a miscalibrated bucket until it survives multiple-comparison control | Adapted: Benjamini-Hochberg across bucket tests. The paper itself needs trader IDs, which are not public |
| Kalshi Research, *Who Trades Prediction Markets?* | Context on who the traders are | Not replicated (survey data) |

Our sample is about 600 events against the papers' millions of markets, so our results can only be *consistent with* or *inconclusive about* their findings, never a confirmation.
