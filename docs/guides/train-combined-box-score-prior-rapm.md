---
last_updated: "2026-08-04"
---

# Train The Combined Box-Score Prior RAPM

After both component forecasts have been built for a target season, run:

```bash
uv run nba-train-combined-box-score-prior-rapm --season 2025-26
```

The combined prior is a hard switch: the returning-player box-score forecast
is used when a prior NBA box profile exists, and the preseason cold-start
profile is used otherwise. The resulting prior table is frozen before the
existing chronological prior-centered RAPM fit, regular-season holdout, and
playoff evaluation.

This is an ablation, not an automatic Leaderboard promotion. It must improve
the locked lineup-level metrics over the forward-lagged RAPM prior.
