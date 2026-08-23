---
last_updated: "2026-08-23"
---

# Train NAIL-RAPM v1.2.1.1 Standard USG%

This release replaces the internal usage-events-per-on-court-possession rate
with conventional, box-score USG% in the additive player profile and
`usage_concentration` context term.

## Full Recursive Fit

```bash
uv run nba-train-nail-v1211-standard-usage \
  --through-season 2025-26 \
  --log-path artifacts/logs/nail-v1211-standard-usage.log
```

Monitor it with:

```bash
tail -f artifacts/logs/nail-v1211-standard-usage.log
```

## Frozen Replay And Bootstrap

```bash
uv run nba-backtest-nail-v1211-standard-usage \
  --log-path artifacts/logs/nail-v1211-standard-usage-frozen.log

uv run nba-bootstrap-nail-v1211-standard-usage
```

The recursive fit is not itself evidence of predictive improvement. Add the
candidate to the leaderboard only after the frozen replay and paired bootstrap
have completed.
