---
last_updated: "2026-08-25"
---

# Evaluate L1-MSP

Run the parameter-free next-game minute-share baseline over the 2024-25 source
season and the 2025-26 holdout:

```bash
uv run nba-evaluate-l1-minute-share-persistence \
  --seasons 2024-25 2025-26
```

The command reads curated regular-season player box scores. It uses actual
team minutes as the denominator, so overtime requires no special case. The
first team-game in each season has no preceding game and is excluded. L1-MSP
does not train: these are direct source-season and holdout evaluations of a
parameter-free forecast rule.

Outputs are immutable under `artifacts/rotation/l1_minute_share_persistence/`:

- `<season>_game_predictions.parquet`: one row per player in the union of
  consecutive team-game supports;
- `<season>_team_game_metrics.parquet`: allocation and player-share errors;
- `metrics.parquet`: one aggregate row per season;
- `metadata.json`: the no-parameter L1-MSP contract.
