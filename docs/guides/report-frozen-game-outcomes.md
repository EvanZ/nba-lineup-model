---
last_updated: "2026-08-07"
---

# Report Frozen Full-Game Outcomes

Aggregate the promoted frozen 2025-26 candidates into one full-game regular-season
report. The command reads existing immutable `regular_game_predictions.parquet`
artifacts; it does not refit a player model or use a target-season result to change
any forecast.

```bash
uv run nba-report-frozen-game-outcomes --season 2025-26
```

The report writes to `artifacts/models/frozen_game_outcomes/2025-26/` and updates
the Full-Game Outcomes section of the [Frozen Preseason
Leaderboard](../models/preseason-leaderboard.md). It retains three tables:

| Artifact | Contents |
| --- | --- |
| `game_outcome_metrics.parquet` | One row per frozen model with margin RMSE, margin MAE, and winner accuracy. |
| `game_outcome_predictions.parquet` | One full final-margin forecast per model and regular-season game. |
| `sources.parquet` | Source run IDs, paths, timestamps, and manifest hashes. |

Full-game margin metrics use every allocated regular-season stint, including rows
outside the eligible-possession reconstruction boundary. Winner accuracy is the
sign of the model's final home-margin forecast. An exactly zero forecast receives
half credit, so it is not arbitrarily treated as an away winner.

Playoff full-game outcomes are not yet included: the retained frozen playoff
predictions are eligible-possession aggregates, not all allocated game stints.
The existing playoff table therefore remains explicitly labeled as an
eligible-possession game metric.
