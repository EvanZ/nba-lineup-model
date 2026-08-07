---
last_updated: "2026-08-05"
---

# Train Frozen Offense/Defense RAPM

Run the full forward regular-season chain and frozen 2025-26 evaluation with:

```bash
uv run nba-train-frozen-offense-defense-rapm --season 2025-26
```

The command fits 1996-97 through 2024-25 sequentially, selecting lambda within
each season on chronological folds. The completed 2024-25 offense and defense
coefficients are frozen before the 2025-26 regular season and playoffs are
scored.

The run is published under:

```text
artifacts/models/frozen_offense_defense_rapm/<season>/<run-id>/
```

Key outputs are:

- `historical_player_coefficients.parquet`: season-by-season offense, defense,
  net, and prior coefficients;
- `historical_cv_results.parquet`: per-season lambda-selection evidence;
- `frozen_player_priors.parquet`: the completed source-season O/D state;
- `cohort_metrics.parquet`: separate regular-season and playoff metrics;
- team NetRtg and Pythagorean-win prediction tables;
- `source_state.json`: the explicit no-target-refit declaration.

## Publish Completed-Season Rankings

After a regular season is complete, build retrospective offense and defense
rankings with:

```bash
uv run nba-rank-offense-defense-rapm --season 2025-26
```

This command starts from the immutable completed 2024-25 frozen O/D state,
selects lambda on chronological folds within the completed 2025-26 regular
season, and refits all 2025-26 regular-season stints. It writes a distinct
artifact under:

```text
artifacts/models/offense_defense_rapm/<season>/<run-id>/
```

It does not update the frozen preseason forecast or its leaderboard metrics.
The artifact includes `player_coefficients.parquet`, `cv_results.parquet`, a
three-sided `player_rankings.parquet`, separate `top_25_overall.parquet`,
`top_25_offense.parquet`, and `top_25_defense.parquet` tables, the immutable
frozen-prior provenance, and file hashes. Every published list requires at
least 500 relevant possessions; overall RAPM is offense plus defense, and
higher defense is better because it is measured as points prevented per 100
possessions.
