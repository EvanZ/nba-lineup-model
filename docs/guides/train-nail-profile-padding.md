---
title: Study and Train NAIL Profile Padding
last_updated: "2026-08-18"
---

# Study and Train NAIL Profile Padding

NAIL-RAPM v1.0 shrinks every possession-rate profile with the same 300
pseudo-possessions. This controlled study changes only that padding contract;
the player prior, RAPM lambda schedule, context features, context penalty, and
three frozen evaluation seasons remain fixed.

## 1. Estimate Cross-Season Constants

```bash
uv run nba-build-profile-padding-study --through-target-season 2022-23
```

For each primitive statistic, the command minimizes next-season weighted
squared error over adjacent player-season transitions ending by 2022-23. The
2023-24, 2024-25, and 2025-26 leaderboard seasons are excluded from selection.
It writes an immutable artifact containing the selected constants, transition
predictions, metadata, and checksums.

## 2. Train the Controlled Candidates

```bash
uv run nba-train-nail-profile-padding --contract uniform-season
uv run nba-train-nail-profile-padding --contract published
uv run nba-train-nail-profile-padding --contract cross-season
```

The `uniform-season` candidate isolates the effect of changing the league
anchor while keeping 300 pseudo-possessions. The `published` candidate uses the statistic-specific stabilization constants
reported by Krishna Medvedovsky. The `cross-season` candidate uses this
project's forward-safe estimates. Three-point makes are reconstructed from a
separately padded 3PA rate and 3P percentage. Usage is reconstructed from
separately padded FGA, FTA, and turnover rates.

## 3. Run the Frozen Comparison

```bash
uv run nba-evaluate-nail-profile-padding
uv run nba-bootstrap-nail-profile-padding
```

The first command replays canonical uniform-300 NAIL, published-rate NAIL, and cross-season
padding NAIL over the fixed 2023-24 through 2025-26 regular seasons and pooled
playoffs. No target-season state is refit during evaluation.

The second command runs a paired 10,000-draw game-block bootstrap, stratified
by season, for full-game margin RMSE, winner accuracy, possession RMSE, and
possession MAE.

The external starting values come from
[NBA Stabilization Rates and the Padding Approach](https://kmedved.com/2020/08/06/nba-stabilization-rates-and-the-padding-approach/).
