---
last_updated: "2026-08-25"
---

# L1-MSP v0.0

**Lag-1 Minute-Share Persistence** is the parameter-free baseline for the
Rotation Models family. It forecasts only the next team-game, not a full season.

## Target

For player \(i\) in team-game \(g\), define:

\[
y_{i,g} = \frac{m_{i,g}}{\sum_{j \in T_g} m_{j,g}}.
\]

The denominator is actual total team minutes. It therefore includes overtime,
and each team's player shares sum to one. Players who do not play have a zero
share.

## Forecast

\[
\widehat y_{i,g+1} = y_{i,g}.
\]

There are no player attributes, fitted coefficients, hyperparameters, rolling
averages, or NAIL inputs. A player receives a distinct forecast only because
the player had a distinct share in the previous team-game.

For evaluation, the support is the union of players in consecutive team-games.
A player with a zero prior-game share, whether newly available or previously
inactive, has a predicted share of zero. A player absent from the current game
retains the prior predicted share and has an actual share of zero. This
deliberately exposes the cold-start and rotation-change problem.

## Evaluation

The primary metric is team-game allocation error, measured as total variation:

\[
\operatorname{TV}_g = \frac{1}{2}\sum_i |y_{i,g} - \widehat y_{i,g}|.
\]

It ranges from zero (identical allocation) to one (disjoint allocations).
We also report player-share MAE and RMSE.

The distributional Brier score is:

\[
\operatorname{BS}_g = \sum_i (y_{i,g} - \widehat y_{i,g})^2.
\]

It is finite even when L1-MSP misses a newly active player, and unlike the
per-player MSE it does not depend on how many players are in the evaluated
union support.

### Strict cross-entropy

We additionally report strict cross-entropy:

\[
\operatorname{CE}_g = -\sum_{i:y_{i,g}>0} y_{i,g}\log \widehat y_{i,g}.
\]

For L1-MSP, this is (+\infty) whenever a player logs positive minutes after
receiving a zero prior-game share. That is not a numerical failure; it is the
baseline's exact cold-start failure. The aggregate therefore reports the strict
cross-entropy, the fraction of team-games with infinite cross-entropy, and the
mean cross-entropy among finite team-games. A future model may introduce a
pregame candidate roster and a cold-start probability mass; L1-MSP does not.

## Initial benchmark

The first run uses the regular-season curated player-game data currently
available in the repository. L1-MSP has no fitted parameters, so 2024-25 is a
source-season evaluation and 2025-26 is the held-out evaluation season for
later rotation models.

| Season | Coverage | Evaluated team-games | Mean TV | Brier | Player-share MAE | Strict CE | Infinite CE rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-25 | 1,139 available games | 2,248 | 0.2038 | 0.0278 | 0.0287 | \(\infty\) | 72.7% |
| 2025-26 | 1,230 games | 2,430 | 0.2025 | 0.0265 | 0.0229 | \(\infty\) | 74.7% |

The 2024-25 source data currently covers 1,139 games rather than a full 1,230
game regular season. The 2025-26 row is complete: one first team-game is
excluded for each of the 30 teams because it has no preceding allocation.

A mean TV of 0.2025 means copying the prior game's allocation leaves roughly
20.25% of a team's next-game minute-share mass allocated to the wrong players
or in the wrong amounts. That is deliberately a hard, cold-start-sensitive
baseline for subsequent rotation models to beat.

## Run

```bash
uv run nba-evaluate-l1-minute-share-persistence \
  --seasons 2024-25 2025-26
```

The command writes per-player predictions, per-team-game metrics, aggregate
season metrics, and metadata beneath `artifacts/rotation/`.
