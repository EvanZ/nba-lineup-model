---
last_updated: "2026-09-09"
---

# Forward Availability v0.1

> Superseded by [Forward Availability v0.2](forward-availability-v02.md).

This pilot estimates a player's medical availability share for a future
season. It is deliberately separate from the minutes-allocation model: a
player may be available and still receive no minutes because the coach does
not select them.

Its training panel and binary label are defined in the
[player-game availability contract](../data/player-availability.md). The
target is a player's available games divided by their known rostered
player-games, not games played.

## Forward Contract

For target season \(t\), the model uses only information from completed
seasons before \(t\):

\[
\operatorname{logit}(p_{i,t}) = a(\operatorname{age}_{i,t})
+ \rho^{g_{i,t}}\left(z_{i,t-1} + w\,\widetilde{m}_{i,t-1}\right).
\]

\(a(\cdot)\) is an exposure-weighted quadratic age baseline. \(z\) is the
player's prior filtered deviation from that baseline, \(g\) is the number of
seasons since that observation, and \(\widetilde{m}\) is prior-season NBA
minutes per available game after standardization. The observed availability
share updates the player state with a beta-binomial-style pseudo-game strength
\(q\):

\[
\widehat{p}^{\mathrm{post}}_{i,t} =
\frac{q p_{i,t} + A_{i,t}}{q + N_{i,t}},
\]

where \(A\) is available games and \(N\) is known rostered player-games.

This is a state model for medical availability, not a claim that a player's
availability is determined by prior minutes. The workload term is a small,
lagged candidate transition signal and must earn its place in forward tests.

## First Frozen Replay

The panel contains every completed season from 2015-16 through 2025-26. For
each target season, the age curve and each player's filtered state use every
earlier available season; no target-season availability leaks into its
prediction. The model selected its hyperparameters on the three completed
tuning targets, 2020-21 through 2022-23, then evaluated once on frozen
2023-24 through 2025-26. Thus the first tuning target already has five source
seasons, 2015-16 through 2019-20, behind it. The selected configuration was:

| Parameter | Value |
| --- | ---: |
| State persistence \(\rho\) | 0.50 |
| Pseudo-game strength \(q\) | 60 |
| Lagged workload weight \(w\) | -0.25 |

The artifact is
`artifacts/rotation/forward_availability/forward-availability-20260909T225142Z-94dee6f`.

| Frozen metric, mean of three seasons | Age-only control | Forward availability v0.1 |
| --- | ---: | ---: |
| Exposure-weighted Brier score | **0.06044** | 0.06093 |
| Binomial log loss | **0.55965** | 0.58250 |
| Player-share MAE | 0.19704 | **0.19201** |
| Player-share RMSE | **0.25452** | 0.25604 |

The player-state model modestly improves average absolute error, but it loses
on both probability-sensitive measures. It also overpredicts availability on
the first two frozen seasons. Therefore v0.1 is the documented **first
benchmark** for this new model family; it is not yet integrated into the
minutes or win-projection pipeline.

## Next Decision

The next slice was to improve first-state calibration while preserving the
same binary contract and frozen splits. That work produced
[v0.2](forward-availability-v02.md). Candidate changes must be tested against
the age-only control and v0.2 before they can affect expected minutes:

\[
E[\mathrm{minutes}_{i,t}] =
P(\mathrm{available}_{i,t})
\times E[\mathrm{minutes}_{i,t}\mid\mathrm{available}_{i,t},\ \mathrm{team}].
\]

The availability probability is player-specific. The conditional-minutes
allocation remains a joint team problem, so changing one player's availability
will ultimately require a team-level reallocation only after this layer has
earned promotion.
