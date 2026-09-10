---
last_updated: "2026-09-09"
---

# AC-MSP v0.1

**Availability-Conditioned Minute-Share Projection** is the first game-level
conditional rotation model. It answers a narrower question than the preseason
rotation models: given the players who were medically available for a game,
how should a team allocate its 240 regulation minutes?

It does not yet forecast the availability mask. Forward Availability v0.2
provides a separate season-level probability; integrating that forecast into
game scenarios is a later step.

## Target and Support

For a regulation team-game \(g\), let \(A_g\) be the players marked medically
available in the historical availability mart. Coach DNPs and G-League
assignments remain in \(A_g\) with zero observed minutes. Players marked
unavailable are not candidates.

\[
y_{i,g} = \frac{m_{i,g}}{240}, \qquad \sum_{i \in A_g} y_{i,g}=1.
\]

Only exact-240 team-games with a complete known availability roster are
eligible. Overtime, malformed minute totals, and the small number of
historical team-games with an ambiguous player status are excluded rather than
rescaled or imputed. The initial replay uses all eligible regular-season games
in each season, not a first-20-game window; playoffs are excluded.

## Pregame Features

Every feature is computed before the current team-game, using only completed
earlier games for that same team and season:

\[
x_{i,g} = \left[
\log(1+m^{(1)}_{i,g}),
\log(1+\bar m^{(5)}_{i,g}),
\log(1+\bar m^{(\mathrm{season})}_{i,g})
\right].
\]

The first three terms are prior-game minutes, mean minutes over the prior five
team-games, and season-to-date mean minutes. A player newly acquired by the
team starts with zero team-history minutes; the softmax still assigns them
positive support when they are available.

## Allocation

After standardizing the features on source seasons, a ridge-regularized score
is fitted with fractional multinomial loss:

\[
s_{i,g}=x_{i,g}^{\top}\beta,
\qquad
\widehat y_{i,g}=\frac{\exp(s_{i,g})}
{\sum_{j\in A_g}\exp(s_{j,g})}.
\]

The primary metric is team-game allocation total variation:

\[
\operatorname{TV}_g = \frac12\sum_{i\in A_g}
\left|y_{i,g}-\widehat y_{i,g}\right|.
\]

The frozen replay also reports Brier score, cross-entropy, player-share MAE,
and player-share RMSE. Its like-for-like control is a lag-five minutes rule
with a one-minute floor, normalized over the same observed available roster.

## Frozen Replay

The ridge penalty is selected only from 2020-21 through 2022-23, each fit on
earlier seasons. The final evaluation is frozen over 2023-24 through 2025-26,
with each target season fit only on prior seasons. This establishes the
conditional-minutes layer before availability forecasts are coupled to it.

The selected penalty was \(\alpha=0\). The tuning curve was essentially flat
through \(\alpha=10\), then worsened at larger penalties; the unpenalized
three-feature score is therefore the exact v0.1 contract.

| Frozen metric, mean of three seasons | Conditional L5 control | AC-MSP v0.1 |
| --- | ---: | ---: |
| Allocation total variation | 0.16294 | **0.16222** |
| Brier score | 0.01509 | **0.01336** |
| Cross-entropy | 2.41318 | **2.38992** |
| Player-share MAE | 0.02468 | **0.02455** |
| Player-share RMSE | 0.03380 | **0.03180** |

AC-MSP wins the aggregate comparison, but the TV gain is small and reverses in
2025-26. This is not yet a preseason or win-projection model: it receives the
observed historical availability roster. Its result only validates the
conditional-minute layer and its next-man-up reallocation behavior.

Artifact:
`artifacts/rotation/availability_conditioned_minutes/ac-msp-20260910T002438Z-7892db2`.

Run:

```bash
uv run nba-evaluate-ac-minute-share-projection
```

The command writes predictions, matched-control predictions, frozen metrics,
and standardized coefficients beneath `artifacts/rotation/availability_conditioned_minutes/`.
