---
last_updated: "2026-09-08"
---

# L20-NAIL-MSP v0.1

**Rating-Aware First-20-Game Preseason Minute-Share Persistence** tests whether
completed NAIL-RAPM adds usable information to preseason role allocation.

## Contract

For a target season \(t\), the model uses only:

- transaction-derived opening-roster candidates;
- prior-season total regular-season minutes; and
- completed NAIL-RAPM from \(t-1\).

It never uses a completed rating, box score, or minutes from the target season.
Players missing from the completed NAIL cache receive a neutral rating of zero.

## Forecast

L20-MSP first creates its smoothed prior-minute share \(p_i\). Let \(z_i\)
be player \(i\)'s completed NAIL-RAPM, divided by the cross-league standard
deviation among target-season opening candidates. L20-NAIL-MSP applies a
within-team rating tilt:

\[
\hat y_i =
\frac{p_i\exp(\beta z_i)}
{\sum_{j \in C_T}p_j\exp(\beta z_j)}.
\]

The cold-start mass \(\epsilon\) and NAIL weight \(\beta\) are selected by
mean team allocation total variation. \(\beta=0\) exactly recovers the L20
prior-minute allocation for the same \(\epsilon\), so the experiment is a
nested comparison rather than a new heuristic.

## Evaluation

The runner performs expanding-window frozen tests across `2015-16` through
`2025-26`. To forecast a holdout season, it chooses \(\epsilon,\beta\) from
earlier first-20-game seasons only, then scores the holdout against realized
cumulative team-minute shares.

```bash
uv run nba-evaluate-l20-nail-minute-share
```

Artifacts include every parameter grid, selected parameters, team-level
metrics, player allocations, and an independently tuned `beta = 0`
prior-minute control for each holdout. The production parameter grid uses all
eleven completed source seasons only after frozen scoring is complete.

## Initial Frozen Result

The first rolling evaluation covered ten holdouts, `2016-17` through
`2025-26`. The control selected its own \(\epsilon\) from the same expanding
source window; it selected \(\epsilon=0.32\) in every holdout. The
rating-aware candidate won seven of ten seasons on the primary total-variation
metric:

| Metric | Independently tuned prior-minute control | Rating-aware L20 | Difference |
| --- | ---: | ---: | ---: |
| Mean team allocation total variation | 0.22205 | **0.22076** | **-0.00129** |
| Mean Brier score | 0.02110 | **0.02039** | **-0.00071** |
| Player-share MAE | 0.02704 | **0.02689** | **-0.00015** |

The frozen selections kept the NAIL tilt deliberately small: \(\beta\) was
`0.10` in five holdouts and `0.05` in five. Fitting all eleven completed source
seasons for the current preseason selects \(\epsilon=0.32\) and
\(\beta=0.05\). At that value, a one-standard-deviation NAIL advantage
multiplies a player's relative minute-allocation score by only
\(\exp(0.05) \approx 1.05\) before the team total is renormalized.

This is evidence that player value helps break role ties, not evidence that it
should override established minutes. It is not yet wired into the Win
Projections page: this initial test uses the prior season's completed NAIL
rating. A promotion candidate must repeat the same frozen contract using the
exact preseason NAIL forecast that would be shown for the target season.
