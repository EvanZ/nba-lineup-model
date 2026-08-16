---
last_updated: "2026-08-15"
---

# Frozen Model Tournament

The tournament is the promotion gate for successive HIPSTER PM feature
releases. It tests each challenger against the incumbent that survived every
earlier round, rather than interpreting small differences in a global ranking
table as automatic wins.

## Contract

All models use the same frozen forecasts for 2023-24 through 2025-26. Games
are resampled with replacement *within each season*, preserving season balance
and all within-game possession dependence. Each draw evaluates both models on
the identical sampled games.

The promotion metric is full-game margin RMSE. For each match we compute

\[
\Delta = \operatorname{RMSE}_{\mathrm{challenger}}
       - \operatorname{RMSE}_{\mathrm{incumbent}}.
\]

The challenger normally advances only when the upper endpoint of its paired 95%
bootstrap interval is below zero. Possession RMSE, possession MAE, and winner
accuracy are retained as supporting diagnostics.

For a coherent **strict simplification** of the active feature contract, the
published reference may instead be changed under a parsimony non-inferiority
decision: the full-game interval must not indicate a practically material
degradation (upper endpoint no greater than `+0.015` RMSE), and the challenger
must remove an interpretable feature family rather than add capacity. This is a
model-selection preference, not evidence that the simpler model is
statistically superior. Future simplification decisions use this same margin.

## Current Tournament

10,000 paired game-block bootstrap draws, seed `20260814`.

| Round | Incumbent | Challenger | Incumbent RMSE | Challenger RMSE | Delta | Paired 95% CI | P(challenger better) | Result |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| 1 | Complete player-prior RAPM | Original HPM v1 | 14.4756 | 14.3802 | -0.0954 | [-0.1763, -0.0153] | 99.03% | Promoted |
| 2 | Original HPM v1 | HPM v2 shooting composition | 14.3802 | 14.3908 | +0.0105 | [-0.0072, +0.0284] | 12.45% | Retained incumbent |
| 3 | Original HPM v1 | HPM v2.1 rebound capacity | 14.3802 | 14.3869 | +0.0067 | [-0.0155, +0.0288] | 27.73% | Retained incumbent |
| 4 | Original HPM v1 | HPM v2.2 usage allocation | 14.3802 | 14.4050 | +0.0247 | [-0.0106, +0.0595] | 8.49% | Retained incumbent |
| 5 | Original HPM v1 | HPM v2.3 shot portfolio | 14.3802 | 14.4288 | +0.0486 | [+0.0132, +0.0831] | 0.40% | Retained incumbent |
| 6 | Original HPM v1 | HPM x1 ORB claim context | 14.3802 | 14.4665 | +0.0862 | [+0.0083, +0.1637] | 1.45% | Retained incumbent |
| 7 | Original HPM v1 | HPM x3 ORB claim rebound replacement | 14.3802 | 14.3774 | -0.0029 | [-0.0168, +0.0113] | 65.17% | Promoted on parsimony non-inferiority |
| 8 | HPM x3 ORB claim rebound replacement | HPM x4 ORB claims + blocks | 14.3774 | 14.4036 | +0.0262 | [-0.0102, +0.0632] | 8.05% | Retained incumbent |
| 9 | HPM x3 ORB claim rebound replacement | HPM x5 interaction-only creation | 14.3774 | 14.3800 | +0.0027 | [-0.0214, +0.0267] | 42.29% | Retained incumbent |

The published regular-season reference is therefore **HPM x3 ORB claim rebound
replacement**. Its full-game interval does not establish superiority, but it
removes four rebound response functions and the rebound-by-usage interaction
without evidence of a practically material full-game degradation. Original HPM
v1 remains the strict-superiority benchmark.

## Run

```bash
uv run nba-run-frozen-model-tournament --draws 10000
```

Each immutable run writes paired metric intervals and one primary-metric row
per tournament round under `artifacts/models/frozen_model_tournament/`.
