---
title: HPM x2
---

# HPM x2: Raw OREB/100 Context

Last updated: 2026-08-14

HPM x2 is a narrow rebounding context experiment. It retains the same centered,
value-conditioned aging prior and exposure-gated cold-start procedure as HPM x1,
but substitutes the original raw offensive-rebounding-rate ingredient:

\[
\operatorname{OREB100}(U) = \sum_{i \in U}\operatorname{OREB100}_i.
\]

This is a sum of forward-safe individual player offensive rebounds per 100
possessions. It is deliberately not transformed, capped, or interacted with
usage. Therefore x2 tests whether raw summed OREB/100 itself is a useful lone
contextual feature, relative to x1's summed ORB% player claims.

## Run

```bash
uv run nba-train-hpm-x2 --through-season 2025-26
```

The recursive artifact is written under
`artifacts/models/forward_hpm_x2_orb_per_100_total/`, then replayed on the
frozen 2023-24 through 2025-26 regular-season and playoff targets.

## Frozen Result

The completed run is
`artifacts/models/forward_hpm_x2_orb_per_100_total/2025-26/forward-hpm-x2-orb-per-100-total-2025-26-20260815T041121Z-55b9ac34`.
Across the three frozen regular seasons, x2 recorded a `14.4641` full-game
RMSE, `1.198048` possession RMSE, `67.79%` winner accuracy, and `3.6558` team
NetRtg RMSE. Its pooled playoff checks were `1.192760` possession RMSE and
`16.6382` eligible-game RMSE.

The appropriate control is the recovered-coverage [Complete Player-Prior RAPM
Baseline](complete-player-prior-baseline.md), which preserves the same
value-conditioned aging and exposure-gated cold-start prior while disabling
all lineup context and box-score terms.

| Metric | No-context control | HPM x2 | x2 minus control | Paired 95% interval | P(x2 better) |
| --- | ---: | ---: | ---: | --- | ---: |
| Full-game RMSE | 14.475623 | 14.464065 | -0.011558 | [-0.029468, +0.006483] | 89.68% |
| Winner accuracy | 67.64% | 67.79% | +0.14 pp | [-0.37 pp, +0.66 pp] | 68.46% |
| Possession RMSE | 1.198061 | 1.198048 | -0.000013 | [-0.000023, -0.000003] | 99.35% |
| Possession MAE | 1.141675 | 1.141660 | -0.000015 | [-0.000026, -0.000005] | 99.80% |

Thus raw summed OREB/100 improves the possession-level forecast relative to
the proper no-context control, but its full-game improvement is not yet
precise enough to satisfy the promotion rule. Against HPM x1, the two
single-feature variants remain effectively tied on full-game RMSE: x2's
`-0.0024` difference has a paired 95% interval of `[-0.0084, +0.0036]`.
