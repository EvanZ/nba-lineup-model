---
last_updated: "2026-08-18"
---

# NAIL-RAPM v1.1: Stat-Specific Profile Padding

NAIL-RAPM v1.1 is a controlled profile-shrinkage update to
[NAIL-RAPM v1.0](nail-rapm-v1.md). It leaves the player-prior model, RAPM
lambda schedule, 14 lineup coordinates, linear context estimator, and context
penalty unchanged. Only the construction of lagged player box-score profiles
changes.

The release replaces one universal 300-possession padding constant with
statistic-specific stabilization constants and uses the source season's league
rate as the shrinkage anchor. It is the new regular-season full-game leader in
the [Three-Season Frozen Leaderboard](three-season-frozen-backtest.md), while
v1.0 remains microscopically better on possession RMSE.

## Why Change 300?

For a possession-rate statistic with observed rate \(r_{i,f,t}\), denominator
\(n_{i,f,t}\), source-season league rate \(\bar r_{f,t}\), and padding constant
\(K_f\), the padded profile is

\[
\widehat r_{i,f,t}
=
\frac{n_{i,f,t}r_{i,f,t}+K_f\bar r_{f,t}}
{n_{i,f,t}+K_f}.
\]

The same form applies to a percentage, but its natural opportunity count is
used as the denominator. For example, 3P% is padded by 3PA rather than player
possessions.

NAIL v1.0 used \(K_f=300\) for every possession-rate count and did not pad
ORB%. Its anchor was an expanding historical league average through the source
season. Neither choice had been optimized for the current NAIL contract.

## Candidate Contracts

Four complete 30-season recursive models isolate the relevant decisions:

| Candidate | League anchor | Padding |
| --- | --- | --- |
| NAIL v1.0 control | Expanding history through source season | Uniform 300 possessions; raw ORB% |
| Anchor-only control | Source season | Same uniform-300 formulas as v1.0 |
| Published stat-specific | Source season | Statistic-specific constants from Medvedovsky |
| Cross-season fitted | Source season | Constants fit on player transitions ending by 2022-23 |

The published starting values come from
[NBA Stabilization Rates and the Padding Approach](https://kmedved.com/2020/08/06/nba-stabilization-rates-and-the-padding-approach/).
That study predicts later performance within a season. The project-specific
fit instead minimizes next-season weighted squared error, matching NAIL's
lagged-profile use case.

| Primitive statistic | Natural denominator | Published (K_f) | Cross-season (K_f) |
| --- | --- | ---: | ---: |
| 3PA rate | Possessions | 29.29 | 86.36 |
| 3P% | 3PA | 242.61 | 283.50 |
| Assist rate | Possessions | 55.57 | 130.31 |
| Turnover rate | Possessions | 425.69 | 721.51 |
| FGA rate | Possessions | 89.62 | 266.40 |
| FTA rate | Possessions | 197.58 | 398.50 |
| Steal rate | Possessions | 632.61 | 839.03 |
| Block rate | Possessions | 151.22 | 183.50 |
| ORB% | Possessions | 98.55 | 150.00 |

The larger cross-season constants are directionally reasonable: preserving a
skill from one season to the next generally requires more evidence than
stabilizing a rate within one season.

## Composite Profiles

The stat-specific candidates no longer pad made threes or usage as opaque
totals. They rebuild them from primitive components:

\[
\widehat{3PM/100}_{i,t}
=
\widehat{3PA/100}_{i,t}\;\widehat{3P\%}_{i,t},
\]

\[
\widehat{USG}_{i,t}
=
\widehat{FGA/100}_{i,t}
+0.44\widehat{FTA/100}_{i,t}
+\widehat{TOV/100}_{i,t}.
\]

This lets shooting volume, shooting accuracy, foul drawing, and ball security
shrink at their own empirical rates.

## Forward-Safe Selection

The project-specific constants use 9,909 adjacent player-season transitions
across 26 source seasons. Selection stops at target season 2022-23. The frozen
2023-24, 2024-25, and 2025-26 seasons do not enter the optimization.

Every selected cross-season constant improved its primitive next-season
weighted MSE over uniform 300. That did **not** make the cross-season contract
the best lineup predictor. This is an important distinction: minimizing
individual profile error is only a proxy for minimizing downstream game error.

## Frozen Results

All values pool the same three frozen seasons. Lower is better except winner
accuracy.

| Contract | Poss. RMSE | Eligible game RMSE | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE | Playoff game RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NAIL v1.0 | **1.197977** | 14.1119 | 14.3516 | 68.33% | 3.4307 | 7.3460 | 16.6636 |
| Uniform 300, source-season anchor | 1.197980 | 14.1012 | 14.3419 | 68.13% | 3.4146 | 7.3041 | 16.6537 |
| **Published stat-specific (v1.1)** | 1.197979 | **14.0864** | **14.3236** | **68.53%** | **3.3898** | **7.2757** | **16.6267** |
| Cross-season fitted | 1.197980 | 14.0957 | 14.3316 | 68.41% | 3.3939 | 7.3012 | 16.6463 |

The downstream improvements are broad but small. Possession RMSE is a tie for
practical purposes. The useful movement appears after possession predictions
are aggregated into games and teams.

## Paired Bootstrap

The uncertainty audit resamples games within each frozen season for 10,000
paired draws. Differences below zero favor the challenger for full-game RMSE.

| Comparison | RMSE difference | 95% interval | P(challenger better) |
| --- | ---: | ---: | ---: |
| Source-season anchor minus v1.0 | -0.0097 | [-0.0250, +0.0059] | 88.8% |
| Published padding minus anchor-only | -0.0184 | [-0.0368, +0.0000] | 97.5% |
| Published padding minus v1.0 | **-0.0281** | **[-0.0491, -0.0073]** | **99.5%** |
| Cross-season padding minus v1.0 | -0.0200 | [-0.0421, +0.0022] | 95.9% |
| Published minus cross-season | **-0.0080** | **[-0.0153, -0.0007]** | **98.3%** |

The anchor explains part of the gain, but not all of it. Published padding is
better than the anchor-only control in 97.5% of draws; its conservative
two-sided 95% upper endpoint is essentially zero. Against canonical v1.0, the
combined v1.1 contract has a clearly negative interval.

## Decision

Promote the published stat-specific contract as **NAIL-RAPM v1.1**. It wins the
primary full-game metric, improves the principal team-level metrics and pooled
playoff game RMSE, and has a defensible statistic-specific construction.

Do not promote the cross-season constants. They are best at the primitive
profile objective but not at the downstream lineup objective. A future
optimization should select padding jointly against forward game-level loss,
not fit each profile coordinate independently.

The context Ridge penalty has since received its own
[controlled study](nail-context-regularization.md). A normalized penalty was
selected without frozen-season leakage but lost materially on regular-season
full-game prediction, so the fixed raw `alpha=10000` contract remains part of
published v1.1.

## Artifacts

- Profile fit: `artifacts/models/profile_padding_study/2022-23/profile-padding-2022-23-20260818T132540Z-c9e2664d`
- v1.1 recursion: `artifacts/models/forward_nail_rapm_v1_medvedovsky_padding/2025-26/forward-nail-rapm-v1-medvedovsky-padding-2025-26-20260818T140419Z-2d5d58cb`
- Four-way frozen replay: `artifacts/models/nail_profile_padding_frozen_backtest/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260818T160628Z-497f1bbf`
- Paired bootstrap: `artifacts/models/nail_profile_padding_bootstrap/2023-24_to_2025-26/nail-profile-padding-bootstrap-20260818T160749Z-00dac677`

Reproduction commands are in
[Study and Train NAIL Profile Padding](../guides/train-nail-profile-padding.md).
