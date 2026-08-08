---
last_updated: "2026-08-07"
---

# Frozen No-Prior Window RAPM

These are deliberately simple preseason controls. They fit ordinary
zero-centered ridge RAPM on completed regular-season stints, with no lagged
player prior, age, draft, box-score, exposure gate, or other player-profile
input. They test whether the forward model’s gains come merely from retaining
more historical possessions.

## Specification

For each training stint, the model estimates one coefficient per player:

\[
\widehat{\operatorname{NetRtg}} =
h + \sum_{i \in H}\beta_i - \sum_{j \in A}\beta_j,
\]

by minimizing possession-weighted squared error plus
\(\lambda\sum_i\beta_i^2\). The player-prior mean is zero for every player.
Each candidate selects \(\lambda\) through chronological folds that end before
the 2025-26 target season, then refits on all of its completed training window.

| Candidate | Regular-season training window | Selected lambda | Cold starts |
| --- | --- | ---: | --- |
| One-year no-prior RAPM | 2024-25 | 0.03 | 0.0 |
| Pooled three-year no-prior RAPM | 2022-23 through 2024-25 | 0.01 | 0.0 |

The three-year model does not have a formal player prior. Its stability comes
from constraining a player to one coefficient across the entire three-season
window. That reduces variance but can lag genuine improvement or decline.

## Frozen 2025-26 Results

The realized 2025-26 lineup allocation is supplied as an oracle, while every
player coefficient, lambda, home-court intercept, and scoring mean is fixed
before opening night. The full comparison is maintained in the
[Frozen Preseason Leaderboard](preseason-leaderboard.md).

| Candidate | Regular possession RMSE | Regular game-margin RMSE | Team NetRtg RMSE | Pythagorean-wins RMSE |
| --- | ---: | ---: | ---: | ---: |
| One-year no-prior RAPM | 1.199061 | 15.0694 | 4.9792 | 11.5828 |
| Pooled three-year no-prior RAPM | 1.198919 | 14.7820 | 4.6618 | 10.7999 |
| Forward lagged RAPM | 1.199000 | 14.8894 | 4.8538 | 10.7006 |
| Forward contextual RAPM | 1.199008 | **14.6525** | **4.1572** | **9.3153** |

The pooled model is clearly more stable than the one-year model. It improves on
the lagged state in possession RMSE, game-margin RMSE, and team NetRtg, while
the lagged state retains a small Pythagorean-wins advantage. Forward contextual
RAPM remains the strongest game, team-NetRtg, and win forecast in this cohort.

## Artifacts

The one-year coefficient/CV artifact is
`frozen-one_year-rapm-2025-26-20260807T224659Z-420d7857` under
`artifacts/models/frozen_one_year_rapm/2025-26/`; its immutable evaluation is
`frozen-one-year-rapm-2025-26-20260807T224708Z-b39c8bd3` under
`artifacts/models/frozen_prior_evaluation/2025-26/`.

The three-year coefficient/CV artifact is
`frozen-three_year-rapm-2025-26-20260807T224747Z-bb4eb660` under
`artifacts/models/frozen_three_year_rapm/2025-26/`; its immutable evaluation is
`frozen-three-year-rapm-2025-26-20260807T224756Z-e8e90b17` under
`artifacts/models/frozen_prior_evaluation/2025-26/`.

Each coefficient artifact includes the frozen player vector, chronological CV
results, selected lambda, fitted home-court intercept, and a season-level
training summary. Each evaluation artifact includes regular-season and playoff
possession predictions, eligible-possession game margins, team NetRtg, and
Pythagorean-wins tables.
