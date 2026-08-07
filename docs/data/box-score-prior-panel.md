---
last_updated: "2026-08-04"
---

# Box-Score Prior Panel

The box-score prior panel is the possession-native feature boundary for the
next RAPM prior experiment. Each row is one player entering target season
\(t\). Dynamic inputs are drawn only from season \(t-1\); target-season RAPM
is a supervised label, never a model feature.

## Possession-Native Rates

For a prior-season counting statistic \(c_{i,t-1}\) and reconstructed
on-court possessions \(n_{i,t-1}\), the panel uses

\[
100 \times \frac{c_{i,t-1}}{n_{i,t-1}}.
\]

The first profile includes FGA, 3PA, FTA, assists, turnovers, offensive and
defensive rebounds, steals, blocks, and personal fouls per 100 on-court
possessions. Minutes and per-36 rates are not model features.

## Shooting Stabilization

Shooting features use source-season league rates and fixed pseudo-attempt
amounts:

| Statistic | Pseudo-attempts |
| --- | ---: |
| Effective FG% | 300 FGA |
| 3P% | 150 3PA |
| FT% | 100 FTA |

For example:

\[
\widetilde{3P\%}_{i,t-1} =
\frac{3PM_{i,t-1} + 150 \cdot \mathrm{League3P\%}_{t-1}}
     {3PA_{i,t-1}+150}.
\]

The raw attempts remain in the panel for audit and the corresponding per-100
rates represent player role and volume.

## Cold Starts

Every target RAPM player receives a row. If the player was absent in \(t-1\),
lagged RAPM and box-score values are null and the row is tagged `no_prior`.
The panel also assigns `low_exposure` below 500 prior on-court possessions,
`developing` from 500 through 1,499, and `established` at 1,500 or more.

Static preseason features remain available for every cohort: age, experience,
rookie status, draft fields, height, weight, and listed position.

## Storage

```text
data/analytical/box_score_prior_panel/
  _manifest.json
  player_prior_features.parquet
  season_cohort_summary.parquet
  league_shooting_references.parquet
```

The manifest records the exact validated player-season panel, target labels,
label-free feature list, thresholds, pseudo-attempt settings, output hashes,
and builder fingerprint.

## Current Build

The current historical build covers 14,109 player-target-season rows from
1997-98 through 2025-26. Its 2025-26 target cohort contains 582 players:

| Prior cohort | Players | Prior RAPM available | Prior box profile available |
| --- | ---: | ---: | ---: |
| Established | 276 | 276 | 276 |
| Developing | 109 | 109 | 109 |
| Low exposure | 77 | 77 | 77 |
| No prior | 120 | 0 | 0 |

This confirms that 2024-25 Stats V3 player boxes are available for every
2025-26 player with a prior season. The no-prior group is intentional and is
the cold-start population for the later profile model.
