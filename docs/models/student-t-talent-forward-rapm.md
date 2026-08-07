# Student-t Talent-Prior RAPM

This experiment restores the canonical Gaussian error model for lineup stints
and changes only the player-talent prior. It uses the same recursive,
regular-season-only exposure-gated state and frozen 2025-26 evaluation as the
Gaussian Forward Exposure-Gated RAPM model.

## Model

For each stint, the observed home net rating remains Gaussian:

\[
y_i \sim \mathcal{N}(\alpha + x_i^\top\beta, \sigma_i^2).
\]

Each player coefficient is centered on their strictly forward prior \(m_j\),
but its adjustment follows a Student-t distribution:

\[
\beta_j - m_j \sim \operatorname{StudentT}(\nu=3, 0, s=3).
\]

The three-point scale is the tail threshold in RAPM points per 100
possessions: near zero adjustment, the local penalty is exactly the completed
Gaussian model's ridge penalty; farther from the prior, shrinkage relaxes as

\[
q_j = \frac{1}{1 + ((\beta_j-m_j)/s)^2 / \nu}.
\]

The fit alternates between updating \(q_j\) and solving a sparse,
prior-centered ridge problem with a player-specific penalty multiplier. This
is the normal-scale-mixture/IRLS representation of a Student-t coefficient
prior. It allows a strongly identified outlier to depart from its prior while
keeping weakly identified player effects near their forward value.

## Comparison Contract

The first run fixes \(\nu=3\), \(s=3\), and the per-season lambda schedule
from the completed Gaussian recursive run. It neither tunes those parameters
on 2025-26 nor combines this change with the earlier Student-t observation
experiment. The 2025-26 prior is frozen after 2024-25, then scored against
realized regular-season and playoff lineup exposures without a player refit.

Run it with [Train Student-t Talent-Prior RAPM](../guides/train-student-t-talent-forward-rapm.md).

## Current Result

The completed \(\nu=3\), \(s=3\) run is
`artifacts/models/student_t_talent_forward_rapm/2025-26/student-t-talent-forward-rapm-2025-26-20260806T203508Z-8e8a2698/`.
All 30 seasonal fits converged, requiring 9 to 81 IRLS updates (mean 24.4).

| Cohort | Possession RMSE | Possession MAE | Game-margin RMSE | Team NetRtg RMSE | Pythagorean win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regular season | 1.198989 | 1.141996 | 14.7993 | **4.6069** | **10.1176** |
| Playoffs | 1.192895 | 1.136105 | 17.4713 | - | - |

Relative to the otherwise identical Gaussian recursive state, it improves all
regular-season point estimates: possession RMSE (1.198993 to 1.198989),
game-margin RMSE (14.8225 to 14.7993), team NetRtg RMSE (4.6680 to 4.6069),
and Pythagorean win RMSE (10.2778 to 10.1176). The playoff metrics worsen, so
this is a promising regular-season result rather than a universal replacement.
The published 2026-27 rankings use the selected 2025-26 final-season
\(\lambda=0.03\).[^lambda-sensitivity]

The completed-state [2026-27 Student-t Talent-Prior Rankings](2026-27-student-t-talent-rankings.md)
page publishes the sortable top 100 returning players.
The [lambda sensitivity report](student-t-talent-lambda-sensitivity.md) holds
the 2025-26 entering state fixed and measures the effect of a 0.10 final-season
refit against the selected 0.03 value.

[^lambda-sensitivity]: **Lambda sensitivity.** Holding all 2025-26 entering
    priors and Student-t settings fixed, the \(\lambda=0.10\) refit has a
    0.926 Pearson rating correlation with the \(\lambda=0.03\) fit, but a
    0.740 mean absolute RAPM difference and 44.5 mean absolute rank movement
    across 582 players. Victor Wembanyama moves from +13.84 (rank 1) to +7.33
    (rank 5). See the full [lambda sensitivity report](student-t-talent-lambda-sensitivity.md).

## Published Outputs

| File | Contents |
| --- | --- |
| `historical_player_coefficients.parquet` | Per-season Student-t-MAP player coefficients and prior adjustments |
| `season_player_priors.parquet` | Strictly forward player prior entering each season |
| `season_cold_start_metadata.parquet` | Cold-start settings plus coefficient-prior IRLS diagnostics |
| `frozen_2025_26_player_priors.parquet` | Player vector fixed before the target season |
| `next_season_top_100_returning_rankings.parquet` | Completed-2025-26 returning-player state for 2026-27 |
| Frozen evaluation files | Regular/playoff possession and game predictions plus regular team outputs |
