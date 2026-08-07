---
last_updated: "2026-08-06"
---

# Student-t Forward RAPM

This model is the first robust-likelihood version of the recursive,
one-number Forward Exposure-Gated RAPM state. It retains the same strictly
forward player priors, replacement-token cold starts, regular-season training
set, and frozen 2025-26 evaluation as the Gaussian model. Only the observation
model for stints changes.

## Model

For stint (i), with observed home net rating (y_i), signed player design
row (x_i), lineup coefficients \(\beta\), and intercept \(\alpha\), the
Gaussian RAPM fit assumes residuals are normally distributed. Here they follow
a Student-t distribution:

\[
y_i \sim \operatorname{StudentT}(\nu,\ \alpha + x_i^\top\beta,\ \sigma).
\]

The player prior remains the same prior-centered Gaussian ridge penalty,
\(\beta_j \sim \mathcal{N}(m_j, \tau^2)\), where \(m_j\) is the frozen
returning-player or exposure-gated cold-start prior. The first exemplar fixes
\(\nu=5\), which gives large-residual stints less influence than a Gaussian
likelihood while retaining finite variance.

It is fitted by iteratively reweighted ridge regression. At each IRLS update,
the usual possession weight \(p_i\) is multiplied by

\[
w_i = \frac{\nu + 1}{\nu + ((y_i - \hat y_i)/\sigma)^2}.
\]

Thus an ordinary residual has weight near one, while a highly unusual stint is
downweighted. The model does **not** yet use a Student-t distribution for
player talent or for the prior; this experiment isolates robust observation
errors first.

## Comparison Contract

Each season uses the per-season lambda selected by the completed Gaussian
forward exposure-gated run. Keeping that schedule fixed means the comparison
answers a narrow question: does a Student-t error model improve the frozen
preseason forecast, holding the recursive prior system and coefficient
regularization policy constant?

The prior entering 2025-26 is frozen after 2024-25. Evaluation then uses the
realized 2025-26 regular-season and playoff lineup exposures without refitting
players. The published artifact contains possession, game-margin, team NetRtg,
and Pythagorean-win outputs so it can enter the Frozen Preseason Leaderboard.

Run the model with [Train Student-t Forward RAPM](../guides/train-student-t-forward-rapm.md).

## Current Result

The completed five-degree-of-freedom run is
`artifacts/models/student_t_forward_rapm/2025-26/student-t-forward-rapm-2025-26-20260806T131834Z-b1cc6592/`.
All 30 seasonal IRLS fits converged, with 14 to 15 updates per season.

| Cohort | Possession RMSE | Possession MAE | Game-margin RMSE | Team NetRtg RMSE | Pythagorean win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regular season | 1.199014 | 1.142242 | 14.8908 | 4.8263 | 10.6358 |
| Playoffs | 1.192621 | 1.136193 | 17.1300 | - | - |

Against the otherwise identical Gaussian recursive model, Student-t is worse
on regular-season possession RMSE (1.198993) and game-margin RMSE (14.8225),
as well as the regular team metrics. It improves playoff game-margin RMSE from
17.4188 to 17.1300, but the frozen aging prior remains the current playoff
leader at 16.4946. This first robust-likelihood specification is therefore a
useful negative result, not a promoted predictive state.

## Published Outputs

| File | Contents |
| --- | --- |
| `historical_player_coefficients.parquet` | Per-season Student-t MAP coefficients and prior adjustments |
| `season_player_priors.parquet` | Strictly forward player priors entering each season |
| `season_cold_start_metadata.parquet` | Cold-start settings plus Student-t IRLS scale and convergence diagnostics |
| `frozen_2025_26_player_priors.parquet` | Player prior vector fixed before the target season |
| `next_season_top_100_returning_rankings.parquet` | Completed-2025-26 returning-player state for 2026-27 |
| Frozen evaluation files | Regular/playoff possession and game predictions plus regular team outputs |
