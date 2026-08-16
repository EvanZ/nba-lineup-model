---
last_updated: "2026-08-16"
---

# Forward Compiled-Additive-Prior HPM x3

This experiment moves the learned **additive** portion of canonical linear HPM
x3 from the carried lineup function into the following season's player prior.
It tests the model boundary directly: additive player-profile signals should be
player state; only genuinely non-additive unit shape should remain context.

For completed season \(t\), the linear context fit is

\[
C_t(H,A)=
\sum_k \beta_{k,t}
\left(\sum_{i\in H}z_{i,t,k}-\sum_{j\in A}z_{j,t,k}\right)
+C^{\mathrm{shape}}_t(H,A),
\]

where the eight \(z_k\) coordinates are three-point attempts and makes,
assists, turnovers, usage, steals, blocks, and offensive-rebound claim. The
six remaining x3 coordinates are lineup-shape terms: shooting depth, credible
shooter count, top-two assists, usage concentration, shooting-by-usage, and
shooter-by-passing.

The prior for season \(t+1\) is then

\[
\mu_{i,t+1}=g_t(r_{i,t}, \mathrm{age}, \mathrm{exposure}, \ldots)
+\sum_k \beta_{k,t}\bigl(z_{i,t+1,k}-\bar z_{t,k}\bigr),
\]

using only the prior-season fitted \(\beta_t\) and the target season's lagged
player profile. \(\bar z_{t,k}\) is a centering convention; it cancels from
any five-versus-five margin.

To avoid double counting, the carried context is only

\[
C^{\mathrm{shape}}_t(H,A)=C_t(H,A)-
\sum_k \beta_{k,t}
\left(\sum_{i\in H}z_{i,t+1,k}-\sum_{j\in A}z_{j,t+1,k}\right).
\]

For fixed player coefficients and \(\beta_t\), this is an exact accounting
identity. The experiment can differ from ordinary HPM only because the
enhanced player prior changes later RAPM refits and subsequent learned state.

This is distinct from Context-Reattributed RAPM. It does not project total
context back onto players using another regression; it transfers only terms
that are algebraically player-additive in the fitted HPM equation.

## Outputs

Each training run stores:

- `season_compiled_additive_prior_coefficients.parquet`: the eight raw
  prior-season \(\beta\) coefficients carried into each target season;
- `season_player_priors.parquet`: the resulting centered player priors;
- `season_player_prior_metadata.parquet`: transfer coverage and centering
  metadata;
- `season_context_models.joblib`: the full completed-season 14-term context
  fit, from which the six-term residual context is evaluated.

The frozen three-season result will determine whether this state allocation
improves prediction relative to NAIL-RAPM v1.0.

## Frozen Three-Season Result

The completed replay pools 584,970 regular-season possessions from 3,284 games
and 39,967 playoff possessions from 238 games. The transfer reached every
eligible player in the three target seasons: 547 in 2023-24, 547 in 2024-25,
and 562 in 2025-26.

| Cohort | Metric | Canonical x3 | Additive-prior x3 | Candidate minus canonical |
| --- | --- | ---: | ---: | ---: |
| Regular season | Possession RMSE | **1.197977** | 1.198079 | +0.000102 |
| Regular season | Possession MAE | **1.141387** | 1.141438 | +0.000051 |
| Regular season | Eligible-game RMSE | **14.1119** | 14.3203 | +0.2084 |
| Regular season | Full-game RMSE | **14.3517** | 14.5908 | +0.2391 |
| Regular season | Winner accuracy | **68.33%** | 67.45% | -0.88 pp |
| Regular season | Team NetRtg RMSE | **3.4307** | 3.7410 | +0.3103 |
| Regular season | Pythagorean-win RMSE | **7.3460** | 8.0203 | +0.6743 |
| Playoffs | Possession RMSE | 1.192752 | **1.192679** | -0.000073 |
| Playoffs | Possession MAE | 1.137640 | **1.137448** | -0.000192 |
| Playoffs | Eligible-game RMSE | 16.6636 | **16.5899** | -0.0738 |

This is a clear regular-season rejection. The modest pooled-playoff improvement
does not compensate for worse regular game, team, and win forecasts, so the
canonical compiled-linear x3 model remains the predictive reference. The
experiment does, however, validate the accounting boundary: all additive terms
can be moved into player state exactly, but that allocation is not presently
the better recursive forecasting state.

Artifact:
`artifacts/models/analysis/compiled_additive_prior_hpm_x3_frozen/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260816T155428Z-44dcf39f`.
