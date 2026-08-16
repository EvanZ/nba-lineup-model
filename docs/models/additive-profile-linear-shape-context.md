---
title: Additive Prior Plus Linear Non-Additive Context
---

# Additive Prior Plus Linear Non-Additive Context

Last updated: 2026-08-15

This is the controlled test of the project’s attribution contract:

1. Player-additive profile information belongs in a strictly lagged player prior.
2. Only genuinely non-additive five-player properties remain in lineup context.

The player prior is the existing eight-feature [Additive Profile-Prior
RAPM](additive-profile-prior-rapm.md): three-point attempts and makes, assists,
turnovers, usage, offensive rebound percentage, steals, and blocks. Its
coefficients are learned recursively using only completed prior seasons.

The context layer is standardized linear Ridge over exactly six non-additive
features:

\[
C(H,A)=\sum_{k=1}^{6}\gamma_k\,[z_k(H)-z_k(A)].
\]

The six (z_k) values are bottom-two three-point makes, credible-shooter
count, top-two assists, usage concentration, shooting-by-usage, and
shooter-by-passing. They are handcrafted nonlinear functions of the five
players, but their effects on net rating are deliberately linear. There are no
P-splines, quadratic terms, thresholds beyond the declared shooter definition,
or target-season outcomes in the fitted state.

## Evaluation

The primary comparison is against additive profile-prior RAPM with no context.
The frozen evaluator will replay 2023-24, 2024-25, and 2025-26 from the state
available before each season, then report regular season and pooled playoff
metrics separately.

## Frozen Result

The three-season replay covers 584,970 eligible regular-season possessions
(3,284 games) and 39,967 playoff possessions (238 games).

| Cohort | Metric | Additive player prior only | Additive prior plus linear non-additive context | Non-additive minus control |
| --- | --- | ---: | ---: | ---: |
| Regular season | Possession RMSE | 1.198042 | **1.198026** | -0.000015 |
| Regular season | Possession MAE | 1.141429 | **1.141366** | -0.000063 |
| Regular season | Eligible game RMSE | 14.2457 | **14.2154** | -0.0303 |
| Regular season | Full-game RMSE | 14.4902 | **14.4542** | -0.0360 |
| Regular season | Winner accuracy | **68.07%** | 67.76% | -0.31 pp |
| Playoffs | Possession RMSE | **1.192751** | 1.192814 | +0.000063 |
| Playoffs | Eligible game RMSE | **16.6886** | 16.7224 | +0.0339 |

The non-additive lineup layer improves every regular-season error metric versus the exact
additive-prior-only control. Its paired full-game RMSE interval is
\([-0.0765, +0.0047]\), with a 95.84% bootstrap probability of improvement,
but the upper endpoint remains above zero. It therefore does not yet meet the
project’s strict primary-metric promotion rule. The playoff check favors the
control.

This establishes that the declared non-additive lineup features contain
some incremental regular-season signal after additive player profile credit is
moved into the prior. It does not establish that they outperform the current
NAIL-RAPM v1.0 reference.

Artifact: `artifacts/models/analysis/additive_profile_linear_shape_context_frozen/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260816T063830Z-ae6c82db`.
