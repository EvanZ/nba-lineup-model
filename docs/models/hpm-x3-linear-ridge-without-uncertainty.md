---
title: NAIL-RAPM v1.0
---

# NAIL-RAPM v1.0

Last updated: 2026-08-16

NAIL-RAPM is **Non-Additive Interactions in Lineups RAPM**. Version 1.0 is
the current canonical linear feature contract, established by this profile-quality
ablation.

It supersedes the earlier [Compiled-Additive Linear HPM x3](hpm-x3-linear-ridge.md)
feature contract.
It preserves the same value-conditioned aging RAPM prior, exposure-gated cold
starts, linear Ridge estimator, and 14 remaining coordinates. It drops
only:

- `imputed_count`: the number of players in a unit with an imputed lagged profile.
- `replacement_weight`: the total amount of replacement-profile blending in the unit.

These are profile-quality calibration terms, not basketball skills. The
ablation tests whether their incremental forecast value is real or whether the
exposure-gated player prior should already absorb their uncertainty.

## Published Player Rating

The player rating shown in the app and rankings compiles the additive profile
terms back to individual players:

\[
R_i^{\mathrm{NAIL}} = R_{i,\mathrm{prior}}
+ \Delta R_{i,\mathrm{season}}
+ A_i^{\mathrm{profile}}.
\]

Here, \(R_{i,\mathrm{prior}} + \Delta R_{i,\mathrm{season}}\) is the regularized
player RAPM state, while \(A_i^{\mathrm{profile}}\) is that player's compiled
share of the model's additive profile signal. The latter is centered within a
season for presentation; five-player margins are unchanged by this centering.
The remaining lineup layer contains only deliberately non-additive lineup
terms, so it is not assigned to any single player. The [attribution
contract](linear-hpm-x3-compilation-audit.md) gives the exact reconstruction
and explains why this compilation does not double count the player update.

The result will be evaluated against the full compiled-linear HPM x3 on the
same frozen 2023-24 through 2025-26 regular-season and pooled-playoff cohorts.

## Frozen Result

The completed replay covers 584,970 eligible regular-season possessions
(3,284 games) and 39,967 playoff possessions (238 games).

| Cohort | Metric | Full compiled-linear HPM x3 | Without uncertainty context | Ablated minus full |
| --- | --- | ---: | ---: | ---: |
| Regular season | Possession RMSE | 1.197978 | **1.197977** | -0.0000002 |
| Regular season | Possession MAE | **1.141349** | 1.141387 | +0.000038 |
| Regular season | Eligible game RMSE | **14.1051** | 14.1119 | +0.0068 |
| Regular season | Full-game RMSE | 14.3529 | **14.3517** | -0.0012 |
| Regular season | Winner accuracy | 68.13% | **68.33%** | +0.20 pp |
| Playoffs | Possession RMSE | 1.192766 | **1.192752** | -0.000014 |
| Playoffs | Eligible game RMSE | 16.6817 | **16.6636** | -0.0180 |

The primary full-game difference is effectively unresolved: the paired 95%
interval is \([-0.0207, +0.0182]\), with a 55.42% bootstrap probability of
improvement. Possession MAE, however, is reliably worse after removing the
terms: (+0.000038), with paired interval \([+0.000027, +0.000049]\).

The ablation does not produce a decisive full-game change. We nevertheless
retire these two terms from new NAIL-RAPM v1.0 runs: they are profile-quality
diagnostics rather than basketball context, and their tiny possession-MAE gain
does not justify the added interpretation burden.

Artifact: `artifacts/models/analysis/hpm_x3_linear_without_uncertainty_frozen/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260816T143204Z-21f36f70`.
