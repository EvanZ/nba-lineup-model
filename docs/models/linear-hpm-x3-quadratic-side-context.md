---
title: Linear HPM x3 Quadratic Side Context
---

# Linear HPM x3 Quadratic Side Context

Last updated: 2026-08-15

This candidate tests the first nonlinear extension of the [Linear HPM x3
Attribution Contract](linear-hpm-x3-compilation-audit.md), without using
splines.

For each of the ten player-compilable additive feature totals \(X_f(U)\), it
fits:

\[
C(H,A)=
\sum_f \beta_f\big(X_f(H)-X_f(A)\big)
+\sum_f \gamma_f\big(X_f(H)^2-X_f(A)^2\big).
\]

The \(\beta\) component can be compiled into the player prior. The \(\gamma\)
component remains explicit lineup context: it permits diminishing returns or
threshold-like curvature based on each unit's own completed total, while
preserving antisymmetry between the two sides.

Unlike the P-spline models, this has exactly one nonlinear degree of freedom
per additive feature. It is the lowest-complexity test of whether aggregate
nonlinearity adds predictive value beyond the compiled additive player layer.

## Run

```bash
uv run nba-train-hpm-x3-linear-quadratic-side-context --through-season 2025-26
uv run nba-run-frozen-multiseason-backtest
```

The frozen leaderboard will compare this candidate against the equivalent
compiled-additive linear baseline and the existing HPM models.

## Frozen Result

The completed three-season replay evaluates 2023-24 through 2025-26 using
only the state available before each target season. It covers 584,970 regular
season possessions (3,284 games) and 39,967 playoff possessions (238 games).

| Cohort | Metric | Canonical Compiled-Linear HPM x3 | Linear plus quadratic side context | Quadratic minus linear |
| --- | --- | ---: | ---: | ---: |
| Regular season | Possession RMSE | **1.197978** | 1.198002 | +0.000024 |
| Regular season | Possession MAE | **1.141349** | 1.141366 | +0.000017 |
| Regular season | Eligible game RMSE | **14.1051** | 14.1311 | +0.0260 |
| Regular season | Full-game RMSE | **14.3529** | 14.3922 | +0.0393 |
| Regular season | Winner accuracy | **68.13%** | 67.36% | -0.77 pp |
| Playoffs | Possession RMSE | 1.192766 | **1.192719** | -0.000047 |
| Playoffs | Eligible game RMSE | 16.6817 | **16.5978** | -0.0838 |

The per-side quadratic terms do not earn promotion: the linear baseline is
better on every regular-season aggregate metric. The small playoff advantage
is a useful follow-up signal, but it is not sufficient to outweigh the larger
regular-season cohort or justify the additional degrees of freedom.

Artifact: `artifacts/models/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260816T033818Z-07bc6a53`.
