---
title: Linear-Ridge HPM x3 Context
---

# Linear-Ridge HPM x3 Context

Last updated: 2026-08-15

This is the canonical simplest contextual comparison for HPM x3. It keeps the
same forward player prior and 14-feature basketball lineup contract, but replaces
bounded hierarchical P-splines with one standardized linear Ridge regression:

\[
C(H,A)=\beta^\top\big(x(H)-x(A)\big).
\]

It uses no spline basis, curvature penalty, temporal spline hierarchy, or
feature clipping. Because the fitted context is linear in the feature
difference, its derived decomposition has no matchup remainder:

\[
C(H,A)=h(H)-h(A),\qquad q(H,A)=0.
\]

The model is therefore the appropriate baseline for testing whether the
nonlinear spline machinery earns its complexity with the same inputs.

## Feature Contract

The current contract excludes `imputed_count` and `replacement_weight`. Those
are provenance and cold-start calibration diagnostics, not player or lineup
basketball attributes. A frozen three-season ablation found no meaningful
full-game advantage from retaining them, while possession MAE changed by only
0.000038. The older 16-feature run remains available as a historical artifact;
new canonical x3 runs use the 14 basketball features.

## Run

```bash
uv run nba-train-hpm-x3-linear-ridge --through-season 2025-26
```

## Frozen Result

| Metric | HPM x3 | Linear Ridge | Ridge minus x3 | Paired 95% interval | P(Ridge better) |
| --- | ---: | ---: | ---: | --- | ---: |
| Full-game RMSE | 14.377365 | 14.352888 | -0.024476 | [-0.049522, +0.000776] | 97.03% |
| Winner accuracy | 67.73% | 68.13% | +0.40 pp | [-0.14 pp, +0.94 pp] | 91.84% |
| Possession RMSE | 1.198000 | 1.197978 | -0.000022 | [-0.000039, -0.000006] | 99.66% |
| Possession MAE | 1.141400 | 1.141349 | -0.000051 | [-0.000068, -0.000033] | 100.00% |

The linear model is statistically better on possession RMSE and MAE. Its
full-game point estimate is better, but the paired interval crosses zero by
0.0008 RMSE, so it does not meet the strict tournament promotion rule.

## Compilation Identity Audit

The [NAIL-RAPM Attribution Contract](linear-hpm-x3-compilation-audit.md)
verifies that eight additive basketball context coordinates can move exactly into the player
prior, while six lineup-shape coordinates remain context-only. The two forms
produce identical frozen forecasts from 2023-24 through 2025-26.
