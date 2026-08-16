---
title: HIPSTER PM v2.3
---

# HIPSTER PM v2.3: Shot Portfolio

Last updated: 2026-08-14

HPM v2.3 retains v2.2's empirical rebound and usage-allocation features, then
adds a forward-safe shot-portfolio test. It asks whether a unit's available rim
pressure is more useful when it is paired with genuine three-point spacing.

For a target season \(t\), every player trait is drawn from \(t-1\). Rim and
three-point attempt rates are shrunk toward their source-season league rate
with 300 pseudo-possessions, then standardized within that source-season
environment. Players without a source-season shot profile receive the league
mean, represented by zero after standardization.

For a unit \(U\), the new signals are

\[
R(U)=\sum_{i\in U}\mathrm{rimPressure}_i,
\qquad
S(U)=\sum_{i\in U}\mathrm{spacingCapacity}_i,
\qquad
I(U)=R(U)S(U).
\]

The bounded hierarchical P-spline model learns separate clipped response
functions for \(R\), \(S\), and \(I\). The interaction is strongly
regularized and is an initial low-cost approximation to a full two-dimensional
rim-pressure-by-spacing surface. It is retained only if it improves the frozen
three-season evaluation.

This is a context test, not a claim that rim attempts or threes mechanically
cause a lineup outcome. Player shooting and finishing quality remain available
to the RAPM prior; these features describe how shot-location capacity combines
within a five-player unit.

## Run

```bash
uv run nba-train-hpm-v23 --through-season 2025-26
```

The run writes its recursive state and target-season outputs under
`artifacts/models/forward_hpm_v23_shot_portfolio/`.

## Frozen Result

On the pooled 2023-24 through 2025-26 frozen regular-season holdouts, v2.3
records a possession RMSE of 1.198025 and a full-game margin RMSE of 14.4288.
It does not improve on v2.1 or the value-conditioned aging HPM across the
principal regular-season metrics, so the shot-portfolio interaction remains an
interpretable experiment rather than a promoted default. Its strongest result
is playoff possession MAE, where it leads this comparison at 1.137641. See the
[Three-Season Frozen Leaderboard](three-season-frozen-backtest.md) for the
complete comparison.
