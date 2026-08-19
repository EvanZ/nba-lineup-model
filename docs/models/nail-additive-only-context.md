---
title: NAIL-RAPM Additive-Only Context Ablation
last_updated: "2026-08-17"
---

# NAIL-RAPM Additive-Only Context Ablation

Last updated: 2026-08-17

This is the direct all-or-none test of NAIL-RAPM v1.0's six non-additive
lineup coordinates. It keeps the recursive value-conditioned aging prior,
exposure-gated cold starts, historical data, annual RAPM penalty schedule,
and linear Ridge estimator fixed. The only difference is the context design
matrix.

## Contract

The ablation retains the eight player-additive basketball profile totals:

- three-point attempts and makes per 100 possessions;
- assists, turnovers, and usage events per 100 possessions;
- offensive-rebound claim percentage; and
- steals and blocks per 100 possessions.

It removes the non-additive lineup bundle: bottom-two shooting, credible
shooter count, top-two assists, usage concentration, shooting-by-usage, and
shooter-by-passing. Consequently, the contextual contribution is exactly

\[
C(H,A)=\sum_k \gamma_k\,[x_k(H)-x_k(A)],
\]

where each \(x_k(U)\) is a sum of the five players' lagged profiles. This is
not the earlier Additive Profile-Prior RAPM experiment: the eight coordinates
remain in the lineup-level Ridge fit rather than being moved into next season's
player prior.

## Frozen Comparison

The completed replay compares this artifact directly to
[NAIL-RAPM v1.0](nail-rapm-v1.md) on 2023-24 through
2025-26. The player prior, annual lambda schedule, target cohorts, and scoring
procedure are identical; the six removed context coordinates are the sole model
difference.

| Cohort | Metric | NAIL-RAPM v1.0 | Additive-only context | Additive-only minus NAIL |
| --- | --- | ---: | ---: | ---: |
| Regular season | Possession RMSE | **1.197977** | 1.197989 | +0.000012 |
| Regular season | Possession MAE | **1.141387** | 1.141430 | +0.000043 |
| Regular season | Eligible game RMSE | **14.1119** | 14.1286 | +0.0166 |
| Regular season | Full-game RMSE | **14.3517** | 14.3754 | +0.0237 |
| Regular season | Winner accuracy | **68.33%** | 67.93% | -0.40 pp |
| Regular season | Team NetRtg RMSE | **3.4307** | 3.4783 | +0.0477 |
| Regular season | Pythagorean-win RMSE | **7.3460** | 7.4728 | +0.1267 |
| Playoffs | Possession RMSE | 1.192752 | **1.192713** | -0.000039 |
| Playoffs | Possession MAE | 1.137640 | **1.137590** | -0.000050 |
| Playoffs | Eligible game RMSE | 16.6636 | **16.5724** | -0.0912 |

For regular-season uncertainty, a 10,000-draw paired bootstrap stratified by
season compares additive-only minus NAIL. Full-game RMSE is directionally worse
at +0.0237, but its 95% interval \([-0.0105, +0.0594]\) crosses zero. Possession
MAE is reliably worse: +0.000042, 95% interval \([+0.000022, +0.000063]\), with
only 0.01% of draws favoring the additive-only ablation.

The result is therefore nuanced: the bundle's aggregate regular-season signal
is real but small at the game level. It is not evidence that every one of the
six coordinates is independently identified or necessary; that requires a
within-bundle ablation. The playoff point estimates favor additive-only, which
is useful caution against treating the regular-season result as universal.

Artifact:

```text
artifacts/models/analysis/nail_additive_only_context_frozen/
frozen_multiseason_backtest/2023-24_to_2025-26/
frozen_multiseason_backtest-2023-24-to-2025-26-20260818T012932Z-40442bd8
```
