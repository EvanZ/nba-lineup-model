---
last_updated: "2026-08-23"
---

# NAIL-RAPM v1.2.1.1: Standard USG%

NAIL-RAPM v1.2.1.1 is a convention-correction patch to
[v1.2.1](nail-rapm-v121-pruned-nonadditive.md). It retains the same eight
additive profile coordinates, two non-additive lineup terms, value-conditioned
aging prior, exposure-gated cold starts, gap-returner bridge, profile padding,
and Ridge estimators. It changes only the definition of usage.

## Usage Contract

The prior release used a player action-event rate per RAPM on-court possession:

\[
100 \cdot \frac{FGA_i + 0.44 FTA_i + TOV_i}{\text{on-court possessions}_i}.
\]

v1.2.1.1 uses the conventional box-score usage percentage, computed at the
game level and aggregated across a player-season:

\[
USG_i = 100 \cdot
\frac{\sum_g (FGA_{ig} + 0.44 FTA_{ig} + TOV_{ig})(TM_g/5)}
{\sum_g MIN_{ig}(FGA_{tg} + 0.44 FTA_{tg} + TOV_{tg})}.
\]

Here, \(TM_g\) is the team’s game minutes, \(MIN_{ig}\) is the player’s game
minutes, and the subscript \(t\) denotes the player’s team. This makes the
denominator an estimate of the team’s offensive opportunities while the player
was on the court. It puts familiar player values, such as a 31.8% season for
Anthony Edwards in 2025-26, on the same scale used in standard NBA analysis.

The feature receives the existing 300 pseudo-opportunity profile shrinkage and
replaces the legacy usage-event coordinate in both the additive profile and
the non-additive `usage_concentration` term. No target-season information is
used in frozen evaluation: each target season uses only its pre-season player
profile state and prior-year fitted context state.

## Frozen Results

The strict frozen replay forecasts 2023-24 through 2025-26 from each target
season’s pre-season information set. Values below are pooled over the three
regular seasons; lower is better except winner accuracy.

| Model | Poss. RMSE | Poss. MAE | Eligible game RMSE | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1.2.1 legacy usage rate | 1.197952 | 1.141355 | **14.0246** | 14.2521 | **68.24%** | 3.2706 | 7.0351 |
| **v1.2.1.1 standard USG%** | **1.197951** | **1.141332** | 14.0264 | **14.2516** | 68.10% | **3.2658** | **7.0279** |

The changes are deliberately small. The candidate improves full-game RMSE by
0.00055 and possession MAE by 0.000023, while eligible-game RMSE and winner
accuracy are slightly worse. The predeclared paired game-block bootstrap is
therefore the decision criterion rather than point-estimate cherry-picking.

| Scope | Full-game RMSE difference (v1.2.1.1 - v1.2.1) | Paired 95% interval | Gate threshold | Gate |
| --- | ---: | ---: | ---: | --- |
| Pooled | -0.00055 | [-0.00743, +0.00608] | +0.07126 | Pass |
| 2023-24 | -0.01101 | [-0.02278, +0.00067] | +0.06976 | Pass |
| 2024-25 | +0.00799 | [-0.00558, +0.02171] | +0.07189 | Pass |
| 2025-26 | +0.00096 | [-0.00834, +0.01040] | +0.07205 | Pass |

## Decision

v1.2.1.1 passes the non-promotion gate, but is **not promoted**. Conventional
USG% is cleaner as a display convention, yet it does not materially change the
Minnesota-style concentration interpretation. The candidate also gives up small
point-estimate advantages in eligible-game RMSE and winner accuracy. v1.2.1
therefore remains the product model and source for the website bundle.

## Artifacts

- Recursive fit: `artifacts/models/forward_nail_rapm_v1211_standard_usage/2025-26/forward-nail-rapm-v1211-standard-usage-2025-26-20260823T141226Z-ddb46a76`
- Frozen replay: `artifacts/models/nail_v1211_standard_usage_frozen_backtest/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260823T142112Z-d5cea78f`
- Paired bootstrap: `artifacts/models/nail_v1211_standard_usage_bootstrap/2023-24_to_2025-26/nail-v1211-standard-usage-bootstrap-20260823T142343Z-07ea4b70`

## Reproduction

See [Train NAIL-RAPM v1.2.1.1 Standard USG%](../guides/train-nail-v1211-standard-usage.md).
