---
title: NAIL Token Residual Models
last_updated: "2026-08-17"
---

# NAIL Token Residual Models

Last updated: 2026-08-17

This experiment asks whether a learned set function can recover lineup effects
that the six handcrafted non-additive NAIL features miss. It is a controlled
residual model, not a replacement for RAPM.

## Frozen baseline

The base prediction is the
[NAIL additive-only context ablation](nail-additive-only-context.md). For a
target season $t$, that baseline uses the recursively learned player prior
and context coefficients available at the end of $t-1$. Target-season
outcomes do not enter either state.

For historical training stint $s$, the neural target is

\[
r_s = y_s - \widehat y_s^{\text{additive NAIL}},
\]

where $y_s$ is home net rating over the stint. Identical season/lineup
matchups are collapsed to one row, and squared error is weighted by their
total possessions. This has the same optimizer as repeating the fitted row
once per possession, up to a constant that does not depend on the model.

## Token contract

Each side contains five player tokens. Every token contains exactly eight
strictly lagged NAIL profile fields:

| Field | Meaning |
| --- | --- |
| `three_pa_per_100` | Three-point attempts per 100 prior-season possessions |
| `three_pm_per_100` | Three-point makes per 100 prior-season possessions |
| `assists_per_100` | Assists per 100 prior-season possessions |
| `turnovers_per_100` | Turnovers per 100 prior-season possessions |
| `usage_per_100` | FGA + 0.44 FTA + turnovers per 100 prior-season possessions |
| `offensive_rebound_pct` | Prior-season offensive-rebound claim percentage |
| `steals_per_100` | Steals per 100 prior-season possessions |
| `blocks_per_100` | Blocks per 100 prior-season possessions |

There is no player ID, team, position, age, draft field, target-season box
score, or learned identity embedding. Cold-start profile construction is the
same forward exposure-gated replacement/cohort process used by NAIL. Token
normalization is fit separately inside each frozen training window.

## Shared side score

Let $X(H_s)\in\mathbb{R}^{5\times8}$ and
$X(A_s)\in\mathbb{R}^{5\times8}$ denote the home and away token sets. Both
sides pass through the same learned function $h_\theta$:

\[
\widehat r_s = h_\theta(X(H_s)) - h_\theta(X(A_s)).
\]

The shared function makes the correction antisymmetric. Reversing the two
units reverses the sign, and permuting players within either unit leaves the
prediction unchanged. The completed prediction is

\[
\widehat y_s
=
\widehat y_s^{\text{additive NAIL}}
+
\widehat r_s.
\]

## Architecture ladder

Two candidates separate generic token nonlinearity from teammate attention.

### Token MLP control

Each player is transformed independently by the same MLP. The five transformed
tokens are mean-pooled, then a side-level MLP emits one scalar. This model can
learn nonlinear transformations of individual profiles, but players cannot
communicate before pooling.

### Within-unit Set Attention

Each side is projected to width 32 and passed through two Transformer encoder
layers with four attention heads. There is no positional encoding. Every
player token can attend to the other four players on the same unit, after
which mean pooling and a side head emit the scalar score. The first experiment
does not use cross-attention between opponents; that remains a later ablation
if within-unit attention earns predictive support.

Both output heads are initialized to zero, so the initial combined prediction
is exactly additive-only NAIL.

## Frozen evaluation

Separate models are trained for 2023-24, 2024-25, and 2025-26. Each training
corpus ends before its target season. Regular seasons and playoffs are scored
separately, and regular-season full-game, team net-rating, winner, and
Pythagorean-win metrics use the same leaderboard contract as NAIL.

The residual history begins in 1998-99. NAIL itself still begins in 1996-97;
the two earlier seasons only bootstrap the recursive player and context state
needed to construct a strictly frozen neural residual target. This restriction
does not remove those seasons from the underlying NAIL model.

<!-- nail-token-results:start -->
### Pooled results

Lower is better except for winner accuracy. The additive-only column is the
unchanged frozen baseline to which each neural residual is added.

| Regular-season metric | Additive-only NAIL | Token MLP | Set Attention |
| --- | ---: | ---: | ---: |
| Possession RMSE | 1.197989 | **1.197986** | 1.197997 |
| Possession MAE | 1.141430 | 1.141411 | **1.141396** |
| Eligible game RMSE | **14.1286** | 14.1207 | 14.1458 |
| Full-game RMSE | 14.3754 | **14.3698** | 14.4098 |
| Winner accuracy | 67.93% | **67.99%** | 67.64% |
| Team NetRtg RMSE | 3.4783 | **3.4669** | 3.5075 |
| Pythagorean-win RMSE | 7.4728 | **7.3841** | 7.5605 |

| Playoff metric | Additive-only NAIL | Token MLP | Set Attention |
| --- | ---: | ---: | ---: |
| Possession RMSE | **1.192713** | 1.192747 | 1.192768 |
| Possession MAE | 1.137590 | 1.137607 | **1.137571** |
| Eligible game RMSE | **16.5724** | 16.6393 | 16.7235 |

### Paired uncertainty

A 10,000-draw paired game-cluster bootstrap compares each candidate with the
additive-only baseline. Differences below zero favor the neural candidate for
RMSE and MAE.

| Cohort | Candidate | Metric | Difference | 95% interval | Probability candidate is better |
| --- | --- | --- | ---: | ---: | ---: |
| Regular | Token MLP | Full-game RMSE | -0.0056 | [-0.0230, +0.0118] | 73.98% |
| Regular | Token MLP | Possession RMSE | -0.000003 | [-0.000013, +0.000007] | 72.72% |
| Regular | Token MLP | Possession MAE | -0.000019 | [-0.000029, -0.000009] | 99.97% |
| Regular | Set Attention | Full-game RMSE | +0.0344 | [+0.0035, +0.0642] | 1.38% |
| Regular | Set Attention | Possession RMSE | +0.000008 | [-0.000009, +0.000026] | 17.86% |
| Playoffs | Token MLP | Eligible game RMSE | +0.0669 | [+0.0109, +0.1242] | 0.82% |
| Playoffs | Set Attention | Eligible game RMSE | +0.1511 | [+0.0515, +0.2479] | 0.19% |

### Decision

Within-unit attention did **not** recover useful incremental lineup signal in
this token contract. It is significantly worse than the additive-only baseline
on regular-season full-game RMSE and playoff eligible-game RMSE. The token MLP
produces a statistically detectable but practically negligible regular-season
MAE improvement, while its full-game interval crosses zero and its playoff
game prediction is worse.

Neither neural residual replaces the current incumbent. Both remain on the
frozen leaderboard and model tree because those surfaces record tested model
lineage, including negative results. The controlled result narrows the next
neural question: additional capacity is not enough when the model sees only
these eight lagged player-profile fields. Future work would need a materially
different information contract or target, not merely more self-attention
layers.

Immutable training artifact:

```text
artifacts/models/analysis/nail_token_residual_frozen/2023-24_to_2025-26/
nail-token-residual-2023-24-to-2025-26-20260818T052735Z-bb21b3ef
```

Immutable paired-bootstrap artifact:

```text
artifacts/models/analysis/nail_token_residual_bootstrap/2023-24_to_2025-26/
nail-token-residual-bootstrap-20260818T055117Z-23de73b3
```
<!-- nail-token-results:end -->

## Reproduce

See [Train NAIL Token Residual Models](../guides/train-nail-token-residual.md).
