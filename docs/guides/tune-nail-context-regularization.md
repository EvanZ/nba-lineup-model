---
title: Tune NAIL Context Regularization
last_updated: "2026-08-18"
---

# Tune NAIL Context Regularization

This study tunes only the Ridge penalty on NAIL-RAPM v1.1's linear lineup
context model. The player prior, statistic-specific profile padding, RAPM
lambda schedule, 14 context coordinates, and frozen evaluation seasons remain
fixed.

## Normalized Objective

The published v1.1 model passes a raw `alpha=10000` to scikit-learn. That value
penalizes a weighted sum of squared errors, so its effective strength depends
on the total possession weight in the fitted season.

The candidates instead fit

\[
\frac{1}{W_t}\sum_{s \in t} w_s
\left(y_s - x_s^\mathsf{T}\beta_t\right)^2
+ \lambda_C\lVert\beta_t\rVert_2^2,
\qquad
W_t=\sum_{s \in t}w_s.
\]

The implementation adds the reversed signed orientation of every stint to
enforce exact antisymmetry. This doubles both weighted SSE and total weight
without changing the mean loss. For scikit-learn's unnormalized Ridge
objective, the equivalent season-specific parameter is therefore

\[
\alpha_t=2\lambda_C W_t
=\lambda_C W_t^{\pm},
\]

where (W_t^{\pm}) is the total weight of the orientation-augmented sample.

Thus one dimensionless \(\lambda_C\) has the same meaning in a shortened season
and an 82-game season and is invariant to a common rescaling of stint weights.

## 1. Train Candidate Recursive States

Each command performs the complete forward recursion through 2022-23. Target
evaluation is skipped because the selection stage replays all validation
seasons from persisted historical state.

```bash
uv run nba-train-nail-context-regularization \
  --through-season 2022-23 \
  --skip-target-evaluation \
  --context-lambda 0.00275 \
  --context-lambda 0.022 \
  --context-lambda 0.044 \
  --context-lambda 0.088 \
  --context-lambda 0.176 \
  --context-lambda 0.352 \
  --context-lambda 0.704 \
  --context-lambda 1.408
```

Additional values can be appended with repeated `--context-lambda` options.
Every artifact records the configured \(\lambda_C\), season weight total, and
effective scikit-learn \(\alpha_t\).

## 2. Select Before the Frozen Seasons

```bash
uv run nba-select-nail-context-regularization \
  --context-lambda 0.00275 \
  --context-lambda 0.022 \
  --context-lambda 0.044 \
  --context-lambda 0.088 \
  --context-lambda 0.176 \
  --context-lambda 0.352 \
  --context-lambda 0.704 \
  --context-lambda 1.408
```

The selector forecasts each season from 2000-01 through 2022-23 using only the
recursive state available before that season. It minimizes the equal-season
mean full-game margin squared error. The one-standard-error rule selects the
strongest penalty whose paired season-by-season loss difference from the exact
minimum remains within one standard error. Pairing removes variation in season
difficulty from the uncertainty estimate. `--reuse-latest` preserves prior
season-level results and replays only new candidate values when extending an
existing study; omit it for the first complete replay shown above.

The 2023-24, 2024-25, and 2025-26 seasons do not participate in selection.

## 3. Train and Evaluate the Selection

```bash
uv run nba-finalize-nail-context-regularization train
uv run nba-finalize-nail-context-regularization evaluate
uv run nba-finalize-nail-context-regularization bootstrap
```

The first command fits the selected contract through 2025-26. The second
replays the three untouched frozen seasons against published NAIL-RAPM v1.1,
including regular-season, playoff, game, and team metrics. The final command
runs a paired 10,000-draw game bootstrap stratified by season.

## Fixed-Raw Sensitivity Audit

The normalized candidate is a different regularization contract. To test
whether its frozen loss reflects a conversion bug or an overly strong selected
penalty, the follow-up audit holds the original raw-loss contract fixed and
changes only `alpha`:

```bash
uv run nba-audit-nail-fixed-context-regularization train --raw-alpha 1000
uv run nba-audit-nail-fixed-context-regularization train --raw-alpha 5000
uv run nba-audit-nail-fixed-context-regularization train --raw-alpha 20000
uv run nba-audit-nail-fixed-context-regularization select
uv run nba-audit-nail-fixed-context-regularization evaluate
uv run nba-audit-nail-fixed-context-regularization bootstrap
```

The published `alpha=10000` artifact is reused. Selection again ends at
2022-23; all four fixed values are replayed on the three frozen seasons only
after the pre-frozen comparison has been persisted. The exact pre-frozen
minimum was 5,000 and the paired one-standard-error selection was 20,000, but
neither beat the published 10,000 model on the frozen regular-season contract.
See [NAIL-RAPM Context Regularization](../models/nail-context-regularization.md)
for the complete tables and bootstrap intervals.
