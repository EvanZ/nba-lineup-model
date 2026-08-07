---
last_updated: "2026-08-04"
---

# Train The Box-Score RAPM Prior

Build the validated possession-native panel first:

```bash
uv run nba-build-box-score-prior-panel
```

Then train the returning-player forecast with its default latest-target holdout:

```bash
uv run nba-train-box-score-prior
```

To make the locked evaluation explicit:

```bash
uv run nba-train-box-score-prior --holdout-season 2025-26
```

The command uses chronological expanding target-season folds to select ridge
regularization, fits through the season before the holdout, and writes an
immutable run below:

```text
artifacts/models/box_score_prior/<holdout-season>/<run-id>/
```

Each run includes fold-level validation metrics, the candidate summary,
out-of-fold and holdout predictions, cohort-level holdout metrics, standardized
feature coefficients, the serialized pipeline, and a hash-validated manifest.
Completed runs are indexed in local MLflow; see [Track experiments with
MLflow](mlflow.md).

The current model accepts only returning players with prior RAPM and a complete
prior box profile. It is a component experiment, not yet a complete RAPM prior
or a Leaderboard entrant. The holdout metrics compare it with lagged RAPM
persistence at the player-season target level; see [Box-Score RAPM
Prior](../models/box-score-prior.md).
