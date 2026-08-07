---
last_updated: "2026-08-01"
---

# Train Age-Informed Prior RAPM

This command fits possession-level prior-centered RAPM using the frozen output
of a forward RAPM aging model. It does not refit the aging model and rejects a
prior table containing target-season outcomes.

## Prerequisites

Build the multi-season player panel and train the target-season aging model:

```bash
uv run nba-train-aging-model --holdout-season 2025-26
```

The aging run must publish a label-free `player_priors.parquet` for the same
target season. Pin its immutable run ID explicitly:

```bash
uv run nba-train-aging-prior-rapm \
  --season 2025-26 \
  --aging-run-id aging-2025-26-20260801T220356Z-4de5f001
```

Without `--aging-run-id`, the command uses `artifacts/models/aging/{season}/latest.json`.

## Contract

For each player in the target RAPM design, the aging forecast is the prior
mean. Players absent from the aging table receive the explicit zero cold-start
prior. Lambda is selected only on chronological folds within the target regular
season. The final regular holdout and playoff outputs use the model fitted on
the first 1,044 2025-26 regular-season games; playoff outcomes never enter
fitting or selection.

## Outputs

Each immutable run is written under:

```text
artifacts/models/aging_prior_rapm/{season}/{run_id}/
```

| File | Purpose |
| --- | --- |
| `aging_player_priors.parquet` | Pinned label-free aging-model input |
| `holdout_metrics.parquet` | Stint-level regular-holdout results |
| `holdout_predictions.parquet` | Regular-holdout stint predictions |
| `player_rankings.parquet` | Frozen final-training coefficients, priors, and adjustments |
| `final_training_player_coefficients.parquet` | Coefficients fit on the first 1,044 games |
| `final_training_state.json` | Frozen intercept and selected lambda |
| `frozen_playoff_metrics.parquet` | Common eligible-possession playoff metrics without refitting |
| `frozen_playoff_predictions.parquet` | Frozen playoff predictions |
| `metadata.json` / `manifest.json` | Aging-run provenance, hashes, split identity, and artifact integrity |

The completed run is indexed in MLflow under the `aging_prior_rapm` kind.
