---
last_updated: "2026-07-30"
---

# Train RAPM + Transformer

Train the single-season frozen-RAPM residual model with:

```bash
uv run nba-train-rapm-transformer 2025-26
```

By default, the command resolves the latest validated RAPM run for the season.
Pin the source explicitly for a reproducible comparison:

```bash
uv run nba-train-rapm-transformer 2025-26 \
  --rapm-run-id baseline-2025-26-20260727T230533Z-72eac627
```

The command first builds the
[RAPM base-prediction mart](../data/rapm-base-predictions.md), then selects
Transformer optimization settings, evaluates three predetermined seeds on the
untouched regular-season holdout, and refits all three seeds on the complete
regular season for playoff inference.

## Default architecture

| Setting | Value |
| --- | ---: |
| Tokens | 13 |
| Model width | 32 |
| Attention heads | 4 |
| Encoder layers | 2 |
| Feedforward width | 128 |
| Dropout | `0.1` |
| Positional encoding | None |

The sequence contains `[STATE]`, `[OFFENSE]`, five offense players,
`[DEFENSE]`, and five defense players. Player and role embeddings are added.
The `[STATE]` token also receives the home-offense sign. Without positional
encoding, shuffling players within offense or defense cannot change the
prediction.

The output is

\[
\widehat y_i
= \widehat y_i^{RAPM}
+ f_\theta(\text{lineup}_i,s_i).
\]

The residual output layer starts at zero, so a newly initialized model predicts
RAPM exactly.

## Selection budget

The CPU-conscious first search contains four candidates:

- learning rate: `0.0003`, `0.001`;
- AdamW weight decay: `0`, `0.01`;
- at most 10 epochs;
- early-stopping patience of three;
- batch size 8,192;
- the same three expanding validation folds as every other model.

Candidate ranking uses validation-possession-weighted MSE across all folds. The
winning candidate's best epoch from the latest fold determines each fixed-seed
refit duration. Seeds 17, 18, and 19 are declared before holdout evaluation;
seed 17 is always the Leaderboard checkpoint.

The current 13-token exemplar is
`rapm-transformer-2025-26-20260729T233233Z-e316a73e`. It selected learning
rate `0.0003`, weight decay `0.01`, and one refit epoch. Its predetermined
seed-17 regular-holdout possession RMSE was `1.199526`, compared with
`1.199460` for its frozen RAPM base. See
[Neural Networks](../models/neural-networks.md#2025-26-result) for seed
stability, residual magnitude, playoff results, and the planned role-pooled
ablation.

Use a small smoke run with:

```bash
uv run nba-train-rapm-transformer 2025-26 \
  --max-epochs 1 \
  --patience 0 \
  --learning-rate 0.001 \
  --weight-decay 0.01 \
  --d-model 8 \
  --attention-heads 2 \
  --transformer-layers 1 \
  --feedforward-dim 16 \
  --dropout 0 \
  --no-progress-bar
```

## Runtime options

| Option | Default |
| --- | ---: |
| `--batch-size` | 8,192 |
| `--max-epochs` | 10 |
| `--patience` | 3 |
| `--seed` | 17 |
| `--accelerator` | `cpu` |
| `--num-workers` | 0 |
| `--d-model` | 32 |
| `--attention-heads` | 4 |
| `--transformer-layers` | 2 |
| `--feedforward-dim` | 128 |
| `--dropout` | `0.1` |

Repeat `--learning-rate` or `--weight-decay` to replace the corresponding
default grid.

## Model artifacts

Each atomic run is stored under:

```text
artifacts/models/rapm_transformer/<season>/<run-id>/
```

| Artifact | Purpose |
| --- | --- |
| `selection_model.ckpt` | Winning latest-fold checkpoint |
| `test_model.ckpt` | Canonical final-training checkpoint |
| `model.ckpt` | Canonical all-regular-season checkpoint |
| `test_model_seed_<seed>.ckpt` | Alternate fixed-seed holdout checkpoint |
| `model_seed_<seed>.ckpt` | Alternate fixed-seed all-season checkpoint |
| `hyperparameter_trials.parquet` | Candidate results on every fold |
| `hyperparameter_summary.parquet` | Weighted candidate ranking and winner |
| `training_history.parquet` | Epoch metrics for search and refits |
| `test_metrics.parquet` | Mean, RAPM, and canonical Transformer metrics |
| `seed_metrics.parquet` | Holdout metrics for all fixed seeds |
| `test_predictions.parquet` | RAPM, residual, and total holdout predictions |
| `seed_predictions.parquet` | Holdout decomposition for every fixed seed |
| `lineup_residuals.parquet` | Residual summaries by observed lineup pair |
| `game_splits.parquet` | Exact chronological game membership |
| `rapm_player_coefficients.parquet` | Frozen all-season RAPM coefficients |
| `rapm_state.json` | Frozen all-season intercept and offense mean |
| `player_columns.json` | NBA player ID to embedding-row mapping |
| `model_parameters.json` | Token, architecture, selection, and leakage contract |
| `manifest.json` | Source hashes, versions, dimensions, seeds, and integrity |

## Inspect results

```bash
uv run python - <<'PY'
import json
from pathlib import Path

import pandas as pd

root = Path("artifacts/models/rapm_transformer/2025-26")
run_id = json.loads((root / "latest.json").read_text())["run_id"]
run = root / run_id

print(pd.read_parquet(run / "test_metrics.parquet").to_string(index=False))
print(pd.read_parquet(run / "seed_metrics.parquet").to_string(index=False))
print(pd.read_parquet(run / "hyperparameter_summary.parquet").to_string(index=False))
PY
```

After training, regenerate the common regular-holdout and playoff report:

```bash
uv run nba-evaluate-models 2025-26
```

Both completed runs are indexed in MLflow. Run focused correctness checks with:

```bash
uv run pytest -q \
  tests/test_transformer_modeling.py \
  tests/test_model_evaluation.py
```
