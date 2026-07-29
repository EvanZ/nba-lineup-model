# Train Neural Models

The initial command builds the single-lineup possession dataset and trains the
additive neural RAPM boundary:

```bash
uv run nba-train-neural-rapm 2025-26
```

The default is deterministic CPU training with batch size 2,048, at most 30
selection epochs, and early-stopping patience of five epochs. Learning rate
and AdamW weight decay are selected by a grid search over all expanding
validation folds.

## Hyperparameter selection

The default grid contains 20 candidates:

- learning rate: `0.0001`, `0.0003`, `0.001`, `0.003`;
- weight decay: `0`, `0.001`, `0.01`, `0.1`, `1`.

For candidate \(c\) and expanding fold \(f\), early stopping records the best
validation possession MSE \(L_{c,f}\). Candidates are ranked by
validation-possession-weighted MSE:

\[
L_c =
\frac{\sum_f N_f L_{c,f}}
{\sum_f N_f},
\]

where \(N_f\) is the eligible validation-possession count in fold \(f\).
Deterministic candidate order breaks exact ties. The selected candidate uses
its best epoch count from the most recent fold for the final-train and
full-season refits.

The final regular-season holdout and playoffs are never used to select the
learning rate, weight decay, or epoch count.

## Runtime options

Use a short run to validate a new environment:

```bash
uv run nba-train-neural-rapm 2025-26 \
  --max-epochs 2 \
  --patience 1 \
  --no-progress-bar
```

Benchmark Apple MPS explicitly:

```bash
uv run nba-train-neural-rapm 2025-26 --accelerator mps
```

Do not compare CPU and MPS wall times from different batch sizes or model
configurations.

Relevant controls include:

| Option | Default |
| --- | ---: |
| `--batch-size` | 2,048 |
| `--max-epochs` | 30 |
| `--patience` | 5 |
| `--learning-rate` | `0.0001`, `0.0003`, `0.001`, `0.003` |
| `--weight-decay` | `0`, `0.001`, `0.01`, `0.1`, `1` |
| `--seed` | 17 |
| `--accelerator` | `cpu` |
| `--num-workers` | 0 |
| `--minimum-ranking-possessions` | 500 |

Repeat `--learning-rate` or `--weight-decay` to provide a custom grid. One of
each produces a single-candidate smoke run:

```bash
uv run nba-train-neural-rapm 2025-26 \
  --learning-rate 0.001 \
  --weight-decay 0.01
```

## Analytical output

The command rebuilds and validates:

```text
data/analytical/neural_possessions/<season>/regular/
  _manifest.json
  part-00000.parquet
```

The manifest records the source segment partition, hashes, included possession
count, excluded multi-segment count, player count, and home/away offense
counts.

## Model artifacts

Each atomic run is written under:

```text
artifacts/models/neural_rapm/<season>/<run-id>/
```

| Artifact | Purpose |
| --- | --- |
| `selection_model.ckpt` | Winning candidate's best latest-fold checkpoint |
| `test_model.ckpt` | Refit used for untouched test predictions |
| `model.ckpt` | Full-season refit used for rankings |
| `hyperparameter_trials.parquet` | Best result for every candidate and fold |
| `hyperparameter_summary.parquet` | Weighted candidate scores, ranks, and winner |
| `training_history.parquet` | Epoch losses for every grid fit and both refits |
| `test_metrics.parquet` | Mean and additive-neural test metrics |
| `test_predictions.parquet` | Possession predictions in offense and home frames |
| `player_rankings.parquet` | Centered embeddings and neural RAPM values |
| `game_splits.parquet` | Exact chronological game membership |
| `player_columns.json` | NBA player ID to embedding-row mapping |
| `model_parameters.json` | Equation, target, conversion, and fitted summaries |
| `manifest.json` | Versions, hashes, hyperparameters, and artifact integrity |

`latest.json` points to the newest completed run. Temporary directories are
removed after a failed run and never become `latest`.

## Inspect results

```bash
uv run python - <<'PY'
import json
from pathlib import Path

import pandas as pd

root = Path("artifacts/models/neural_rapm/2025-26")
run_id = json.loads((root / "latest.json").read_text())["run_id"]
run = root / run_id

print(pd.read_parquet(run / "test_metrics.parquet").to_string(index=False))
print(
    pd.read_parquet(run / "player_rankings.parquet")
    .query("exposure_eligible")
    .head(25)
    .loc[:, ["eligible_rank", "player_name", "neural_rapm", "possessions"]]
    .to_string(index=False)
)
PY
```

The first review should compare predictive skill, game-margin error, and rank
stability against ridge RAPM. The additive model is a pipeline and objective
boundary, not yet the nonlinear lineup model. Regenerate the common
regular-holdout and playoff scoreboard after a completed model run:

```bash
uv run nba-evaluate-models 2025-26
```

See the [Leaderboard](../models/leaderboard.md) for the frozen cohort and
metric definitions.
