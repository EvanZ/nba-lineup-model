# Train CatBoost

Train the categorical player-state baseline with:

```bash
uv run nba-train-catboost 2025-26
```

The command builds or validates the shared single-lineup possession dataset,
fits three chronological selection folds, evaluates an untouched final
regular-season holdout, and fits one model on the full regular season for
playoff inference.

The default 2025-26 CPU run takes about two minutes on the reference
development machine.

## Runtime options

| Option | Default | Purpose |
| --- | ---: | --- |
| `--iterations` | 1,000 | Maximum trees per selection fold |
| `--cv-folds` | 3 | Expanding chronological validation folds |
| `--validation-fraction` | `0.1` | Validation share within pre-test games |
| `--test-fraction` | `0.15` | Untouched final regular-season share |
| `--seed` | 17 | CatBoost random seed |

Use a short environment smoke test with:

```bash
uv run nba-train-catboost 2025-26 --iterations 10
```

The model intentionally exposes no other CatBoost parameters in this first
slice. CatBoost resolves unspecified defaults, and the run records all of them
in `resolved_parameters.json`. This is defaults-first training, not a hidden
hyperparameter search.

## Feature representation

Each player has one categorical feature with values:

| Value | State |
| ---: | --- |
| 0 | Absent |
| 1 | On offense |
| 2 | On defense |

A final binary categorical feature records whether the home team is on
offense. Every feature is passed directly to CatBoost; no external one-hot
encoder is fitted.

See [Tree Models](../models/tree-models.md) for the equations, invariance
contract, and interpretation boundary.

## Model artifacts

Each atomic run is stored under:

```text
artifacts/models/catboost/<season>/<run-id>/
```

| Artifact | Purpose |
| --- | --- |
| `selection_model.cbm` | Latest-fold model truncated to its best iteration |
| `test_model.cbm` | Final-training refit used for untouched holdout predictions |
| `model.cbm` | Full-regular-season refit used for playoff predictions |
| `fold_metrics.parquet` | Validation metrics and selected trees by fold |
| `training_history.parquet` | Learn and validation RMSE at every iteration |
| `test_metrics.parquet` | Mean and CatBoost holdout metrics |
| `test_predictions.parquet` | Holdout predictions in offense and home frames |
| `feature_importance.parquet` | PredictionValuesChange importance by feature |
| `game_splits.parquet` | Exact chronological game membership |
| `player_columns.json` | NBA player ID to categorical feature mapping |
| `resolved_parameters.json` | Requested and all fitted CatBoost parameters |
| `model_parameters.json` | Feature, target, and selection contract |
| `manifest.json` | Versions, hashes, counts, and artifact integrity |

`latest.json` points to the newest completed run. A failed or interrupted run
does not replace it.

## Inspect results

```bash
uv run python - <<'PY'
import json
from pathlib import Path

import pandas as pd

root = Path("artifacts/models/catboost/2025-26")
run_id = json.loads((root / "latest.json").read_text())["run_id"]
run = root / run_id

print(pd.read_parquet(run / "test_metrics.parquet").to_string(index=False))
print(
    pd.read_parquet(run / "feature_importance.parquet")
    .head(25)
    .loc[:, ["rank", "feature_name", "player_name", "feature_importance"]]
    .to_string(index=False)
)
print(json.loads((run / "resolved_parameters.json").read_text())["selection"])
PY
```

Feature importance has no sign and must not be reported as player value. Use it
to inspect the fitted tree model, then use contextual counterfactuals or SHAP
analysis for player-level interpretation.

## Refresh evaluation

After training, regenerate the common regular-holdout and playoff report:

```bash
uv run nba-evaluate-models 2025-26
```

Both commands register their immutable artifacts in the project-local MLflow
store. See [Track experiments with MLflow](mlflow.md) for the UI and backfill
workflow.

Run focused correctness checks with:

```bash
uv run pytest -q tests/test_catboost_modeling.py tests/test_model_evaluation.py
```
