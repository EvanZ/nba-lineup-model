# Track Experiments with MLflow

MLflow is the searchable experiment index for model training and evaluation.
The project run directory remains the canonical reproducibility artifact.
MLflow does not replace typed manifests, Parquet outputs, checkpoints, hashes,
or the Leaderboard.

## Storage

The default local configuration uses:

| Component | Path |
| --- | --- |
| Metadata database | `artifacts/mlflow/mlflow.db` |
| Copied run artifacts | `artifacts/mlflow/artifacts/` |
| Canonical model runs | `artifacts/models/` |
| Canonical evaluation runs | `artifacts/reports/` |

The entire `artifacts/mlflow/` runtime store is ignored by Git except for its
`.gitkeep`. SQLite is the default MLflow backend and is appropriate for this
local workflow. A tracking server is optional because the Python client can
write directly to SQLite.

## Automatic tracking

These commands index a run after its immutable artifact directory has been
written and validated:

- `nba-train-rapm`;
- `nba-train-bayesian-rapm`;
- `nba-train-neural-rapm`;
- `nba-train-deep-sets`;
- `nba-evaluate-models`.

The command output includes the MLflow run ID. If MLflow logging fails, the
canonical run directory already exists and can be indexed later with
`nba-sync-mlflow`.

Each season has two experiments:

| Experiment | Contents |
| --- | --- |
| `nba-lineup-model-<season>-models` | Fitted model runs |
| `nba-lineup-model-<season>-reports` | Leaderboards and diagnostics |

Hyperparameter candidates from neural searches are nested child runs. Fold
validation MSE and selected epochs are metric histories indexed by fold.

## Start the UI

From the repository root:

```bash
uv run mlflow server \
  --backend-store-uri "sqlite:///$(pwd)/artifacts/mlflow/mlflow.db" \
  --default-artifact-root "file://$(pwd)/artifacts/mlflow/artifacts" \
  --no-serve-artifacts \
  --host 127.0.0.1 \
  --port 5000
```

Open `http://127.0.0.1:5000`. Stop the server with `Ctrl+C`.

The local server and direct Python clients point to the same SQLite database
and artifact directory. Training does not require the server to be running.
Binding to `127.0.0.1` keeps the unauthenticated development UI local to the
machine.

## Backfill existing runs

Index the run referenced by every `latest.json` pointer:

```bash
uv run nba-sync-mlflow
```

Index every immutable run, including superseded experiments:

```bash
uv run nba-sync-mlflow --all
```

Restrict discovery to one season:

```bash
uv run nba-sync-mlflow --season 2025-26
```

Index one explicit run:

```bash
uv run nba-sync-mlflow \
  artifacts/models/deep_sets/2025-26/<run-id>
```

Synchronization is idempotent. MLflow stores the SHA-256 of `manifest.json`
under the `project.manifest_sha256` tag. A repeated sync reuses the existing
MLflow run. If the same project run ID has a different manifest hash,
synchronization fails rather than merging incompatible artifacts.

## Logged data

The primary MLflow run contains:

- flattened manifest parameters;
- flattened `model_parameters.json` and `resolved_parameters.json` values;
- metrics from model, seed, comparison, calibration, and Leaderboard tables;
- selected hyperparameter-summary metrics;
- a copy of the complete immutable run under `immutable_run/`;
- project run ID, season, run kind, source directory, creation time, and
  manifest hash tags.

The original files remain authoritative. MLflow's artifact copy makes the UI
self-contained and supports later migration to a shared tracking server.

For CatBoost, `resolved_parameters.json` will contain `get_all_params()`,
including dynamically selected defaults. The manifest will distinguish:

- the requested iteration ceiling;
- best validation iteration;
- saved tree count;
- frozen refit tree count;
- explicitly requested parameters;
- resolved CatBoost parameters.

## Configuration

Point clients at a running or remote tracking server:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Change the direct local storage root:

```bash
export NBA_MLFLOW_ROOT=/absolute/path/to/mlflow
```

Disable automatic post-run indexing:

```bash
export NBA_MLFLOW_TRACKING_ENABLED=0
```

The explicit `nba-sync-mlflow` command still runs when automatic tracking is
disabled. This allows recovery after a server outage or temporary tracking
configuration problem.

## Run interpretation

Resolved defaults are not the same as searched hyperparameters. For example,
CatBoost can calculate a default learning rate from the dataset and requested
iteration count. MLflow records that value as resolved configuration. It
should not be described as the winner of a hyperparameter search unless
multiple candidates were actually evaluated under the documented validation
protocol.

The implementation follows MLflow's documented
[local database workflow](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database)
and [tracking server architecture](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).
