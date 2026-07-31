# nba-lineup-model

Experimental NBA lineup modeling stack for context-sensitive player and lineup value.

The first target is a reproducible data spine:

1. Fetch direct NBA source JSON.
2. Cache raw responses locally with provenance metadata.
3. Normalize source actions into canonical basketball events.
4. Reconstruct event-level lineups and stable lineup stints.
5. Segment possessions and intersect them with lineup stints.
6. Build baseline linear models before nonlinear lineup models.

## Setup

```bash
uv sync --group dev
```

## Build One Game

```bash
uv run nba-build-game 0022000180
```

The command stores each NBA response body byte-for-byte under `data/raw/`. Fetch
metadata and a SHA-256 digest are stored in a separate `.meta.json` sidecar. It then writes:

```text
data/processed/
  events/{game_id}.parquet
  players/{game_id}.parquet
  event_lineups/{game_id}.parquet
  lineup_stints/{game_id}.parquet
  possessions/{game_id}.parquet
  possession_segments/{game_id}.parquet
```

Raw and processed data are intentionally ignored by Git.

`possessions` contains one row per team possession. `possession_segments` splits
those rows only when a substitution changes either lineup, so a free-throw trip
can remain one possession while contributing to multiple fixed-lineup samples.

## Audit Across Seasons

Run the committed 21-game, seven-season audit matrix:

```bash
uv run nba-audit-games config/audit_manifest.json
```

The command reuses cached raw responses when available and writes compact reports
to `data/audit/games.parquet` and `data/audit/summary.parquet`. A game fails only
on reconstruction or exact accounting invariants. Differences from the approximate
box-score possession formula are reported as warnings.

The committed matrix covers 2019-20 through 2025-26 with three games per season:
one same-ordinal regular-season game, one Finals opener, and one confirmed overtime
game. The overtime stratum includes double-overtime games.

To generate a larger deterministic manifest from a Parquet or CSV game catalog:

```bash
uv run nba-sample-audit data/external/game_catalog.parquet \
  --games-per-stratum 25 \
  --seed 7 \
  --output config/audit_manifest_sample.json
```

The catalog must contain `game_id`, `season`, and `season_type`. An optional
`sample_group` column supports strata such as overtime, playoffs, and feed-edge
cases.

## Season Data Contracts

Discover a season directly from the NBA Stats schedule endpoint:

```bash
uv run nba-discover-season 2025-26
```

The exact response is cached at
`data/raw/scheduleleaguev2/2025-26.json`; the normalized multi-season inventory
is written to `data/catalog/games.parquet`. Re-running discovery replaces that
season while preserving other catalog seasons.

Fetch play-by-play and boxscore responses for every final catalog game with:

```bash
uv run nba-fetch-season 2025-26 --max-workers 4
```

The local Prefect flow uses one task per game, resumes from validated raw cache
files, retries transient source failures, and appends terminal outcomes to
`data/manifests/fetches.parquet`. Use `--limit 1 --max-workers 1` for a smoke
test.

Process a representative, validation-gated pilot entirely from local raw data:

```bash
uv run nba-process-season 2025-26 \
  --sample-per-stratum 3 \
  --seed 7
```

Process every final catalog game with:

```bash
uv run nba-process-season 2025-26 --max-workers 4
```

The Prefect flow writes six per-game Parquet tables, checkpoints terminal
attempts in `data/manifests/builds.parquet`, and maintains canonical game and
aggregate quality reports under `data/quality/`.

Compact every quality-gated game into season-level analytical datasets:

```bash
uv run nba-compact-season 2025-26 --max-workers 4
```

This lossless Prefect flow writes self-contained Parquet shards under
`data/curated/{table}/{season}/{season_type}/`. Each partition has a hashed
manifest, uses deterministic 100-game shards by default, and carries catalog,
quality, build, processing-code, and source-hash provenance on every row.

Collect the historical player universe and season-specific physical and
background fields directly from NBA Stats:

```bash
uv run nba-fetch-player-bios 2025-26
```

The command makes two bulk requests, preserves both response bodies under
`data/raw/`, writes `data/catalog/players.parquet`, and publishes leakage-safe
season rows under `data/curated/player_seasons/2025-26/regular/`.

Run the regular-season historical pipeline from 2019-20 through 2024-25 with a
checkpointed parent manifest:

```bash
uv run nba-backfill-history \
  --run-id history-2019-2024 \
  --max-workers 2 \
  --min-request-interval 0.5
```

Rerunning the same command skips completed stages and resumes from validated
game-level artifacts. See the
[historical backfill guide](docs/guides/backfill-history.md) for stage ranges,
failure policy, and Prefect behavior.

Preserve cloud-backed Stats V3 responses for historical regular-season games
whose liveData CDN artifacts are missing:

```bash
uv run nba-fetch-stats-history \
  --season 2019-20 \
  --cdn-missing-only \
  --run-id stats-gaps-2019-20
```

This fallback flow tracks play-by-play and box scores independently under
`data/raw/stats/`. See the
[historical Stats guide](docs/guides/fetch-stats-history.md) for full-history,
endpoint-specific, and rotation commands.

Build the canonical regular-season modeling stints and train the mean, team,
and one-number RAPM baselines:

```bash
uv run nba-train-rapm 2025-26
```

The command selects ridge regularization with expanding chronological game
folds, evaluates once on the final 15% of games, and then refits the rankings
on the complete regular season. Modeling tables are written under
`data/analytical/`; reproducible run artifacts are written under
`artifacts/models/rapm/`.

After historical RAPM runs exist, build the leakage-safe player-season feature
boundary shared by future box-score, aging, and neural models:

```bash
uv run nba-build-player-season-panel \
  2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 2025-26
```

The same-season table retains research outcomes. The transition table exposes
only prior-season performance for each target-season player, with explicit
cold starts for players without a prior row.

Train the first forward-only aging prior, using the latest transition season as
an untouched holdout:

```bash
uv run nba-train-aging-model
```

The immutable run compares zero, training-mean, prior-RAPM persistence, and
aging-ridge predictions. Its label-free `player_priors.parquet` is the future
handoff to prior-centered RAPM and Transformer player tokens.

Run stability, context, influence, and possession-allocation diagnostics
against the latest validated RAPM run:

```bash
uv run nba-diagnose-rapm 2025-26
```

Immutable diagnostic reports are written under `artifacts/reports/rapm/`.
See the [RAPM guide](docs/guides/train-rapm.md) for interpretation and output
contracts.

Generate the reproducible one-season diagnostic case study for the
documentation:

```bash
uv run --group docs nba-build-rapm-case-study 2025-26
```

Validate and normalize an already canonical CSV or Parquet game catalog:

```bash
uv run nba-import-catalog source_games.csv \
  --output data/catalog/games.parquet
```

Season operations retain per-game raw and processed artifacts for retries, while
validated outputs compact into `data/curated/{table}/{season}/{season_type}/`
partitions. Terminal build attempts are represented by a typed Parquet ledger.

## Documentation

Install the documentation dependencies and start the Zensical preview server:

```bash
uv sync --group docs
uv run --group docs zensical serve
```

Build the documentation with strict validation:

```bash
uv run --group docs zensical build --strict
```

See the
[documentation workflow](docs/guides/documentation.md) for preview options,
clean builds, configuration, and output locations.

## Experiment Tracking

Model and evaluation CLIs index completed immutable runs in a project-local
MLflow SQLite store. Backfill the latest run from every model and report family:

```bash
uv run nba-sync-mlflow
```

Start the local UI:

```bash
uv run mlflow server \
  --backend-store-uri "sqlite:///$(pwd)/artifacts/mlflow/mlflow.db" \
  --default-artifact-root "file://$(pwd)/artifacts/mlflow/artifacts" \
  --no-serve-artifacts \
  --host 127.0.0.1 \
  --port 5000
```

Open `http://127.0.0.1:5000`. See the
[MLflow guide](docs/guides/mlflow.md) for the storage and synchronization
contract.

Train the categorical CatBoost lineup baseline and refresh the common
Leaderboard with:

```bash
uv run nba-train-catboost 2025-26
uv run nba-evaluate-models 2025-26
```

See [Tree Models](docs/models/tree-models.md) for the feature contract and
[Train CatBoost](docs/guides/train-catboost.md) for artifacts and controls.

Train the frozen-RAPM plus Transformer residual with:

```bash
uv run nba-train-rapm-transformer 2025-26
```

The command builds stage-specific RAPM base predictions before training the
position-free lineup attention residual. See the
[Transformer guide](docs/guides/train-transformer.md) for the leakage and
artifact contracts.

## Test

```bash
uv run pytest
```

## Project Layout

```text
src/nba_lineup_model/
  ingest/       Direct NBA source clients and raw JSON cache
  events/       Canonical typed event stream
  normalize/    Source table normalization
  lineups/      On-court reconstruction
  possessions/  Possession segmentation
  audit/        Cross-season manifests, sampling, and invariant reports
  players/      Historical identities and season-specific player bios
  season/       Game catalogs, build ledgers, and curated dataset layout
  flows/        Thin Prefect orchestration around project-owned operations
  modeling/     Modeling marts, chronological splits, and training runs
  models/       Ridge, tree, and later nonlinear models
  evaluation/   Validation and benchmark metrics
```

## Data Policy

Code, schemas, tests, and small fixtures belong in Git. Raw NBA responses, processed Parquet files, trained models, and large reports do not.
