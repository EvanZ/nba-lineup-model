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
  season/       Game catalogs, build ledgers, and curated dataset layout
  flows/        Thin Prefect orchestration around project-owned operations
  models/       Ridge, tree, and later nonlinear models
  evaluation/   Validation and benchmark metrics
```

## Data Policy

Code, schemas, tests, and small fixtures belong in Git. Raw NBA responses, processed Parquet files, trained models, and large reports do not.
