# Getting Started

## Prerequisites

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Network access when a requested NBA game is not already in the raw cache

## Install dependencies

Install runtime and development dependencies:

```bash
uv sync --group dev
```

Include the documentation toolchain:

```bash
uv sync --group dev --group docs
```

Activation of `.venv` is optional. Commands prefixed with `uv run` use the
project environment automatically.

## Verify the repository

```bash
uv run pytest
uv run ruff check src tests
```

## Process one game

```bash
uv run nba-build-game 0022000180
```

The command reuses valid cached responses by default. Pass `--refresh` to fetch
both source documents again.

## Run the cross-season audit

```bash
uv run nba-audit-games config/audit_manifest.json
```

## Discover a season

```bash
uv run nba-discover-season 2025-26
```

The command reads the season-parameterized NBA Stats schedule endpoint directly,
preserves the raw response, and updates `data/catalog/games.parquet`. Existing
rows from other seasons are retained. Pass `--refresh` to bypass the schedule
cache.

## Fetch a season

Start with one game:

```bash
uv run nba-fetch-season 2025-26 --limit 1 --max-workers 1
```

Then fetch every final catalog game:

```bash
uv run nba-fetch-season 2025-26 --max-workers 4
```

The Prefect flow runs locally, validates cached documents before skipping work,
and writes a durable Parquet fetch manifest. See
[Fetch a season](guides/fetch-season.md) for filters, retries, and resume
semantics.

Inspect orchestration history in the local Prefect UI:

```bash
uv run prefect server start
```

Open `http://127.0.0.1:4200`. See
[Use the Prefect web UI](guides/prefect-ui.md) to connect future runs to the
persistent server instead of the automatic temporary API.

## Process a season

Run a representative local-data pilot:

```bash
uv run nba-process-season 2025-26 \
  --sample-per-stratum 3 \
  --seed 7
```

Then process every final catalog game:

```bash
uv run nba-process-season 2025-26 --max-workers 4
```

See [Process a season](guides/process-season.md) for quality gates, outputs,
checkpointing, and resume semantics.

## Compact a season

After every final game has a successful build and a passing or warning quality
record, create the season-level analytical datasets:

```bash
uv run nba-compact-season 2025-26 --max-workers 4
```

The flow writes deterministic Parquet shards and partition manifests under
`data/curated/`. Re-running the command validates and skips unchanged
partitions. See [Compact a season](guides/compact-season.md) for the row-level
provenance contract, analytical reads, and resume behavior.

## Collect player bios

Fetch the historical player index and regular-season bio table:

```bash
uv run nba-fetch-player-bios 2025-26
```

This uses two direct NBA Stats requests rather than one request per player. It
writes a historical player catalog plus a season-specific bio partition without
including same-season performance columns. See
[Collect player bios](guides/collect-player-bios.md) for fields, cache behavior,
and leakage policy.

## Import a canonical game catalog

```bash
uv run nba-import-catalog source_games.csv \
  --output data/catalog/games.parquet
```

The input must follow the canonical fields documented in
[Season catalog and storage](data/season-storage.md).

## Preview documentation

```bash
uv run --group docs zensical serve
```

Open `http://127.0.0.1:8000`.

Build the static site in strict mode:

```bash
uv run --group docs zensical build --strict
```

Rendered files are written to `site/` and are not tracked by Git.

See [Build and serve the documentation](guides/documentation.md) for live
preview options, clean builds, and the documentation layout.
