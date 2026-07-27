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
