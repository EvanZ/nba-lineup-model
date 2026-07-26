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
```

Raw and processed data are intentionally ignored by Git.

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
  models/       Ridge, tree, and later nonlinear models
  evaluation/   Validation and benchmark metrics
```

## Data Policy

Code, schemas, tests, and small fixtures belong in Git. Raw NBA responses, processed Parquet files, trained models, and large reports do not.
