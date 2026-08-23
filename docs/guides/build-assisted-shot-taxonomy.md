---
last_updated: "2026-08-21"
---

# Build Assisted Shot Profiles

Build the validated player-season panel, curated regular-season event
partitions, and processed player box scores first. Then run:

```bash
uv run nba-build-assisted-shot-taxonomy
```

For a faster historical build, write validated seasonal parts concurrently and
combine them automatically:

```bash
uv run nba-build-assisted-shot-taxonomy --workers 4
```

The command atomically refreshes `data/analytical/assisted_shot_taxonomy/`.
It creates player-season assisted, unassisted, and unknown-status made-shot
counts for rim, non-rim two, and three-point families. It also writes a
team-game reconciliation against official player box-score FGM, 2PM, 3PM, and
assists.

Validate the published contract with:

```bash
uv run python -c "from nba_lineup_model.modeling.assisted_shot_taxonomy import validate_assisted_shot_taxonomy; validate_assisted_shot_taxonomy('data/analytical/assisted_shot_taxonomy')"
```

Review `season_assisted_shot_reconciliation.parquet` before using these
features in a model. It is the source-of-truth audit for whether description
assist markers agree sufficiently with the official box score in each season.

## Resumable Historical Build

On a constrained machine, build one season at a time into ignored checkpoint
directories, then atomically combine the validated parts:

```bash
uv run nba-build-assisted-shot-taxonomy \
  --season 2025-26 \
  --output-dir artifacts/assisted_shot_taxonomy_parts/2025-26

uv run nba-build-assisted-shot-taxonomy \
  --combine-parts-dir artifacts/assisted_shot_taxonomy_parts
```

Repeat the first command for every season before combining. The combine command
requires exactly one validated part per season and checks that all parts use
the same source player-panel and processed player-box contracts.
