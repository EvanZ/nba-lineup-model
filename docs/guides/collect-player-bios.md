---
last_updated: "2026-07-27"
---

# Collect Player Bios

Collect the historical player index and one season bio table:

```bash
uv run nba-fetch-player-bios 2025-26
```

This is a synchronous two-request operation. Prefect is unnecessary because
the workflow does not fan out by player.

## Outputs

```text
data/raw/playerindex/2025-26.json
data/raw/playerindex/2025-26.meta.json
data/raw/leaguedashplayerbiostats/2025-26/regular.json
data/raw/leaguedashplayerbiostats/2025-26/regular.meta.json
data/catalog/players.parquet
data/curated/player_seasons/2025-26/regular/_manifest.json
data/curated/player_seasons/2025-26/regular/part-00000.parquet
```

Validated caches are reused by default. Bypass both with:

```bash
uv run nba-fetch-player-bios 2025-26 --refresh
```

The player catalog is a complete replacement from the historical index. The
player-season partition is atomically published through a validated temporary
directory and replacement.

## Other season types

The CLI accepts:

```bash
uv run nba-fetch-player-bios 2025-26 --season-type playoffs
```

Supported values are `regular`, `playoffs`, `preseason`, and `all_star`. The
historical player catalog remains common; each season type has an independent
raw response and normalized partition.

## Read the data

```python
import pandas as pd

players = pd.read_parquet("data/catalog/players.parquet")
bios = pd.read_parquet(
    "data/curated/player_seasons/2025-26/regular"
)
```

Player IDs, team IDs, height, weight, and numeric draft values use nullable
64-bit integers. The part is self-contained and retains explicit season fields.

See [Player bios](../data/player-bios.md) for source mapping, missing-value
semantics, and the same-season leakage policy.
