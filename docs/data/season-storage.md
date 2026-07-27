# Season Catalog and Storage

Season processing uses separate contracts for discovery, game-level execution,
and analytical data.

## Storage layers

| Layer | Granularity | Role |
| --- | --- | --- |
| Raw cache | Endpoint and game | Byte-preserved NBA responses and provenance |
| Processed | Contract and game | Retryable reconstruction artifacts |
| Catalog | One row per game | Canonical season inventory |
| Build ledger | One row per attempt | Durable terminal execution outcomes |
| Curated | Season and season type | Compact analytical datasets |

The catalog and ledger are operational metadata. They do not replace raw source
provenance or reconstruction audit reports.

## Game catalog

The conventional catalog path is:

```text
data/catalog/games.parquet
```

Every row contains:

- a lexical ten-character `game_id`;
- `season`, normalized `season_type`, schedule date, and optional UTC game time;
- game status, teams, period count, and overtime status;
- source status fields, source URL, and UTC fetch time;
- a row-level schema version.

Season values use `YYYY-YY` and the suffix must be the immediately following
year. Schedule dates must fall within one of those two calendar years. Team IDs
are integer fields; game IDs remain strings.

Discover and normalize a season directly from the NBA Stats schedule endpoint:

```bash
uv run nba-discover-season 2025-26
```

The command replaces that season in the catalog while preserving rows from
other seasons. See [Season discovery](season-discovery.md) for source mapping,
cache behavior, and overtime semantics.

Import an already canonical CSV or Parquet catalog with:

```bash
uv run nba-import-catalog source_games.csv \
  --output data/catalog/games.parquet
```

The importer validates every row, rejects duplicate game IDs, orders games by
date and ID, and writes atomically. It remains useful for project-owned or
manually audited canonical inventories.

## Build ledger

The build ledger is an append-oriented history of terminal game attempts. A
conventional path is:

```text
data/manifests/builds.parquet
```

Each record identifies its run, attempt, game, season partition, timestamps,
cache policy, terminal status and stage, source hashes, processing counts, and
failure or skip details.

Statuses are:

| Status | Required state |
| --- | --- |
| `succeeded` | Complete stage, source hashes, all counts, and written outputs |
| `failed` | Non-complete stage plus exception type and message |
| `skipped` | Preflight stage plus a skip reason |

Attempt IDs and game attempt numbers must be unique. Durations must agree with
UTC start and finish timestamps. Ledger appends use an atomic whole-file rewrite
and therefore require one owning writer. Parallel game workers return terminal
records to that writer rather than modifying the ledger themselves.

!!! note "Orchestration boundary"

    Prefect or another orchestrator may track task state and retries. The Parquet
    ledger remains the portable project-owned history and can be regenerated or
    compared independently of the orchestration database.

## Curated layout

Validated game artifacts compact into Hive-style partitions:

```text
data/curated/
  events/season=2025-26/season_type=regular/part-00000.parquet
  possessions/season=2025-26/season_type=regular/part-00000.parquet
  possession_segments/season=2025-26/season_type=playoffs/part-00000.parquet
```

The same layout applies to `players`, `event_lineups`, and `lineup_stints`.
Every curated row retains `game_id`, `season`, and `season_type`.

Consumers must read the partition directory as a Parquet dataset. A partition
may contain one or several part files.
