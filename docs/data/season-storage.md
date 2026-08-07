---
last_updated: "2026-07-27"
---

# Season Catalog and Storage

Season processing uses separate contracts for discovery, game-level execution,
and analytical data.

## Storage layers

| Layer | Granularity | Role |
| --- | --- | --- |
| Raw cache | Endpoint and game | Byte-preserved NBA responses and provenance |
| Processed | Contract and game | Retryable reconstruction artifacts |
| Catalog | One row per game | Canonical season inventory |
| Fetch manifest | One row per game and fetch run | Durable raw acquisition outcomes |
| Build ledger | One row per attempt | Durable reconstruction outcomes |
| Curated | Season and season type | Compact analytical datasets |
| Analytical | Model, season, and season type | Experiment-specific modeling marts |
| Player catalog | One row per player | Historical identity and listed attributes |
| Player seasons | Player and season type | Physical and background priors |

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

## Fetch manifest

The Prefect season fetch flow appends terminal task outcomes to:

```text
data/manifests/fetches.parquet
```

The record contract includes project and Prefect run IDs, game partition keys,
UTC timing, cache provenance, exact-byte hashes and sizes, plus failure or skip
details. A successful or skipped record requires both validated raw artifacts.
A failed record retains evidence for either endpoint that completed before the
failure.

Records are unique by project run ID and game ID. The flow collects task results
and performs one atomic manifest write after the batch settles, avoiding
concurrent Parquet writers.

## Build ledger

The build ledger is an append-oriented history of terminal game attempts. A
conventional path is:

```text
data/manifests/builds.parquet
```

Each record identifies its run, attempt, game, season partition, timestamps,
cache policy, Prefect flow and task IDs, terminal status and stage, source
hashes, processing-code fingerprint, reconstruction counts, and failure or skip
details.

The fingerprint is scoped to raw-game validation, reconstruction, auditing,
and processed-table contract modules. Player reference and modeling code do not
invalidate completed game reconstructions.

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

    Prefect tracks live flow state, task state, concurrency, and retries. The
    Parquet fetch manifest and build ledger remain the portable project-owned
    history and can be inspected independently of the orchestration database.

## Quality reports

Season processing maintains:

```text
data/quality/games.parquet
data/quality/summary.parquet
```

`games.parquet` contains the latest quality result for each game, tied to exact
source hashes and a processing-code fingerprint. It records hard invariant
results, event, lineup, possession, and segment diagnostic counts, issue codes,
and Prefect run IDs. `event_warning_count` includes source-level normalization
flags such as a nonmonotonic source clock; the matching named flags remain in
`issue_codes`.

`summary.parquet` aggregates pass, warning, failure, and error counts by season
and season type. Quality records are checkpointed before matching build-ledger
records so an interrupted run cannot skip work based on incomplete validation
metadata.

Warnings are accepted outputs, not silent corrections. Consumers can filter
them by `status` or specific `issue_codes` when constructing stricter modeling
datasets.

## Curated layout

The season compaction flow losslessly combines validated per-game artifacts
into plain-directory partitions with self-contained Parquet files:

```text
data/curated/
  events/
    2025-26/
      regular/
        _manifest.json
        part-00000.parquet
        part-00001.parquet
  possessions/
    2025-26/
      regular/
        _manifest.json
```

The same layout applies to `players`, `event_lineups`, and `lineup_stints`.
The default shard size is 100 source games, not a target byte size, so part
boundaries are deterministic across tables and repeated builds.

Each Parquet row retains its source `game_id` and adds:

| Metadata | Meaning |
| --- | --- |
| `game_date`, `game_time_utc` | Canonical schedule time |
| `catalog_home_*`, `catalog_away_*` | Canonical team IDs and tricodes |
| `quality_status` | `pass` or `warning` |
| `quality_issue_codes_json` | Canonical JSON array of named quality issues |
| `quality_recorded_at`, `quality_run_id` | Quality-report provenance |
| `source_build_run_id`, `source_build_attempt_id` | Successful build provenance |
| `processing_code_version` | Fingerprint of processing Python sources |
| `play_by_play_sha256`, `boxscore_sha256` | Exact raw-source identities |

`season` and `season_type` are stored as strings in every part file. The plain
directory names organize files for people and targeted reads, but no analytical
identity depends on path-derived virtual columns. A copied shard remains
self-describing.

Read a specific partition directly:

```python
import pandas as pd

segments = pd.read_parquet(
    "data/curated/possession_segments/2025-26/regular",
    filters=[
        ("season", "==", "2025-26"),
        ("season_type", "==", "regular"),
    ],
)
```

Reading the table root also works across multiple seasons because every shard
has the same self-contained schema.

Each `_manifest.json` records exact source metadata and file fingerprints,
ordered game IDs, per-game row counts, part hashes and sizes, quality counts,
and input/output row totals. Publication uses a complete temporary directory
and an atomic directory swap. A partition is resumable only when its selected
inputs, curation code, shard policy, files, hashes, schemas, and row counts all
still agree.

Warnings remain in the canonical curated layer with their issue codes. Modeling
marts can impose stricter filters without deleting accepted source evidence.

## Player reference data

Player reference collection writes:

```text
data/catalog/players.parquet
data/curated/player_seasons/2025-26/regular/part-00000.parquet
```

The catalog is the historical identity universe returned by `PlayerIndex`.
Player-season rows come from `LeagueDashPlayerBioStats` and retain explicit
season, season type, team, age, height, weight, position, college, country, and
draft fields.

The NBA season bio response also contains aggregate performance statistics.
Those remain in byte-preserved raw JSON but are deliberately absent from the
bio table so a full-season value cannot leak into predictions for earlier
games. See [Player bios](player-bios.md) for normalization and provenance.

## Analytical modeling data

The initial RAPM mart is stored separately from canonical curated data:

```text
data/analytical/rapm_stints/2025-26/regular/
  _manifest.json
  part-00000.parquet
```

This layer is allowed to make experiment-specific choices. The first contract
restricts games to the regular season, removes zero-exposure stints, allocates
multi-lineup possessions across their fixed-lineup segments, and derives a
possession-weighted home net-rating target. The manifest ties the mart to exact
curated partition manifests and modeling code.
