# Compact a Season

The compaction flow publishes the six per-game processed tables as
season-level analytical Parquet datasets. It does not alter basketball rows or
make network requests.

## Prerequisites

Discover, fetch, and process the complete season first:

```bash
uv run nba-discover-season 2025-26
uv run nba-fetch-season 2025-26 --max-workers 4
uv run nba-process-season 2025-26 --max-workers 4
```

Every selected final game must have:

- a successful build containing all six processed tables;
- a latest quality status of `pass` or `warning`;
- matching processing-code and raw-source hashes in both records;
- readable per-game Parquet with the expected lexical `game_id`.

Compaction fails preflight rather than silently dropping an incomplete game.

## Full season

```bash
uv run nba-compact-season 2025-26 --max-workers 4
```

The Prefect flow creates one task for every table and season-type combination.
Each task reads one game file at a time and writes deterministic shards
containing at most 100 source games. This bounds memory without creating 1,400
tiny analytical files per table.

Limit execution to one or more complete season types with a repeatable option:

```bash
uv run nba-compact-season 2025-26 \
  --season-type regular \
  --season-type playoffs
```

The command does not offer individual-game or prefix selection because
publishing a partial canonical partition would make the dataset ambiguous.

## Outputs

Each of the six tables uses the same partition contract:

```text
data/curated/{table}/
  2025-26/
    regular/
      _manifest.json
      part-00000.parquet
      part-00001.parquet
```

The tables are:

- `events`
- `players`
- `event_lineups`
- `lineup_stints`
- `possessions`
- `possession_segments`

Every row retains its original processed columns and appends catalog team and
time fields, quality status and issue codes, successful build IDs, processing
code version, exact source JSON hashes, `season`, and `season_type`. The
directory names organize partitions, but each Parquet shard is independently
self-describing.

Warnings remain included. They are accepted, explicitly labeled source
evidence, and can be filtered when a modeling dataset requires a narrower
quality policy.

## Read analytically

Read a table root with partition filters:

```python
import json

import pandas as pd

segments = pd.read_parquet(
    "data/curated/possession_segments/2025-26/regular",
    filters=[
        ("season", "==", "2025-26"),
        ("season_type", "==", "regular"),
    ],
)
segments["quality_issue_codes"] = segments[
    "quality_issue_codes_json"
].map(json.loads)
```

The JSON representation keeps the Parquet schema stable even when one shard
contains no issue codes and another contains named warnings.

## Manifests and row conservation

Every partition manifest records:

- ordered game IDs and exact per-game row counts;
- source-file hashes and all metadata through an aggregate input fingerprint;
- curation-code fingerprint and games-per-part policy;
- quality pass and warning game counts;
- every part filename, row count, byte count, and SHA-256 digest;
- equal input and output row totals.

The flow also writes a run-level summary under:

```text
data/curated/_manifests/2025-26/{run_id}.json
```

That summary retains each Prefect partition outcome and total games, rows,
parts, skips, and failures.

## Resume and recovery

A partition is skipped only when:

- the selected catalog, build, quality, and processed-file fingerprints match;
- the curation-code fingerprint and shard size match;
- the exact declared part files exist;
- every part's byte count and SHA-256 digest match;
- required metadata columns, game IDs, per-game rows, and total rows validate.

Any mismatch rebuilds the complete partition in a temporary sibling directory.
The new directory is fully validated before an atomic swap replaces the old
partition. Use `--force` to rebuild all selected partitions deliberately.

Changing `--games-per-part` also rebuilds partitions:

```bash
uv run nba-compact-season 2025-26 \
  --games-per-part 200 \
  --max-workers 4
```

## Prefect UI

Start the persistent server:

```bash
uv run prefect server start
```

Then run compaction from another terminal:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
  uv run nba-compact-season 2025-26 --max-workers 4
```

The UI shows one task per table and season-type partition, including timing,
retries, skips, and failures. See [Use the Prefect web UI](prefect-ui.md) for
persistent profile configuration.

## Validation baseline

The complete 2025-26 catalog contains 1,400 final games across six season
types. The approved plain-directory build reconciles as follows:

| Table | Games | Rows | Partitions | Parts |
| --- | ---: | ---: | ---: | ---: |
| `events` | 1,400 | 805,813 | 6 | 18 |
| `players` | 1,400 | 48,984 | 6 | 18 |
| `event_lineups` | 1,400 | 805,813 | 6 | 18 |
| `lineup_stints` | 1,400 | 45,465 | 6 | 18 |
| `possessions` | 1,400 | 278,524 | 6 | 18 |
| `possession_segments` | 1,400 | 310,648 | 6 | 18 |

All 2,295,247 source rows are conserved. Every table-root read contains the
same 1,400 lexical game IDs and explicit `season` and `season_type` columns.
An immediate identical run validates and skips all 36 partitions.
