# Backfill Historical Seasons

The historical runner composes the existing season operations into one
checkpointed, regular-season-only pipeline. Its default range is 2019-20
through 2024-25; 2025-26 remains the current-season build.

## Stages

Each season advances through these stages in order:

| Stage | Operation | Durable output |
| --- | --- | --- |
| `discover` | Fetch and normalize the season schedule | `data/catalog/games.parquet` |
| `bios` | Collect player identity, bio, and draft fields | `data/catalog/players.parquet`, `data/curated/player_seasons/` |
| `fetch` | Fetch final regular-season game feeds | `data/raw/`, `data/manifests/fetches.parquet` |
| `process` | Reconstruct and validate each game | `data/processed/`, `data/quality/` |
| `compact` | Publish season Parquet partitions | `data/curated/` |
| `rapm` | Train the canonical one-season baselines | `data/analytical/`, `artifacts/models/rapm/` |

Seasons and stages run serially. Individual game work inside the fetch and
process stages uses bounded Prefect thread pools.

## Run the backfill

Supply a stable run ID so an interrupted invocation can resume the same plan:

```bash
uv run nba-backfill-history \
  --run-id history-2019-2024 \
  --max-workers 2 \
  --min-request-interval 1.0 \
  --request-interval-jitter 0.25
```

The minimum request interval is process-wide, including concurrent workers.
The defaults use at least one second between requests plus up to 0.25 seconds
of jitter. Cached documents are validated and skipped without consuming this
interval.

To run a smaller range, repeat `--season`:

```bash
uv run nba-backfill-history \
  --season 2019-20 \
  --season 2020-21 \
  --run-id history-pilot
```

## Resume

The parent manifest is written after every terminal stage:

```text
data/manifests/history_backfill/{run_id}.json
```

Rerun the exact command with the same `--run-id`. Completed stages are skipped,
and a failed stage is replaced only after its retry completes. The underlying
fetch, processing, and compaction operations also resume from validated
game-level artifacts.

A run ID is tied to one immutable season and stage plan. Changing that plan
requires a new run ID.

## Select a stage range

The endpoints are inclusive:

```bash
uv run nba-backfill-history \
  --season 2019-20 \
  --from-stage fetch \
  --through-stage compact \
  --run-id history-2019-data
```

Use stage ranges only when all prerequisite outputs already exist. `--refresh`
forces source refreshes; `--force` rebuilds processed and curated outputs.

For the historical RAPM panel, where some game reconstructions are
intentionally excluded by quality gates, add `--quality-eligible-only` while
running the `compact` through `rapm` stages. The compaction manifest records
the subset policy and counts; incomplete catalog games are not silently
promoted.

## Failure policy

The default runner never promotes a partial season. A fetch stage fails when any final
regular-season game remains unavailable after retries, processing fails when
any game build fails, and compaction requires every selected game to pass its
quality gate. `--quality-eligible-only` is the sole documented exception: it
publishes only the successful pass/warning subset for historical modeling and
records the exclusion count in its compaction manifest.

Network failures, NBA CDN `408`, `425`, and server errors receive bounded
retries. A CDN `403` or `429` instead opens the request circuit for at least 15
minutes and prevents queued work from making more requests. Stop the run, allow
the cooldown, perform the one-game smoke test in
[Fetch a Season](fetch-season.md#recover-from-an-access-denial), and rerun the
same run ID. Existing raw files remain valid and are not downloaded again.
