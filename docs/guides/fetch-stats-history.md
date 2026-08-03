# Fetch Historical NBA Stats Responses

The historical Stats flow preserves archived game responses directly from
`stats.nba.com`. It is the primary acquisition path for historical games.
Previously retained liveData files are used only as a compatibility fallback
when a matching V3 document is absent.

## Endpoints

The default run fetches two endpoints independently:

| Endpoint | Purpose | Raw path |
| --- | --- | --- |
| `playbyplayv3` | Historical event stream | `data/raw/stats/playbyplayv3/{game_id}.json` |
| `boxscoretraditionalv3` | Players, minutes, and traditional game totals | `data/raw/stats/boxscoretraditionalv3/{game_id}.json` |

`gamerotation` is supported as an opt-in auxiliary endpoint. It can provide
interval-level substitution evidence, but coverage and latency differ from the
two cloud-backed V3 feeds. A missing rotation response never invalidates a
retained play-by-play or box score.

> **Current status:** Game Rotation acquisition is suspended. The endpoint's
> coverage varies sharply by season and produces frequent HTTP 500 responses.
> Retained responses remain available for local recovery experiments, but no
> historical backfill is part of the active data pipeline.

Canonical processing selects each endpoint independently. It prefers a valid
Stats V3 artifact and otherwise uses the corresponding retained liveData
artifact.
The build ledger and quality report record the selected source and SHA-256
digest of the exact raw response.

## Archive From 1996-97

`playbyplayv3` has been verified as populated from 1996-97 onward. Begin a
regular-season archive with play-by-play, then retain box scores in a separate
resumable pass. Explicit `--season` values keep the source range visible in the
run provenance:

```bash
uv run nba-fetch-stats-history \
  --season 1996-97 \
  --season 1997-98 \
  --season 1998-99 \
  --season 1999-00 \
  --endpoint playbyplayv3 \
  --season-type regular \
  --max-workers 2 \
  --min-request-interval 1.0 \
  --request-interval-jitter 0.25 \
  --run-id stats-v3-pbp-1996-1999
```

Repeat the same command for subsequent season blocks. The raw cache validates
and skips completed responses, so an interrupted archive is resumed by rerunning
the identical selection. Retain `boxscoretraditionalv3` in a second pass after
the play-by-play archive is complete.

## Progress Logs

Each newly archived game endpoint emits a Prefect task log with the game ID,
local game date, matchup, and final score from the cached V3 play-by-play event.
For example:

```text
Archived 0029600001 | 1996-11-01 | CHI @ BOS | final CHI 107-98 BOS
```

Validated cache hits do not produce a second progress line. Prefect still tracks
them as completed tasks, while the raw cache remains the authoritative resume
boundary.

## One-game smoke test

Start with the historical game already used to validate the endpoint contract:

```bash
uv run nba-fetch-stats-history \
  --season 2019-20 \
  --game-id 0021900194 \
  --max-workers 1 \
  --min-request-interval 2.0
```

The flow creates one Prefect task per game and endpoint. A second invocation
validates and skips both cache artifacts without making a request.

## Legacy Cache Reconciliation

`--cdn-missing-only` is retained only to reconcile a legacy cache. It is not
the recommended historical acquisition path:

```bash
uv run nba-fetch-stats-history \
  --season 2019-20 \
  --cdn-missing-only \
  --max-workers 2 \
  --min-request-interval 1.0 \
  --request-interval-jitter 0.25 \
  --run-id stats-gaps-2019-20
```

For a new historical pull, omit `--cdn-missing-only` and archive the V3 pair
directly.

## Preserve all historical V3 feeds

Without `--season`, the command selects final regular-season games from
2019-20 through 2024-25:

```bash
uv run nba-fetch-stats-history \
  --max-workers 2 \
  --min-request-interval 1.0 \
  --request-interval-jitter 0.25 \
  --run-id stats-v3-2019-2024
```

For an acquisition-first pass, request only play-by-play:

```bash
uv run nba-fetch-stats-history \
  --endpoint playbyplayv3 \
  --run-id stats-pbp-2019-2024
```

Then run `boxscoretraditionalv3` separately. Endpoint-specific passes reduce
the amount of work exposed to any single source interruption.

## Suspended Game Rotation experiment

Request rotation data explicitly:

```bash
uv run nba-fetch-stats-history \
  --season 2024-25 \
  --endpoint gamerotation \
  --limit 10 \
  --max-workers 1
```

Treat this as a coverage probe only. The endpoint can return a server error for
games that are otherwise available through the V3 feeds. Historical acquisition
is currently suspended because observed coverage was materially nonuniform.

To probe only games whose latest regular-season processing attempt failed, use
the durable build ledger. The command samples a deterministic number of failed
games from each available season, archives only `gamerotation`, validates the
two team interval tables, and writes a Parquet report under `artifacts/reports/`:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
  uv run nba-probe-game-rotation --per-season 3
```

This is a coverage and structural-availability test, not a claim that Game
Rotation resolves the associated lineup ambiguity. Fetch it for the complete
failure manifest only after the probe demonstrates useful coverage.
The probe uses no per-game retry because an HTTP 500 is itself coverage evidence
for this auxiliary endpoint; archival endpoint runs retain the default retries.

When a cached response is structurally valid, use the documented
[`nba-recover-game-rotation`](process-season.md#game-rotation-recovery) command
to measure whether the existing processing quality contract accepts the repair.

## Fetch remaining lineup failures

After the coverage probe and recovery cohort establish that the source can fix
period-start ambiguity, archive it only for current failures whose terminal
message is `Period lineup remains ambiguous` or `No legal period lineup can be
inferred`. This avoids spending endpoint capacity on unrelated score and
possession defects.

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
  uv run nba-fetch-game-rotation-failures \
  --max-workers 1 \
  --min-request-interval 1.0 \
  --request-interval-jitter 0.25 \
  --run-id game-rotation-lineup-failures-r1
```

The command writes its selected games to
`artifacts/reports/game_rotation_fetch/{run_id}/selection.parquet` before any
network activity and requests no retries: an HTTP 500 is recorded as coverage
evidence. Rerunning the same command skips retained valid responses and tries
only still-unavailable games.

The command also supports a season-by-season reverse backfill for a future
re-evaluation. It is intentionally not part of the active pipeline:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
  uv run nba-fetch-game-rotation-failures \
  --max-season 2018-19 \
  --reverse-seasons \
  --max-workers 4 \
  --min-request-interval 1.0 \
  --request-interval-jitter 0.25 \
  --run-id game-rotation-backfill-through-2018-r1
```

The pacing gate remains process-wide: four workers improve throughput around
slow responses but do not multiply the one-request-per-second source rate.

## Resume, provenance, and failure policy

Every successful response is written byte-for-byte before the task reports
success. Its `.meta.json` sidecar stores the requested URL, fetch time,
SHA-256 digest, and selected source response headers. Cache reads validate the
digest, endpoint, game ID, and minimum endpoint schema.

Terminal endpoint outcomes are appended to:

```text
data/manifests/stats_fetches.parquet
```

The manifest is written after all selected tasks finish, but the raw cache is
the resume boundary. If a process stops before the manifest write, rerun the
same selection; valid endpoint files are skipped.

Network errors, HTTP 408, 425, and 5xx responses receive bounded retries. HTTP
403 or 429 opens a process-wide 15-minute circuit. The default global pacing
is one request per second plus up to 0.25 seconds of jitter, including across
concurrent Prefect workers.

## Tests

The endpoint contract tests are in
`tests/test_stats_history_fetch.py`. They cover:

- exact query parameters and browser headers for all three endpoints;
- byte preservation and SHA-256 sidecars;
- source game-ID and minimum-schema validation;
- cache resume behavior and typed Parquet manifest round trips;
- transient server-error classification.
- final-score extraction used by play-by-play progress logs.

Run the focused suite with:

```bash
uv run pytest -q tests/test_stats_history_fetch.py
```
