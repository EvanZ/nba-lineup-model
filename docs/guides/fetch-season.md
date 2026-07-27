# Fetch a Season

The season fetch flow downloads byte-preserved play-by-play and boxscore
responses for every final game selected from the canonical catalog.

## Prerequisite

Discover the season before fetching game feeds:

```bash
uv run nba-discover-season 2025-26
```

This updates `data/catalog/games.parquet`. The 2025-26 catalog currently
contains 1,400 final games across preseason, regular season, the NBA Cup final,
All-Star games, play-in games, and playoffs.

## Smoke test

Exercise the Prefect flow with one catalog game:

```bash
uv run nba-fetch-season 2025-26 --limit 1 --max-workers 1
```

The command runs locally. Prefect starts a temporary local API for run state;
no Prefect Cloud account or deployment is required.

## Prefect web UI

Start the local UI with:

```bash
uv run prefect server start
```

Open [http://127.0.0.1:4200](http://127.0.0.1:4200). See
[Use the Prefect web UI](prefect-ui.md) for persistent-server configuration,
connecting future flow runs, and restoring temporary-server mode.

## Full season

```bash
uv run nba-fetch-season 2025-26 --max-workers 4
```

The flow creates one task per game. A task owns both source documents for that
game, while the thread-pool limit controls how many games are fetched
concurrently.

Four workers is the conservative default. Increasing concurrency makes the NBA
source more likely to throttle the run and should be justified by observed
throughput rather than local CPU capacity.

## Selection

Only final catalog games are eligible. Narrow the selection with repeatable
filters:

```bash
uv run nba-fetch-season 2025-26 \
  --season-type regular \
  --season-type playoffs

uv run nba-fetch-season 2025-26 \
  --game-id 0022500001 \
  --game-id 0042500401
```

`--limit N` is applied after deterministic ordering by game date and game ID.

## Resume and refresh

Before making a request, each task validates the JSON and SHA-256 sidecar for
both endpoint files.

| Existing state | Behavior |
| --- | --- |
| Both documents valid | Record `skipped`; make no network request |
| One document valid | Keep it and fetch the missing or invalid document |
| Invalid cache document | Replace it from the source |
| `--refresh` | Refetch both documents |

This makes an interrupted season run resumable from the raw cache. The fetch
manifest is written atomically after all selected tasks reach terminal states.
If the process stops before that write, rerun the command: completed raw files
will be validated and skipped.

## Retries and failures

Each game task allows three retries after the initial attempt, with delays of
2, 10, and 30 seconds plus jitter. Retries are limited to network failures,
HTTP 408, 425, 429, and 5xx responses. Permanent HTTP failures and invalid
source structures fail immediately.

The command exits nonzero when any selected game still fails after retries.
Successful and partial raw files remain available for the next run.

## Durable manifest

Terminal outcomes are appended to:

```text
data/manifests/fetches.parquet
```

Every record includes:

- project run ID and Prefect flow/task run IDs;
- game, season, and season type;
- attempt number and UTC timing;
- success, skip, or failure status;
- cache provenance for each endpoint;
- exact-byte SHA-256 digests and byte counts;
- structured failure or skip details.

Prefect owns live orchestration state. The Parquet manifest remains the
portable, project-owned execution history.
