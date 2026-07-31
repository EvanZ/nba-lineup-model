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
uv run nba-fetch-season 2025-26
```

The flow creates one task per game. A task owns both source documents for that
game, while the thread-pool limit controls how many games are fetched
concurrently.

The conservative defaults use two workers and space requests by at least one
second plus up to 0.25 seconds of random jitter:

```bash
uv run nba-fetch-season 2025-26 \
  --max-workers 2 \
  --min-request-interval 1.0 \
  --request-interval-jitter 0.25
```

The interval is process-wide, so concurrency can overlap response latency but
cannot increase the source request rate. Cached documents do not consume
request slots.

For a cold-cache historical run, retain these defaults unless a smoke test
shows the CDN needs a slower rate:

```bash
uv run nba-fetch-season 2019-20 \
  --season-type regular \
  --max-workers 2 \
  --min-request-interval 1.0 \
  --request-interval-jitter 0.25
```

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
5, 30, and 120 seconds plus jitter. Retries are limited to network failures,
HTTP 408, 425, and 5xx responses. Other permanent HTTP failures and invalid
source structures fail immediately.

The NBA CDN uses a web application firewall and can return HTML `403 Access
Denied` responses. The direct client sends the required
`Referer: https://www.nba.com/` header, checks HTTP status before parsing JSON,
and includes content type plus a short body preview in the failure record. This
behavior follows the failure reported in
[`nba_api` issue #678](https://github.com/swar/nba_api/issues/678) and the
header fix in
[`nba_api` pull request #671](https://github.com/swar/nba_api/pull/671).

An HTTP 403 or 429 opens a process-wide request circuit immediately. The default
cooldown is 15 minutes, and a longer source `Retry-After` value takes
precedence. Queued tasks then fail fast without making more CDN requests. This
is deliberate: repeated short retries against a WAF denial can extend the
block.

The command exits nonzero when any selected game still fails after retries.
Successful and partial raw files remain available for the next run.

## Recover from an access denial

1. Stop the fetch and make no further CDN probes during the cooldown.
2. Wait at least 15 minutes; use the reported `retry_after_seconds` when longer.
3. Run one missing game with one worker and a two-second interval.
4. Resume the original command only after that smoke test succeeds.

```bash
uv run nba-fetch-season 2019-20 \
  --season-type regular \
  --game-id MISSING_GAME_ID \
  --max-workers 1 \
  --min-request-interval 2.0 \
  --request-interval-jitter 0.5
```

The cache makes the resumed run incremental. Do not use `--refresh`: it would
redownload valid raw files and add avoidable requests.

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
