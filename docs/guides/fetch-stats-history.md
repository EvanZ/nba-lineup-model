# Fetch Historical NBA Stats Responses

The historical Stats flow preserves archived game responses directly from
`stats.nba.com`. It is both a source fallback for games whose liveData CDN
objects are unavailable and an independently retained comparison source for
games available from both systems.

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

Canonical processing selects each endpoint independently. It prefers a valid
liveData artifact and otherwise adapts the corresponding Stats V3 artifact.
The build ledger and quality report record the selected source and SHA-256
digest of the exact raw response.

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

## Fill current CDN gaps first

The fastest high-value acquisition pass is the set of regular-season endpoint
artifacts missing from the existing liveData cache:

```bash
uv run nba-fetch-stats-history \
  --season 2019-20 \
  --cdn-missing-only \
  --max-workers 2 \
  --min-request-interval 1.0 \
  --request-interval-jitter 0.25 \
  --run-id stats-gaps-2019-20
```

`--cdn-missing-only` is evaluated per endpoint. For example, a valid CDN box
score does not prevent acquisition of missing Stats V3 play-by-play.

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

## Rotation experiment

Request rotation data explicitly:

```bash
uv run nba-fetch-stats-history \
  --season 2024-25 \
  --endpoint gamerotation \
  --limit 10 \
  --max-workers 1
```

Treat this as a coverage probe before a full run. The endpoint can return a
server error for games that are otherwise available through the V3 feeds.

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

Run the focused suite with:

```bash
uv run pytest -q tests/test_stats_history_fetch.py
```
