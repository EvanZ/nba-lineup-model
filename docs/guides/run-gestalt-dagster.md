# Run NBA GESTALT orchestration

Dagster replaces the three entries previously installed in
`/etc/cron.d/nba-gestalt-schedule`. It runs on the existing EC2 instance with
DuckDB for bounded analytical queries and S3 as durable storage.

## Workloads

- `schedule_refresh_every_30_minutes` refreshes today's and the next seven
  available ESPN boards, lines, official injury reports, projections, and
  immutable snapshots.
- `nba_market_odds_game_day` archives league-wide DraftKings, FanDuel, BetMGM,
  BetRivers, theScore Bet, Hard Rock Bet, betPARX, Bally Bet, BetOnline, and
  Polymarket responses at 08:20, 12:20, and hourly from 15:20
  through 19:20 America/Los_Angeles. The request is limited to spreads and
  moneylines, so the normal maximum is 14 credits per day and 434 credits in a
  31-day month. A 50-credit reserve is enforced from the provider's response
  headers. The displayed benchmark uses the median spread and mean de-vigged
  moneyline probability from the nine sportsbooks when at least two books are
  available. Polymarket remains outside the sportsbook consensus, while its
  response and the complete raw bookmaker payload remain archived for research.
- `yesterday_finalization_overnight` revisits yesterday's board from 04:00
  through 09:30 UTC.
- `live_score_game_window` evaluates every 75 seconds but creates score-update
  runs only from 30 minutes before the first game through six hours after the
  last scheduled start. When a game becomes final, the run immediately writes
  and byte-verifies its immutable audit summary and source ESPN payload in S3
  before advancing local state. Each run is short; there is no minute-by-minute
  job running around the clock.
- `final_game_archive_morning_reconciliation` runs at 08:15
  America/New_York and byte-verifies every local final artifact from the prior
  day's completed games against its immutable S3 key. Missing local artifacts
  fail the run instead of being silently skipped.
- `in_season_rosters_four_times_daily` checks the official NBA transaction
  ledger and all 30 current ESPN team rosters at 06:00, 09:00, 12:00, and
  15:00 America/Los_Angeles. Unchanged membership is a no-op. Newly rostered
  players extend the frozen ratings/minutes vocabulary without rewriting any
  incumbent preseason prior, and a changed roster digest makes a quarantined
  same-date ratings/minutes slate eligible for repair.
- `daily_s3_lake_health` reads a small immutable Parquet object directly from
  S3 using DuckDB's AWS credential chain.
- `completed_game_stats_v3` evaluates every 15 minutes and launches ingestion
  only when a game has been final for at least 60 minutes and either its V3
  play-by-play or traditional box score is missing from the lake. It examines
  only today's and yesterday's compact schedule files and reads a persistent
  local coverage index; it does not list the full season on every tick.
- `stats_v3_morning_reconciliation` re-fetches the prior day's final games at
  08:00 America/New_York. Byte-identical responses are no-ops; corrected
  responses become new immutable content-addressed versions.

The instance permits one run at a time. DuckDB is limited to two threads and
512 MB by default. Dagster schedule/sensor history is retained for 30 days.
The UI and daemon share one persistent loopback-only code server. Orchestration
definitions use lazy workload imports so NumPy, pandas, DuckDB, and model code
are loaded only by the active run rather than duplicated in every Dagster
control-plane process.

High-frequency raw schedule snapshots are staged locally only while their
immutable S3 uploads are verified, then removed. Official injury-report PDFs
use a SHA-256 content-addressed S3 key and a small local marker, so an unchanged
report is stored once rather than copied into every date and half-hour snapshot.
Only the compact `latest`, `frozen`, and `final` serving state remains local.

The Odds API credential is read only from `GESTALT_ODDS_API_KEY` in
`/etc/nba-gestalt/dagster.env`; never place it in the repository or a Dagster
run config. Restart the code server and daemon after adding it. Immutable raw
responses and the mutable latest pointer are stored under:

```text
s3://nba-gestalt-data/nba-gestalt/schedule/season=2026-27/odds-api/
```

Each successful run records `x-requests-used`, `x-requests-remaining`, and
`x-requests-last` as Dagster metadata. Retries after an S3 failure reuse the
locally staged provider response rather than consuming another API request.

## Deploy with automations stopped

Publish and deploy a normal GESTALT release, then on EC2 run:

```bash
cd ~/nba-lineup-model
./scripts/deploy_gestalt_orchestration.sh
```

The initial cutover keeps automations stopped unless
`GESTALT_AUTOMATIONS_DEFAULT_RUNNING=1` is set. The production systemd units
set that flag after cutover so schedules and sensors remain enabled when a new
immutable release changes Dagster's physical code-location path. Confirm both
services before cutover:

```bash
systemctl status gestalt-dagster-code-server gestalt-dagster-webserver gestalt-dagster-daemon
journalctl -u gestalt-dagster-daemon -n 100 --no-pager
```

## Secure remote UI

The webserver binds only to EC2 loopback. From a local terminal:

```bash
ssh -N -L 3000:127.0.0.1:3000 gestalt-web-server
```

Open `http://127.0.0.1:3000` locally. Port 3000 must remain closed in the EC2
security group and must not be proxied by the public Nginx site.

## DuckDB/S3 smoke test

Install DuckDB's signed extensions once, then query S3 without copying the lake:

```bash
cd /opt/nba-gestalt/current
.venv/bin/nba-check-s3-lake --install-extensions
```

Subsequent health checks load the cached extensions and use the EC2 AWS
credential chain. The default target is an 8 KB analytical Parquet object.

## Parity and cutover

1. In the UI, manually materialize `s3_lake_query_health`.
2. Launch `refresh_schedule_window` with today's Eastern date and compare its
   game count, frozen count, latest JSON, and S3 keys with the cron output.
3. Stop the current live watcher and move the cron file out of `/etc/cron.d`.
4. Start both schedules and the `live_score_game_window` sensor in the UI.
5. Confirm the next 30-minute tick succeeds before deleting the disabled cron
   file.

The disabled cron file is the rollback path. Stop the Dagster schedules and
sensor, restore `/etc/cron.d/nba-gestalt-schedule`, and let cron launch the
original commands again.

## Operations

The active release deployment restarts Dagster after moving the `current`
symlink, so jobs and the API always load the same code. Persistent Dagster
state lives in `/var/lib/nba-gestalt/dagster`; schedule state remains in
`/var/lib/nba-gestalt/schedule` and immutable copies remain in S3.

## Current-season CDN play-by-play

The active completed-game sensor checks only today's and yesterday's compact
serving snapshots plus `/var/lib/nba-gestalt/cdn/coverage.json`. It does not
scan the season or list S3 on each evaluation. One hour after a game is first
observed final, it archives the raw CDN play-by-play bytes at:

```text
s3://nba-gestalt-data/lake/raw/cdn/playbyplay/season=<season>/game_id=<game-id>/<sha256>.json
```

Append-only records and run summaries live under
`lake/manifests/cdn_live_data/`. A reconciliation run at 08:00
America/New_York re-fetches the prior day's final games and preserves a new
revision only when the checksum changes. The existing schedule pipeline remains
the source for final player minutes; this collector does not download a
duplicate box score.

Run or resume a date-bounded backfill with the same implementation used by
Dagster:

```bash
nba-ingest-cdn-play-by-play \
  --season 2026-27 \
  --start-date 2026-10-20 \
  --end-date 2026-10-24 \
  --ignore-grace
```

Current CDN and historical Stats V3/V2 actions are reconciled only at the
canonical event boundary. A training build selects one immutable source
revision per game and records the source and checksum in provenance; it never
unions overlapping raw source actions.

## Stats V3 backfill and coverage

The optional Stats V3 repair collector uses the compact serving snapshots as
its completion catalog and writes no durable local PBP cache. Raw source bytes
are stored at:

```text
s3://nba-gestalt-data/lake/raw/stats/<endpoint>/season=<season>/game_id=<game-id>/<sha256>.json
```

Append-only terminal records and run summaries are stored under
`lake/manifests/stats_v3/`. A successful response is represented once per
checksum, while failures retain their run and attempt IDs.

Run or resume a date-bounded backfill with the same implementation used by
Dagster:

```bash
nba-ingest-stats-v3 \
  --season 2026-27 \
  --start-date 2026-10-20 \
  --end-date 2026-10-24 \
  --discover-from-nba-schedule \
  --ignore-grace
```

Re-running that command validates S3 coverage and skips complete endpoint
pairs. Use `--refresh` only for a reconciliation pass that should detect an
upstream correction.

`stats.nba.com` may drop traffic from data-center IP ranges. If the EC2 egress
address is affected, set `GESTALT_NBA_STATS_PROXY` in
`/etc/nba-gestalt/dagster.env` to an approved HTTP(S) proxy and restart the
code server and daemon. The V3 sensor and reconciliation schedule remain
stopped until a proxy is present. For a host whose direct egress has been
verified, `GESTALT_STATS_V3_ENABLED=1` explicitly enables them without a proxy.
Proxy credentials must never be committed or printed.
Normal current-season collection does not depend on this optional V3 path.

DuckDB can audit the append-only success/failure records directly:

```sql
select
  game_date,
  game_id,
  endpoint,
  status,
  sha256,
  finished_at
from read_json_auto(
  's3://nba-gestalt-data/lake/manifests/stats_v3/records/**/*.json',
  hive_partitioning = true,
  union_by_name = true
)
order by game_date, game_id, endpoint, finished_at;
```
