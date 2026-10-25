# Run NBA GESTALT orchestration

Dagster replaces the three entries previously installed in
`/etc/cron.d/nba-gestalt-schedule`. It runs on the existing EC2 instance with
DuckDB for bounded analytical queries and S3 as durable storage.

## Workloads

- `schedule_refresh_every_30_minutes` refreshes today's and the next seven
  available ESPN boards, lines, official injury reports, projections, and
  immutable snapshots.
- `yesterday_finalization_overnight` revisits yesterday's board from 04:00
  through 09:30 UTC.
- `live_score_game_window` evaluates every 75 seconds but creates score-update
  runs only from 30 minutes before the first game through six hours after the
  last scheduled start. Each run is short; there is no minute-by-minute job
  running around the clock.
- `daily_s3_lake_health` reads a small immutable Parquet object directly from
  S3 using DuckDB's AWS credential chain.

The instance permits one run at a time. DuckDB is limited to two threads and
512 MB by default. Dagster schedule/sensor history is retained for 30 days.

High-frequency raw schedule snapshots are staged locally only while their
immutable S3 uploads are verified, then removed. Official injury-report PDFs
use a SHA-256 content-addressed S3 key and a small local marker, so an unchanged
report is stored once rather than copied into every date and half-hour snapshot.
Only the compact `latest`, `frozen`, and `final` serving state remains local.

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
systemctl status gestalt-dagster-webserver gestalt-dagster-daemon
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
