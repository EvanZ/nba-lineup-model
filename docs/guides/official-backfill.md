# Official NBA catch-up and backfill

Run on EC2. Production orchestration stays stopped until explicitly approved.

## Routine operation

The `complete_slate_daily_release` sensor launches `backfill_official_history`.
It checks every past Pacific date against the canonical NBA catalog and local
coverage, not only games still visible on today's scoreboard. Once the current
slate is fully final it also permits a same-night update.

The CLI and Dagster op call the same `run_backfill` function:

```sh
python -m nba_lineup_model.orchestration.backfill \
  --config /var/lib/nba-gestalt-dev/daily-pipeline/config.json \
  --start-date 2026-11-01 --end-date 2026-11-01
```

Omit dates to check the season through yesterday. Repeat `--game-id` to select
games; the operation expands to their whole slates and required predecessor
dates. It never silently skips a predecessor to update a later date.

Dagster manual launch uses op `recover_official_history` with `config_path`,
`end_date`, optional `start_date`, and optional `game_ids`.

## Guarantees

- Historical final boxes are fetched directly by NBA game ID. No live-board
  lookup or current-rating API is used to finalize a historical forecast.
- Pregame forecasts and market snapshots are retained byte-for-byte. Missing
  historical priors are explicit failures, never fabricated predictions.
- Each date is collected and validated before its data artifact, model artifact,
  and full serving candidate are built. A failed date stops model advancement.
- All games within a date are attempted so the report lists failures together.
  Successfully collected boxes/PBP remain available on retry.
- Data collection is separate from forecasts: missing frozen inputs do not
  prevent retaining valid official boxes/PBP, but do prevent publishing a
  fabricated historical prediction.
- Checkpoints are scoped to source-schedule fingerprint and code revision.
  Validated dates are reused on retry; subsequent days use private schedule
  copies, never edit the previous validated candidate.
- Already complete backfills do not regenerate daily artifacts or forecasts.
- The sensor permits at most one daily/backfill job in flight, waits 15 minutes
  between failures, and stops after three attempts per source tip/target date.
  Successful candidates awaiting review are not repeatedly queued. A manual
  launch remains available after correcting a reported failure.
- Activation is opt-in (`activate_after_validation`, false in dev). Publication
  atomically switches the full release with stale-source protection and API
  rollback. Backfill never starts Dagster services.

## Evidence and failures

`<pipeline_root>/reports/backfill/report.html` lists processed dates, their
status, and any failing game IDs. Per-date reports remain under
`<pipeline_root>/reports/YYYY-MM-DD/`. The existing daily-report API surfaces
the latest report without requiring activation of a failed candidate.

Retry the same command after correcting a failure. Do not delete frozen data,
edit chain pointers, invent missing player ratings, or bypass quality gates.
If a previously published daily model/data artifact itself needs correction,
that is a separately reviewed historical revision—not a silent backfill.
