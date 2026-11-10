# GESTALT pipeline map

The Dagster code location is **nba_gestalt_pipelines**. Its definitions live in [the GESTALT project](gestalt/src/gestalt_pipelines/definitions.py). Both pipelines use the season set by `GESTALT_NBA_SEASON`. The injury partitions start on the date set by `GESTALT_INJURY_REPORT_START_DATE`. See the [example configuration](gestalt/code-location.env.example).

## Data flow

~~~mermaid
flowchart LR
    scheduleSource["Official NBA season schedule"] --> schedule["nba_season_schedule"]
    schedule --> catalog["Versioned schedule JSON in S3"]
    nba["Official NBA injury PDFs"] --> capture["injury_pdf_archive"]
    capture --> extract["injury_report_extractions"]
    extract --> reconcile["injury_status_by_game_date"]
    catalog --> reconcile
    rosters["Official team roster snapshots in S3"] --> reconcile
    repairs["Season player repair dictionary"] --> reconcile
    reconcile --> statuses["Versioned statuses and date latest.json"]
~~~

The arrows show which data each stage reads. Reconciliation loads the existing catalog through [NBAScheduleResource](gestalt/src/gestalt_pipelines/resources/nba_schedule.py), so the schedule job runs on its own cadence. Odds and live scores have their own collection code location. Website and forecast integration will use the injury status handoff in separate work.

## Season preparation

| Item | Details |
| --- | --- |
| Asset and job | `nba_season_schedule` and `capture_nba_season_schedule` |
| Input | Official NBA CDN schedule at https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json |
| Output | Verified, versioned schedule JSON and a season-level `latest.json` with its URI and SHA-256 |
| Timing | On demand, normally once per season |
| Refresh | Launch `capture_nba_season_schedule` in Dagster. The job uses `GESTALT_NBA_SEASON` and needs no date range. |

The schedule supplies the NBA game IDs, team IDs, and scheduled starts. Reconciliation groups games by the Pacific date of their scheduled start and records the exact schedule reference it used.

## Injury reports

| Asset | Dagster job | Inputs and outputs | Timing and partitions |
| --- | --- | --- | --- |
| `injury_pdf_archive` | `capture_injury_pdfs` | Downloads official NBA PDFs and saves each with a checksum and a `capture.json` record. | `injury_pdf_poll` runs at :02, :17, :32, and :47 Eastern and checks the preceding hour, including across midnight. A manual run checks the full Eastern publication date. |
| `injury_report_extractions` | `extract_injury_pdf` | Reads committed `capture.json` records and their PDFs. Saves parsed player rows and team submission statuses in a versioned `report.json`. | `new_injury_pdf` checks about once a minute. It scans today's and yesterday's Eastern folders for new S3 commits and requests one extraction per exact PDF revision and parser version. |
| `injury_status_by_game_date` | `reconcile_injury_date` | Reads saved extractions, the official schedule, each scheduled team's roster, and the season repair dictionary. Writes a full-slate status and advances `latest.json` for an accepted result. | `injury_report_dates_ready` checks about once a minute. New extractions can trigger today's and tomorrow's Pacific partitions when they affect games still ahead of scheduled tipoff. |

Reconciliation selects a report separately for each game. Its original publication timestamp must be earlier than the game's official scheduled start. A PDF can cover more than one game date. An accepted status can still show pending teams. It records its exact input references and checksums, and `latest.json` points to the immutable status and lists the changed game IDs. Reconciliation checks that its inputs are still current before writing the pointer. Unresolved player identities and missing game reports leave the candidate visible for diagnosis.

## Manual backfill and recovery

1. In Dagster, open the **Partitions** tab for `injury_pdf_archive`. Select the Eastern publication dates and click **Materialize…**. The job reuses archived captures and reports unavailable slots.
2. Materialize the same Eastern dates on `injury_report_extractions`. A date run processes every archived PDF that still needs an extraction for the current parser version. Recent captures can also be picked up by the sensor.
3. Select the affected **Pacific game dates** on `injury_status_by_game_date` and materialize them. The game dates can differ from the publication dates. The three date-range runs need no YAML.

| Situation | Action |
| --- | --- |
| Extract one exact PDF | Select its Eastern publication-date partition and set `capture_uri` under `ops.injury_report_extractions.config`. |
| Re-extract the newest saved PDF for a date | Select that partition and set `publication_date` under the same config. |
| Retry a failed extraction | Choose **Re-execute from failure** on the original run. A saved failure record gives the retry the exact PDF and parser. |
| Replay after a parser upgrade | Select the original Eastern partition and use `repair_failed_run_id` under `ops.injury_report_extractions.config`. The current parser writes a new extraction version and a linked recovery. |
| Repair an unmatched player | Add the name printed on the report and the NBA player ID to the season's `player_repairs.json`. Re-execute the original failed reconciliation run. Dagster keeps its reports, schedule, and rosters and reads the updated dictionary. |
| Reconcile with newer inputs | Materialize the Pacific date again. A fresh run selects its inputs from S3. |

Each extraction selector takes one input. Failures and recoveries use the same season-scoped S3 error folder. The occurrence date is Pacific. The JSON carries the exact timestamp, run ID, and input references. A recovery links to its original failure.

## S3 locations

All keys below are in s3://nba-gestalt-data. The publication date is Eastern, the game date and error occurrence date are Pacific, and `published_at` in an object key is UTC.

| Data | Key pattern |
| --- | --- |
| Versioned NBA schedule | `lake/raw/scheduleleaguev2/season=<season>/sha256=<schedule-sha256>/schedule.json` |
| Schedule pointer | `lake/raw/scheduleleaguev2/season=<season>/latest.json` |
| Archived PDF and capture record | `lake/raw/nba/injury-reports/season=<season>/publication_date=<date>/published_at=<utc>/sha256=<pdf-sha256>/{report.pdf,capture.json}` |
| Extraction | `lake/artifacts/injury_report_extraction/season=<season>/publication_date=<date>/published_at=<utc>/sha256=<pdf-sha256>/parser_version=<version>/report.json` |
| Official roster input and metadata | `lake/raw/commonteamroster/<season>/{<team-id>.json,<team-id>.meta.json}` |
| Operator repair dictionary | `lake/artifacts/injury_status_by_game_date/season=<season>/player_repairs.json` |
| Versioned game statuses | `lake/artifacts/injury_status_by_game_date/season=<season>/game_date_pacific=<date>/selection_sha256=<selection-sha256>/statuses.json` |
| Accepted game-date pointer | `lake/artifacts/injury_status_by_game_date/season=<season>/game_date_pacific=<date>/latest.json` |
| Failures and recoveries | `lake/artifacts/errors/pipeline=injury_reports/season=<season>/occurred_date_pacific=<date>/<event-id>.{failure,recovery}.json` |

Backfill and replay use these same locations. The PDF checksum identifies the original bytes. The selection checksum identifies the reconciliation inputs and rules. A pointer also stores the checksum of the saved status file.

### Verified 2026–27 examples

The PDF, capture, and extraction below belong to the same report recovered by the Nov 9 backfill. The error and recovery are a linked pair from an operator rerun.

~~~text
s3://nba-gestalt-data/lake/raw/scheduleleaguev2/season=2026-27/latest.json
s3://nba-gestalt-data/lake/raw/scheduleleaguev2/season=2026-27/sha256=88fb7c58d088d6243d3bd23e5d68ccbe4faaffe3b85de71f7b0345c9c658d3f4/schedule.json
s3://nba-gestalt-data/lake/raw/nba/injury-reports/season=2026-27/publication_date=2026-11-09/published_at=20261109T191500Z/sha256=7dd1ca210f25b2d87199bdc990df4f7d9516e23ab2c53b6207245226e8747bca/report.pdf
s3://nba-gestalt-data/lake/raw/nba/injury-reports/season=2026-27/publication_date=2026-11-09/published_at=20261109T191500Z/sha256=7dd1ca210f25b2d87199bdc990df4f7d9516e23ab2c53b6207245226e8747bca/capture.json
s3://nba-gestalt-data/lake/artifacts/injury_report_extraction/season=2026-27/publication_date=2026-11-09/published_at=20261109T191500Z/sha256=7dd1ca210f25b2d87199bdc990df4f7d9516e23ab2c53b6207245226e8747bca/parser_version=v2/report.json
s3://nba-gestalt-data/lake/raw/commonteamroster/2026-27/1610612760.json
s3://nba-gestalt-data/lake/raw/commonteamroster/2026-27/1610612760.meta.json
s3://nba-gestalt-data/lake/artifacts/injury_status_by_game_date/season=2026-27/player_repairs.json
s3://nba-gestalt-data/lake/artifacts/injury_status_by_game_date/season=2026-27/game_date_pacific=2026-11-10/selection_sha256=a79977ae08910276dc3c0aafe5c32682c1814b687809fc9aa07169da13f3bc32/statuses.json
s3://nba-gestalt-data/lake/artifacts/injury_status_by_game_date/season=2026-27/game_date_pacific=2026-11-10/latest.json
s3://nba-gestalt-data/lake/artifacts/errors/pipeline=injury_reports/season=2026-27/occurred_date_pacific=2026-11-09/2a7e7ca1ea5c752d6d71ad9ad7f69e19f629e99153b6268aed860cb4579970cf.failure.json
s3://nba-gestalt-data/lake/artifacts/errors/pipeline=injury_reports/season=2026-27/occurred_date_pacific=2026-11-09/8a380af46e10a5fd001b47499803c5aac523f328d65ae65d87569a39e7ea493e.recovery.json
~~~

A `latest.json` object can advance as newer inputs are accepted. A consumer can reproduce an earlier result from the exact immutable reference recorded with that result.
