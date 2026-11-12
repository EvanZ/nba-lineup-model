# GESTALT pipeline map

The Dagster code location is **nba_gestalt_pipelines**. Its definitions live in [the GESTALT project](gestalt/src/gestalt_pipelines/definitions.py). The pipelines use the season set by `GESTALT_NBA_SEASON`. The injury and box-score partitions start on the date set by `GESTALT_NBA_SEASON_START_DATE`. See the [example configuration](gestalt/code-location.env.example).

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
    oddsSource["The Odds API"] --> odds["nba_market_odds"]
    catalog -->|Upcoming games| odds
    odds --> oddsArchive["Immutable odds snapshots in S3"]
    odds --> oddsLatest["Season odds latest.json in S3"]
    nbaBoxes["Official NBA CDN box scores"] --> boxScores["nba_box_scores"]
    catalog --> boxScores
    boxScores --> boxArchive["Final box-score JSON in S3"]
    nbaPlays["Official NBA CDN play-by-play"] --> playByPlay["nba_play_by_play"]
    catalog --> playByPlay
    playByPlay --> playArchive["Versioned final play-by-play JSON and sidecars"]
    playArchive --> stints["nba_stints"]
    boxArchive --> stints
    catalog --> stints
    stints --> stintsArchive["Date stints JSON versions and latest.json"]
    playByPlay -->|"Scheduled capture succeeds: same-date parsing"| stints
    nbaBoard["Official NBA CDN scoreboard"] --> scoreboard["nba_scoreboard"]
    catalog --> scoreboard
    scoreboard --> scoresLatest["Date scoreboard latest.json in S3"]
    movements["Official NBA player movement feed"] --> transactions["transactions_by_date"]
    transactions --> transactionFiles["Date transaction history in S3"]
    rosters --> rosterBuilder["rosters_by_date"]
    transactionFiles --> rosterBuilder
    transactions -->|"New rows: affected date(s)"| rosterBuilder
    rosterMorning["3:30 a.m. Pacific: yesterday"] --> rosterBuilder
    boxArchive --> rosterBuilder
    catalog --> rosterBuilder
    rosterBuilder --> rosterVersions["30-team roster versions in S3"]
~~~

The arrows show which data each stage reads. Injury reconciliation and odds capture load the existing catalog through [NBAScheduleResource](gestalt/src/gestalt_pipelines/resources/nba_schedule.py), so the schedule job runs on its own cadence. The scoreboard runs in this same code location. Website and forecast integration will use the injury status handoff in separate work.

## Season preparation

| Item | Details |
| --- | --- |
| Asset and job | `nba_season_schedule` and `capture_nba_season_schedule` |
| Input | Official NBA CDN schedule at https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json |
| Output | Verified, versioned schedule JSON and a season-level `latest.json` with its URI and SHA-256 |
| Timing | On demand, normally once per season |
| Refresh | Launch `capture_nba_season_schedule` in Dagster. The job uses `GESTALT_NBA_SEASON` and needs no date range. |

The schedule supplies the NBA game IDs, team IDs, and scheduled starts. Reconciliation groups games by the Pacific date of their scheduled start and records the exact schedule reference it used.

## Market odds

Launch the `archive_nba_market_odds` job in `nba_gestalt_pipelines` to materialize `nba_market_odds`. It needs `GESTALT_ODDS_API_KEY` in the runtime environment and uses the official season schedule to check for upcoming games through seven Pacific dates ahead. An eligible run captures league-wide spreads and moneylines and records the event count and quota in Dagster.

| Output | S3 key in `nba-gestalt-data` |
| --- | --- |
| Immutable snapshot | `lake/raw/odds-api/season=<season>/snapshots/<YYYYMMDDTHHMMSSZ>.json` |
| Latest snapshot | `lake/raw/odds-api/season=<season>/latest.json` |

The schedule runs daily at 8:20 a.m., 12:20 p.m., and hourly from 3:20 through 7:20 p.m. Pacific.

If publication fails after a response is saved, choose **Re-execute from failure** to publish the saved snapshot. Historical provider backfill is unavailable.

## Live scoreboard

Launch the `capture_nba_scoreboard` job in Dagster to publish the official NBA CDN scoreboard for the Pacific date of the slate. The job verifies the game IDs and teams against the official season schedule, then replaces `lake/raw/cdn/scoreboard/season=<season>/game_date_pacific=<date>/latest.json` with the raw NBA response. The S3 object metadata records the source, fetch time, and SHA-256 checksum.

The scoreboard_poll schedule checks every minute. It launches a capture during the six hours following a scheduled tipoff while a game still needs a final scoreboard update. It continues past midnight for the original slate, skips an already queued or running capture, and stops requesting runs when the saved games are final. Manual captures remain available without configuration. The endpoint only provides the current slate, so historical backfill is unavailable.

## Box scores

In Dagster, launch the `capture_nba_box_scores` job with a Pacific game-date partition. It gets the game IDs from the official season schedule. For each game, it checks the archived box score's checksum, game ID, teams, and final status. If the archive is empty, it fetches the box score from the NBA CDN and saves the verified response.

Today, games before scheduled tipoff are skipped and NBA game statuses that are still pregame or live are recorded as pending. Pending runs record their captured boxes and remaining game IDs as an asset observation. The date materializes once every final box is archived. Historical dates require all final boxes. Each materialization records the schedule and box-score references along with the reused and newly captured counts. If a game fails, the run reports the game ID and keeps the verified boxes already archived. Rerunning the date reuses those files.

Each final box has a visible JSON sidecar at the same S3 prefix: the box is named <sha256>.json and its sidecar is <sha256>.meta.json. The sidecar records the season, game ID, box-score checksum, home and away team IDs, and each team's roster hash. Roster hashes use sorted NBA player IDs and names, including listed players who did not play; game statistics and source row order do not affect them. Existing sidecars are checked against the archived box. Missing sidecars are created from the saved box without fetching that game again. Dagster reports metadata_created_count and metadata_reused_count alongside captured_count and reused_count.

To populate sidecars for older boxes, materialize the desired Pacific partitions of nba_box_scores in Dagster. Existing box-score JSON remains unchanged. A metadata mismatch fails visibly and leaves the existing sidecar intact.

The enabled box-score schedule checks every five minutes. It launches the same job when a game has reached scheduled tipoff and still needs a final box. The capture window runs through six hours after tipoff and can continue past midnight under the original Pacific date. Complete slates and dates with an active run are skipped.

## Play-by-play

The nba_play_by_play asset and capture_nba_play_by_play job use completed Pacific game-date partitions starting at GESTALT_NBA_SEASON_START_DATE. The play_by_play_morning schedule selects yesterday at 3 a.m. America/Los_Angeles each morning. Manual backfills select daily partitions of the same asset without YAML.

NBAScheduleResource supplies the complete slate and exact schedule reference. Each selected game is fetched from https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_<game_id>.json on every run, including reruns, so NBA corrections can be detected. This asset archives the NBA response as received. It checks HTTP availability and a nonempty body; event ordering, completion markers, scores, player data, and JSON/event parsing belong to the separate parsing job in this pipeline. Downloads already saved are retained if another endpoint is unavailable, and the date fails visibly until every selected response is captured.

Raw bytes remain at lake/raw/cdn/playbyplay/season=<season>/game_id=<game_id>/<sha256>.json with a matching <sha256>.meta.json. Identical content reuses the saved version; changed content creates a new immutable version. Existing sidecars are checked against the source and source checksum, and missing sidecars are created. Valid existing metadata remains unchanged. The sidecar preserves the NBA source URL, response headers, collection times, game and team identities, checksum, byte count, and run details. Prohibited legacy source fields fail validation and require a separately approved repair. Date results record the exact raw and metadata references in Dagster. Each game also has latest.json in that same game directory. It contains the accepted raw and sidecar URIs and checksums, the Pacific game date, the check time, and the run ID. It advances after the raw response and sidecar are saved and their checksums verified. Conditional writes prevent an overlapping run from overwriting another accepted selection. A fresh response can select an earlier preserved version if the NBA reverts a correction.

Play-by-play failures use the injury pipeline's error envelope at lake/artifacts/errors/pipeline=play_by_play/season=<season>/occurred_date_pacific=<date>/<event_id>.failure.json. The record includes the Dagster run ID, selected game date, schedule reference, failed game IDs, source URLs, and error messages; its URI and checksum also appear in Dagster failure metadata. Re-executing that failed run successfully writes a linked <event_id>.recovery.json in the same date-organized error root. Recovery reads only the failed parent run and verifies its saved failure record. An ordinary fresh run has no failed parent and does not claim recovery. The original failure files are immutable. S3 or archive-integrity failures remain visible; collection does not bypass them.

## Parsed stints

The nba_stints asset and parse_nba_games job belong to the existing play_by_play pipeline. Select one or several completed Pacific date partitions in Dagster; no per-game configuration is required. The job reads saved inputs through NBAScheduleResource, NBAPlayByPlayArchive.read_game, and NBABoxScoreArchive.read_saved. It never downloads missing sources or rematerializes the collectors.

For each game, the PBP selection comes from its existing latest.json. The box selection matches the collector: greatest raw-object LastModified, then key as deterministic tie-break. These are archive selection rules, not a claim of NBA correction timestamps. Checksums, source metadata, final box status, and official matchup identities are verified before parsing. The JSON records the selected immutable schedule, raw PBP, box scores, and sidecar references/checksums.

One accepted date is saved at:

```text
s3://nba-gestalt-data/lake/artifacts/stints/season=<season>/game_date_pacific=<date>/<sha256>.json
s3://nba-gestalt-data/lake/artifacts/stints/season=<season>/game_date_pacific=<date>/latest.json
```

The immutable file's SHA-256 covers its canonical JSON bytes. Its envelope contains schema_version, season, game_date_pacific, processing_version, inputs, validation, and rows. Validation records the expected/processed game IDs and per-game diagnostics. The pointer contains the accepted URI, checksum, season/date, and processing version. New source or processing versions preserve earlier files. Identical results reuse their file and leave an unchanged pointer untouched. Conditional pointer writes detect overlapping updates. A failed date never advances the pointer or publishes an accepted partial slate; a no-game date has explicit empty coverage and rows.

Rows contain game identity/time, stint index and period, team/player IDs, elapsed start/end/duration and source event boundaries, observed points/margin, home/away offensive possession exposure, paired possessions, and observed net rating. All lineup intervals remain, including zero-possession intervals and overtime. Net rating is null at zero exposure. Following the existing frame builder, each team possession is split equally across its reconstructed lineup segments; paired possessions are half the combined team-possession counts. These are observed data conventions, not fitted model predictions.

The NBA event/lineup/possession reconstruction is owned by play_by_play/utils/reconstruction inside gestalt_pipelines. Its basketball rules are ported without behavioral changes; unused DataFrame/Parquet exporters are excluded. Reconstruction errors, scoring inconsistencies, and incomplete or overlapping time coverage fail visibly. No missing-time redistribution, 44-minute role cutoff, availability classification, model parameters, or invented player minutes are added. The 1e-6-second numeric comparison only accommodates floating-point arithmetic; real timeline gaps are not filled. Actual minutes remain in the referenced final box scores.

The play_by_play_ready_for_stints run-success sensor requests the same date after capture_nba_play_by_play succeeds with the play_by_play_morning schedule tag. This orders parsing after the daily 3 a.m. Pacific refresh without another clock delay. Source run ID is the request key. Manual collection/backfills do not implicitly launch parsing; manual parsing/backfills use parse_nba_games independently. The sensor starts stopped in definitions and is enabled through the existing Dagster UI/instance for deployment.

Parsing errors and linked recoveries use the existing lake/artifacts/errors/pipeline=play_by_play/season=<season>/occurred_date_pacific=<date>/ root with stage nba_stints; capture keeps stage nba_play_by_play. Failure metadata includes the exact available input selection and per-game reconstruction diagnostics. Re-execute the failed parsing run to record recovery; a fresh unrelated run does not claim recovery. Source files and previous accepted stints remain unchanged.

Runtime setup is self-contained: uv sync --project pipelines/gestalt --group dev installs the locked pipeline dependencies. No root-project installation, nba_lineup_model import, parent-checkout source path, pandas, or HTTPX is required for reconstruction. Parser fixtures are included under the pipeline tests. No fitted models, legacy releases, live APIs, or legacy orchestration definitions are loaded by parsing.

## Transactions

The daily-partitioned transactions_by_date asset is captured by capture_nba_transactions. It reads the official NBA movement feed at https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json and groups rows by their reported TRANSACTION_DATE. Historical manual runs select only the requested date. A current-day poll checks today and yesterday, catching rows added late under yesterday's date; it fetches the source once and does not scan older dates. The source supplies date-only values encoded at midnight without a timezone; these values are not shifted from UTC.

The hourly schedule transactions_hourly refreshes today's Pacific partition at the top of each hour from 8 a.m. through 8 p.m. Pacific and at 11:59 p.m. Pacific, including daylight-saving changes. Manual runs and backfills use the same job: choose Pacific date partitions in Dagster, with no YAML needed. The season and partition start use GESTALT_NBA_SEASON and GESTALT_NBA_SEASON_START_DATE.

The output is lake/raw/nba/transactions/season=<season>/game_date_pacific=<date>/latest.json in nba-gestalt-data. Its visible provenance includes the official source URL, the full fetched feed's checksum, and last_polled_at. Each date's file contains only that date's rows. Yesterday's file is left untouched if no new rows are found. Each original NBA row is preserved and receives a UTC first_observed_at when first captured. Later polls append distinct rows and retain previously observed rows and their timestamps, including rows no longer returned by the source. The original NBA fields identify distinct rows; identical rows are not appended twice.

Historical backfills record the actual collection time, not a fabricated historical observation time. Roster reconstruction and replay rules remain a separate consumer decision. A date with no published transactions produces an empty transaction list. A failed fetch or invalid source leaves the saved history unchanged. Concurrent publication fails if the saved file changed after it was read. The next day's polls also catch late transactions dated yesterday. Older dates require an explicit manual date rerun/backfill.

## Roster snapshots

The build_nba_rosters job materializes rosters_by_date for selected Pacific date partitions. Partitions are independent: each date reads the opening evidence and dated sources directly, without depending on yesterday's roster output. Single-date runs and backfills use the same job without YAML, using the shared NBA season and season-start settings.

The approved inputs are the existing 30 opening CommonTeamRoster snapshots under lake/raw/commonteamroster/<season>/, daily transaction files under lake/raw/nba/transactions/season=<season>/game_date_pacific=<date>/latest.json, and final box scores with their metadata sidecars. NBAScheduleResource reads the official S3 schedule once for the required date range. The job selects each team's latest available final box through the requested date. Within a run, upstream metadata hashes allow reuse of already-read membership lists. Missing historical games or sidecars are reported for box-score backfill.

Opening memberships carry through days off. Dated signings, trades, waivers, waiver awards, and contract conversions update memberships. Box scores supply explicit positive membership evidence; absence alone does not imply removal. For historical end-of-day reconstruction, reported same-day transactions apply after game evidence. Invalid IDs, unsupported operations, ambiguous moves, or unexplained team conflicts fail visibly in Dagster with source references.

The teams field is a map of all 30 NBA team IDs to sorted player-ID lists. No player names or name-repair dictionary are required or read. A separate consumer can perform name lookup when it needs one. Team hashes and evidence dates are recorded in team_metadata, and exact source URIs/checksums and observation times are recorded in inputs.

Accepted output is lake/artifacts/rosters/season=<season>/game_date_pacific=<date>/<sha256>.json in nba-gestalt-data. Processing timestamps appear in Dagster metadata rather than changing the immutable payload on each run. Identical inputs reuse the file; changed inputs create a new version, preserving earlier versions. Failed runs publish no partial roster.

The rosters_morning schedule runs at 3:30 a.m. Pacific for yesterday's date, including dates without transactions. The roster_transactions_ready sensor also rebuilds affected roster dates after a successful capture_nba_transactions run adds new rows. If yesterday gains a late transaction, it rebuilds yesterday and today so the membership correction carries forward. Each source run ID and date identifies one request, preventing duplicate launches on repeated sensor ticks. Polls with no new rows do not trigger an extra roster run. Manual single-date jobs and backfills remain available independently. Roster outputs provide team-to-player-ID membership for the model updater; injury reconciliation is unchanged. No additional latest.json paths are created.

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
| Final NBA CDN box score | `lake/raw/cdn/boxscore/season=<season>/game_id=<game-id>/<sha256>.json` |
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
