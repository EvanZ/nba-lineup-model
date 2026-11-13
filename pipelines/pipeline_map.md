# GESTALT pipeline map

The Dagster code location is **nba_gestalt_pipelines**. Its definitions live in [the GESTALT project](gestalt/src/gestalt_pipelines/definitions.py). The pipelines use the season set by `GESTALT_NBA_SEASON`. The daily partitions start on the date set by `GESTALT_NBA_SEASON_START_DATE`. See the [example configuration](gestalt/code-location.env.example).

Current reference reviewed November 13, 2026 Pacific. The sections below cover inputs, accepted S3 selections, automatic timing, manual/backfill operation, and recovery. Runtime enablement is a dated snapshot; use Dagster for current run state.

## Data flow

~~~mermaid
flowchart LR
    playerIndex["Official PlayerIndex JSON and metadata in S3"] --> playerNames["Shared NBAPlayerNamesResource"]
    playerNames --> reconcile
    scheduleSource["Official NBA season schedule"] --> schedule["nba_season_schedule"]
    schedule --> catalog["Versioned schedule JSON in S3"]
    nba["Official NBA injury PDFs"] --> capture["injury_pdf_archive"]
    capture --> extract["injury_report_extractions"]
    extract --> reconcile["injury_status_by_game_date"]
    catalog --> reconcile
    rosters["Official opening roster snapshots in S3"]
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
    rosterRepairs["Season repair_transactions.json"] --> rosterBuilder
    catalog -->|"10-day contract terms"| rosterBuilder
    transactionFiles --> rosterBuilder
    transactions -->|"New rows: affected date(s)"| rosterBuilder
    rosterMorning["3:30 a.m. Pacific: yesterday"] --> rosterBuilder
    rosterBuilder --> rosterVersions["30-team roster versions and date latest.json in S3"]
    rosterVersions -->|"Newest roster date <= game date"| reconcile
    priorMinutes["Frozen conditional priors and historical-role lookup"] --> minutes["player_minutes"]
    stintsArchive --> minutes
    rosterVersions --> minutes
    statuses --> minutes
    catalog --> minutes
    minutes --> minutesArchive["Versioned conditional-minutes state and date latest.json"]
    minutesArchive -->|"Previous Pacific date"| minutes
    minutesArchive --> gameMinutes["game_minutes"]
    priorMinutes --> gameMinutes
    minutesMorning["4 a.m. Pacific: yesterday"] --> minutes
    projectionSensor["project_game_minutes_sensor: today/tomorrow"] --> gameMinutes
    rosterVersions --> gameMinutes
    statuses --> gameMinutes
    catalog --> gameMinutes
    gameMinutes --> gameAllocations["Per-game minutes JSON and latest.json"]

~~~

The arrows show saved-data inputs and the explicitly labelled automatic triggers. Box scores feed stint reconstruction; the daily minutes updater reads the resulting stints rather than reading box scores directly. Injury reconciliation and odds capture load the existing catalog through [NBAScheduleResource](gestalt/src/gestalt_pipelines/resources/nba_schedule.py), so the schedule job runs on its own cadence. The scoreboard runs in this same code location. Website and forecast integration will use the injury status handoff in separate work.

## Timing and automatic triggers

Times are Pacific (`America/Los_Angeles`) except the injury publication schedule, which uses Eastern (`America/New_York`). Date boundaries follow those named timezones through daylight-saving changes.

| Job | Schedule or sensor | When it requests work |
| --- | --- | --- |
| `capture_nba_season_schedule` | On demand | Normally once per season |
| `capture_injury_pdfs` | `injury_pdf_poll` | Eastern :02, :17, :32, :47; checks the preceding hour |
| `extract_injury_pdf` | `new_injury_pdf` | About every minute; new saved PDFs in today's/yesterday's Eastern partitions |
| `reconcile_injury_date` | `injury_report_dates_ready` | About every minute; changed accepted inputs for open games today/tomorrow |
| `archive_nba_market_odds` | `nba_market_odds_game_day` | 08:20, 12:20, and 15:20–19:20; source/window and quota checks apply |
| `capture_nba_scoreboard` | `scoreboard_poll` | Every minute during the configured game window until final scores are saved |
| `capture_nba_box_scores` | `box_score_poll` | Every five minutes during the game window until final boxes are saved |
| `capture_nba_play_by_play` | `play_by_play_morning` | 03:00 for yesterday |
| `parse_nba_games` | `play_by_play_ready_for_stints` | After that scheduled PBP capture succeeds, for the same date |
| `capture_nba_transactions` | `transactions_hourly` | Hourly 08:00–20:00, plus 23:59; current polls check today and yesterday |
| `build_nba_rosters` | `rosters_morning`; `roster_transactions_ready` | 03:30 for yesterday, plus affected dates when transaction collection adds rows |
| `update_player_minutes` | `update_player_minutes_schedule` | 04:00 for yesterday; requires successful parser and roster runs that morning |
| `project_game_minutes` | `project_game_minutes_sensor` | About every minute; eligible games today/tomorrow when accepted inputs differ |

All eight listed recurring schedules and five sensors were verified RUNNING in the permanent instance on November 13, 2026. The former `minutes_ready` / `update_player_minutes_sensor` is retired; `game_minutes_ready` was renamed `project_game_minutes_sensor`. Definition defaults may be STOPPED; the table describes enabled instance state. Manual partition runs and backfills remain available alongside automation.

The morning sequence is 03:00 PBP capture → successful capture triggers same-date stints parsing; 03:30 yesterday's roster build; 04:00 yesterday's minutes update. The updater fails visibly if the scheduled parser/roster prerequisites are missing, failed, still running, or stale. Repair them and re-execute the updater manually. Injury reconciliation and game allocation read their approved accepted inputs; neither requires the injury artifact and allocation roster to reference an identical roster version.

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

The NBA event/lineup/possession reconstruction is owned by play_by_play/utils/reconstruction inside gestalt_pipelines. The box score supplies each team's starting five, confirms final status and matchup identities, and provides full-game player minutes for reconstruction diagnostics. Starting lineups plus PBP substitutions establish the on-court intervals; minute mismatches are warnings. Its basketball rules are ported without behavioral changes; unused DataFrame/Parquet exporters are excluded. Reconstruction errors, scoring inconsistencies, and incomplete or overlapping time coverage fail visibly. No missing-time redistribution, 44-minute role cutoff, availability classification, model parameters, or invented player minutes are added. The 1e-6-second numeric comparison only accommodates floating-point arithmetic; real timeline gaps are not filled. The full stint timeline supports deriving actual playing time; referenced final box scores supply independent totals for validation.

The play_by_play_ready_for_stints run-success sensor requests the same date after capture_nba_play_by_play succeeds with the play_by_play_morning schedule tag. This orders parsing after the daily 3 a.m. Pacific refresh without another clock delay. Source run ID is the request key. Manual collection/backfills do not implicitly launch parsing; manual parsing/backfills use parse_nba_games independently. The sensor is enabled in the existing Dagster instance; its definition default is STOPPED.

Parsing errors and linked recoveries use the existing lake/artifacts/errors/pipeline=play_by_play/season=<season>/occurred_date_pacific=<date>/ root with stage nba_stints; capture keeps stage nba_play_by_play. Failure metadata includes the exact available input selection and per-game reconstruction diagnostics. Re-execute the failed parsing run to record recovery; a fresh unrelated run does not claim recovery. Source files and previous accepted stints remain unchanged.

Runtime setup is self-contained: uv sync --project pipelines/gestalt --group dev installs the locked pipeline dependencies. No root-project installation, nba_lineup_model import, parent-checkout source path, pandas, or HTTPX is required for reconstruction. Parser fixtures are included under the pipeline tests. No fitted models, legacy releases, live APIs, or legacy orchestration definitions are loaded by parsing.

## Transactions

The daily-partitioned transactions_by_date asset is captured by capture_nba_transactions. It reads the official NBA movement feed at https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json and groups rows by their reported TRANSACTION_DATE. Historical manual runs select only the requested date. A current-day poll checks today and yesterday, catching rows added late under yesterday's date; it fetches the source once and does not scan older dates. The source supplies date-only values encoded at midnight without a timezone; these values are not shifted from UTC.

The hourly schedule transactions_hourly refreshes today's Pacific partition at the top of each hour from 8 a.m. through 8 p.m. Pacific and at 11:59 p.m. Pacific, including daylight-saving changes. Manual runs and backfills use the same job: choose Pacific date partitions in Dagster, with no YAML needed. The season and partition start use GESTALT_NBA_SEASON and GESTALT_NBA_SEASON_START_DATE.

The output is lake/raw/nba/transactions/season=<season>/game_date_pacific=<date>/latest.json in nba-gestalt-data. Its visible provenance includes the official source URL, the full fetched feed's checksum, and last_polled_at. Each date's file contains only that date's rows. Yesterday's file is left untouched if no new rows are found. Each original NBA row is preserved and receives a UTC first_observed_at when first captured. Later polls append distinct rows and retain previously observed rows and their timestamps, including rows no longer returned by the source. The original NBA fields identify distinct rows; identical rows are not appended twice.

Historical backfills record the actual collection time, not a fabricated historical observation time. Roster reconstruction and replay rules remain a separate consumer decision. A date with no published transactions produces an empty transaction list. A failed fetch or invalid source leaves the saved history unchanged. Concurrent publication fails if the saved file changed after it was read. The next day's polls also catch late transactions dated yesterday. Older dates require an explicit manual date rerun/backfill.

## Roster snapshots

The `rosters_by_date` asset and `build_nba_rosters` job produce all 30 teams as a team-ID → sorted player-ID map for a selected Pacific date. Names and contract-type classifications are not part of membership. Partitions are independent: each run reconstructs from the opening seed plus dated changes, without loading the previous roster. Carry-forward optimization is deferred.

| Input | How it is used |
| --- | --- |
| Opening CommonTeamRoster snapshots | Bootstrap membership from all 30 saved official snapshots under `lake/raw/commonteamroster/<season>/`, with their sidecars and observed date |
| Saved official transactions | Apply rows after the seed date through the requested date from `lake/raw/nba/transactions/season=<season>/game_date_pacific=<date>/latest.json` |
| Operator membership repairs | Read `lake/artifacts/rosters/season=<season>/repair_transactions.json` on every run; apply repairs on their effective dates |
| Official schedule | Use `NBAScheduleResource` when a saved signing describes a 10-day contract; record the exact immutable schedule reference |

This builder reads no box scores or box-score sidecars. Opening memberships persist through days off until a dated transaction, repair, or contract expiration changes them. Unsupported operations, invalid IDs, or ambiguous destination teams fail visibly. The output records exact input references/checksums and per-team roster hashes/evidence dates.

### Operator membership repairs

The season's repair file is an editable JSON list. For example, the approved Nance repair is:

```json
[
  {
    "effective_date": "2026-10-19",
    "action": "add",
    "team_id": 1610612749,
    "player_id": 1631250
  }
]
```

Use `add` or `remove`; no name or source field is required. Repairs may correct the opening roster on its own date but cannot predate it. A future-dated repair does not affect an earlier partition. Official transactions and repairs are processed chronologically; official moves apply first on a shared date, followed by repairs in file order. Later official moves still take effect. Repeated additions do not duplicate membership. An absent repair file means no repairs; invalid content fails the run.

After editing the S3 file, run `build_nba_rosters` for the affected dates. The edit itself does not launch a sensor. Any historical minutes suffix replay is a separate explicitly approved operation. Membership repairs are distinct from the injury pipeline's name → ID repair dictionary.

### Ten-day contract ledger

Each roster JSON includes `ten_day_contracts`, derived during that roster run. It records player/team IDs, signing date, third scheduled team game ID/date, last active date, removal date, and any superseding transaction date. The calculation follows NBA–NBPA CBA Article II §9: signing day is day one, and membership lasts through the later of day ten and the third team game. Removal takes effect the following Pacific date. Until three team games are established through the selected date, expiry stays unset and membership remains.

Expirations apply before new official transactions on their removal date. Renewals, conversions, trades, waivers, and membership repairs supersede old terms. Duplicate signing rows are idempotent. The ledger is inside the existing roster JSON; there is no extra ledger job, schedule, or S3 path. Wesley's Oct 29 signing yields a Nov 2 third game, Nov 7 last active date, and Nov 8 removal.

### Publication, warnings, and timing

Accepted outputs are `lake/artifacts/rosters/season=<season>/game_date_pacific=<date>/<sha256>.json` and that date's `latest.json`. Write and verify the immutable file before conditionally advancing the pointer. Identical inputs reuse it; changed inputs preserve old versions; failed validation or stale competing publication cannot replace the accepted selection. There is no season-level roster pointer.

More than **18 total IDs** on a team emits a Dagster warning and records `roster_size_warnings` in materialization metadata; publication continues. This allows 15 standard plus 3 two-way memberships without separately validating contract types. The run also saves a `.warning.json` under `lake/artifacts/errors/pipeline=rosters/season=<season>/occurred_date_pacific=<processing-date>/`, with affected teams/counts/IDs, exact inputs, roster artifact and run ID. The earlier 15-player total threshold was superseded; old warning events remain historical records.

`rosters_morning` runs at **03:30 Pacific for yesterday**, even without new transactions. `roster_transactions_ready` rebuilds affected dates when successful transaction collection adds rows. A late row for yesterday causes yesterday and today to rebuild. Polls with no new rows do not trigger a roster run. Manual date runs/backfills remain available in any order. Consumers select accepted date pointers rather than whichever version was most recently uploaded.

## Injury reports

| Asset | Dagster job | Inputs and outputs | Timing and partitions |
| --- | --- | --- | --- |
| `injury_pdf_archive` | `capture_injury_pdfs` | Downloads official NBA PDFs and saves each with a checksum and a `capture.json` record. | `injury_pdf_poll` runs at :02, :17, :32, and :47 Eastern and checks the preceding hour, including across midnight. A manual run checks the full Eastern publication date. |
| `injury_report_extractions` | `extract_injury_pdf` | Reads committed `capture.json` records and their PDFs. Saves parsed player rows and team submission statuses in a versioned `report.json`. | `new_injury_pdf` checks about once a minute. It scans today's and yesterday's Eastern folders for new S3 commits and requests one extraction per exact PDF revision and parser version. |
| `injury_status_by_game_date` | `reconcile_injury_date` | Reads saved extractions, the official schedule, newest eligible dated roster, shared PlayerIndex names, and the season injury-name repair dictionary. Writes a full-slate status and advances `latest.json` for an accepted result. | `injury_report_dates_ready` checks about once a minute. Changed extraction, roster, PlayerIndex, repair, or schedule selections can trigger today's and tomorrow's Pacific partitions for games still ahead of scheduled tipoff. |

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
| Repair an unmatched player | Add the name printed on the report and the NBA player ID to the season's `player_repairs.json`. Re-execute the original failed reconciliation run. Dagster keeps its selected reports and schedule, then rereads the newest eligible roster, PlayerIndex, and updated dictionary. |
| Reconcile with newer inputs | Materialize the Pacific date again. A fresh run selects its inputs from S3. |

Each extraction selector takes one input. Failures and recoveries use the same season-scoped S3 error folder. The occurrence date is Pacific. The JSON carries the exact timestamp, run ID, and input references. A recovery links to its original failure.

## S3 locations

All keys below are in s3://nba-gestalt-data. The publication date is Eastern, the game date and error occurrence date are Pacific, and `published_at` in an object key is UTC.

| Data | Key pattern |
| --- | --- |
| Versioned NBA schedule | `lake/raw/scheduleleaguev2/season=<season>/sha256=<schedule-sha256>/schedule.json` |
| Schedule pointer | `lake/raw/scheduleleaguev2/season=<season>/latest.json` |
| Odds snapshots and pointer | `lake/raw/odds-api/season=<season>/snapshots/<YYYYMMDDTHHMMSSZ>.json`; season `latest.json` |
| Scoreboard | `lake/raw/cdn/scoreboard/season=<season>/game_date_pacific=<date>/latest.json` |
| Final NBA CDN box score and sidecar | `lake/raw/cdn/boxscore/season=<season>/game_id=<game-id>/<sha256>.{json,meta.json}` |
| PBP versions, sidecars, and game pointer | `lake/raw/cdn/playbyplay/season=<season>/game_id=<game-id>/<sha256>.{json,meta.json}`; game `latest.json` |
| Parsed stints | `lake/artifacts/stints/season=<season>/game_date_pacific=<date>/<sha256>.json`; date `latest.json` |
| Daily transactions | `lake/raw/nba/transactions/season=<season>/game_date_pacific=<date>/latest.json` |
| Opening roster seed and metadata | `lake/raw/commonteamroster/<season>/{<team-id>.json,<team-id>.meta.json}` |
| Roster membership repairs | `lake/artifacts/rosters/season=<season>/repair_transactions.json` |
| Derived rosters, including contract ledger | `lake/artifacts/rosters/season=<season>/game_date_pacific=<date>/<sha256>.json`; date `latest.json` |
| PlayerIndex and metadata | `lake/raw/playerindex/<season>.{json,meta.json}` |
| Archived PDF and capture record | `lake/raw/nba/injury-reports/season=<season>/publication_date=<date>/published_at=<utc>/sha256=<pdf-sha256>/{report.pdf,capture.json}` |
| Extraction | `lake/artifacts/injury_report_extraction/season=<season>/publication_date=<date>/published_at=<utc>/sha256=<pdf-sha256>/parser_version=<version>/report.json` |
| Injury name repairs | `lake/artifacts/injury_status_by_game_date/season=<season>/player_repairs.json` |
| Versioned game statuses | `lake/artifacts/injury_status_by_game_date/season=<season>/game_date_pacific=<date>/selection_sha256=<selection-sha256>/statuses.json` |
| Accepted injury date selection | `lake/artifacts/injury_status_by_game_date/season=<season>/game_date_pacific=<date>/latest.json` |
| Conditional minutes priors | `lake/artifacts/player_minutes/season=<season>/model_version=<version>/priors/<sha256>.json` |
| Conditional minutes state | `lake/artifacts/player_minutes/season=<season>/model_version=<version>/game_date_pacific=<date>/<sha256>.json`; date `latest.json` |
| Per-game minutes allocation | `lake/artifacts/game_minutes/season=<season>/model_version=<version>/game_date_pacific=<date>/game_id=<game-id>/<sha256>.json`; game `latest.json` |
| Injury processing events | `lake/artifacts/errors/pipeline=injury_reports/season=<season>/occurred_date_pacific=<date>/<event-id>.{failure,recovery,warning}.json` |
| Roster size warnings | `lake/artifacts/errors/pipeline=rosters/season=<season>/occurred_date_pacific=<date>/<event-id>.warning.json` |
| PBP capture/stints failures and recoveries | `lake/artifacts/errors/pipeline=play_by_play/season=<season>/occurred_date_pacific=<date>/<event-id>.{failure,recovery}.json` |
| Minutes updater/projection failures and recoveries | `lake/artifacts/errors/pipeline=minutes/season=<season>/occurred_date_pacific=<date>/<event-id>.{failure,recovery}.json` |

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

## Conditional player minutes

The `player_minutes` asset and `update_player_minutes` job maintain conditional-minutes estimates: how much a player plays when available. This is separate from the player-ratings/HCA updater, seasonal availability probabilities, forecast minute allocation, and website publication. Daily Pacific partitions use the configured season start. The first estimator version is `v1`.

| Input | Selection |
| --- | --- |
| Official slate | `NBAScheduleResource`, including its immutable schedule reference |
| First-44-minute exposure | Same-date accepted `nba_stints` JSON |
| Actual participation/minutes | Full-game playing time from the same validated stints; no direct box-score reads |
| Membership | Same-date roster `latest.json` selects the accepted `rosters_by_date` JSON for all 30 teams |
| Injury labels | Same-date accepted `injury_status_by_game_date/latest.json` selection, using resolved NBA IDs |
| Historical initialization | Verified conditional-only priors below; no refitting or seasonal injury overrides |
| Previous state | Immediately preceding Pacific date of `player_minutes`, same season/model version |

All output paths below are under `s3://nba-gestalt-data/`.

| Output | S3 key |
| --- | --- |
| Conditional priors | `lake/artifacts/player_minutes/season=<season>/model_version=<version>/priors/<sha256>.json` |
| Immutable daily state | `lake/artifacts/player_minutes/season=<season>/model_version=<version>/game_date_pacific=<date>/<sha256>.json` |
| Accepted date selection | Same date prefix, `latest.json`, containing the immutable URI and checksum |
| Failures and recoveries | `lake/artifacts/errors/pipeline=minutes/season=<season>/occurred_date_pacific=<date>/<event_id>.{failure,recovery}.json` |

The registered 2026-27/v1 prior checksum is `08081b51d754bd3fa0abef0a9996d7de8cffaaaa67480833a4c53ca701f4ed47`. It preserves 528 historical starting estimates as a lookup, 1,599 historical-role entries, initialization provenance, and the accepted cold-start cohort. The 528 IDs do not define opening-night membership. Dated roster membership determines introductions. Box-score or injury-report-only players do not enter the observation population. Frozen rookie estimates are reused; later entrants use historical role when available or the approved cold-start median. Known players retain their state when returning after a gap.

The daily JSON contains exact input and parent references, estimator/settings/code identities, every current roster player plus retained off-roster histories, up to three dated eligible observations per player, estimates, and the day's classifications and coverage checks. The date means **updated through games on that date**, available for the following day's forecasts.

A played observation measures regulation exposure before minute 44, scales it by 48/44, and applies the approved prior-relative clipping. The estimate combines the original historical prior (weight 0.2 games) with up to three eligible observations. Healthy DNPs, Available/zero-minute cases, inactive-only cases, and G League assignments count as real zeros. Zero-minute Probable/Questionable/Doubtful/Out cases are excluded, except G League assignments. Medical gaps and days without a team game add no observation. No-game dates still carry state forward.

For a single date, select its partition on `update_player_minutes`; no per-player YAML is needed. For a season backfill, select `player_minutes` on the **asset page**, choose the date range, and launch an asset backfill. Its previous-date dependency enforces date order with one partition per run. The first configured date initializes from priors; later dates require the exact previous state. A failure holds the requested later dates. Do not use a job backfill to assume this ordering.

Rerunning an unchanged date reuses identical JSON. Changed inputs produce a new immutable file before conditionally advancing the pointer. A correction to an earlier date needs an explicitly approved replay of the following dates; automation does not rebuild historical suffixes. Old versions and frozen forecasts remain untouched. Model versions have isolated priors, parents, and pointers.

`update_player_minutes_schedule` runs `update_player_minutes` once daily at **4 a.m. America/Los_Angeles**, for the previous Pacific date. It replaces the former updater sensor; no automatic update is requested merely because an input changes during the day. The existing 3 a.m. PBP capture triggers `parse_nba_games`; `build_nba_rosters` runs at 3:30 a.m. Scheduled updates verify that the latest parser and roster runs for that partition succeeded after the scheduled morning's Pacific midnight. Missing, failed, running, or stale upstream work causes a visible failure through the existing S3 error path. Repair the upstream job, then manually re-execute the failed updater. Manual date launches and ordered backfills use saved S3 inputs without requiring recent upstream run history. Missed dates are not automatically backfilled.

A failed run records its exact inputs and diagnostics in S3 and Dagster. After repairing the input, **Re-execute from failure** writes a linked recovery record on success. A new run from scratch does not claim recovery. Failed validation leaves the accepted state untouched.

Shared NBA team IDs, Pacific/Eastern timezone objects, the default data bucket, and the error-root prefix live in `gestalt_pipelines/utils/__init__.py`. Estimator settings and pipeline-specific paths remain with their owning pipeline.

Initial v1 validation covered October 20 through November 10, 2026: 22 ordered daily states and all 160 games, including the November 3 no-game carry. The updater now uses the fixed 4 a.m. schedule described above. The November 12 roster-pointer backfill verified all 23 dates from October 20 through November 11 without changing selected roster versions. Detailed arithmetic, named-player checks, remaining acceptance, and the exact accepted references are recorded in [issue #74](https://github.com/EvanZ/nba-lineup-model/issues/74).

## Game-specific minutes projections

The game_minutes asset and project_game_minutes job live in the minutes pipeline. Each Pacific date partition writes separate game files, each containing both teams and all players in the selected rosters. This supplies allocations for the future prediction/API integration; it does not run ratings/HCA updates, collect odds, or publish a website release.

The calculation ports the existing Schedule-tab behavior: apply Out=0, Doubtful=0.25, Questionable=0.5, Probable=0.75, Available=1; keep the top 15 injury-adjusted weights per team; allocate 240 minutes proportionally. Unlisted players in a pending team report retain an unknown public status and use multiplier 1 for allocation. Individual allocation caps remain deferred. The existing daily updater's prior-relative clipping also remains unchanged.

Player identity and injury reconciliation belong to injury_status_by_game_date. This asset consumes that accepted data directly by NBA player ID. It does not parse reports, repeat name matching, read the repair dictionary, or require every injury-report row to appear in the allocation roster. Only rostered players enter the allocation pool; additional accepted Out rows do not block a projection.

| Output | S3 key under nba-gestalt-data |
| --- | --- |
| Immutable per-game JSON | lake/artifacts/game_minutes/season=<season>/model_version=v1/game_date_pacific=<date>/game_id=<id>/<sha256>.json |
| Accepted game selection | Same game prefix, latest.json |
| Failures/recoveries | lake/artifacts/errors/pipeline=minutes/season=<season>/occurred_date_pacific=<date>/<event_id>.{failure,recovery}.json; stage game_minutes |

The JSON records each player's conditional estimate, injury status and multiplier, adjusted weight, selected-rotation flag, final minutes, both team totals, exact input references/checksums, model settings, and implementation identity. Identical inputs reuse the immutable output and leave its selection unchanged. Conditional pointer writes prevent competing runs from silently replacing each other.

For live partitions (today/tomorrow), the job requires the accepted daily minutes state for exactly yesterday and the newest accepted roster snapshot dated no later than today. Evaluated on November 13, both November 13 and November 14 use minutes updated through November 12. Reconciled injury data is selected for each game's Pacific date. Each game stops advancing at scheduled tipoff. A job that finishes after tipoff leaves the accepted pointer intact.

`project_game_minutes_sensor` checks accepted source selections about once a minute and requests upcoming date jobs when the daily baseline, roster, injury, or schedule version changes. On enable, its first tick immediately checks existing inputs: it creates missing projections and refreshes older ones for today/tomorrow without waiting for another PDF. An unchanged tick creates no new work. It skips dates already running and unchanged projections; it does not automatically backfill history. A failed input selection can be repaired and the job re-executed from failure to create a linked recovery record.

For historical backfill, select the past date partitions of game_minutes, or the project_game_minutes job. Each date reads the immediately previous date's accepted daily state: November 4 uses November 3. Opening day uses the approved frozen starting priors directly, without inventing a previous-day state file. Other dates fail if their required previous state is missing. Historical membership comes from that date's accepted rosters_by_date selection, and injuries come from the producer's saved pre-scheduled-tipoff selection. These game projections are independent date calculations, so they do not require another ordered learning-state replay. The game asset backfill uses one partition per run. A date without games succeeds without creating game files.

Historical files carry purpose=historical_replay and explicitly identify reconstructed game-date roster evidence. They are reconstructed results, not authenticated projections published before the game. Existing files with purpose=pregame_projection are preserved: backfilling a historical date never replaces its genuinely saved pregame selection. Historical runs do not retrain models, alter daily minutes states, or publish/replace website forecasts.

## Shared player names

`NBAPlayerNamesResource` is a shared Dagster resource exported from `gestalt_pipelines.resources`. It reads `s3://nba-gestalt-data/lake/raw/playerindex/<configured-season>.json` and the matching `.meta.json` sidecar. `load()` returns an NBA player ID → official name dictionary plus exact raw/metadata references and SHA-256 checksums. No asset, job, schedule, generated lookup, or new S3 location is required.

The resource verifies the official NBA PlayerIndex URL, season, historical/all-team request parameters, source checksum, and player identities. It reads S3 on each call, so a replacement upload is used by the next run once its metadata checksum is updated. The current-season upload includes historical names. Player identity is separate from dated team membership; the existing season-scoped injury repair dictionary continues to handle spelling/name-order exceptions such as Cenac and Yang. Repairs are not copied into PlayerIndex or a second alias file.

### Manual PlayerIndex acquisition (November 12, 2026 Pacific)

EC2 requests to the NBA Stats PlayerIndex endpoint timed out. Evan obtained the official response through the browser and uploaded it to `s3://nba-gestalt-data/lake/raw/playerindex/2026-27.json`. This is an operator acquisition of official NBA data, not an alternate data provider.

The uploaded file contains **5,197 player identities**, SHA-256 `251af366e1fe85491b2543e736f8e67ff91cb0881c97975be8cfd428e887b129`. Its sidecar is `s3://nba-gestalt-data/lake/raw/playerindex/2026-27.meta.json`, recording the operator-supplied NBA URL/parameters, checksum, and `acquisition_method: manual_browser_response`. The S3 upload time is `2026-11-13T02:54:40+00:00` (November 12 at 6:54:40 p.m. Pacific). The exact download time is unknown, so `fetched_at` is explicitly null; do not interpret upload time as download time.

The source request was `https://stats.nba.com/stats/playerindex?College=&Country=&DraftPick=&DraftRound=&DraftYear=&Height=&Historical=1&LeagueID=00&Season=2026-27&SeasonType=Regular+Season&TeamID=0&Weight=`. Existing historical raw files end at 2025–26. No scheduled PlayerIndex refresh is enabled. The shared name resource reads this saved input directly. Injury reconciliation uses it with the newest accepted roster dated on or before the game date and the existing repair dictionary.

## Injury roster selection and identity

`injury_status_by_game_date` declares its dependency on `rosters_by_date`. Every manual, backfill, live, or failed-run retry selects the newest accepted roster **dated on or before the game-date partition**, then reads that date's `lake/artifacts/rosters/season=<season>/game_date_pacific=<selected-date>/latest.json` through `NBARosterArchive`. It verifies the selected immutable file and records its URI/checksum in `inputs.rosters` and its actual date in `roster_dates_pacific`. November 13 can use November 12 until a November 13 selection exists; historical dates never use later-dated rosters. No further roster run is required when an eligible selection exists. Fail clearly only when no eligible roster exists.

The ID-only membership list is paired with names from `NBAPlayerNamesResource` for existing team-scoped injury name matching. A name absent from the selected team roster is checked against the full official PlayerIndex. A unique match retains the report's team/status, records `identity_source=player_index` and `identity_warning=not_on_selected_roster`, and emits a warning without changing roster membership. Unknown/ambiguous names and conflicting repairs still produce reconciliation errors; the season repair dictionary continues to resolve name exceptions, including IDs absent from the roster. No additional downstream roster-membership gate is introduced. `inputs.player_names` records the raw PlayerIndex and metadata references/checksums. Report selection, scheduled-tipoff cutoff, player repair location, status layout, error/recovery layout, and model version are unchanged.

`injury_report_dates_ready` checks today/tomorrow's open games about once a minute. Its run key includes selected extractions, newest eligible roster pointer, PlayerIndex references, schedule, and repair checksum, so a roster-only or repair change can trigger reconciliation. The asset selects the newest eligible roster again at execution rather than receiving a fixed opening roster in launch config. If no eligible roster exists, the job fails clearly; publishing an eligible roster changes the sensor key and permits a new run. Raw PDF collection/extraction remain independent.

Failed-run retries preserve the failed attempt's selected schedule/reports for recovery comparison, but read the newest eligible roster and current PlayerIndex/repair inputs. Latest status publication checks those selections again, leaving the pointer unchanged if an input changed during processing. Existing old runs whose launch YAML contains removed `roster_metadata` must be launched as fresh date runs under this config; no legacy opening-roster override is supported. New failed runs can be re-executed from failure after repairing S3 inputs.

Reconciliation backfills and any resulting ordered daily-minutes replay/game-minutes correction require their separately agreed date scope. Use the agreed date scope and owning producers; input publication and downstream model replay are separate operations.

### Known-player roster warnings

Known PlayerIndex identities absent from the selected team roster do not fail reconciliation. Their report rows are retained with a warning flag. Each run writes an immutable event to the existing injury error directory:

`lake/artifacts/errors/pipeline=injury_reports/season=<season>/occurred_date_pacific=<run-date>/<event-id>.warning.json`

The event has `status=warning`, player/game/team/report details, exact roster/PlayerIndex/repair/extraction inputs, the status artifact path, and the Dagster run ID. Occurrence date is when processing ran, including during historical backfill. Dagster also emits a WARNING log and includes `warning_count` and warning artifact references in run materialization metadata. No email/push integration is added. Genuine unresolved or ambiguous identities continue to fail the run with `.failure.json`; a linked successful retry can emit both a recovery and a warning.

Examples are Blake Hinson (Utah, October 21) and Tristan Enaruna (Cleveland, November 7): the official injury report lists each player, while the date's reconstructed roster excludes him after a date-only waiver. PlayerIndex identifies them without adding repair-dictionary entries. Resolver code version v5 identifies this processing change; conditional-minutes model_version remains v1. Later approved minutes replays are recorded in the verification section below.

## Verified operation — November 13, 2026 Pacific

The current implementation and real-run evidence are recorded in [#75](https://github.com/EvanZ/nba-lineup-model/issues/75); estimator/allocation decisions are in [#74](https://github.com/EvanZ/nba-lineup-model/issues/74). This map describes the current contracts rather than earlier temporary automation states.

| Check | Verified result |
| --- | --- |
| Pipeline tests | 212 passed, including source selection, contract boundaries/renewals, dated repairs, 18/19-player warnings, minutes lineage, and projection cutoff behavior |
| Initial box-score removal from roster inputs | Backfill `iwobzcip`, Oct 20–Nov 12, 24/24 successful; all 720 team/date memberships checked; downstream pointers unchanged during that roster-only repair |
| Membership repair | Approved Nance signing repair is read from S3 and recorded by checksum; Nov 11/12 user runs verified |
| Contract-expiration correction | Backfill `rjtddkjg`, Nov 8–12, 5/5 successful; Wesley removed from the Clippers, which have 18 players; Nance is on Milwaukee |
| Ordered minutes correction | Five successful `update_player_minutes` runs, Nov 8–12, each using the corrected roster and immediately preceding corrected state; exact preseason priors preserved; no new Wesley observations after expiry |
| Historical game projections | All 174 accepted per-game pointers through Nov 12 unchanged during that latest correction; old immutable roster/minutes artifacts preserved |
| Automatic game projections | Initial enabled tick at 05:52 on Nov 13: 9 games for Nov 13 and 5 for Nov 14, both successful; 14 accepted per-game files subsequently read back with valid checksums and the Nov 12 daily baseline |
| Morning updater | `update_player_minutes_schedule` loaded and RUNNING at 04:00 Pacific; Nov 12 state was manually completed and subsequently corrected above. The first scheduled 04:00 execution after activation is Nov 14 and has not yet been observed. |
| Collection | Injury, odds, and scoreboard schedules remain RUNNING independently |

The source-removal regression was Kevin McCullar Jr.: an archived zero-minute box listing had reintroduced him to New York after his Oct 23 waiver; transaction-based reconstruction keeps him absent until the Nov 5 Toronto signing. Roster builds no longer infer membership from box appearances. The stint parser still reads boxes for its starting lineups and validation, as described above.

Correction evidence is retained on EC2 in `/tmp/75-ten-day-before.json`, `/tmp/75-ten-day-runs.json`, and `/tmp/75-ten-day-validation.json`. The ticket records permanent run IDs and acceptance evidence. Historical data corrections, ordered model replay, and any historical game-projection correction require their agreed scopes; running a roster date does not itself authorize a model replay or website publication.
