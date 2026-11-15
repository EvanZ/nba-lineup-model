# GESTALT pipeline map

The Dagster code location is **nba_gestalt_pipelines**. Its definitions live in [the GESTALT project](gestalt/src/gestalt_pipelines/definitions.py). The pipelines use the season set by `GESTALT_NBA_SEASON`. The daily partitions start on the date set by `GESTALT_NBA_SEASON_START_DATE`. See the [example configuration](gestalt/code-location.env.example).

**Issue #79 code is loaded. Both approved Oct 20–Nov 13 learned-state backfills completed successfully; minutes/ratings schedules and the game-minutes trigger are restored. Protected collection remained enabled. The approved historical projection replay is complete; the next overnight schedule verification remains pending.**

Current reference reviewed November 14, 2026 Pacific. The sections below cover inputs, accepted S3 selections, automatic timing, manual/backfill operation, and recovery. Runtime enablement is a dated snapshot; use Dagster for current run state.

## Partition dates — issue #79

A partition describes the data date, independently of when the job executes. Game dates use the Pacific date of scheduled tipoff, including games finishing after midnight. Execution and source-observation timestamps remain separate.

| Assets / jobs | Partition meaning | First available date | Last available date |
| --- | --- | --- | --- |
| Season schedule, rating initialization | Configured season; no daily partition | — | — |
| Injury capture / extraction | Eastern publication date | Configured season start | Today Eastern |
| Injury reconciliation | Pacific game date | Configured season start | Tomorrow |
| Odds | Pacific capture date; a snapshot can cover many game dates | `GESTALT_ODDS_CAPTURE_START_DATE` (Oct 26) | Today |
| Scoreboard | Pacific slate date | `GESTALT_SCOREBOARD_START_DATE` (Nov 10) | Today |
| Box scores, transactions, rosters | Pacific game / transaction / roster date | Configured season start | Today |
| PBP, stints | Completed Pacific game date | Configured season start | Yesterday |
| Player minutes, ratings/HCA | Learned evidence through completed Pacific date D | Configured season start | Yesterday |
| Game minutes, game margins | Target Pacific game date; reads learned state D−1 | Configured season start | Tomorrow |
| Actual-minutes counterfactual | Completed Pacific game date of the saved prediction | Configured season start | Yesterday |

Tomorrow's provisional forecast uses the same state available today, then refreshes after the morning update. Historical dates require their exact D−1 state. Opening night uses real frozen priors, with no invented prior-day daily state. No-game days still create unchanged learned-state carry-forwards.

Odds retain their timestamped S3 snapshot layout. Historical date materializations validate genuine archived captures and record their URI/checksums without provider calls or writes to `latest.json`. Scoreboard historical dates validate their saved slate; the current endpoint cannot supply historical slates. Active games after midnight continue updating their original slate partition. Genuine missing archive dates fail visibly.

At 03:30, today's roster can be initialized before its first transaction poll using the known earlier dated files. Its inputs explicitly record `current_date_transactions_pending` and the latest transaction date actually read. A later transaction poll refreshes it normally. Missing historical transaction dates still fail; no empty poll is invented.

The first genuine odds/scoreboard dates are configured in the tracked `code-location.env.example`; local `code-location.env` remains ignored. The package rereads that existing local file on code-location reload, so editing its quoted settings does not require a service restart. All agreed S3 paths remain unchanged. The producer/reader switch and Oct 20–Nov 13 learned-state backfills were explicitly approved before loading this build. Additional downstream backfills require their own approval. Original frozen forecasts and pinned inputs remain preserved. Replay validation compares old effective-date D+1 values to new completed-date D values before proceeding downstream.

## Data flow

~~~mermaid
flowchart LR
    playerIndex["Official PlayerIndex JSON and metadata in S3"] --> playerNames["Shared NBAPlayerNamesResource"]
    playerNames --> reconcile
    playerNames -->|New-player warning labels| ratings
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
    rosterMorning["03:30 Pacific: yesterday and today"] --> rosterBuilder
    rosterBuilder --> rosterVersions["30-team roster versions and date latest.json in S3"]
    rosterVersions -->|"Newest roster date <= game date"| reconcile
    frozenRatings["Approved frozen NAIL priors, HCA and fixed B2B"] --> ratingPriors["player_rating_priors"]
    ratingPriors --> ratingPriorsJSON["Initialization JSON versions and priors/latest.json"]
    ratingPriorsJSON --> ratings["player_ratings: completed-date update"]
    stintsArchive --> ratings
    rosterVersions --> ratings
    catalog --> ratings
    ratings --> ratingsJSON["Joint player/HCA state and date latest.json"]
    priorMinutes["Frozen conditional priors and historical-role lookup"] --> minutes["player_minutes"]
    stintsArchive --> minutes
    rosterVersions --> minutes
    statuses --> minutes
    catalog --> minutes
    minutes --> minutesArchive["Versioned conditional-minutes state and date latest.json"]
    minutesArchive -->|"Previous Pacific date"| minutes
    minutesArchive --> gameMinutes["game_minutes"]
    priorMinutes --> gameMinutes
    minutesMorning["04:00 Pacific: completed yesterday"] --> minutes
    projectionSensor["project_game_minutes_sensor: today/tomorrow"] --> gameMinutes
    rosterVersions --> gameMinutes
    statuses --> gameMinutes
    catalog --> gameMinutes
    gameMinutes --> gameAllocations["Per-game minutes JSON and latest.json"]
    gameAllocations --> predictions["game_predictions: predict_game_margins_sensor"]
    ratingsJSON --> predictions
    catalog --> predictions
    predictions --> predictionFiles["Per-game prediction JSON and latest.json"]
    predictionFiles --> actualMinutesPredictions["game_predictions_actual_minutes: daily 04:30 Pacific"]
    boxArchive --> actualMinutesPredictions
    catalog --> actualMinutesPredictions
    actualMinutesPredictions --> counterfactualFiles["Postgame counterfactual JSON and latest.json"]
    predictionFiles --> marketPerformance["market_performance_by_date"]
    counterfactualFiles --> marketPerformance
    boxArchive --> marketPerformance
    oddsArchive --> marketPerformance
    catalog --> marketPerformance
    actualMinutesPredictions -->|"Successful same-date job"| marketPerformance
    marketPerformance --> performanceFiles["Date performance JSON versions and latest.json"]
    performanceFiles --> marketConsumers["Schedule market summary and calendar: integration under #71"]

~~~

The arrows show saved-data inputs and the explicitly labelled automatic triggers. Box scores feed stint reconstruction; the daily minutes updater reads the resulting stints rather than reading box scores directly. Injury reconciliation and odds capture load the existing catalog through [NBAScheduleResource](gestalt/src/gestalt_pipelines/resources/nba_schedule.py), so the schedule job runs on its own cadence. The scoreboard runs in this same code location. Website and forecast integration will use the injury status handoff in separate work.

## Timing and automatic triggers

Times are Pacific (`America/Los_Angeles`) except the injury publication schedule, which uses Eastern (`America/New_York`). Date boundaries follow those named timezones through daylight-saving changes.

| Job | Schedule or sensor | When it requests work |
| --- | --- | --- |
| `capture_nba_season_schedule` | On demand | Normally once per season |
| `predict_game_margins_actual_minutes` | `predict_game_margins_actual_minutes_schedule`, 04:30 Pacific | Yesterday's completed game date; use each saved prediction and final box-score minutes |
| `evaluate_market_performance` | `evaluate_market_performance_sensor` | Same completed date after its actual-minutes counterfactual job succeeds; manual date jobs/backfills use the same asset |
| `predict_game_margins` | `predict_game_margins_sensor` | About every minute; seed or refresh open games today/tomorrow when accepted ratings, game minutes or schedule change |
| `initialize_player_ratings` | On demand | Prepare approved frozen initialization |
| `update_player_ratings` | `update_player_ratings_schedule`, 04:15 Pacific | Yesterday's completed ratings/HCA; first completed date starts from approved initialization |
| `capture_injury_pdfs` | `injury_pdf_poll` | Eastern :02, :17, :32, :47; checks the preceding hour |
| `extract_injury_pdf` | `new_injury_pdf` | About every minute; new saved PDFs in today's/yesterday's Eastern partitions |
| `reconcile_injury_date` | `injury_report_dates_ready` | About every minute; changed accepted inputs for open games today/tomorrow |
| `archive_nba_market_odds` | `nba_market_odds_game_day` | 08:20, 12:20, and 15:20–19:20; source/window and quota checks apply |
| `capture_nba_scoreboard` | `scoreboard_poll` | Every minute during the configured game window until final scores are saved |
| `capture_nba_box_scores` | `box_score_poll` | Every five minutes during the game window until final boxes are saved |
| `capture_nba_play_by_play` | `play_by_play_morning` | 03:00 for yesterday |
| `parse_nba_games` | `play_by_play_ready_for_stints` | After that scheduled PBP capture succeeds, for the same date |
| `capture_nba_transactions` | `transactions_hourly` | Hourly 08:00–20:00 and 23:59 for today; 03:15 explicitly checks yesterday; daytime polls also check yesterday |
| `build_nba_rosters` | `rosters_morning`; `roster_transactions_ready` | 03:30 finalizes yesterday and initializes today; new transaction rows still trigger affected dates |
| `update_player_minutes` | `update_player_minutes_schedule` | 04:00 completes yesterday's learned state; requires successful same-date parser and roster runs that morning |
| `project_game_minutes` | `project_game_minutes_sensor` | About every minute; eligible games today/tomorrow when accepted inputs differ |

The existing automation and both prediction triggers are enabled in the permanent instance. The prediction sensor and 04:30 counterfactual schedule were enabled on November 14, 2026; the first overnight counterfactual run is still future. The former `minutes_ready` / `update_player_minutes_sensor` is retired; `game_minutes_ready` was renamed `project_game_minutes_sensor`. Definition defaults may be STOPPED; the table describes enabled instance state. Manual partition runs and backfills remain available alongside automation.

The morning sequence is 03:00 yesterday's PBP → successful capture triggers same-date parsing; 03:15 yesterday's transaction check; 03:30 yesterday's final roster and today's starting roster; 04:00 yesterday's learned minutes; 04:15 yesterday's ratings/HCA; 04:30 yesterday's actual-minute counterfactuals. Both learned states are used for today's games. The counterfactual uses its saved prediction's exact ratings and final boxes, independently of that morning's new learned states. The updater fails visibly if the scheduled parser/roster prerequisites are missing, failed, still running, or stale. Repair them and re-execute the updater manually. Injury reconciliation and game allocation read their approved accepted inputs; neither requires the injury artifact and allocation roster to reference an identical roster version.

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

The schedule runs daily at 8:20 a.m., 12:20 p.m., and hourly from 3:20 through 7:20 p.m. Pacific. The request includes ten bookmakers, including `espnbet`. Bookmaker entries supplied by The Odds API are preserved, including ESPN-branded names; ESPN websites/APIs are not sources. Publication checks the recorded provider and season rather than rejecting text containing ESPN. The 50-credit reserve remains unchanged.

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

Conditional estimates describe how much a player plays when available. Completed-date partition D includes games through D, is produced on D+1, and supplies forecasts for D+1. It is separate from ratings/HCA, seasonal availability probabilities, game minute allocation and website publication. The estimator stays v1; no numerical model rule changes in #79.

| Input | Selection |
| --- | --- |
| Official slate | `NBAScheduleResource`, including its immutable schedule reference |
| First-44-minute exposure | Accepted `nba_stints` JSON for D−1 |
| Actual participation/minutes | Full-game playing time from the same validated stints; no direct box-score reads |
| Membership | D−1 roster `latest.json` selects the accepted `rosters_by_date` JSON for all 30 teams |
| Injury labels | D−1 accepted `injury_status_by_game_date/latest.json` selection, using resolved NBA IDs and each game’s scheduled-tipoff cutoff |
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

The daily JSON contains exact input and parent references, settings/code identities, current roster players plus retained histories, up to three dated eligible observations per player, estimates, classifications and coverage checks. `game_date_pacific` and `observed_through_game_date_pacific` are both D; `effective_from_game_date_pacific` is D+1. `date_contract=completed_game_date_v1` distinguishes these states from the old effective-date files. November 13 includes games through November 13 and supplies November 14 forecasts.

A played observation measures regulation exposure before minute 44, scales it by 48/44, and applies the approved prior-relative clipping. The estimate combines the original historical prior (weight 0.2 games) with up to three eligible observations. Healthy DNPs, Available/zero-minute cases, inactive-only cases, and G League assignments count as real zeros. Zero-minute Probable/Questionable/Doubtful/Out cases are excluded, except G League assignments. Medical gaps and days without a team game add no observation. No-game dates still carry state forward.

For a single date, select its partition on `update_player_minutes`; no per-player YAML is needed. For ordered backfill, select `player_minutes` on the asset page and launch the approved date range. The first completed date initializes from approved priors and applies that day's observations; later dates require the exact preceding completed state. Its self dependency enforces chronological asset backfills with one date per run. A failed date holds later requested dates.

Rerunning an unchanged date reuses identical JSON. Changed inputs produce a new immutable file before conditionally advancing the pointer. A correction to an earlier date needs an explicitly approved replay of the following dates; automation does not rebuild historical suffixes. Old versions and frozen forecasts remain untouched. Model versions have isolated priors, parents, and pointers.

`update_player_minutes_schedule` requests yesterday at 04:00 Pacific. Scheduled runs require that morning's successful parser and roster runs for the same completed date. Missing, failed, running or stale work causes a visible failure under the existing error path. Repair upstream work, then re-execute the updater. Manual launches/backfills read saved S3 inputs without requiring fresh run history. Missed dates require explicit backfill. The updater has no sensor.

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

For live partitions (today/tomorrow), game minutes read yesterday's completed minutes state and the newest approved roster no later than today. On November 14, both November 14 and the provisional November 15 preview use the November 13 learned state. The next morning refreshes the preview from completed November 14 evidence. Missing required states cause a visible wait/failure. Reconciled injuries remain game-dated. Each game stops advancing at scheduled tipoff.

`project_game_minutes_sensor` checks accepted source selections about once a minute and requests upcoming date jobs when the daily baseline, roster, injury, or schedule version changes. On enable, its first tick immediately checks existing inputs: it creates missing projections and refreshes older ones for today/tomorrow without waiting for another PDF. An unchanged tick creates no new work. It skips dates already running and unchanged projections; it does not automatically backfill history. A failed input selection can be repaired and the job re-executed from failure to create a linked recovery record.

Historical `project_game_minutes` date D reads completed minutes state D−1: November 4 uses the November 3 file. Opening day reads approved frozen priors directly. Every other historical date requires its exact preceding state, without substituting a newer state. Historical rosters retain their game dates; injury reconciliation supplies the saved pre-scheduled-tipoff report selection. These game dates can run independently after their learned states are available. No-game dates succeed without game files.

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


## Ratings/HCA — issue #76

Asset **`player_rating_priors`**, manual job **`initialize_player_ratings`**, group **`ratings`**. Launch in the existing `nba_gestalt_pipelines` code location with no run configuration. The resource uses `GESTALT_NBA_SEASON`; only the approved 2026–27/v1 input selection is currently registered. Initialization is unpartitioned and has no schedule or sensor. The manual daily updater is available as described below. Replacement-level introductions and warnings are implemented below. The ordered season replay and 4:15 a.m. Pacific schedule were validated on November 14; website/forecast integration remains separate.

The job preserves the approved numerical NAIL priors, joint HCA prior and fixed B2B coefficient. A first initialization checks four source objects before parsing them: the catalog, its metadata, HCA settings, and fixed B2B metadata. An existing accepted initialization is checksum-verified and reused without requiring its earlier unused model/profile sources. No stat-profile, lineup-composition, or aging calculation is part of these jobs. No model training, legacy application imports, website publication, minutes update, or historical forecast rewrite occurs.

All paths below are in `s3://nba-gestalt-data/`. Exact source paths and byte hashes are registered in `gestalt/src/gestalt_pipelines/ratings/utils/sources.py` and recorded in the output's `inputs`.

| Input | S3 location |
| --- | --- |
| Original 524 ratings and catalog metadata | `nba-gestalt/releases/gestalt-web-d739c28/artifacts/web/preseason_rankings/forward_nail_rapm_v1212_residualized_lambda/forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25/2026-27.{parquet,json}` |
| Fixed B2B source | `nba-gestalt/releases/gestalt-web-d739c28/artifacts/models/forward_nail_rapm_v1212_residualized_lambda/2025-26/forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25/season_schedule_control_metadata.parquet` |
| Accepted HCA settings | `lake/artifacts/site_release/season=2026-27/game_date_pacific=2026-11-02/runtime_sha256=042869e115b05d0895f14db3f85568dabb72b0788ae97425e7a1f5df21688e3f/code/deploy/causal-hca-replacement-candidate-v2.json` |

| Output | S3 key |
| --- | --- |
| Immutable initialization JSON | `lake/artifacts/player_ratings/season=<season>/model_version=v1/priors/<sha256>.json` |
| Accepted initialization pointer | `lake/artifacts/player_ratings/season=<season>/model_version=v1/priors/latest.json` |
| Failure/recovery records | `lake/artifacts/errors/pipeline=ratings/season=<season>/occurred_date_pacific=<processing-date>/<event-id>.{failure,recovery}.json` |

An identical rerun verifies and reuses the saved initialization. Publication reads the immutable JSON back before creating `latest.json`; a competing selection fails visibly. The job refuses to replace a different already-selected initialization: a replacement requires the separately approved model-version/replay decision. Missing or changed source objects fail without substituting another model or changing the accepted pointer. After repairing an actual failure, use **Re-execute from failure** to record the linked recovery; an unrelated fresh success does not claim recovery.

The original 524 catalog includes later additions documented in its retained metadata; it is the approved numerical lookup, not proof of opening-night team membership. Opening-day initialization retains all 524 saved numerical priors. Later roster inputs supply team membership. The existing immutable initialization can retain unused profile/model provenance, which the updater does not use or refresh.


### Daily ratings/HCA update

Asset **`player_ratings`**, job **`update_player_ratings`**, group **`ratings`**. Partition D means learned ratings/HCA including completed games through D, normally produced at 04:15 on D+1. Available dates run from configured season start through yesterday. The first date applies opening games to the complete approved frozen catalog; later dates require the immediately preceding completed state. Opening-game forecasts use `player_rating_priors` directly. Selecting this job does not rerun initialization or source collection. Scheduled runs require fresh successful same-date parser and roster runs; failures need manual repair. No ratings sensor is added.

Completed date D reads approved initialization, stints and rosters for D, the shared NBA schedule for D and the preceding date's B2B flags, and completed state D−1. Stints/rosters are same-date Dagster dependencies; the learned parent is the previous date. It reads no raw box scores, injury reports, odds, minutes states, web API or legacy Python model. All source URI/checksums remain recorded; played stint evidence is usable even when the player is absent from the final roster.

The calculation uses full-game and overtime evidence, with positive possession weights. Zero-possession intervals remain in the upstream stints artifact and are counted in diagnostics, while contributing no weighted evidence. Frozen starting NAIL and the B2B coefficient remain fixed. For each stint, the scoring residual is `observed home net rating − fixed B2B offset − signed frozen NAIL sum`. HCA is estimated jointly rather than subtracted as a fixed adjustment. No assists, usage, rebounding, additive-profile rescoring, or remaining lineup-composition terms are calculated. The solve updates player residuals and absolute HCA jointly, retaining all player/HCA cross terms and uncertainty; yesterday's learned estimates do not become a replacement prior.

| Output | S3 key under `s3://nba-gestalt-data/` |
| --- | --- |
| Immutable joint ratings/HCA state | `lake/artifacts/player_ratings/season=<season>/model_version=v1/game_date_pacific=<date>/<sha256>.json` |
| Accepted date selection | `lake/artifacts/player_ratings/season=<season>/model_version=v1/game_date_pacific=<date>/latest.json` |
| Daily failures/recoveries/warnings | `lake/artifacts/errors/pipeline=ratings/season=<season>/occurred_date_pacific=<processing-date>/<event-id>.{failure,recovery,warning}.json`; records identify `stage=player_ratings` and the game date |

In the JSON, `game_date_pacific` and `observed_through_game_date_pacific` are D, `effective_from_game_date_pacific` is D+1, and `date_contract` is `completed_game_date_v1`. Forecast D reads state D−1. New daily updates reject old effective-date predecessors. A saved frozen forecast can still read its exact old immutable source URI/checksum and recorded cutoff; this does not permit selecting an old-date file as a new predecessor. Original immutable versions are preserved.

The JSON contains all retained player priors/ratings, current roster team IDs, exposure, uncertainty, initialization lineage, HCA, ordered player IDs, the full joint matrix and score/exposure vectors, cumulative game IDs, and exact parent/input references. No-game dates carry the evidence forward. An identical rerun reconstructs the date from the same predecessor and inputs; it does not add its games twice. Source changes, missing coverage, or concurrent publication fail visibly while preserving accepted selections. Repair failed inputs and use **Re-execute from failure** for a linked recovery. Earlier corrections requiring later model dates to be rebuilt remain a separately approved suffix replay.

Player initialization follows the operator-approved rule: retain an existing season state; otherwise reuse an approved saved prior from the 524-player catalog; otherwise assign the fixed replacement starting NAIL **-5.109488431706889** with `forecast_source=in_season_replacement_cold_start_prior`. This is ID-based and has no named exceptions. Injury, delayed first appearance, absence and team changes do not discard a saved prior or reinitialize a retained player. Historical aging prediction and rookie-model generation are not applied to new arrivals under this decision.

Each replacement introduction emits a Dagster warning and a `.warning.json` in the ratings error path before advancing the accepted date selection. The warning identifies the NBA player ID, official name when available, assigned starting rating, game date, run ID and exact inputs. The existing shared `NBAPlayerNamesResource` reads saved NBA PlayerIndex names only when new replacement IDs are present; a missing name is reported by ID. The warning itself does not fail the job. A failure to persist its warning record blocks publication visibly. The materialization metadata includes `new_replacement_players` and `ratings_warnings`. Later dates retain the state and do not generate another introduction warning; re-executing the introduction date can produce another run-linked warning.

The original frozen initialization is retained. Its earlier model/profile provenance is unused; no fitted-model loader or replacement stat-profile defaults remain in ratings. Existing roster/minutes/game-projection history and the final coordinated backfill/release remain deferred at Evan's request.

Validation of this calculation used all three Oct 20 games and the original 524-player numerical priors, without Dagster materialization or S3 writes. The accepted legacy contribution and joint-HCA solver were evaluated with fixed B2B as the only fixed adjustment. Maximum absolute differences: matrix/exposure `0`, score `4.55e-13`, estimates `2.78e-16`, uncertainty `2.23e-16`. This checks the agreed B2B-only calculation; it does not claim parity with checkpoints that included lineup-composition adjustments.

Daily states carry `calculation_contract=frozen_priors_stints_joint_hca_fixed_b2b_v1`. New predecessors also require the completed-date contract. The following validation describes the preserved pre-#79 effective-date checkpoint: The approved November 14 replay rebuilt ratings effective Oct 20–Nov 14 through the canonical asset, preserving older immutable versions. All 26 dates and parent links were read back successfully, covering all 183 games through Nov 13. A fresh aggregation of those same game contributions exactly matches the accumulated matrix, response and exposure; maximum estimate difference is 8.88e-16. The final file retains 533 players and jointly estimated HCA 1.8073343269384732. The schedule is RUNNING; its first scheduled morning execution is still future. No website release or frozen forecast was rewritten.


## Historical checkpoint: November 14 effective-date migration before #79

**Preserved evidence for the earlier running build; superseded by the staged #79 convention above after cutover.**

Both `player_minutes` and `player_ratings` now label a file by the date its estimates are effective for; each declares `observed_through_game_date_pacific=D−1`. The 4 a.m. minutes job and 4:15 a.m. ratings job target today. Their numerical estimators remain separate; neither job depends on the other.

The approved minutes backfill `igdnkkuh` succeeded on all 25 effective dates Oct 21–Nov 14. Every player state, observation, three-game window and included-game list is exactly equal to its prior observation-date counterpart; only the output date contract/lineage changes. Opening-day game allocations continue to use the existing frozen minutes priors, with no artificial first-day observation.

Ratings backfill `kwlsknbm` succeeded on all 26 dates Oct 20–Nov 14. Ratings Oct 20 contains the original 524 scalar priors and initial HCA only. Both daily pipelines include all 183 games through Nov 13 in their Nov 14 states. Their final retained player counts are both 533. The combined minutes/ratings test suites pass 130 tests plus the legacy-pointer preservation regression (131 total), including date mappings, same-day leakage rejection, DST scheduling, old-state rejection, and frozen game selections.

Game-minutes backfill `shvsjdzn` succeeded on all 26 dates Oct 20–Nov 14. Readback covers all **188 games** (183 completed-season games through Nov 13 plus five Nov 14 games): 175 refreshed selections and 13 preserved frozen pregame selections. All 188 injury selections were checked against original report publication times before scheduled tipoff. The preservation check reads purpose from the immutable forecast, including older pointers that omit that field; their S3 timestamps verify they were not moved. No frozen pregame selection was replaced.

Both daily schedules are RUNNING: minutes at 04:00 and ratings at 04:15 Pacific, next targeting Nov 15 with games through Nov 14. `project_game_minutes_sensor` is RUNNING again. The protected injury, odds and scoreboard collectors were left enabled throughout. Evan approved committing and pushing this completed migration and closing #76. Website/full-game forecast integration and the optional context experiment (#77) remain separate work.


## Game-margin predictions — manual first slice (#78)

Asset **`game_predictions`**, job **`predict_game_margins`**, group **`predictions`**, in the existing `nba_gestalt_pipelines` code location. Select a Pacific game-date partition in the launchpad; no YAML or hand-prepared inputs are needed. The job consumes accepted per-game `game_minutes`, accepted daily `player_ratings`, and the shared `NBAScheduleResource`. It does not recalculate injuries, rosters, minutes, ratings, or models. The job selection contains only this new asset; upstream dependencies are visible in the graph without rerunning their producers.

Historical game date D reads completed ratings D−1. Open games today and tomorrow's provisional preview read yesterday's completed state. Each saved prediction records the actual evidence cutoff; game-minute inputs still belong to the target game date. Opening-night ratings/HCA and minutes read their approved frozen priors directly. No daily state with an invented prior-season date is created.

For each player, contribution is `projected_minutes / 48 * nail`. The home margin is the sum of home contributions minus away contributions, plus the selected daily HCA and `(home_is_b2b − away_is_b2b) * fixed_b2b_coefficient`. The schedule identifies previous-day games for those flags. Positive margin favors the home team. The approved **100 possessions** assumption makes this the predicted point margin. The saved home win probability is `min(1, max(0, 0.499583 + 0.030250 * predicted_home_margin))`. No pace fitting, profile statistics, non-additive lineup context, or odds enter this producer.

The JSON contains both team/player lists, conditional and allocated minutes, injury statuses/reasons/probabilities, rotation membership, scalar NAIL and each weighted contribution. It also includes team submission status and minute totals, explicit rating/minute evidence dates, HCA/B2B components, fixed probability parameters, source URIs/checksums, implementation identity, and generation time. Injury report publication time, PDF URI and source URL follow the exact injury/extraction references already recorded in `game_minutes`; a newer live report is never independently selected here. An upstream allocation with no eligible report retains a null report summary. All numbers retain full floating-point precision; display rounding belongs in the website.

| Output | S3 key in `nba-gestalt-data` |
| --- | --- |
| Immutable prediction | `lake/artifacts/game_predictions/season=<season>/model_version=v1/game_date_pacific=<date>/game_id=<NBA-game-id>/<sha256>.json` |
| Accepted per-game prediction | Same per-game directory, `latest.json` containing the immutable URI and SHA-256 |
| Failure/recovery/warning | `lake/artifacts/errors/pipeline=predictions/season=<season>/occurred_date_pacific=<processing-date>/<event-id>.{failure,recovery,warning}.json` |

The source fingerprint includes the selected inputs, calculation settings and implementation. Repeating unchanged inputs reuses the original immutable JSON and generation time without rewriting `latest.json`. The producer verifies immutable and pointer read-back, uses conditional pointer writes, and checks source selections before publishing. A missing required input fails visibly; all pending games are calculated and validated before publishing new selections for the slate. A player missing from daily ratings uses the approved frozen scalar prior if present, otherwise the existing approved replacement rating with a nonfatal warning in S3 and Dagster. This does not modify the rating state.

For historical backfill, select daily partitions of **`game_predictions`** in Dagster. Dates are independent and use one partition per run. Historical calculations are explicitly `purpose=historical_replay`. Existing `pregame_projection` selections remain unchanged, including older pointers where the purpose is only recorded in the immutable file. Live selections stop advancing at scheduled tipoff, with a final time check immediately before pointer publication. Started games lacking a prediction are reported in run metadata and logs rather than assigned a newly fabricated pregame snapshot. Dates without games succeed without writing game files.

Failures leave saved accepted files intact and write a `.failure.json`. Repair the owning upstream job/input and choose **Re-execute from failure** on the failed prediction run; an actual successful child run writes the linked `.recovery.json`. Fresh runs do not claim recovery of a failed parent.

`predict_game_margins_sensor` is enabled and checks today/tomorrow about every minute. It requests the same partitioned job when an upcoming game has no forecast or its accepted allocation, ratings or schedule changes. Unchanged inputs and dates already queued/running are skipped. Missing prerequisites produce a visible waiting reason. Small source selections are checked first; the larger ratings state is read only when work is needed. The existing producer still verifies current inputs and the per-game tipoff cutoff before publication. Odds changes do not trigger predictions. Manual partition runs and historical backfills remain available. Website API/UI integration is tracked separately in #71.


## Postgame actual-minutes counterfactual (#78)

Asset **`game_predictions_actual_minutes`**, job **`predict_game_margins_actual_minutes`**, group **`predictions`**, in the existing code location. Select a completed Pacific game-date partition, through yesterday, without YAML. Partitions are independent and support the same asset backfill UI. This job selects only the counterfactual asset and declares dependencies on `game_predictions`, `nba_box_scores` and `nba_season_schedule`; it does not rerun those producers. A no-game date succeeds without game files. Missing predictions or missing/non-final saved boxes fail visibly.

Read each game's accepted prediction through `GamePredictionArchive` and its final official NBA box score through the existing `NBABoxScoreArchive`. Enumerate the slate through shared `NBAScheduleResource`. Preserve every baseline player's scalar NAIL and the baseline's HCA, fixed B2B, 100-possession assumption and probability conversion. For each player, calculate `5 * actual_minutes / team_total_actual_minutes * saved_nail`; subtract away contributions from home contributions and add the unchanged HCA/B2B. These are full-game minutes, including overtime. Regulation totals of 240 make this equivalent to actual_minutes/48; overtime minutes remain normalized to five players. No first-44-minute role measurement, injury recomputation, model fitting or updated ratings are involved.

Retain forecast-only players with actual zeroes and include additional players listed in the final box. For an additional player, `RatingsArchive.read_state` reads the exact immutable daily state recorded in the baseline prediction and verifies its recorded cutoff, including old effective-date files, without consulting today's selection. A frozen-prior baseline reads that exact initialization instead. Its recorded prior can be read with `read_priors`; an unknown player uses the already-approved replacement rating and emits a player-specific warning. Existing baseline player ratings always take precedence. No new roster/name source is introduced.

| Output | S3 key in `nba-gestalt-data` |
| --- | --- |
| Immutable counterfactual | `lake/artifacts/game_predictions_actual_minutes/season=<season>/model_version=v1/game_date_pacific=<date>/game_id=<NBA-game-id>/<sha256>.json` |
| Accepted postgame selection | Same per-game counterfactual directory, `latest.json` |
| Failures/recoveries/warnings | Existing `lake/artifacts/errors/pipeline=predictions/season=<season>/occurred_date_pacific=<processing-date>/<event-id>.{failure,recovery,warning}.json`, with `stage=game_predictions_actual_minutes` |

The JSON uses `artifact_kind=game_prediction_actual_minutes` and `purpose=postgame_actual_minutes_counterfactual`. It retains `baseline_prediction_purpose`, distinguishing original pregame forecasts from historical replays. Fields include original and counterfactual margin/probability, actual scores/margin, unchanged HCA/B2B components, both teams, and every player's projected/actual minutes, saved NAIL, projected/actual rating contributions and contribution delta. Inputs pin the baseline prediction, final box plus sidecar, schedule, and any additional ratings/prior references actually read. Numeric values retain full precision.

The resource reuses immutable files, checksum/read-back verification and conditional pointer writes. An identical input fingerprint preserves the original generation time and accepted pointer. A changed final box creates another immutable counterfactual version and can advance only the separate postgame latest.json. Baseline prediction files/pointers remain unchanged. All required games are calculated before publishing new selections; source selections are checked again before publication. Repair a failed input and choose Re-execute from failure to obtain a stage-scoped recovery record; unrelated successful runs do not claim recovery.

`predict_game_margins_actual_minutes_schedule` is enabled at **04:30 America/Los_Angeles for yesterday's Pacific game date**. It uses the same job as manual runs and backfills. Missed ticks older than a day are left for an explicit backfill. A missing baseline or non-final box fails visibly rather than substituting an input. Website integration remains separate work.

Initial read-only validation covered complete saved slates for Oct 20 (3 games), Nov 4 (14) and Nov 13 (9), including three overtime games. Evan then accepted the real manual output; the complete season backfill and fresh S3 readback are recorded below.


### Manual counterfactual acceptance and date-redesign checkpoint

On November 14, Evan ran `predict_game_margins_actual_minutes` for game date Nov 4 (run `3ad5d047-3273-41c0-8d81-181f1216c218`). All 14 counterfactuals were read back and reproduced from the saved predictions and official final boxes, including two overtime games. Saved NAIL, HCA/B2B, baseline forecast objects and baseline pointer timestamps remained unchanged.

The initial backfill `pjqhzhdn` was canceled during the date-convention discussion: 21 runs succeeded and three were canceled. That history and its existing outputs remain preserved. The completed #79 migration and replacement backfill `nxtzkjoi` are verified below. The prediction triggers were subsequently enabled as recorded in the final automation section.

## Market performance — issue #80

Asset **`market_performance_by_date`**, job **`evaluate_market_performance`**, group **`market_performance`**, in the existing code location. Partition **D means completed Pacific game date D**; its normal run follows D's 04:30 actual-minute counterfactual on D+1. Partitions use the configured season start through yesterday. The sensor is `evaluate_market_performance_sensor`; it monitors successful `predict_game_margins_actual_minutes` runs, uses the same date, and keys requests by saved counterfactual references so unchanged source successes do not request duplicate work. Historical asset backfills can be evaluated explicitly without rerunning upstream producers.

Inputs are accepted per-game predictions, their accepted postgame counterfactuals, final NBA boxes and sidecars, the shared official NBA schedule, and immutable Odds API snapshots. No API key or paid provider request is required by this evaluator. The asset graph declares those dependencies; the odds dependency covers the capture stream because a game's last pre-tipoff quote can have been captured on an earlier date. Exact used objects are recorded in the output. Missing required prediction/counterfactual/final inputs fail visibly; refresh the owning upstream job before rerunning this date.

| Output | S3 key under `s3://nba-gestalt-data/` |
| --- | --- |
| Immutable whole-date JSON | `lake/artifacts/market_performance/season=<season>/model_version=<prediction-version>/game_date_pacific=<D>/<sha256>.json` |
| Accepted date selection | Same date directory, `latest.json` containing the immutable URI/checksum |
| Failure/recovery | `lake/artifacts/errors/pipeline=market_performance/season=<season>/occurred_date_pacific=<processing-date>/<event-id>.{failure,recovery}.json` |

The JSON keeps every completed game, team IDs/names, start time and final score; model, actual-minute counterfactual, Polymarket and bookmaker-consensus comparisons; selected quotes/timestamps; source checksums; missing-market coverage; and additive daily totals. The summary and totals use the explicitly named common comparison cohort. Individual rows remain available even when market quotes are missing. Actual-minute values are labeled postgame, and original predictions retain their saved pregame-versus-historical-replay purpose. The website can use game rows for its calendar and combine totals/counts for arbitrary summary ranges. Website/API integration remains #71.

The calculations preserve the existing scripts: latest eligible Polymarket h2h/spread snapshots are selected separately, with both collection and update times strictly before the earlier of scheduled tipoff and provider start. Each book consensus uses the **same snapshot as that market's selected Polymarket quote**. It is the median of normalized two-sided moneyline probabilities or negative home spreads, excluding Polymarket and preserving `espnbet`. No independent book snapshot or future quote is substituted. A missing Polymarket anchor leaves that comparison unavailable, with explicit coverage. Snapshots are searched newest-first and reading stops after all games' two anchors are found.

Brier is mean squared home-win probability error; winner picks use home at probability >= 0.5; margin errors are predicted minus actual home margin. ATS selects the side favored relative to the consensus margin, retaining the earlier tolerance, no-edge and push rules. The consensus itself is the benchmark, so its ATS field is `not_applicable` (the old scripts called these baseline entries `no_pick`). The separate Polymarket-different-spread totals preserve the same-game ATS comparison. Cross-date summaries sum squared/absolute errors and counts before division/square root; they never average daily RMSEs or percentages. Numeric values keep full precision.

Unchanged reruns preserve the accepted JSON and pointer timestamp. Changed accepted inputs produce another immutable performance file. The producer checks source selections and eligible raw-odds inventory before publication and conditionally advances only its own pointer. It never changes forecasts, counterfactuals, ratings, minutes, or raw odds. Repair an owning input and rerun this partition; **Re-execute from failure** produces a linked recovery for the actual failed parent. A fresh success does not claim recovery of another run.

All **387 tests pass**, including **29 new market-performance tests**. Real Nov 4 run `5f55bc12-599e-4052-9c1c-c5b5967df48e` succeeded and all 14 games match the prior script's values and exact market/source references. Approved backfill `mkdkjkli` covers Oct 26–Nov 13 (19 dates, 140 games) at priority -2, leaving collectors at higher priority. Backfill `mkdkjkli` is **COMPLETED_SUCCESS: 19 successful dates, zero failures**. All **140 games**, all four series and all combined Brier/accuracy/RMSE/MAE/ATS results match the earlier scripts, including the same 28-game ATS subset. Exact selected market snapshot hashes and prediction/counterfactual/box references also match. All **499 historical upstream objects/selections** checked remain unchanged, and the backfilled Nov 4 rerun preserved its original performance selection timestamp.

`evaluate_market_performance_sensor` is **RUNNING**. A read-only evaluation of the real successful Nov 4 counterfactual run verified the same-date request and stable duplicate key without launching or rewriting that upstream job. The first normal overnight automatic handoff is still future, after the Nov 15 04:30 counterfactual for Nov 14. All protected collectors remain enabled/healthy; the scoreboard already contains final results for all five Nov 14 games. No paid provider request, model replay, source repair or website deployment was made by this evaluator.

Evidence remains on EC2 under `/tmp/market-performance-validation/`: `initial-comparison.json`, `final-comparison.json`, `backfill-status.json`, `upstream-before.json`, `real-source-trigger-check.json`, `collector-check.json` and the automation snapshots. Code remains uncommitted pending review.

## Completed-date migration verification — issue #79

The approved November 14 cutover used Dagster reload in the existing code location, with no service restart. The development checkout is on `codex/79-data-date-partitions`; these changes are not committed yet.

| Asset backfill | Permanent-instance ID | Result |
| --- | --- | --- |
| player_minutes, Oct 20–Nov 13 | aneevsxi | 25 successful dates, no failures |
| player_ratings, Oct 20–Nov 13 | licdbttd | 25 successful dates, no failures |

All 50 immutable states were read back with matching checksums, completed-date markers and matching observation/effective-from dates. Each new completed D was compared to its saved old effective D+1 equivalent using the same accepted source inputs. Maximum difference was **0** for minutes estimates/histories, player ratings, HCA and the cumulative joint matrix/vectors. Both chains cover all 183 games through Nov 13. All 77 upstream selection checks matched before replay. Original versions remain in S3.

The 366 saved historical game-minute/prediction pointers and their immutable file checksums remained unchanged after replay. The pinned ratings reader also verified 25 old immutable states with their original recorded date meanings; selecting a new daily state never reinterprets those old files.

A retained old effective-date selection is unavailable as a new completed-date state. Live input selection verifies the learned artifact, waits when the required completed state has not been published, and does not fall back to an older state. A real check confirmed Nov 14 and Nov 15 preview selections use completed Nov 13, and the next morning window waits for completed Nov 14 despite its old effective-date pointer still existing. No S3 file or out-of-range pointer was deleted for this readiness rule.

The full pre-cutover suite passed 335 tests; all 178 affected minutes/ratings/prediction tests passed after the additional overnight readiness regression. The two updater schedules and project_game_minutes_sensor were restored after validation. Collectors remained RUNNING: injury capture succeeded at 16:32 Pacific, odds at 16:20 (five events), scoreboard at 16:41 (all five slate games, one live). Next expected checks at that checkpoint were 16:47, 17:20 and 16:42 respectively.

The first approval covered learned states only; the subsequent historical projection replay and its approval are recorded below. No website deployment, model fitting, release publication or Git commit occurred.

After automation resumed, the normal Nov 14 and Nov 15 project_game_minutes runs both succeeded. All 12 still-upcoming game allocations (three Nov 14, nine Nov 15) were read back and reference completed Nov 13. The original canceled counterfactual backfill pjqhzhdn still has status CANCELED.

### Scoreboard handoff exception

The initial partition switch missed one old unpartitioned scheduler request: run `0b0d408c-3704-4e60-843a-f072d7f26da0` failed at 16:16:08 Pacific because it had no partition key. The previous capture completed 16:15:13 and the next genuine dated capture completed 16:17:04, a 111-second gap. Subsequent dated polls succeeded; all five slate games remain covered by latest.json. The schedule stayed enabled. The current-only endpoint cannot reproduce the missed 16:16 response; preserve this failure and interval rather than inventing a historical capture. Issue #79 records the full cutover audit. Injury and odds histories showed no failures during this interval.

## Historical projection replay and market comparison — issue #79

Backfill `nxtzkjoi` completed successfully for Oct 20–Nov 13: 25 dates for each of `game_minutes`, `game_predictions`, and `game_predictions_actual_minutes`, across 50 successful runs and no failures. All three assets cover all 183 scheduled games (549 validated files). The 17 previously missing counterfactuals are now present. All 13 genuine pregame minute allocations and 532 original immutable versions are preserved. Existing numerical outputs match their earlier versions exactly; the 50 learned-state selections remain unchanged. Live collection and automation continued.

The refreshed comparison covers all 140 games on Oct 26–Nov 13. It reads the new accepted historical game-margin predictions, official final boxes, and the latest archived Odds API quotes collected and updated before scheduled tipoff. Book consensus is the median of the other bookmakers, excluding Polymarket. Probabilities use normalized two-sided moneylines; market margin errors use quoted spreads. Actual-minute counterfactuals remain separately labeled and are not used as pregame forecasts.

| Metric | Model | Polymarket | Median books |
| --- | ---: | ---: | ---: |
| Brier score | 0.192299 | 0.178650 | 0.179562 |
| Winner accuracy | 72.86% | 73.57% | 72.86% |
| Margin RMSE (points) | 14.749948 | 14.106989 | 14.122756 |
| Margin MAE (points) | 11.594430 | 11.000000 | 11.026786 |
| ATS versus median books | 72–68 | 18–10; 112 no-picks | Baseline |

All metrics match the pre-migration comparison exactly. Polymarket had a spread difference from the book median in only 28 games; on that same subset, the model went 13–15 and Polymarket 18–10. No spread pushes occurred. All nine non-Polymarket bookmaker keys supplied by the Odds API, including espnbet, were eligible; no ESPN website/API basketball data was used.

Full-precision reports and per-game source references remain on EC2 in `/tmp/partition79-evaluation/`: `updated-prediction-evaluation.json`, `updated-three-way-evaluation.json`, `three-way-summary.json`, `oct26-nov13-three-way-games.csv`, and `downstream-verification.json`. Raw market snapshots were checked against their provider identity and recorded checksums. All 140 evaluated prediction references match the validated final selections.

During this replay, protected histories show two injury successes, one odds success and 31 scoreboard successes, with no failures. Final checked captures (Pacific): injury 17:17, next 17:32, one new PDF/none unavailable; odds 17:20, next 18:20, five events; scoreboard 17:29, next 17:30, all five games live. The prior 16:16 cutover failure remains preserved in its earlier audit. No code commit, service restart or website deployment occurred in this replay.

The next overnight schedule sequence still needs observation. Margin live automation and counterfactual scheduling were subsequently enabled under #78. Historical odds/scoreboard Dagster materializations were not added to this projection-replay approval.

## Historical odds partitions — completed validation

Backfill `cbkneowy` completed all **19 Pacific capture-date partitions, Oct 26–Nov 13**, with 19 successful runs and zero failures. Each genuine archive-only materialization records its existing snapshot URIs and checksums. All **133 archived reports** were validated, with no missing dates. Historical file bytes, sizes and ETags are unchanged; the live `latest.json` is unchanged. The historical branch made no Odds API requests and did not move S3 files.

The complete Oct 26–Nov 13 market report was recomputed from S3 after materialization. All 140 per-game rows and aggregate metrics match the preceding report exactly. The report uses historical game-margin predictions and official final boxes, normalized pre-tipoff moneylines, and actual market spreads; the book median excludes Polymarket. No counterfactual values enter the pregame comparison.

| Metric | Model | Polymarket | Median other books |
| --- | ---: | ---: | ---: |
| Brier score | 0.192299 | 0.178650 | 0.179562 |
| Winner accuracy | 72.86% | 73.57% | 72.86% |
| Margin RMSE (points) | 14.749948 | 14.106989 | 14.122756 |
| Margin MAE (points) | 11.594430 | 11.000000 | 11.026786 |
| ATS versus median books | 72–68 | 18–10; 112 no-picks | Baseline |

On the same 28 games where Polymarket differs from the median book spread, model ATS is 13–15 and Polymarket ATS is 18–10. There are no pushes. All original predictions and market data are unchanged by this odds backfill.

All three protected schedules stayed RUNNING with no failures during this backfill. Final checked successful captures / next checks (Pacific): injury 18:02 / 18:17 (one new PDF, none unavailable); odds 17:20 / 18:20 (five events); scoreboard 18:14 / 18:15 (all five games live). Capture freshness is read from the live job history, rather than the newest historical archive materialization.

Full-precision reports and audit files are on EC2 in `/tmp/partition79-odds-validation/`: `partition-verification.json`, `updated-prediction-evaluation.json`, `updated-three-way-evaluation.json`, `market-summary.json`, `market-comparison-games.csv`, and `collector-check.json`. No pipeline code, S3 layout, live automation or website was changed in this step.

The historical scoreboard materializations and next-morning schedule observation remain separate pending checks.

## Projected versus actual minutes: market comparison

The actual-minutes backfill was already complete: permanent-instance backfill nxtzkjoi has 25 successful actual-minute date partitions, covering all 183 games through Nov 13. This report freshly verified all 183 counterfactuals against their current baseline predictions and official final box scores; there were no missing games and no additional backfill was needed.

All 6,364 baseline player-game rating entries and their projected minutes match the counterfactual rows exactly. Extra actual players use the saved baseline rating source. HCA, B2B adjustment, rating evidence dates, the 100-possession assumption and probability conversion are unchanged. Actual full-game minutes, including overtime, are normalized to five players per team. This is a postgame diagnostic.

The paired comparison uses all 140 Oct 26–Nov 13 games and the same pre-tipoff Polymarket/book quotes as the preceding report. All 65 cited market snapshot files were rechecked against their saved SHA-256 hashes. Existing projected-minute, Polymarket and book metrics reproduce the previous report exactly.

| Metric | Model: projected minutes | Model: actual minutes (postgame) | Polymarket | Median books |
| --- | ---: | ---: | ---: | ---: |
| Brier score | 0.192299 | 0.185030 | 0.178650 | 0.179562 |
| Winner accuracy | 72.86% | 72.14% | 73.57% | 72.86% |
| Margin RMSE (points) | 14.749948 | 14.648878 | 14.106989 | 14.122756 |
| Margin MAE (points) | 11.594430 | 11.620371 | 11.000000 | 11.026786 |
| ATS versus median books | 72–68 | 73–67 | 18–10; 112 no-picks | Baseline |

Actual minutes reduce Brier by 0.007268505 and RMSE by 0.101069865 points. MAE increases by 0.025940258 points and winner accuracy falls from 102 to 101 correct out of 140. Thus actual minutes improve probability error while aggregate point-margin errors change little in this sample.

On the same 28 games where Polymarket differs from the book median, ATS is 13–15 with projected minutes, 15–13 with actual minutes, and 18–10 for Polymarket. There are no pushes. The postgame column uses information learned after the game and is kept separately labeled from the historical projected-minute forecasts.

Full-precision report and per-game data are on EC2 under `/tmp/partition79-actual-minutes-report/`: `market-comparison.json`, `summary.json`, `games.csv`, `coverage.json`, `rating-preservation.json` and `validated-game-inputs.json`. All source predictions, counterfactuals and official final boxes are recorded by URI/checksum. No source data, ratings, forecasts, schedules or pipeline code were changed for this report.


## Prediction automation enabled — November 14 (#78)

`predict_game_margins_sensor` and `predict_game_margins_actual_minutes_schedule` are RUNNING in the existing `nba_gestalt_pipelines` location. No collector, existing schedule/sensor, code location, service, S3 path or model calculation was changed. The existing one-worker limit and lower live-model priority remain in place.

The first sensor-created run, `0ba26f6c-42e0-4a62-b23e-f8db6f91c887`, succeeded for the **Nov 15** partition. S3 readback verified all **nine upcoming games**, each labeled pregame and using completed **Nov 13** ratings for the provisional next-day preview. Later ticks skipped unchanged inputs without launching duplicate runs. All **188 existing current/past prediction selections**, including their timestamps, remained unchanged. The morning learned-state updates will refresh open Nov 15 games through Nov 14 when those states become available.

All **355 tests passed**, including **19 new automation checks** for source changes, opening-day priors, missing prerequisites, duplicate suppression, today/tomorrow scope, scheduled-tipoff freezing, unchanged fallback inputs, and Pacific daily schedule dates across both DST transitions. The next real counterfactual tick is **Nov 15 at 04:30 Pacific**, targeting **Nov 14 games**. Its overnight execution is still to be observed; it was not run early or recorded as a past success.

At 19:31 Pacific, protected collection remained healthy: injury capture succeeded at 19:17 (one new report, three reused, zero unavailable; next 19:32), Odds API capture succeeded at 19:20 (eight events; next scheduled Nov 15 at 08:20), and scoreboard capture succeeded at 19:31 (all five games, one still live; next 19:32). No recent collector failures were found. Existing automation states were verified unchanged after the code-location reload. Validation records remain on EC2 in `/tmp/prediction-automation-validation.json`, `/tmp/prediction-automation-runs.json`, and `/tmp/prediction-automation-collector-check.json`. Code remains uncommitted pending review.
