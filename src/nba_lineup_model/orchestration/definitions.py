"""Resource-bounded Dagster jobs for the production NBA GESTALT instance."""

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dagster import (
    AssetExecutionContext,
    DagsterRunStatus,
    DefaultScheduleStatus,
    DefaultSensorStatus,
    Definitions,
    Field,
    MetadataValue,
    OpExecutionContext,
    RetryPolicy,
    RunRequest,
    RunsFilter,
    ScheduleDefinition,
    SensorEvaluationContext,
    SkipReason,
    asset,
    define_asset_job,
    in_process_executor,
    job,
    op,
    sensor,
)

from nba_lineup_model.modeling.in_season_minutes import (
    UPDATER_VERSION as IN_SEASON_MINUTES_UPDATER_VERSION,
)
from nba_lineup_model.season.slate_time import PACIFIC, summary_slate_date

EASTERN = ZoneInfo("America/New_York")
RETRY_POLICY = RetryPolicy(max_retries=3, delay=30)
WORKLOAD_TAG = "gestalt/workload"
DEFAULT_SCHEDULE_STATE_DIR = Path("/var/lib/nba-gestalt/schedule")
DEFAULT_SCHEDULE_API_URL = "http://127.0.0.1:8010/api/win-projections"
DEFAULT_SCHEDULE_S3_URI = "s3://nba-gestalt-data/nba-gestalt/schedule"
DEFAULT_SCHEDULE_DAYS_AHEAD = 7
DEFAULT_ODDS_API_BOOKMAKERS = (
    "draftkings",
    "fanduel",
    "betmgm",
    "betrivers",
    "espnbet",
    "hardrockbet",
    "betparx",
    "ballybet",
    "betonlineag",
    "polymarket",
)
DEFAULT_STATS_V3_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_STATS_V3_SEASON = "2026-27"
DEFAULT_STATS_V3_GRACE_MINUTES = 60
DEFAULT_STATS_V3_COVERAGE_PATH = Path("/var/lib/nba-gestalt/stats-v3/coverage.json")
DEFAULT_CDN_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_CDN_SEASON = "2026-27"
DEFAULT_CDN_GRACE_MINUTES = 60
DEFAULT_CDN_COVERAGE_PATH = Path("/var/lib/nba-gestalt/cdn/coverage.json")
DEFAULT_IN_SEASON_NAIL_ROOT = Path("/var/lib/nba-gestalt/in-season-nail")
DEFAULT_IN_SEASON_NAIL_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_IN_SEASON_NAIL_PRIOR_POSSESSIONS = 6000
DEFAULT_IN_SEASON_MINUTES_ROOT = Path("/var/lib/nba-gestalt/in-season-minutes")
DEFAULT_IN_SEASON_MINUTES_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_IN_SEASON_MINUTES_PARTICIPATION_PRIOR_GAMES = 4
DEFAULT_IN_SEASON_MINUTES_CONDITIONAL_PRIOR_GAMES = 0.2
DEFAULT_IN_SEASON_MINUTES_CONDITIONAL_LOOKBACK_GAMES = 3
DEFAULT_IN_SEASON_MINUTES_ROLE_CUTOFF_MINUTES = 44
DEFAULT_CURRENT_SEASON_SERVING_ROOT = Path(
    "/var/lib/nba-gestalt/current-season-serving"
)
DEFAULT_CURRENT_SEASON_SERVING_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_CURRENT_SEASON_LINEUPS_ROOT = Path(
    "/var/lib/nba-gestalt/current-season-lineups"
)
DEFAULT_RELEASE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HEALTHCHECK_PARQUET = "s3://nba-gestalt-data/lake/analytical/box_score_prior_panel/season_cohort_summary.parquet"
AUTOMATIONS_DEFAULT_RUNNING = (
    os.environ.get("GESTALT_AUTOMATIONS_DEFAULT_RUNNING", "0") == "1"
)
SCHEDULE_DEFAULT_STATUS = (
    DefaultScheduleStatus.RUNNING
    if AUTOMATIONS_DEFAULT_RUNNING
    else DefaultScheduleStatus.STOPPED
)
SENSOR_DEFAULT_STATUS = (
    DefaultSensorStatus.RUNNING
    if AUTOMATIONS_DEFAULT_RUNNING
    else DefaultSensorStatus.STOPPED
)
STATS_V3_SOURCE_CONFIGURED = bool(os.environ.get("GESTALT_NBA_STATS_PROXY")) or (
    os.environ.get("GESTALT_STATS_V3_ENABLED", "0") == "1"
)
STATS_V3_SCHEDULE_DEFAULT_STATUS = (
    SCHEDULE_DEFAULT_STATUS
    if STATS_V3_SOURCE_CONFIGURED
    else DefaultScheduleStatus.STOPPED
)
STATS_V3_SENSOR_DEFAULT_STATUS = (
    SENSOR_DEFAULT_STATUS if STATS_V3_SOURCE_CONFIGURED else DefaultSensorStatus.STOPPED
)
ODDS_API_CONFIGURED = bool(os.environ.get("GESTALT_ODDS_API_KEY", "").strip())
ODDS_API_SCHEDULE_DEFAULT_STATUS = (
    SCHEDULE_DEFAULT_STATUS if ODDS_API_CONFIGURED else DefaultScheduleStatus.STOPPED
)


def _state_dir() -> Path:
    return Path(
        os.environ.get("GESTALT_SCHEDULE_STATE_DIR", DEFAULT_SCHEDULE_STATE_DIR)
    )


def _release_root() -> Path:
    """Return the artifact-bearing release root independently of process CWD."""

    return Path(os.environ.get("GESTALT_RELEASE_ROOT", DEFAULT_RELEASE_ROOT)).resolve()


def _schedule_s3_uri() -> str | None:
    value = os.environ.get("GESTALT_SCHEDULE_S3_URI", DEFAULT_SCHEDULE_S3_URI).strip()
    return value or None


def _stats_v3_lake_uri() -> str:
    return os.environ.get(
        "GESTALT_STATS_V3_LAKE_URI", DEFAULT_STATS_V3_LAKE_URI
    ).rstrip("/")


def _stats_v3_season() -> str:
    return os.environ.get("GESTALT_STATS_V3_SEASON", DEFAULT_STATS_V3_SEASON)


def _stats_v3_coverage_path() -> Path:
    return Path(
        os.environ.get("GESTALT_STATS_V3_COVERAGE_PATH", DEFAULT_STATS_V3_COVERAGE_PATH)
    )


def _cdn_lake_uri() -> str:
    return os.environ.get("GESTALT_CDN_LAKE_URI", DEFAULT_CDN_LAKE_URI).rstrip("/")


def _cdn_season() -> str:
    return os.environ.get("GESTALT_CDN_SEASON", DEFAULT_CDN_SEASON)


def _cdn_coverage_path() -> Path:
    return Path(os.environ.get("GESTALT_CDN_COVERAGE_PATH", DEFAULT_CDN_COVERAGE_PATH))


def _in_season_nail_root() -> Path:
    return Path(
        os.environ.get("GESTALT_IN_SEASON_NAIL_ROOT", DEFAULT_IN_SEASON_NAIL_ROOT)
    )


def _in_season_nail_lake_uri() -> str:
    return os.environ.get(
        "GESTALT_IN_SEASON_NAIL_LAKE_URI", DEFAULT_IN_SEASON_NAIL_LAKE_URI
    ).rstrip("/")


def _in_season_minutes_root() -> Path:
    return Path(
        os.environ.get("GESTALT_IN_SEASON_MINUTES_ROOT", DEFAULT_IN_SEASON_MINUTES_ROOT)
    )


def _in_season_minutes_lake_uri() -> str:
    return os.environ.get(
        "GESTALT_IN_SEASON_MINUTES_LAKE_URI", DEFAULT_IN_SEASON_MINUTES_LAKE_URI
    ).rstrip("/")


def _current_season_serving_root() -> Path:
    return Path(
        os.environ.get(
            "GESTALT_CURRENT_SEASON_SERVING_ROOT",
            DEFAULT_CURRENT_SEASON_SERVING_ROOT,
        )
    )


def _current_season_serving_lake_uri() -> str:
    return os.environ.get(
        "GESTALT_CURRENT_SEASON_SERVING_LAKE_URI",
        DEFAULT_CURRENT_SEASON_SERVING_LAKE_URI,
    ).rstrip("/")


def _current_season_lineups_root() -> Path:
    return Path(
        os.environ.get(
            "GESTALT_CURRENT_SEASON_LINEUPS_ROOT",
            DEFAULT_CURRENT_SEASON_LINEUPS_ROOT,
        )
    )


def _scheduled_time(context: object) -> datetime:
    value = getattr(context, "scheduled_execution_time", None)
    return value.astimezone(UTC) if isinstance(value, datetime) else datetime.now(UTC)


def _schedule_run_config(*, start_date: str, days_ahead: int) -> dict[str, object]:
    return {
        "ops": {
            "refresh_schedule": {
                "config": {"start_date": start_date, "days_ahead": days_ahead}
            }
        }
    }


@op(
    name="refresh_schedule",
    config_schema={"start_date": str, "days_ahead": Field(int, default_value=7)},
    retry_policy=RETRY_POLICY,
)
def refresh_schedule_op(context: OpExecutionContext) -> dict[str, object]:
    """Refresh ESPN, lines, injury reports, projections, and immutable snapshots."""

    from nba_lineup_model.web_api.win_projections import refresh_schedule_window

    payload = refresh_schedule_window(
        start_date=str(context.op_config["start_date"]),
        days_ahead=int(context.op_config["days_ahead"]),
        state_dir=_state_dir(),
        projection_url=os.environ.get(
            "GESTALT_SCHEDULE_API_URL", DEFAULT_SCHEDULE_API_URL
        ),
        s3_uri=_schedule_s3_uri(),
    )
    context.add_output_metadata(
        {
            "archived_days": int(payload["archived_day_count"]),
            "games": int(payload["game_count"]),
            "newly_frozen_games": int(payload["newly_frozen_game_count"]),
        }
    )
    return payload


@job(
    name="refresh_schedule_window",
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "schedule_refresh"},
)
def refresh_schedule_job() -> None:
    refresh_schedule_op()


def _regular_refresh_config(context: object) -> dict[str, object]:
    game_date = _scheduled_time(context).astimezone(EASTERN).date().isoformat()
    return _schedule_run_config(
        start_date=game_date, days_ahead=DEFAULT_SCHEDULE_DAYS_AHEAD
    )


def _yesterday_refresh_config(context: object) -> dict[str, object]:
    game_date = (
        _scheduled_time(context).astimezone(EASTERN).date() - timedelta(days=1)
    ).isoformat()
    return _schedule_run_config(start_date=game_date, days_ahead=0)


schedule_refresh_schedule = ScheduleDefinition(
    name="schedule_refresh_every_30_minutes",
    job=refresh_schedule_job,
    cron_schedule="*/30 * * * *",
    execution_timezone="UTC",
    run_config_fn=_regular_refresh_config,
    default_status=SCHEDULE_DEFAULT_STATUS,
    description="Refresh the current and next seven NBA boards every 30 minutes.",
)

yesterday_finalization_schedule = ScheduleDefinition(
    name="yesterday_finalization_overnight",
    job=refresh_schedule_job,
    cron_schedule="0,30 4-9 * * *",
    execution_timezone="UTC",
    run_config_fn=_yesterday_refresh_config,
    default_status=SCHEDULE_DEFAULT_STATUS,
    description="Revisit yesterday's board overnight until late games are final.",
)


@op(
    name="collect_nba_market_odds_snapshot",
    config_schema={
        "bookmakers": Field(
            [str],
            default_value=list(DEFAULT_ODDS_API_BOOKMAKERS),
        ),
        "markets": Field([str], default_value=["spreads", "h2h"]),
        "minimum_remaining_credits": Field(int, default_value=50),
    },
    retry_policy=RetryPolicy(max_retries=1, delay=60),
)
def archive_nba_market_odds_op(context: OpExecutionContext) -> dict[str, object]:
    """Archive one quota-bounded, league-wide NBA market snapshot."""

    from nba_lineup_model.web_api.the_odds_api import (
        OddsApiQuotaGuard,
        collect_nba_odds_snapshot,
    )

    now = datetime.now(UTC)
    upcoming_games = 0
    for offset in range(DEFAULT_SCHEDULE_DAYS_AHEAD + 1):
        game_date = (
            now.astimezone(PACIFIC).date() + timedelta(days=offset)
        ).isoformat()
        path = _state_dir() / "latest" / f"{game_date}.json"
        if not path.is_file():
            continue
        try:
            board = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(board, dict):
            continue
        upcoming_games += sum(
            1
            for game in board.get("games", [])
            if isinstance(game, dict)
            and isinstance(game.get("status"), dict)
            and game["status"].get("state") == "pre"
        )
    if not upcoming_games:
        reason = "No pregame NBA events are present in the next seven schedule boards"
        context.log.info(reason)
        context.add_output_metadata({"skipped": True, "reason": reason})
        return {"skipped": True, "reason": reason}

    try:
        result = collect_nba_odds_snapshot(
            api_key=os.environ.get("GESTALT_ODDS_API_KEY", ""),
            state_dir=_state_dir(),
            s3_uri=_schedule_s3_uri(),
            bookmakers=tuple(context.op_config["bookmakers"]),
            markets=tuple(context.op_config["markets"]),
            minimum_remaining_credits=int(
                context.op_config["minimum_remaining_credits"]
            ),
            request_id=context.run_id,
        )
    except OddsApiQuotaGuard as error:
        context.log.warning(str(error))
        context.add_output_metadata({"skipped": True, "reason": str(error)})
        return {"skipped": True, "reason": str(error)}
    quota = result.get("quota", {})
    metadata = {
        "events": len(result.get("events", [])),
        "credits_used": quota.get("requests_used"),
        "credits_remaining": quota.get("requests_remaining"),
        "request_cost": quota.get("request_cost"),
        "bookmakers": ", ".join(result.get("request", {}).get("bookmakers", [])),
        "markets": ", ".join(result.get("request", {}).get("markets", [])),
    }
    context.add_output_metadata(
        {key: value for key, value in metadata.items() if value is not None}
    )
    return result


@job(
    name="archive_nba_market_odds",
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "market_odds"},
)
def archive_nba_market_odds_job() -> None:
    archive_nba_market_odds_op()


nba_market_odds_schedule = ScheduleDefinition(
    name="nba_market_odds_game_day",
    job=archive_nba_market_odds_job,
    cron_schedule="20 8,12,15-19 * * *",
    execution_timezone="America/Los_Angeles",
    default_status=ODDS_API_SCHEDULE_DEFAULT_STATUS,
    description=(
        "Archive league-wide spreads and moneylines at 08:20, 12:20, and hourly "
        "from 15:20 through 19:20 Pacific; normal usage is at most 434 credits in "
        "a 31-day month."
    ),
)


@op(
    name="update_live_score_state",
    config_schema={"game_date": str},
    retry_policy=RETRY_POLICY,
)
def update_live_scores_op(context: OpExecutionContext) -> dict[str, object]:
    """Perform one short score/clock update; the sensor controls when it runs."""

    from nba_lineup_model.web_api.win_projections import refresh_live_score_state

    game_date = str(context.op_config["game_date"])
    result = refresh_live_score_state(
        game_date=game_date,
        state_dir=_state_dir(),
        s3_uri=_schedule_s3_uri(),
    )
    context.add_output_metadata(
        {
            "game_date": game_date,
            "games": int(result["game_count"]),
            "any_live": bool(result["any_live"]),
            "all_final": bool(result["all_final"]),
            "newly_finalized_games": int(result["newly_finalized_game_count"]),
            "uploaded_final_artifacts": int(result["uploaded_final_artifact_count"]),
        }
    )
    return result


@job(
    name="update_live_scores",
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "live_scores"},
)
def update_live_scores_job() -> None:
    update_live_scores_op()


@op(
    name="verify_final_game_archives_s3",
    config_schema={"game_date": str},
    retry_policy=RETRY_POLICY,
)
def reconcile_final_game_archives_op(context: OpExecutionContext) -> dict[str, object]:
    """Verify yesterday's immutable final audit artifacts against S3."""

    from nba_lineup_model.web_api.win_projections import (
        reconcile_final_game_artifacts_to_s3,
    )

    game_date = str(context.op_config["game_date"])
    s3_uri = _schedule_s3_uri()
    if not s3_uri:
        raise ValueError("GESTALT_SCHEDULE_S3_URI must be set for final reconciliation")
    result = reconcile_final_game_artifacts_to_s3(
        game_date=game_date,
        state_dir=_state_dir(),
        s3_uri=s3_uri,
    )
    context.add_output_metadata(
        {
            "game_date": game_date,
            "final_games": int(result["final_game_count"]),
            "verified_final_artifacts": int(result["verified_artifact_count"]),
        }
    )
    return result


@job(
    name="reconcile_final_game_archives",
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "final_archive_reconciliation"},
)
def reconcile_final_game_archives_job() -> None:
    reconcile_final_game_archives_op()


def _final_archive_reconciliation_config(context: object) -> dict[str, object]:
    game_date = (
        _scheduled_time(context).astimezone(EASTERN).date() - timedelta(days=1)
    ).isoformat()
    return {
        "ops": {"verify_final_game_archives_s3": {"config": {"game_date": game_date}}}
    }


final_game_archive_morning_reconciliation_schedule = ScheduleDefinition(
    name="final_game_archive_morning_reconciliation",
    job=reconcile_final_game_archives_job,
    cron_schedule="15 8 * * *",
    execution_timezone="America/New_York",
    run_config_fn=_final_archive_reconciliation_config,
    default_status=SCHEDULE_DEFAULT_STATUS,
    description="Verify yesterday's local final audit artifacts against immutable S3.",
)


def active_live_game_date(
    *,
    state_dir: Path | str,
    now: datetime | None = None,
) -> str | None:
    """Return a live slate or a recently completed slate missing its final audit."""

    current = (now or datetime.now(UTC)).astimezone(UTC)
    eastern_date = current.astimezone(EASTERN).date()
    state_root = Path(state_dir)
    for game_date in (
        (eastern_date - timedelta(days=1)).isoformat(),
        eastern_date.isoformat(),
    ):
        path = state_root / "latest" / f"{game_date}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        games = payload.get("games", []) if isinstance(payload, dict) else []
        games = [game for game in games if isinstance(game, dict)]
        states = [str(game.get("status", {}).get("state", "pre")) for game in games]
        starts = [
            datetime.fromisoformat(str(game["start_time"]).replace("Z", "+00:00"))
            for game in games
            if game.get("start_time")
        ]
        if not states or not starts:
            continue
        missing_final_audit = any(
            isinstance(game.get("status"), dict)
            and bool(game["status"].get("completed"))
            and bool(str(game.get("espn_game_id", "")).strip())
            and (state_root / "frozen" / f"{game['espn_game_id']}.json").is_file()
            and (
                not (
                    state_root / "final" / f"{game['espn_game_id']}.summary.json"
                ).is_file()
                or not (
                    state_root / "final" / f"{game['espn_game_id']}.espn.json"
                ).is_file()
            )
            for game in games
        )
        if missing_final_audit and current <= max(starts) + timedelta(hours=12):
            return game_date
        if all(state == "post" for state in states):
            continue
        if (
            min(starts) - timedelta(minutes=30)
            <= current
            <= max(starts) + timedelta(hours=6)
        ):
            return game_date
    return None


@sensor(
    name="live_score_game_window",
    job=update_live_scores_job,
    minimum_interval_seconds=75,
    default_status=SENSOR_DEFAULT_STATUS,
    description=(
        "Emit short score-update runs while a slate is active or a recently completed "
        "game is still missing its immutable final audit."
    ),
)
def live_score_sensor(context: SensorEvaluationContext) -> RunRequest | SkipReason:
    now = datetime.now(UTC)
    game_date = active_live_game_date(state_dir=_state_dir(), now=now)
    if game_date is None:
        return SkipReason("No live or unfinalized slate is inside its recovery window")
    inflight = context.instance.get_runs(
        filters=RunsFilter(
            job_name="update_live_scores",
            statuses=(
                DagsterRunStatus.QUEUED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.CANCELING,
            ),
            tags={"gestalt/game_date": game_date},
        ),
        limit=1,
    )
    if inflight:
        return SkipReason(
            f"Live-score update already queued or running for {game_date}: {inflight[0].run_id}"
        )
    interval_bucket = int(now.timestamp() // 75)
    return RunRequest(
        run_key=f"{game_date}:{interval_bucket}",
        run_config={
            "ops": {"update_live_score_state": {"config": {"game_date": game_date}}}
        },
        tags={WORKLOAD_TAG: "live_scores", "gestalt/game_date": game_date},
    )


@op(
    name="archive_completed_cdn_play_by_play",
    config_schema={
        "season": Field(str, default_value=DEFAULT_CDN_SEASON),
        "start_date": Field(str, default_value=""),
        "end_date": Field(str, default_value=""),
        "game_ids": Field([str], default_value=[]),
        "refresh": Field(bool, default_value=False),
        "ignore_grace": Field(bool, default_value=False),
        "grace_minutes": Field(int, default_value=DEFAULT_CDN_GRACE_MINUTES),
    },
    retry_policy=RetryPolicy(max_retries=1, delay=300),
)
def archive_completed_cdn_play_by_play_op(
    context: OpExecutionContext,
) -> dict[str, object]:
    """Archive immutable CDN play-by-play for eligible final games."""

    from datetime import date

    from nba_lineup_model.season.cdn_lake import ingest_completed_cdn

    config = context.op_config
    start_date = (
        date.fromisoformat(config["start_date"]) if config["start_date"] else None
    )
    end_date = date.fromisoformat(config["end_date"]) if config["end_date"] else None
    result = ingest_completed_cdn(
        state_dir=_state_dir(),
        lake_uri=_cdn_lake_uri(),
        coverage_path=_cdn_coverage_path(),
        season=str(config["season"]),
        start_date=start_date,
        end_date=end_date,
        game_ids=list(config["game_ids"]),
        refresh=bool(config["refresh"]),
        ignore_grace=bool(config["ignore_grace"]),
        grace_period=timedelta(minutes=int(config["grace_minutes"])),
    )
    context.add_output_metadata(
        {
            "selected_games": int(result["selected_game_count"]),
            "archived": int(result["archived_count"]),
            "revised": int(result["revised_count"]),
            "unchanged": int(result["unchanged_count"]),
            "skipped": int(result["skipped_count"]),
            "manifest_uri": MetadataValue.url(str(result["manifest_uri"])),
        }
    )
    return result


@job(
    name="ingest_completed_cdn_play_by_play",
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "cdn_play_by_play_ingest"},
)
def ingest_completed_cdn_play_by_play_job() -> None:
    archive_completed_cdn_play_by_play_op()


@sensor(
    name="completed_game_cdn_play_by_play",
    job=ingest_completed_cdn_play_by_play_job,
    minimum_interval_seconds=900,
    default_status=SENSOR_DEFAULT_STATUS,
    description="Archive CDN play-by-play once each game has been final for one hour.",
)
def completed_game_cdn_play_by_play_sensor(
    context: SensorEvaluationContext,
) -> RunRequest | SkipReason:
    from nba_lineup_model.season.cdn_lake import pending_completed_cdn_game_ids

    now = datetime.now(UTC)
    pending = pending_completed_cdn_game_ids(
        state_dir=_state_dir(),
        season=_cdn_season(),
        coverage_path=_cdn_coverage_path(),
        now=now,
        grace_period=timedelta(minutes=DEFAULT_CDN_GRACE_MINUTES),
    )
    if not pending:
        return SkipReason("No final games are missing CDN play-by-play")
    pending_digest = hashlib.sha256(",".join(pending).encode()).hexdigest()[:12]
    pending_tag = {"gestalt/cdn_pending_digest": pending_digest}
    prior_attempts = context.instance.get_runs(
        filters=RunsFilter(
            job_name="ingest_completed_cdn_play_by_play",
            tags=pending_tag,
        ),
    )
    inflight = [
        run
        for run in prior_attempts
        if run.status
        in (
            DagsterRunStatus.QUEUED,
            DagsterRunStatus.STARTING,
            DagsterRunStatus.STARTED,
            DagsterRunStatus.CANCELING,
        )
    ]
    if inflight:
        return SkipReason(
            f"Equivalent CDN play-by-play ingestion already queued or running: {inflight[0].run_id}"
        )
    # The pending-set digest stays stable while an attempt is queued or running.
    # A terminal attempt increments the suffix so failed/canceled runs remain
    # retryable without allowing every 15-minute sensor tick to create a copy.
    attempt = len(prior_attempts)
    return RunRequest(
        run_key=f"cdn-pbp:{pending_digest}:attempt-{attempt}",
        run_config={
            "ops": {
                "archive_completed_cdn_play_by_play": {
                    "config": {
                        "season": _cdn_season(),
                        "game_ids": pending,
                        "refresh": False,
                        "ignore_grace": False,
                        "grace_minutes": DEFAULT_CDN_GRACE_MINUTES,
                    }
                }
            }
        },
        tags={
            WORKLOAD_TAG: "cdn_play_by_play_ingest",
            "gestalt/game_count": str(len(pending)),
            **pending_tag,
        },
    )


def _cdn_reconciliation_config(context: object) -> dict[str, object]:
    game_date = (
        _scheduled_time(context).astimezone(EASTERN).date() - timedelta(days=1)
    ).isoformat()
    return {
        "ops": {
            "archive_completed_cdn_play_by_play": {
                "config": {
                    "season": _cdn_season(),
                    "start_date": game_date,
                    "end_date": game_date,
                    "game_ids": [],
                    "refresh": True,
                    "ignore_grace": True,
                    "grace_minutes": DEFAULT_CDN_GRACE_MINUTES,
                }
            }
        }
    }


cdn_morning_reconciliation_schedule = ScheduleDefinition(
    name="cdn_play_by_play_morning_reconciliation",
    job=ingest_completed_cdn_play_by_play_job,
    cron_schedule="0 8 * * *",
    execution_timezone="America/New_York",
    run_config_fn=_cdn_reconciliation_config,
    default_status=SCHEDULE_DEFAULT_STATUS,
    description="Re-fetch yesterday's final CDN play-by-play and preserve revisions.",
)


@op(
    name="run_in_season_nail_production",
    config_schema={
        "through_date": str,
        "prior_possessions": Field(
            int,
            default_value=DEFAULT_IN_SEASON_NAIL_PRIOR_POSSESSIONS,
        ),
    },
    retry_policy=RetryPolicy(max_retries=1, delay=300),
)
def update_in_season_nail_production_op(
    context: OpExecutionContext,
) -> dict[str, object]:
    """Advance the production ratings state and retain the frozen-model audit."""

    import argparse

    from nba_lineup_model.modeling.in_season_nail_monitor import run
    from nba_lineup_model.season.stats_v3_lake import (
        AwsCliImmutableObjectStore,
        _join_s3_uri,
    )

    through_date = str(context.op_config["through_date"])
    root = _in_season_nail_root()
    output_dir = root / "runs" / f"through_date={through_date}" / context.run_id
    result = run(
        argparse.Namespace(
            state_dir=_state_dir(),
            coverage_path=_cdn_coverage_path(),
            release_root=_release_root(),
            cdn_s3_prefix=(f"{_cdn_lake_uri()}/raw/cdn/playbyplay"),
            matchup_url=os.environ.get(
                "GESTALT_MATCHUP_API_URL", "http://127.0.0.1:8010/api/matchups"
            ),
            prior_possessions=float(context.op_config["prior_possessions"]),
            through_date=through_date,
            cache_dir=root,
            output_dir=output_dir,
        )
    )
    local_artifacts = result.pop("local_artifacts")
    store = AwsCliImmutableObjectStore()
    lake_uri = _in_season_nail_lake_uri()
    published: dict[str, str] = {}
    for label in (
        "summary",
        "game_predictions",
        "player_state",
        "player_game_trajectory",
    ):
        path = Path(str(local_artifacts[label]))
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        uri = _join_s3_uri(
            lake_uri,
            "analytical",
            "in_season_nail_shadow",
            f"season={_cdn_season()}",
            f"through_date={through_date}",
            f"{label}-{digest}{path.suffix}",
        )
        store.put_bytes(uri, payload)
        published[label] = uri

    publication_marker_root = root / "s3-published-contributions"
    for directory_value in local_artifacts["contribution_directories"]:
        directory = Path(str(directory_value))
        relative = directory.relative_to(root / "contributions")
        marker_path = publication_marker_root / relative.with_suffix(".json")
        if marker_path.exists():
            continue
        contribution_uris: list[str] = []
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            uri = _join_s3_uri(
                lake_uri,
                "analytical",
                "in_season_nail_shadow",
                "game_contributions",
                relative.as_posix(),
                path.name,
            )
            store.put_bytes(uri, path.read_bytes())
            contribution_uris.append(uri)
        marker_payload = {
            "schema_version": 1,
            "published_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "directory": str(directory),
            "objects": contribution_uris,
        }
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_tmp = marker_path.with_suffix(".json.tmp")
        marker_tmp.write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True) + "\n"
        )
        marker_tmp.replace(marker_path)

    publication = {
        "schema_version": 1,
        "season": _cdn_season(),
        "through_date": through_date,
        "updater_version": str(result["updater_version"]),
        "dagster_run_id": context.run_id,
        "generated_at": result["generated_at"],
        "updater_version": result["updater_version"],
        "artifacts": published,
        "metrics": result["metrics"],
        "comparison": result["comparison"],
        "coverage": {
            "games": result["game_count"],
            "contributions": result["contribution_count"],
            "excluded": len(result["excluded_contributions"]),
        },
    }
    publication_bytes = (
        json.dumps(publication, indent=2, sort_keys=True) + "\n"
    ).encode()
    publication_sha = hashlib.sha256(publication_bytes).hexdigest()
    publication_uri = _join_s3_uri(
        lake_uri,
        "analytical",
        "in_season_nail_shadow",
        "manifests",
        f"season={_cdn_season()}",
        f"through_date={through_date}",
        f"{publication_sha}.json",
    )
    store.put_bytes(publication_uri, publication_bytes)
    latest_path = root / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest_path.with_suffix(".json.tmp")
    temporary.write_bytes(publication_bytes)
    temporary.replace(latest_path)

    current = result["metrics"]["current_model"]
    updater = result["metrics"]["in_season_updater"]
    context.add_output_metadata(
        {
            "through_date": through_date,
            "games": int(result["game_count"]),
            "contributions": int(result["contribution_count"]),
            "excluded_contributions": len(result["excluded_contributions"]),
            "current_brier": float(current["brier_score"]),
            "updater_brier": float(updater["brier_score"]),
            "current_margin_mae": float(current["margin_mae"]),
            "updater_margin_mae": float(updater["margin_mae"]),
            "current_ats": f"{current['ats_wins']}-{current['ats_losses']}-{current['ats_pushes']}",
            "updater_ats": f"{updater['ats_wins']}-{updater['ats_losses']}-{updater['ats_pushes']}",
            "publication": MetadataValue.url(publication_uri),
        }
    )
    return {**result, "publication_uri": publication_uri, "published": published}


@job(
    name="update_in_season_nail_production",
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "in_season_nail_production"},
)
def update_in_season_nail_production_job() -> None:
    update_in_season_nail_production_op()


def _in_season_nail_production_config(through_date: str) -> dict[str, object]:
    return {
        "ops": {
            "run_in_season_nail_production": {
                "config": {
                    "through_date": through_date,
                    "prior_possessions": DEFAULT_IN_SEASON_NAIL_PRIOR_POSSESSIONS,
                }
            }
        }
    }


def _next_production_rating_slate() -> tuple[str, str] | SkipReason:
    """Return the next complete slate once every game has immutable PBP coverage."""

    rating_latest = _in_season_nail_root() / "latest.json"
    latest_rating_date = ""
    if rating_latest.is_file():
        try:
            latest_rating_date = str(
                json.loads(rating_latest.read_text()).get("through_date", "")
            )
        except (OSError, json.JSONDecodeError, TypeError):
            return SkipReason("The production ratings pointer is invalid")

    coverage_path = _cdn_coverage_path()
    if not coverage_path.is_file():
        return SkipReason("CDN play-by-play coverage is unavailable")
    try:
        coverage = json.loads(coverage_path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return SkipReason("CDN play-by-play coverage is invalid")
    covered_games = coverage.get("games", {}) if isinstance(coverage, dict) else {}
    if not isinstance(covered_games, dict):
        return SkipReason("CDN play-by-play game coverage is invalid")

    final_root = _state_dir() / "final"
    summaries: dict[str, tuple[Path, dict[str, object]]] = {}
    for path in sorted(final_root.glob("*.summary.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        game_id = str(payload.get("game_id", "")) if isinstance(payload, dict) else ""
        if game_id.startswith("002"):
            summaries[game_id] = (path, payload)

    today = datetime.now(UTC).astimezone(PACIFIC).date().isoformat()
    for slate_path in sorted((_state_dir() / "latest").glob("*.json")):
        through_date = slate_path.stem
        if through_date <= latest_rating_date or through_date > today:
            continue
        try:
            slate = json.loads(slate_path.read_text())
        except (OSError, json.JSONDecodeError, TypeError):
            return SkipReason(f"Schedule slate is invalid: {through_date}")
        games = [
            game
            for game in slate.get("games", [])
            if isinstance(game, dict) and str(game.get("game_id", "")).startswith("002")
        ]
        if not games:
            continue
        if not all(
            isinstance(game.get("status"), dict)
            and bool(game["status"].get("completed"))
            for game in games
        ):
            return SkipReason(f"The {through_date} slate is not final")
        game_ids = [str(game["game_id"]) for game in games]
        missing_pbp = [
            game_id for game_id in game_ids if not covered_games.get(game_id)
        ]
        if missing_pbp:
            return SkipReason(
                f"The {through_date} slate is waiting for CDN play-by-play: "
                + ", ".join(missing_pbp)
            )
        missing_finals = [game_id for game_id in game_ids if game_id not in summaries]
        if missing_finals:
            return SkipReason(
                f"The {through_date} slate is waiting for final archives: "
                + ", ".join(missing_finals)
            )

        source_parts = [hashlib.sha256(slate_path.read_bytes()).hexdigest()]
        for game_id in sorted(game_ids):
            summary_path, summary = summaries[game_id]
            espn_path = final_root / f"{summary.get('espn_game_id')}.espn.json"
            if not espn_path.is_file():
                return SkipReason(
                    f"The {through_date} slate is waiting for ESPN box score: {game_id}"
                )
            source_parts.extend(
                [
                    game_id,
                    str(covered_games[game_id][-1]),
                    hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    hashlib.sha256(espn_path.read_bytes()).hexdigest(),
                ]
            )
        snapshot = hashlib.sha256(":".join(source_parts).encode()).hexdigest()[:16]
        return through_date, snapshot
    return SkipReason("No newer complete production slate is ready")


@sensor(
    name="in_season_nail_production_ready",
    job=update_in_season_nail_production_job,
    minimum_interval_seconds=900,
    default_status=SENSOR_DEFAULT_STATUS,
    description=(
        "Advance production ratings as soon as the full slate is final and every game "
        "has immutable CDN play-by-play coverage."
    ),
)
def in_season_nail_production_sensor(
    context: SensorEvaluationContext,
) -> RunRequest | SkipReason:
    readiness = _next_production_rating_slate()
    if isinstance(readiness, SkipReason):
        return readiness
    through_date, snapshot = readiness
    inflight = context.instance.get_runs(
        filters=RunsFilter(
            job_name="update_in_season_nail_production",
            statuses=(
                DagsterRunStatus.QUEUED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.CANCELING,
            ),
            tags={"gestalt/ratings_snapshot": snapshot},
        ),
        limit=1,
    )
    if inflight:
        return SkipReason(
            f"Equivalent production ratings update already queued or running: {inflight[0].run_id}"
        )
    return RunRequest(
        run_key=f"production-ratings:{through_date}:{snapshot}",
        run_config=_in_season_nail_production_config(through_date),
        tags={
            WORKLOAD_TAG: "in_season_nail_production",
            "gestalt/through_date": through_date,
            "gestalt/ratings_snapshot": snapshot,
        },
    )


@op(
    name="run_in_season_minutes_production",
    config_schema={
        "through_date": str,
        "participation_prior_games": Field(
            int, default_value=DEFAULT_IN_SEASON_MINUTES_PARTICIPATION_PRIOR_GAMES
        ),
        "conditional_prior_games": Field(
            float, default_value=DEFAULT_IN_SEASON_MINUTES_CONDITIONAL_PRIOR_GAMES
        ),
        "conditional_lookback_games": Field(
            int, default_value=DEFAULT_IN_SEASON_MINUTES_CONDITIONAL_LOOKBACK_GAMES
        ),
        "role_cutoff_minutes": Field(
            int, default_value=DEFAULT_IN_SEASON_MINUTES_ROLE_CUTOFF_MINUTES
        ),
    },
    retry_policy=RetryPolicy(max_retries=1, delay=300),
)
def update_in_season_minutes_production_op(
    context: OpExecutionContext,
) -> dict[str, object]:
    """Advance the production participation/minutes state and publish its audit."""

    import argparse

    from nba_lineup_model.modeling.in_season_minutes_monitor import run
    from nba_lineup_model.season.stats_v3_lake import (
        AwsCliImmutableObjectStore,
        _join_s3_uri,
    )

    through_date = str(context.op_config["through_date"])
    root = _in_season_minutes_root()
    output_dir = root / "runs" / f"through_date={through_date}" / context.run_id
    result = run(
        argparse.Namespace(
            state_dir=_state_dir(),
            release_root=_release_root(),
            through_date=through_date,
            participation_prior_games=float(
                context.op_config["participation_prior_games"]
            ),
            conditional_prior_games=float(context.op_config["conditional_prior_games"]),
            conditional_lookback_games=int(
                context.op_config["conditional_lookback_games"]
            ),
            role_cutoff_minutes=int(context.op_config["role_cutoff_minutes"]),
            rating_prior_possessions=float(DEFAULT_IN_SEASON_NAIL_PRIOR_POSSESSIONS),
            rating_cache_dir=_in_season_nail_root(),
            rating_coverage_path=_cdn_coverage_path(),
            cdn_coverage_path=_cdn_coverage_path(),
            cdn_lake_uri=_cdn_lake_uri(),
            cache_dir=root,
            output_dir=output_dir,
        )
    )
    rating_publication = json.loads(
        (_in_season_nail_root() / "latest.json").read_text()
    )
    input_rating_artifacts = {
        "player_state": _published_artifact_uri(rating_publication, "player_state"),
        "player_game_trajectory": _published_artifact_uri(
            rating_publication, "player_game_trajectory"
        ),
    }
    local_artifacts = result.pop("local_artifacts")
    store = AwsCliImmutableObjectStore()
    lake_uri = _in_season_minutes_lake_uri()
    published: dict[str, str] = {}
    for label in (
        "report",
        "summary",
        "player_game_evaluation",
        "four_cell_game_evaluation",
        "player_state",
    ):
        path = Path(str(local_artifacts[label]))
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        uri = _join_s3_uri(
            lake_uri,
            "analytical",
            "in_season_minutes_shadow",
            f"season={_cdn_season()}",
            f"through_date={through_date}",
            f"{label}-{digest}{path.suffix}",
        )
        store.put_bytes(uri, payload)
        published[label] = uri

    marker_root = root / "s3-published-contributions"
    for directory_value in local_artifacts["contribution_directories"]:
        directory = Path(str(directory_value))
        relative = directory.relative_to(root / "contributions")
        marker = marker_root / relative.with_suffix(".json")
        if marker.exists():
            continue
        objects: list[str] = []
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            uri = _join_s3_uri(
                lake_uri,
                "analytical",
                "in_season_minutes_shadow",
                "game_contributions",
                relative.as_posix(),
                path.name,
            )
            store.put_bytes(uri, path.read_bytes())
            objects.append(uri)
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "published_at": datetime.now(UTC)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "objects": objects,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary.replace(marker)

    publication = {
        "schema_version": 1,
        "season": _cdn_season(),
        "through_date": through_date,
        "dagster_run_id": context.run_id,
        "generated_at": result["generated_at"],
        "artifacts": published,
        "coverage": {
            "games": result["games"],
            "player_game_rows": result["player_game_rows"],
            "excluded": len(result["excluded_games"]),
        },
        "player_metrics": result["player_metrics"],
        "four_cell_game_metrics": result["four_cell_game_metrics"],
        "minutes_ab_game_metrics": result["minutes_ab_game_metrics"],
        "early_role_coverage": result["early_role_coverage"],
        "input_rating_artifacts": input_rating_artifacts,
    }
    publication_bytes = (
        json.dumps(publication, indent=2, sort_keys=True) + "\n"
    ).encode()
    publication_sha = hashlib.sha256(publication_bytes).hexdigest()
    publication_uri = _join_s3_uri(
        lake_uri,
        "analytical",
        "in_season_minutes_shadow",
        "manifests",
        f"season={_cdn_season()}",
        f"through_date={through_date}",
        f"{publication_sha}.json",
    )
    store.put_bytes(publication_uri, publication_bytes)
    latest = root / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest.with_suffix(".json.tmp")
    temporary.write_bytes(publication_bytes)
    temporary.replace(latest)
    context.add_output_metadata(
        {
            "through_date": through_date,
            "games": int(result["games"]),
            "player_game_rows": int(result["player_game_rows"]),
            "excluded_games": len(result["excluded_games"]),
            "created_contributions": int(result["created_contributions"]),
            "cached_contributions": int(result["cached_contributions"]),
            "publication": MetadataValue.url(publication_uri),
            "report": MetadataValue.url(published["report"]),
        }
    )
    return {**result, "publication_uri": publication_uri, "published": published}


@job(
    name="update_in_season_minutes_production",
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "in_season_minutes_production"},
)
def update_in_season_minutes_production_job() -> None:
    update_in_season_minutes_production_op()


def _in_season_minutes_production_config(through_date: str) -> dict[str, object]:
    return {
        "ops": {
            "run_in_season_minutes_production": {
                "config": {
                    "through_date": through_date,
                    "participation_prior_games": (
                        DEFAULT_IN_SEASON_MINUTES_PARTICIPATION_PRIOR_GAMES
                    ),
                    "conditional_prior_games": (
                        DEFAULT_IN_SEASON_MINUTES_CONDITIONAL_PRIOR_GAMES
                    ),
                    "conditional_lookback_games": (
                        DEFAULT_IN_SEASON_MINUTES_CONDITIONAL_LOOKBACK_GAMES
                    ),
                    "role_cutoff_minutes": (
                        DEFAULT_IN_SEASON_MINUTES_ROLE_CUTOFF_MINUTES
                    ),
                }
            }
        }
    }


@sensor(
    name="in_season_minutes_production_ready",
    job=update_in_season_minutes_production_job,
    minimum_interval_seconds=900,
    default_status=SENSOR_DEFAULT_STATUS,
    description=(
        "Advance production minutes immediately after a new exclusion-free ratings "
        "publication; equivalent queued or active work is suppressed."
    ),
)
def in_season_minutes_production_sensor(
    context: SensorEvaluationContext,
) -> RunRequest | SkipReason:
    rating_path = _in_season_nail_root() / "latest.json"
    if not rating_path.is_file():
        return SkipReason("Production minutes are waiting for a ratings publication")
    try:
        rating_publication = json.loads(rating_path.read_text())
        through_date = str(rating_publication["through_date"])
        rating_coverage = rating_publication["coverage"]
        rating_state_uri = _published_artifact_uri(rating_publication, "player_state")
        rating_trajectory_uri = _published_artifact_uri(
            rating_publication, "player_game_trajectory"
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        return SkipReason(f"Production ratings publication is invalid: {error}")
    if not isinstance(rating_coverage, dict) or int(
        rating_coverage.get("excluded", -1)
    ):
        return SkipReason("Production ratings publication has exclusions")

    minutes_path = _in_season_minutes_root() / "latest.json"
    if minutes_path.is_file():
        try:
            minutes_publication = json.loads(minutes_path.read_text())
            if (
                str(minutes_publication.get("through_date")) == through_date
                and str(minutes_publication.get("updater_version"))
                == IN_SEASON_MINUTES_UPDATER_VERSION
                and minutes_publication.get("input_rating_artifacts")
                == {
                    "player_state": rating_state_uri,
                    "player_game_trajectory": rating_trajectory_uri,
                }
            ):
                return SkipReason(
                    "Production minutes already match the ratings publication"
                )
            serving_path = _current_season_serving_root() / "latest.json"
            if (
                str(minutes_publication.get("through_date")) == through_date
                and str(minutes_publication.get("updater_version"))
                == IN_SEASON_MINUTES_UPDATER_VERSION
                and serving_path.is_file()
            ):
                serving_pointer = json.loads(serving_path.read_text())
                manifest_path = (
                    _current_season_serving_root()
                    / str(serving_pointer["bundle_path"])
                    / "manifest.json"
                )
                sources = json.loads(manifest_path.read_text())["identity"]["sources"]
                if (
                    sources["rating_state"]["uri"] == rating_state_uri
                    and sources["rating_trajectory"]["uri"] == rating_trajectory_uri
                    and sources["minutes_state"]["uri"]
                    == _published_artifact_uri(minutes_publication, "player_state")
                ):
                    return SkipReason(
                        "Production minutes and serving state already match ratings"
                    )
        except (KeyError, OSError, ValueError, json.JSONDecodeError, TypeError):
            pass

    final_root = _state_dir() / "final"
    source_parts: list[str] = [
        IN_SEASON_MINUTES_UPDATER_VERSION,
        rating_state_uri,
        rating_trajectory_uri,
    ]
    final_game_count = 0
    for summary_path in sorted(final_root.glob("*.summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(summary, dict)
            or not summary.get("final", {}).get("status", {}).get("completed")
            or str(summary.get("game_id", ""))[:3] != "002"
            or summary_slate_date(summary).isoformat() > through_date
        ):
            continue
        espn_path = final_root / f"{summary.get('espn_game_id')}.espn.json"
        if not espn_path.is_file():
            continue
        source_parts.extend(
            [
                hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                hashlib.sha256(espn_path.read_bytes()).hexdigest(),
            ]
        )
        final_game_count += 1
    if not final_game_count:
        return SkipReason(f"No completed games are available through {through_date}")
    coverage_path = _cdn_coverage_path()
    if coverage_path.is_file():
        source_parts.append(hashlib.sha256(coverage_path.read_bytes()).hexdigest())
    snapshot = hashlib.sha256(":".join(source_parts).encode()).hexdigest()[:16]
    inflight = context.instance.get_runs(
        filters=RunsFilter(
            job_name="update_in_season_minutes_production",
            statuses=(
                DagsterRunStatus.QUEUED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.CANCELING,
            ),
            tags={"gestalt/minutes_snapshot": snapshot},
        ),
        limit=1,
    )
    if inflight:
        return SkipReason(
            f"Equivalent production minutes update already queued or running: {inflight[0].run_id}"
        )
    return RunRequest(
        run_key=f"production-minutes:{through_date}:{snapshot}",
        run_config=_in_season_minutes_production_config(through_date),
        tags={
            WORKLOAD_TAG: "in_season_minutes_production",
            "gestalt/through_date": through_date,
            "gestalt/minutes_snapshot": snapshot,
        },
    )


def _current_season_child_publications() -> tuple[dict[str, object], dict[str, object]]:
    """Load the locally mirrored immutable child-state publication manifests."""

    paths = (
        _in_season_nail_root() / "latest.json",
        _in_season_minutes_root() / "latest.json",
    )
    payloads: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Current-season child publication is missing: {path}")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Current-season child publication is invalid: {path}")
        payloads.append(payload)
    return payloads[0], payloads[1]


def _published_artifact_uri(publication: dict[str, object], label: str) -> str:
    artifacts = publication.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Child publication artifact inventory is missing")
    uri = artifacts.get(label)
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError(f"Child publication artifact is invalid: {label}")
    return uri


def _published_file_digest(uri: str, *, label: str, suffix: str) -> str:
    filename = uri.rsplit("/", 1)[-1]
    prefix = f"{label}-"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ValueError(f"Published artifact name is invalid: {uri}")
    digest = filename[len(prefix) : -len(suffix)]
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"Published artifact checksum is invalid: {uri}")
    return digest


def _current_season_child_state_paths(
    through_date: str,
    rating_publication: dict[str, object],
    minutes_publication: dict[str, object],
) -> tuple[Path, Path, Path, str, str, str]:
    """Resolve exact local child states and verify their published checksums."""

    rating_uri = _published_artifact_uri(rating_publication, "player_state")
    trajectory_uri = _published_artifact_uri(
        rating_publication, "player_game_trajectory"
    )
    minutes_uri = _published_artifact_uri(minutes_publication, "player_state")
    rating_digest = _published_file_digest(
        rating_uri, label="player_state", suffix=".parquet"
    )
    minutes_digest = _published_file_digest(
        minutes_uri, label="player_state", suffix=".parquet"
    )
    trajectory_digest = _published_file_digest(
        trajectory_uri,
        label="player_game_trajectory",
        suffix=".parquet",
    )
    rating_matches = list(
        _in_season_nail_root().glob(
            f"player_states/season={_cdn_season()}/cutoff=*/"
            f"state_sha256={rating_digest}/player_state.parquet"
        )
    )
    if len(rating_matches) != 1:
        raise ValueError(
            f"Expected one local rating state for {rating_digest}, found {len(rating_matches)}"
        )
    minutes_run_id = minutes_publication.get("dagster_run_id")
    if not isinstance(minutes_run_id, str) or not minutes_run_id:
        raise ValueError("Minutes publication Dagster run ID is missing")
    minutes_path = (
        _in_season_minutes_root()
        / "runs"
        / f"through_date={through_date}"
        / minutes_run_id
        / "player_state.parquet"
    )
    if not minutes_path.is_file():
        raise ValueError(f"Local minutes state is missing: {minutes_path}")
    if hashlib.sha256(minutes_path.read_bytes()).hexdigest() != minutes_digest:
        raise ValueError("Local minutes state does not match its published checksum")
    rating_run_id = rating_publication.get("dagster_run_id")
    if not isinstance(rating_run_id, str) or not rating_run_id:
        raise ValueError("Rating publication Dagster run ID is missing")
    trajectory_path = (
        _in_season_nail_root()
        / "runs"
        / f"through_date={through_date}"
        / rating_run_id
        / "player_game_trajectory.parquet"
    )
    if not trajectory_path.is_file():
        raise ValueError(f"Local player-game trajectory is missing: {trajectory_path}")
    if hashlib.sha256(trajectory_path.read_bytes()).hexdigest() != trajectory_digest:
        raise ValueError(
            "Local player-game trajectory does not match its published checksum"
        )
    return (
        rating_matches[0],
        minutes_path,
        trajectory_path,
        rating_uri,
        minutes_uri,
        trajectory_uri,
    )


def _preseason_release_run_id() -> str:
    release_path = _release_root() / "release.json"
    if not release_path.is_file():
        raise ValueError(f"Release metadata is missing: {release_path}")
    payload = json.loads(release_path.read_text())
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Release metadata has no promoted preseason run ID")
    return run_id


@op(
    name="publish_current_season_serving_state",
    config_schema={"through_date": str},
    retry_policy=RetryPolicy(max_retries=1, delay=300),
)
def promote_current_season_serving_op(
    context: OpExecutionContext,
) -> dict[str, object]:
    """Publish and atomically promote one synchronized current-season state."""

    from nba_lineup_model.season.current_serving import (
        MANIFEST_FILENAME,
        RATING_TRAJECTORY_FILENAME,
        REPORT_FILENAME,
        SERVING_STATE_FILENAME,
        ServingPromotionConfig,
        promote_serving_bundle,
        publish_serving_bundle,
    )
    from nba_lineup_model.season.stats_v3_lake import (
        AwsCliImmutableObjectStore,
        _join_s3_uri,
    )

    through_date = str(context.op_config["through_date"])
    datetime.fromisoformat(through_date)
    rating_publication, minutes_publication = _current_season_child_publications()
    child_dates = {
        str(rating_publication.get("through_date")),
        str(minutes_publication.get("through_date")),
    }
    if child_dates != {through_date}:
        raise ValueError(
            "Rating and minutes publications must match the requested through date: "
            + ", ".join(sorted(child_dates))
        )
    for label, publication in (
        ("rating", rating_publication),
        ("minutes", minutes_publication),
    ):
        coverage = publication.get("coverage")
        if not isinstance(coverage, dict) or int(coverage.get("excluded", -1)) != 0:
            raise ValueError(
                f"{label.title()} publication has exclusions or invalid coverage"
            )
    (
        rating_path,
        minutes_path,
        rating_trajectory_path,
        rating_uri,
        minutes_uri,
        rating_trajectory_uri,
    ) = _current_season_child_state_paths(
        through_date,
        rating_publication,
        minutes_publication,
    )
    next_midnight_pacific = datetime.fromisoformat(f"{through_date}T00:00:00").replace(
        tzinfo=PACIFIC
    ) + timedelta(days=1)
    config = ServingPromotionConfig(
        preseason_run_id=_preseason_release_run_id(),
        rating_prior_possessions=float(DEFAULT_IN_SEASON_NAIL_PRIOR_POSSESSIONS),
        minutes_participation_prior_games=(
            DEFAULT_IN_SEASON_MINUTES_PARTICIPATION_PRIOR_GAMES
        ),
        minutes_conditional_prior_games=(
            DEFAULT_IN_SEASON_MINUTES_CONDITIONAL_PRIOR_GAMES
        ),
        minutes_conditional_lookback_games=(
            DEFAULT_IN_SEASON_MINUTES_CONDITIONAL_LOOKBACK_GAMES
        ),
        minutes_role_cutoff_minutes=DEFAULT_IN_SEASON_MINUTES_ROLE_CUTOFF_MINUTES,
    )
    root = _current_season_serving_root()
    bundle = publish_serving_bundle(
        rating_path,
        minutes_path,
        root=root,
        season=_cdn_season(),
        evidence_through_date=next_midnight_pacific.date() - timedelta(days=1),
        as_of_cutoff_time=next_midnight_pacific,
        config=config,
        rating_source_uri=rating_uri,
        minutes_source_uri=minutes_uri,
        rating_trajectory_path=rating_trajectory_path,
        rating_trajectory_source_uri=rating_trajectory_uri,
    )
    store = AwsCliImmutableObjectStore()
    base_uri = _join_s3_uri(
        _current_season_serving_lake_uri(),
        "serving",
        "current_season",
        f"season={_cdn_season()}",
        f"through_date={through_date}",
        f"bundle_sha256={bundle.manifest['bundle_sha256']}",
    )
    published: dict[str, str] = {}
    for filename in (
        SERVING_STATE_FILENAME,
        RATING_TRAJECTORY_FILENAME,
        MANIFEST_FILENAME,
        REPORT_FILENAME,
    ):
        uri = f"{base_uri}/{filename}"
        store.put_bytes(uri, (bundle.path / filename).read_bytes())
        published[filename] = uri
    pointer, promoted = promote_serving_bundle(bundle.path, root=root)
    pointer_payload = {
        **pointer,
        "dagster_run_id": context.run_id,
        "artifacts": published,
    }
    pointer_bytes = (
        json.dumps(pointer_payload, indent=2, sort_keys=True) + "\n"
    ).encode()
    pointer_uri = _join_s3_uri(
        _current_season_serving_lake_uri(),
        "serving",
        "current_season",
        "pointers",
        f"{bundle.manifest['bundle_sha256']}.json",
    )
    store.put_bytes(pointer_uri, pointer_bytes)
    coverage = bundle.manifest["identity"]["coverage"]
    context.add_output_metadata(
        {
            "through_date": through_date,
            "bundle_sha256": str(bundle.manifest["bundle_sha256"]),
            "players": int(coverage["serving_player_count"]),
            "teams": int(coverage["team_count"]),
            "games": int(coverage["game_count"]),
            "created": bundle.created,
            "promoted": promoted,
            "manifest": MetadataValue.url(published[MANIFEST_FILENAME]),
            "report": MetadataValue.url(published[REPORT_FILENAME]),
            "pointer": MetadataValue.url(pointer_uri),
        }
    )
    return {
        "through_date": through_date,
        "bundle_path": str(bundle.path),
        "bundle_sha256": bundle.manifest["bundle_sha256"],
        "created": bundle.created,
        "promoted": promoted,
        "published": published,
        "pointer_uri": pointer_uri,
    }


@job(
    name="promote_current_season_serving",
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "current_season_serving"},
)
def promote_current_season_serving_job() -> None:
    promote_current_season_serving_op()


def _current_season_serving_config(through_date: str) -> dict[str, object]:
    return {
        "ops": {
            "publish_current_season_serving_state": {
                "config": {"through_date": through_date}
            }
        }
    }


@sensor(
    name="current_season_serving_ready",
    job=promote_current_season_serving_job,
    minimum_interval_seconds=900,
    default_status=SENSOR_DEFAULT_STATUS,
    description=(
        "Promote one current-season serving bundle after matching, exclusion-free "
        "production ratings and minutes publications are complete."
    ),
)
def current_season_serving_sensor(
    context: SensorEvaluationContext,
) -> RunRequest | SkipReason:
    try:
        rating_publication, minutes_publication = _current_season_child_publications()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return SkipReason(str(error))
    rating_date = str(rating_publication.get("through_date", ""))
    minutes_date = str(minutes_publication.get("through_date", ""))
    if not rating_date or rating_date != minutes_date:
        return SkipReason(
            f"Current-season child dates do not match: {rating_date or '-'} / {minutes_date or '-'}"
        )
    for label, publication in (
        ("ratings", rating_publication),
        ("minutes", minutes_publication),
    ):
        coverage = publication.get("coverage")
        if not isinstance(coverage, dict) or int(coverage.get("excluded", -1)) != 0:
            return SkipReason(f"{label.title()} publication is not exclusion-free")
    try:
        rating_uri = _published_artifact_uri(rating_publication, "player_state")
        trajectory_uri = _published_artifact_uri(
            rating_publication, "player_game_trajectory"
        )
        minutes_uri = _published_artifact_uri(minutes_publication, "player_state")
    except ValueError as error:
        return SkipReason(str(error))
    snapshot = hashlib.sha256(
        f"{rating_uri}:{trajectory_uri}:{minutes_uri}".encode()
    ).hexdigest()[:16]
    inflight = context.instance.get_runs(
        filters=RunsFilter(
            job_name="promote_current_season_serving",
            statuses=(
                DagsterRunStatus.QUEUED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.CANCELING,
            ),
            tags={"gestalt/serving_snapshot": snapshot},
        ),
        limit=1,
    )
    if inflight:
        return SkipReason(
            "Equivalent current-season serving promotion already queued or running: "
            f"{inflight[0].run_id}"
        )
    latest_path = _current_season_serving_root() / "latest.json"
    if latest_path.is_file():
        try:
            latest = json.loads(latest_path.read_text())
            bundle_path = _current_season_serving_root() / str(latest["bundle_path"])
            manifest = json.loads((bundle_path / "manifest.json").read_text())
            sources = manifest["identity"]["sources"]
            if (
                sources["rating_state"]["uri"] == rating_uri
                and sources["rating_trajectory"]["uri"] == trajectory_uri
                and sources["minutes_state"]["uri"] == minutes_uri
            ):
                return SkipReason("Current-season serving state is already promoted")
        except (KeyError, OSError, json.JSONDecodeError, TypeError):
            pass
    return RunRequest(
        run_key=f"current-season-serving:{rating_date}:{snapshot}",
        run_config=_current_season_serving_config(rating_date),
        tags={
            WORKLOAD_TAG: "current_season_serving",
            "gestalt/through_date": rating_date,
            "gestalt/serving_snapshot": snapshot,
        },
    )


@op(
    name="materialize_current_season_lineups",
    retry_policy=RetryPolicy(max_retries=1, delay=300),
)
def materialize_current_season_lineups_op(
    context: OpExecutionContext,
) -> dict[str, object]:
    """Publish observed current lineups tied to the promoted serving checksum."""

    from nba_lineup_model.modeling.current_season_lineup_materialization import (
        materialize_current_season_lineups,
    )
    from nba_lineup_model.web_api.inference import LineupEvaluator

    path = materialize_current_season_lineups(
        evaluator=LineupEvaluator.from_latest_artifact(),
        schedule_state_dir=_state_dir(),
        coverage_path=_cdn_coverage_path(),
        cdn_s3_prefix=f"{_cdn_lake_uri()}/raw/cdn/playbyplay",
        output_root=_current_season_lineups_root(),
    )
    manifest = json.loads((path / "manifest.json").read_text())
    identity = manifest["identity"]
    result = {
        "bundle_path": str(path),
        "bundle_sha256": str(manifest["bundle_sha256"]),
        "serving_bundle_sha256": str(identity["serving_bundle_sha256"]),
        "game_count": len(identity["source_game_ids"]),
        "lineup_count": int(identity["lineup_count"]),
    }
    context.add_output_metadata(result)
    return result


@job(
    name="materialize_promoted_current_season_lineups",
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "current_season_lineups"},
)
def materialize_current_season_lineups_job() -> None:
    materialize_current_season_lineups_op()


@sensor(
    name="promoted_current_season_lineups_ready",
    job=materialize_current_season_lineups_job,
    minimum_interval_seconds=900,
    default_status=SENSOR_DEFAULT_STATUS,
    description=(
        "Materialize observed lineups after a new synchronized current-season "
        "rating/minutes serving state is promoted."
    ),
)
def current_season_lineups_sensor(
    context: SensorEvaluationContext,
) -> RunRequest | SkipReason:
    serving_path = _current_season_serving_root() / "latest.json"
    if not serving_path.is_file():
        return SkipReason("No promoted current-season serving state")
    try:
        serving = json.loads(serving_path.read_text())
        serving_sha = str(serving["bundle_sha256"])
    except (KeyError, OSError, json.JSONDecodeError, TypeError) as error:
        return SkipReason(f"Invalid current-season serving pointer: {error}")
    lineup_path = _current_season_lineups_root() / "latest.json"
    if lineup_path.is_file():
        try:
            lineup = json.loads(lineup_path.read_text())
            if str(lineup.get("serving_bundle_sha256")) == serving_sha:
                return SkipReason(
                    "Current-season lineups already match the serving state"
                )
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    inflight = context.instance.get_runs(
        filters=RunsFilter(
            job_name="materialize_promoted_current_season_lineups",
            statuses=(
                DagsterRunStatus.QUEUED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.CANCELING,
            ),
            tags={"gestalt/serving_bundle": serving_sha[:16]},
        ),
        limit=1,
    )
    if inflight:
        return SkipReason(
            "Equivalent current-season lineup materialization already queued or running: "
            f"{inflight[0].run_id}"
        )
    return RunRequest(
        run_key=f"current-season-lineups:{serving_sha}",
        tags={
            WORKLOAD_TAG: "current_season_lineups",
            "gestalt/serving_bundle": serving_sha[:16],
        },
    )


@op(
    name="archive_completed_stats_v3",
    config_schema={
        "season": Field(str, default_value=DEFAULT_STATS_V3_SEASON),
        "start_date": Field(str, default_value=""),
        "end_date": Field(str, default_value=""),
        "game_ids": Field([str], default_value=[]),
        "refresh": Field(bool, default_value=False),
        "ignore_grace": Field(bool, default_value=False),
        "grace_minutes": Field(int, default_value=DEFAULT_STATS_V3_GRACE_MINUTES),
    },
    retry_policy=RetryPolicy(max_retries=1, delay=300),
)
def archive_completed_stats_v3_op(context: OpExecutionContext) -> dict[str, object]:
    """Archive immutable V3 PBP and box scores for eligible final games."""

    from datetime import date

    from nba_lineup_model.season.stats_v3_lake import ingest_completed_stats_v3

    config = context.op_config
    start_date = (
        date.fromisoformat(config["start_date"]) if config["start_date"] else None
    )
    end_date = date.fromisoformat(config["end_date"]) if config["end_date"] else None
    result = ingest_completed_stats_v3(
        state_dir=_state_dir(),
        lake_uri=_stats_v3_lake_uri(),
        coverage_path=_stats_v3_coverage_path(),
        season=str(config["season"]),
        start_date=start_date,
        end_date=end_date,
        game_ids=list(config["game_ids"]),
        refresh=bool(config["refresh"]),
        ignore_grace=bool(config["ignore_grace"]),
        grace_period=timedelta(minutes=int(config["grace_minutes"])),
    )
    context.add_output_metadata(
        {
            "selected_games": int(result["selected_game_count"]),
            "archived": int(result["archived_count"]),
            "revised": int(result["revised_count"]),
            "unchanged": int(result["unchanged_count"]),
            "skipped": int(result["skipped_count"]),
            "manifest_uri": MetadataValue.url(str(result["manifest_uri"])),
        }
    )
    return result


@job(
    name="ingest_completed_stats_v3",
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "stats_v3_ingest"},
)
def ingest_completed_stats_v3_job() -> None:
    archive_completed_stats_v3_op()


@sensor(
    name="completed_game_stats_v3",
    job=ingest_completed_stats_v3_job,
    minimum_interval_seconds=900,
    default_status=STATS_V3_SENSOR_DEFAULT_STATUS,
    description=(
        "Archive V3 PBP and box scores once each game has been final for one hour."
    ),
)
def completed_game_stats_v3_sensor(
    context: SensorEvaluationContext,
) -> RunRequest | SkipReason:
    from nba_lineup_model.season.stats_v3_lake import pending_completed_game_ids

    now = datetime.now(UTC)
    pending = pending_completed_game_ids(
        state_dir=_state_dir(),
        lake_uri=_stats_v3_lake_uri(),
        season=_stats_v3_season(),
        coverage_path=_stats_v3_coverage_path(),
        now=now,
        grace_period=timedelta(minutes=DEFAULT_STATS_V3_GRACE_MINUTES),
    )
    if not pending:
        return SkipReason("No final games are missing required Stats V3 artifacts")
    interval_bucket = int(now.timestamp() // 900)
    pending_digest = hashlib.sha256(",".join(pending).encode()).hexdigest()[:12]
    return RunRequest(
        run_key=f"stats-v3:{interval_bucket}:{pending_digest}",
        run_config={
            "ops": {
                "archive_completed_stats_v3": {
                    "config": {
                        "season": _stats_v3_season(),
                        "game_ids": pending,
                        "refresh": False,
                        "ignore_grace": False,
                        "grace_minutes": DEFAULT_STATS_V3_GRACE_MINUTES,
                    }
                }
            }
        },
        tags={
            WORKLOAD_TAG: "stats_v3_ingest",
            "gestalt/game_count": str(len(pending)),
        },
    )


def _stats_v3_reconciliation_config(context: object) -> dict[str, object]:
    game_date = (
        _scheduled_time(context).astimezone(EASTERN).date() - timedelta(days=1)
    ).isoformat()
    return {
        "ops": {
            "archive_completed_stats_v3": {
                "config": {
                    "season": _stats_v3_season(),
                    "start_date": game_date,
                    "end_date": game_date,
                    "game_ids": [],
                    "refresh": True,
                    "ignore_grace": True,
                    "grace_minutes": DEFAULT_STATS_V3_GRACE_MINUTES,
                }
            }
        }
    }


stats_v3_morning_reconciliation_schedule = ScheduleDefinition(
    name="stats_v3_morning_reconciliation",
    job=ingest_completed_stats_v3_job,
    cron_schedule="0 8 * * *",
    execution_timezone="America/New_York",
    run_config_fn=_stats_v3_reconciliation_config,
    default_status=STATS_V3_SCHEDULE_DEFAULT_STATUS,
    description="Re-fetch yesterday's final-game V3 artifacts and preserve revisions.",
)


@asset(
    name="s3_lake_query_health",
    group_name="data_lake",
    retry_policy=RETRY_POLICY,
    description="Read a small immutable Parquet object directly from S3 with DuckDB.",
)
def s3_lake_query_health(context: AssetExecutionContext) -> int:
    from nba_lineup_model.storage.duckdb_lake import s3_parquet_row_count

    uri = os.environ.get("GESTALT_LAKE_HEALTHCHECK_URI", DEFAULT_HEALTHCHECK_PARQUET)
    count = s3_parquet_row_count(uri)
    context.add_output_metadata({"rows": count, "s3_uri": MetadataValue.url(uri)})
    return count


daily_lake_health_job = define_asset_job(
    "daily_lake_health",
    selection=[s3_lake_query_health],
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "lake_health"},
)

daily_lake_health_schedule = ScheduleDefinition(
    name="daily_s3_lake_health",
    target=daily_lake_health_job,
    cron_schedule="15 10 * * *",
    execution_timezone="UTC",
    default_status=SCHEDULE_DEFAULT_STATUS,
    description="Prove that DuckDB can query the immutable S3 lake directly each day.",
)


defs = Definitions(
    assets=[s3_lake_query_health],
    jobs=[
        refresh_schedule_job,
        archive_nba_market_odds_job,
        update_live_scores_job,
        reconcile_final_game_archives_job,
        ingest_completed_cdn_play_by_play_job,
        update_in_season_nail_production_job,
        update_in_season_minutes_production_job,
        promote_current_season_serving_job,
        materialize_current_season_lineups_job,
        ingest_completed_stats_v3_job,
        daily_lake_health_job,
    ],
    schedules=[
        schedule_refresh_schedule,
        nba_market_odds_schedule,
        yesterday_finalization_schedule,
        final_game_archive_morning_reconciliation_schedule,
        cdn_morning_reconciliation_schedule,
        stats_v3_morning_reconciliation_schedule,
        daily_lake_health_schedule,
    ],
    sensors=[
        live_score_sensor,
        completed_game_cdn_play_by_play_sensor,
        in_season_nail_production_sensor,
        in_season_minutes_production_sensor,
        current_season_serving_sensor,
        current_season_lineups_sensor,
        completed_game_stats_v3_sensor,
    ],
)
