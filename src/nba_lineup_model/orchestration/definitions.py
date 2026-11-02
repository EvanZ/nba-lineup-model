"""Resource-bounded Dagster jobs for the production NBA GESTALT instance."""

import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
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
DEFAULT_GAME_CATALOG_PATH = Path("/var/lib/nba-gestalt/catalog/games.parquet")
DEFAULT_IN_SEASON_NAIL_ROOT = Path("/var/lib/nba-gestalt/in-season-nail")
DEFAULT_IN_SEASON_NAIL_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_IN_SEASON_NAIL_PRIOR_POSSESSIONS = 6000
DEFAULT_IN_SEASON_MINUTES_ROOT = Path("/var/lib/nba-gestalt/in-season-minutes")
DEFAULT_IN_SEASON_MINUTES_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_IN_SEASON_MINUTES_PARTICIPATION_PRIOR_GAMES = 4
DEFAULT_IN_SEASON_MINUTES_CONDITIONAL_PRIOR_GAMES = 0.2
DEFAULT_IN_SEASON_MINUTES_CONDITIONAL_LOOKBACK_GAMES = 3
DEFAULT_IN_SEASON_MINUTES_ROLE_CUTOFF_MINUTES = 44
DEFAULT_IN_SEASON_ROSTER_ROOT = Path("/var/lib/nba-gestalt/in-season-rosters")
DEFAULT_IN_SEASON_ROSTER_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_CURRENT_SEASON_SERVING_ROOT = Path(
    "/var/lib/nba-gestalt/current-season-serving"
)
DEFAULT_CURRENT_SEASON_SERVING_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_CURRENT_SEASON_SERVING_HEALTH_URL = "http://127.0.0.1:8010/api/health"
DEFAULT_CURRENT_SEASON_SERVING_VERIFY_TIMEOUT_SECONDS = 180
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


def _in_season_roster_root() -> Path:
    return Path(
        os.environ.get("GESTALT_IN_SEASON_ROSTER_ROOT", DEFAULT_IN_SEASON_ROSTER_ROOT)
    )


def _in_season_roster_lake_uri() -> str:
    return os.environ.get(
        "GESTALT_IN_SEASON_ROSTER_LAKE_URI", DEFAULT_IN_SEASON_ROSTER_LAKE_URI
    ).rstrip("/")


def _in_season_roster_artifact(label: str) -> Path | None:
    latest = _in_season_roster_root() / "latest.json"
    if not latest.is_file():
        return None
    manifest = json.loads(latest.read_text())
    artifacts = manifest.get("artifacts")
    checksums = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not isinstance(checksums, dict):
        raise ValueError("In-season roster manifest has no artifact inventory")
    path = Path(str(artifacts.get(label, "")))
    expected = str(checksums.get(label, ""))
    if not path.is_file() or not expected:
        raise ValueError(f"In-season roster artifact is unavailable: {label}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"In-season roster artifact checksum changed: {label}")
    return path


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
    """Refresh official NBA boards, lines, injuries, and immutable snapshots."""

    from nba_lineup_model.season.official_schedule_runtime import (
        refresh_official_schedule_window,
    )

    payload = refresh_official_schedule_window(
        start_date=str(context.op_config["start_date"]),
        days_ahead=int(context.op_config["days_ahead"]),
        state_dir=_state_dir(),
        game_catalog_path=Path(
            os.environ.get("GESTALT_GAME_CATALOG_PATH", DEFAULT_GAME_CATALOG_PATH)
        ),
        projection_url=os.environ.get(
            "GESTALT_SCHEDULE_API_URL", DEFAULT_SCHEDULE_API_URL
        ),
        s3_uri=_schedule_s3_uri(),
    )
    context.add_output_metadata(
        {
            "archived_days": int(payload["archived_day_count"]),
            "games": int(payload["game_count"]),
            "newly_published_candidates": int(
                payload["newly_published_candidate_count"]
            ),
            "newly_frozen_games": int(payload["newly_frozen_game_count"]),
            "missing_pregame_candidates": len(
                payload["missing_pregame_candidate_game_ids"]
            ),
            "provider_fallbacks": 0,
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
    game_date = _scheduled_time(context).astimezone(PACIFIC).date().isoformat()
    return _schedule_run_config(
        start_date=game_date, days_ahead=DEFAULT_SCHEDULE_DAYS_AHEAD
    )


def _yesterday_refresh_config(context: object) -> dict[str, object]:
    game_date = (
        _scheduled_time(context).astimezone(PACIFIC).date() - timedelta(days=1)
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


def active_pregame_projection_date(
    *,
    state_dir: Path | str,
    now: datetime | None = None,
    lead_time: timedelta = timedelta(minutes=45),
) -> str | None:
    """Return a slate with a game approaching tipoff inside a bounded window."""

    current = (now or datetime.now(UTC)).astimezone(UTC)
    slate_date = current.astimezone(PACIFIC).date()
    state_root = Path(state_dir)
    for game_date in (
        (slate_date - timedelta(days=1)).isoformat(),
        slate_date.isoformat(),
        (slate_date + timedelta(days=1)).isoformat(),
    ):
        path = state_root / "latest" / f"{game_date}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        games = payload.get("games", []) if isinstance(payload, dict) else []
        future_starts = sorted(
            datetime.fromisoformat(str(game["start_time"]).replace("Z", "+00:00"))
            for game in games
            if isinstance(game, dict)
            and game.get("start_time")
            and str(game.get("status", {}).get("state", "pre")) == "pre"
            and datetime.fromisoformat(
                str(game["start_time"]).replace("Z", "+00:00")
            )
            > current
        )
        if future_starts and future_starts[0] - lead_time <= current:
            return game_date
    return None


@sensor(
    name="pregame_projection_window",
    job=refresh_schedule_job,
    minimum_interval_seconds=60,
    default_status=SENSOR_DEFAULT_STATUS,
    description=(
        "Refresh official injuries and projection candidates every five minutes only "
        "during the 45 minutes before each tipoff."
    ),
)
def pregame_projection_sensor(
    context: SensorEvaluationContext,
) -> RunRequest | SkipReason:
    now = datetime.now(UTC)
    game_date = active_pregame_projection_date(state_dir=_state_dir(), now=now)
    if game_date is None:
        return SkipReason("No game is within its bounded pregame projection window")
    inflight = context.instance.get_runs(
        filters=RunsFilter(
            job_name="refresh_schedule_window",
            statuses=(
                DagsterRunStatus.QUEUED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.CANCELING,
            ),
        ),
        limit=1,
    )
    if inflight:
        return SkipReason(
            f"Schedule refresh already queued or running: {inflight[0].run_id}"
        )
    interval_bucket = int(now.timestamp() // 300)
    return RunRequest(
        run_key=f"pregame:{game_date}:{interval_bucket}",
        run_config=_schedule_run_config(start_date=game_date, days_ahead=0),
        tags={
            WORKLOAD_TAG: "pregame_projection",
            "gestalt/game_date": game_date,
        },
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

    from nba_lineup_model.season.official_schedule_runtime import (
        refresh_official_live_score_state,
    )

    game_date = str(context.op_config["game_date"])
    result = refresh_official_live_score_state(
        game_date=game_date,
        state_dir=_state_dir(),
        projection_url=os.environ.get("GESTALT_SCHEDULE_API_URL", DEFAULT_SCHEDULE_API_URL),
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
            "provider_fallbacks": 0,
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

    from nba_lineup_model.season.official_schedule_runtime import (
        reconcile_official_final_artifacts_to_s3,
    )

    game_date = str(context.op_config["game_date"])
    s3_uri = _schedule_s3_uri()
    if not s3_uri:
        raise ValueError("GESTALT_SCHEDULE_S3_URI must be set for final reconciliation")
    result = reconcile_official_final_artifacts_to_s3(
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
        _scheduled_time(context).astimezone(PACIFIC).date() - timedelta(days=1)
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
    slate_date = current.astimezone(PACIFIC).date()
    state_root = Path(state_dir)
    for game_date in (
        (slate_date - timedelta(days=1)).isoformat(),
        slate_date.isoformat(),
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
            and bool(str(game.get("game_id", "")).strip())
            and (state_root / "frozen" / f"{game['game_id']}.json").is_file()
            and (
                not (
                    state_root / "final" / f"{game['game_id']}.summary.json"
                ).is_file()
                or not (
                    state_root / "final" / f"{game['game_id']}.nba.json"
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
        _scheduled_time(context).astimezone(PACIFIC).date() - timedelta(days=1)
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
    name="reconcile_current_nba_rosters",
    retry_policy=RetryPolicy(max_retries=1, delay=120),
)
def reconcile_in_season_rosters_op(context: OpExecutionContext) -> dict[str, object]:
    """Poll current rosters and append legitimate in-season player priors."""

    from nba_lineup_model.players.in_season_rosters import run
    from nba_lineup_model.season.stats_v3_lake import (
        AwsCliImmutableObjectStore,
        _join_s3_uri,
    )

    result = run(release_root=_release_root(), state_root=_in_season_roster_root())
    published: dict[str, str] = {}
    if bool(result.get("changed")):
        store = AwsCliImmutableObjectStore()
        for label, path_value in dict(result.get("artifacts", {})).items():
            path = Path(str(path_value))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            uri = _join_s3_uri(
                _in_season_roster_lake_uri(),
                "analytical",
                "in_season_rosters",
                f"season={_cdn_season()}",
                f"source_sha256={result['source_digest']}",
                f"{label}-{digest}{path.suffix}",
            )
            store.put_bytes(uri, path.read_bytes())
            published[str(label)] = uri
        manifest_path = Path(str(result["manifest_path"]))
        manifest_uri = _join_s3_uri(
            _in_season_roster_lake_uri(),
            "analytical",
            "in_season_rosters",
            "manifests",
            f"season={_cdn_season()}",
            f"{result['manifest_sha256']}.json",
        )
        store.put_bytes(manifest_uri, manifest_path.read_bytes())
        published["manifest"] = manifest_uri
    coverage = dict(result.get("coverage", {}))
    context.add_output_metadata(
        {
            "status": str(result.get("status", "")),
            "changed": bool(result.get("changed")),
            "teams": int(coverage.get("teams", 0)),
            "current_players": int(coverage.get("current_players", 0)),
            "catalog_additions": int(coverage.get("catalog_additions", 0)),
            "unresolved_players": int(coverage.get("unresolved_players", 0)),
        }
    )
    return {**result, "published": published}


@job(
    name="reconcile_in_season_rosters",
    executor_def=in_process_executor,
    tags={WORKLOAD_TAG: "in_season_rosters"},
)
def reconcile_in_season_rosters_job() -> None:
    reconcile_in_season_rosters_op()


in_season_roster_schedule = ScheduleDefinition(
    name="in_season_rosters_four_times_daily",
    job=reconcile_in_season_rosters_job,
    cron_schedule="0 6,9,12,15 * * *",
    execution_timezone="America/Los_Angeles",
    default_status=SCHEDULE_DEFAULT_STATUS,
    description=(
        "Reconcile NBA transactions and current rosters at 06:00, 09:00, "
        "12:00, and 15:00 Pacific; unchanged source membership is a no-op."
    ),
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
        _scheduled_time(context).astimezone(PACIFIC).date() - timedelta(days=1)
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


@op(name="build_daily_site_candidate", config_schema={"game_date": str, "config_path": str})
def prepare_daily_site_release_op(context: OpExecutionContext) -> dict[str, object]:
    """One coupled graph. No independent ratings/minutes publication jobs."""
    from datetime import date
    from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig, prepare_daily_release
    result = prepare_daily_release(
        DailyPipelineConfig.read(context.op_config["config_path"]),
        date.fromisoformat(context.op_config["game_date"]),
    )
    context.add_output_metadata({"game_date": result["game_date"], "status": result["status"], "runtime_release": result["runtime_release"]})
    return result


@job(name="prepare_daily_site_release", executor_def=in_process_executor, tags={WORKLOAD_TAG: "daily_site_release"})
def prepare_daily_site_release_job() -> None:
    prepare_daily_site_release_op()


@sensor(name="complete_slate_daily_release", job=prepare_daily_site_release_job, minimum_interval_seconds=300, default_status=DefaultSensorStatus.STOPPED)
def complete_slate_daily_release_sensor(context: SensorEvaluationContext) -> RunRequest | SkipReason:
    from datetime import date
    from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig, require_complete_slate
    config_path = os.environ.get("GESTALT_DAILY_PIPELINE_CONFIG", "")
    if not config_path:
        return SkipReason("No environment-specific daily pipeline configuration")
    config = DailyPipelineConfig.read(config_path)
    pointer = config.live_pointer / "site-releases" / "latest.json"
    if not pointer.is_file():
        return SkipReason("Bootstrap release must be approved before enabling nightly candidates")
    target = date.fromisoformat(json.loads(pointer.read_text())["evidence_through_date"]) + timedelta(days=1)
    try:
        require_complete_slate(config, target)
    except (OSError, ValueError) as error:
        return SkipReason(str(error))
    inflight = context.instance.get_runs(filters=RunsFilter(job_name="prepare_daily_site_release", statuses=(DagsterRunStatus.QUEUED, DagsterRunStatus.STARTING, DagsterRunStatus.STARTED, DagsterRunStatus.CANCELING)), limit=1)
    if inflight:
        return SkipReason(f"Daily pipeline already queued/running: {inflight[0].run_id}")
    tags = {"gestalt/game_date": target.isoformat()}
    previous = context.instance.get_runs(filters=RunsFilter(job_name="prepare_daily_site_release", tags=tags), limit=1)
    if previous:
        return SkipReason(f"Date already attempted; inspect report and explicitly retry or activate: {previous[0].run_id}")
    return RunRequest(run_key=f"daily-site:{target.isoformat()}", run_config={"ops": {"build_daily_site_candidate": {"config": {"game_date": target.isoformat(), "config_path": config_path}}}}, tags={**tags, WORKLOAD_TAG: "daily_site_release"})


defs = Definitions(
    assets=[s3_lake_query_health],
    jobs=[
        prepare_daily_site_release_job,
        refresh_schedule_job,
        archive_nba_market_odds_job,
        update_live_scores_job,
        reconcile_final_game_archives_job,
        ingest_completed_cdn_play_by_play_job,
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
        complete_slate_daily_release_sensor,
        pregame_projection_sensor,
        live_score_sensor,
        completed_game_cdn_play_by_play_sensor,
        completed_game_stats_v3_sensor,
    ],
)
