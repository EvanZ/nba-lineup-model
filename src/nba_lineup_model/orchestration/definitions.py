"""Resource-bounded Dagster jobs for the production NBA GESTALT instance."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dagster import (
    AssetExecutionContext,
    DefaultScheduleStatus,
    DefaultSensorStatus,
    Definitions,
    Field,
    MetadataValue,
    OpExecutionContext,
    RetryPolicy,
    RunRequest,
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

from nba_lineup_model.storage.duckdb_lake import (
    DEFAULT_HEALTHCHECK_PARQUET,
    s3_parquet_row_count,
)
from nba_lineup_model.web_api.win_projections import (
    DEFAULT_SCHEDULE_API_URL,
    DEFAULT_SCHEDULE_DAYS_AHEAD,
    DEFAULT_SCHEDULE_S3_URI,
    DEFAULT_SCHEDULE_STATE_DIR,
    refresh_live_score_state,
    refresh_schedule_window,
)

EASTERN = ZoneInfo("America/New_York")
RETRY_POLICY = RetryPolicy(max_retries=3, delay=30)
WORKLOAD_TAG = "gestalt/workload"
AUTOMATIONS_DEFAULT_RUNNING = os.environ.get(
    "GESTALT_AUTOMATIONS_DEFAULT_RUNNING", "0"
) == "1"
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


def _state_dir() -> Path:
    return Path(os.environ.get("GESTALT_SCHEDULE_STATE_DIR", DEFAULT_SCHEDULE_STATE_DIR))


def _schedule_s3_uri() -> str | None:
    value = os.environ.get("GESTALT_SCHEDULE_S3_URI", DEFAULT_SCHEDULE_S3_URI).strip()
    return value or None


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

    payload = refresh_schedule_window(
        start_date=str(context.op_config["start_date"]),
        days_ahead=int(context.op_config["days_ahead"]),
        state_dir=_state_dir(),
        projection_url=os.environ.get("GESTALT_SCHEDULE_API_URL", DEFAULT_SCHEDULE_API_URL),
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
    return _schedule_run_config(start_date=game_date, days_ahead=DEFAULT_SCHEDULE_DAYS_AHEAD)


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
    name="update_live_score_state",
    config_schema={"game_date": str},
    retry_policy=RETRY_POLICY,
)
def update_live_scores_op(context: OpExecutionContext) -> dict[str, object]:
    """Perform one short score/clock update; the sensor controls when it runs."""

    game_date = str(context.op_config["game_date"])
    result = refresh_live_score_state(game_date=game_date, state_dir=_state_dir())
    context.add_output_metadata(
        {
            "game_date": game_date,
            "games": int(result["game_count"]),
            "any_live": bool(result["any_live"]),
            "all_final": bool(result["all_final"]),
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


def active_live_game_date(
    *,
    state_dir: Path | str,
    now: datetime | None = None,
) -> str | None:
    """Return the active slate only inside its bounded T-30 through T+6h window."""

    current = (now or datetime.now(UTC)).astimezone(UTC)
    eastern_date = current.astimezone(EASTERN).date()
    for game_date in (
        (eastern_date - timedelta(days=1)).isoformat(),
        eastern_date.isoformat(),
    ):
        path = Path(state_dir) / "latest" / f"{game_date}.json"
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
        if not states or not starts or all(state == "post" for state in states):
            continue
        if min(starts) - timedelta(minutes=30) <= current <= max(starts) + timedelta(hours=6):
            return game_date
    return None


@sensor(
    name="live_score_game_window",
    job=update_live_scores_job,
    minimum_interval_seconds=75,
    default_status=SENSOR_DEFAULT_STATUS,
    description="Emit short score-update runs only while an unfinished slate is active.",
)
def live_score_sensor(context: SensorEvaluationContext) -> RunRequest | SkipReason:
    now = datetime.now(UTC)
    game_date = active_live_game_date(state_dir=_state_dir(), now=now)
    if game_date is None:
        return SkipReason("No unfinished slate is inside its live-game window")
    interval_bucket = int(now.timestamp() // 75)
    return RunRequest(
        run_key=f"{game_date}:{interval_bucket}",
        run_config={
            "ops": {"update_live_score_state": {"config": {"game_date": game_date}}}
        },
        tags={WORKLOAD_TAG: "live_scores", "gestalt/game_date": game_date},
    )


@asset(
    name="s3_lake_query_health",
    group_name="data_lake",
    retry_policy=RETRY_POLICY,
    description="Read a small immutable Parquet object directly from S3 with DuckDB.",
)
def s3_lake_query_health(context: AssetExecutionContext) -> int:
    uri = os.environ.get("GESTALT_LAKE_HEALTHCHECK_URI", DEFAULT_HEALTHCHECK_PARQUET)
    count = s3_parquet_row_count(uri)
    context.add_output_metadata(
        {"rows": count, "s3_uri": MetadataValue.url(uri)}
    )
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
    jobs=[refresh_schedule_job, update_live_scores_job, daily_lake_health_job],
    schedules=[
        schedule_refresh_schedule,
        yesterday_finalization_schedule,
        daily_lake_health_schedule,
    ],
    sensors=[live_score_sensor],
)
