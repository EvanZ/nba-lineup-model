"""Poll for final box scores during NBA game windows."""

from datetime import UTC, datetime, timedelta

from dagster import (
    AssetRecordsFilter,
    DagsterRunStatus,
    DefaultScheduleStatus,
    RunRequest,
    RunsFilter,
    ScheduleEvaluationContext,
    SkipReason,
    schedule,
)

from ...resources import NBAScheduleResource
from .. import BOX_SCORE_POLL_NAME
from ..assets import nba_box_scores
from ..jobs import capture_nba_box_scores
from ..partitions import box_score_game_dates

from ...utils import PACIFIC

POLL_INTERVAL = timedelta(minutes=5)
GAME_WINDOW = timedelta(hours=6)


def _archived_game_ids(context: ScheduleEvaluationContext, partition_key: str, schedule_ref: dict) -> set[str]:
    """Read the finals verified by the most recent materialization and observation."""
    filters = AssetRecordsFilter(asset_key=nba_box_scores.key, asset_partitions=[partition_key])
    saved = set()
    for fetch in (context.instance.fetch_materializations, context.instance.fetch_observations):
        records = fetch(filters, limit=1).records
        if records:
            metadata = records[0].asset_event.metadata
            reference = metadata.get("schedule")
            boxes = metadata.get("box_scores")
            if reference and reference.value == schedule_ref and boxes:
                saved.update(boxes.value)
    return saved


@schedule(
    name=BOX_SCORE_POLL_NAME,
    job=capture_nba_box_scores,
    cron_schedule="*/5 * * * *",
    execution_timezone=box_score_game_dates.timezone,
    default_status=DefaultScheduleStatus.STOPPED,
)
def box_score_poll(context: ScheduleEvaluationContext, nba_schedule: NBAScheduleResource):
    """Run while a game is within its capture window and needs a final box."""
    tick = context.scheduled_execution_time.astimezone(PACIFIC)
    tick_utc = tick.astimezone(UTC)
    if datetime.now(UTC) - tick_utc >= POLL_INTERVAL:
        yield SkipReason("This poll was missed. Use the manual job for older slates.")
        return

    today = tick.date()
    midnight_utc = tick.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
    days = [today]
    if tick_utc < midnight_utc + GAME_WINDOW:
        days.insert(0, today - timedelta(days=1))

    requests = []
    for day in days:
        if day < box_score_game_dates.start.date():
            continue
        games, schedule_ref = nba_schedule.games_for_date(day, allow_empty=True)
        eligible = {
            game["game_id"]
            for game in games
            if game["game_time_utc"] <= tick_utc < game["game_time_utc"] + GAME_WINDOW
        }
        partition = day.isoformat()
        if not eligible or eligible <= _archived_game_ids(context, partition, schedule_ref):
            continue

        inflight = context.instance.get_runs(
            filters=RunsFilter(
                job_name=capture_nba_box_scores.name,
                statuses=[
                    DagsterRunStatus.QUEUED, DagsterRunStatus.STARTING,
                    DagsterRunStatus.STARTED, DagsterRunStatus.CANCELING,
                ],
                tags={"dagster/partition": partition},
            ),
            limit=1,
        )
        if not inflight:
            requests.append(RunRequest(run_key=f"{tick_utc.isoformat()}:{partition}", partition_key=partition))

    if requests:
        yield from requests
    else:
        yield SkipReason("No games need a final box score in an active capture window.")


__all__ = ["box_score_poll"]
