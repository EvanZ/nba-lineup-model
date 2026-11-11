"""Capture current NBA scores and clocks during game windows."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from dagster import DefaultScheduleStatus, DagsterRunStatus, RunRequest, RunsFilter, ScheduleEvaluationContext, SkipReason, schedule

from ...resources import NBAScheduleResource
from ..jobs import capture_nba_scoreboard
from ..resources import NBAScoreboardArchive
from ..utils import validate_games

PACIFIC = ZoneInfo("America/Los_Angeles")
GAME_WINDOW = timedelta(hours=6)


@schedule(
    job=capture_nba_scoreboard,
    cron_schedule="* * * * *",
    execution_timezone="America/Los_Angeles",
    default_status=DefaultScheduleStatus.STOPPED,
)
def scoreboard_poll(
    context: ScheduleEvaluationContext, nba_schedule: NBAScheduleResource, nba_scoreboard_archive: NBAScoreboardArchive,
):
    tick = context.scheduled_execution_time.astimezone(PACIFIC)
    tick_utc = tick.astimezone(UTC)
    if datetime.now(UTC) - tick_utc >= timedelta(minutes=1):
        return SkipReason("This scoreboard poll was missed. The next poll fetches current scores.")

    inflight = context.instance.get_runs(
        filters=RunsFilter(
            job_name=capture_nba_scoreboard.name,
            statuses=[
                DagsterRunStatus.QUEUED, DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED, DagsterRunStatus.CANCELING,
            ],
        ),
        limit=1,
    )
    if inflight:
        return SkipReason("A scoreboard capture is already queued or running.")

    days = [tick.date()]
    midnight = tick.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
    if tick_utc < midnight + GAME_WINDOW:
        days.insert(0, tick.date() - timedelta(days=1))

    for day in days:
        scheduled, _ = nba_schedule.games_for_date(day, allow_empty=True)
        eligible = {
            game["game_id"] for game in scheduled
            if game["game_time_utc"] <= tick_utc < game["game_time_utc"] + GAME_WINDOW
        }
        if not eligible:
            continue
        board = nba_scoreboard_archive.read_latest(nba_schedule.season, day)
        if board is None:
            return RunRequest(run_key=tick_utc.isoformat())
        games = validate_games(board.get("games"), scheduled)
        if any(game["gameId"] in eligible and game["gameStatus"] != 3 for game in games):
            return RunRequest(run_key=tick_utc.isoformat())

    return SkipReason("No unfinished games are inside the scoreboard capture window.")


__all__ = ["scoreboard_poll"]
