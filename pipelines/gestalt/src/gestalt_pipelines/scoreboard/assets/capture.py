"""Publish the latest official NBA scoreboard for its Pacific game date."""

from datetime import UTC, date, datetime, timedelta

from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset

from ...resources import NBAScheduleResource
from ..resources import NBAScoreboardArchive
from ..partitions import scoreboard_dates
from ..utils import parse_scoreboard, validate_games

from ...utils import PACIFIC



@asset(group_name="scoreboard", partitions_def=scoreboard_dates)
def nba_scoreboard(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
    nba_scoreboard_archive: NBAScoreboardArchive,
) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    now = datetime.now(UTC)
    today = now.astimezone(PACIFIC).date()
    if day > today:
        raise ValueError("Cannot capture a future Pacific scoreboard slate.")
    scheduled, schedule_ref = nba_schedule.games_for_date(day, allow_empty=True)
    # A game that continues after midnight still belongs to its tipoff date.
    live_window = day == today or (
        day == today-timedelta(days=1)
        and any(game["game_time_utc"] <= now < game["game_time_utc"]+timedelta(hours=6) for game in scheduled)
    )
    if not live_window:
        saved = nba_scoreboard_archive.load(nba_schedule.season, day)
        if saved is None:
            raise FileNotFoundError(f"No saved NBA scoreboard for {day}; today's endpoint cannot backfill an old slate.")
        board, reference = saved
        games = validate_games(board.get("games"), scheduled)
        return MaterializeResult(metadata={
            "game_date_pacific": day.isoformat(), "archive_only": True,
            "game_count": len(games), "schedule": MetadataValue.json(schedule_ref),
            "scoreboard_uri": reference["uri"], "sha256": reference["sha256"],
            "fetched_at": reference["fetched_at"],
        })
    payload = nba_scoreboard_archive.fetch()
    fetched_at = datetime.now(UTC)
    board = parse_scoreboard(payload)
    if board["gameDate"] != day.isoformat():
        raise ValueError(f"NBA scoreboard returned {board['gameDate']}, not the requested slate {day}.")

    games = validate_games(board.get("games"), scheduled)
    published = nba_scoreboard_archive.publish(nba_schedule.season, day, payload, fetched_at)
    return MaterializeResult(metadata={
        "game_date_pacific": day.isoformat(), "archive_only": False,
        "fetched_at": fetched_at.isoformat(),
        "game_count": len(games),
        "live_game_count": sum(game["gameStatus"] == 2 for game in games),
        "schedule": MetadataValue.json(schedule_ref),
        "scoreboard_uri": published["uri"],
        "sha256": published["sha256"],
    })
