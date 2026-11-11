"""Publish the latest official NBA scoreboard for its Pacific game date."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from dagster import MaterializeResult, MetadataValue, asset

from ...resources import NBAScheduleResource
from ..resources import NBAScoreboardArchive
from ..utils import parse_scoreboard, validate_games

PACIFIC = ZoneInfo("America/Los_Angeles")


@asset(group_name="scoreboard")
def nba_scoreboard(nba_schedule: NBAScheduleResource, nba_scoreboard_archive: NBAScoreboardArchive) -> MaterializeResult:
    payload = nba_scoreboard_archive.fetch()
    fetched_at = datetime.now(UTC)
    board = parse_scoreboard(payload)

    day = date.fromisoformat(board["gameDate"])
    today = fetched_at.astimezone(PACIFIC).date()
    if day not in {today, today - timedelta(days=1)}:
        raise ValueError(f"NBA scoreboard returned an unexpected game date {day}.")

    scheduled, schedule_ref = nba_schedule.games_for_date(day, allow_empty=True)
    games = validate_games(board.get("games"), scheduled)
    published = nba_scoreboard_archive.publish(nba_schedule.season, day, payload, fetched_at)
    return MaterializeResult(metadata={
        "game_date_pacific": day.isoformat(),
        "fetched_at": fetched_at.isoformat(),
        "game_count": len(games),
        "live_game_count": sum(game["gameStatus"] == 2 for game in games),
        "schedule": MetadataValue.json(schedule_ref),
        "scoreboard_uri": published["uri"],
        "sha256": published["sha256"],
    })
