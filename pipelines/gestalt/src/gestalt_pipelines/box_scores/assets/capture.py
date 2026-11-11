"""Capture official final NBA box scores for one Pacific game date."""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from dagster import AssetExecutionContext, AssetObservation, Failure, MaterializeResult, MetadataValue, asset

from ...resources import NBAScheduleResource
from .. import BOX_SCORE_POLL_NAME
from ..partitions import box_score_game_dates
from ..resources import NBABoxScoreArchive

PACIFIC = ZoneInfo(box_score_game_dates.timezone)


@asset(group_name="box_scores", partitions_def=box_score_game_dates, output_required=False)
def nba_box_scores(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource, nba_box_score_archive: NBABoxScoreArchive,
) -> Iterator[AssetObservation | MaterializeResult]:
    game_date = date.fromisoformat(context.partition_key)
    games, schedule = nba_schedule.games_for_date(game_date, allow_empty=True)
    now = datetime.now(UTC)
    today = now.astimezone(PACIFIC).date()
    overnight_poll = (
        game_date == today - timedelta(days=1)
        and context.run.tags.get("dagster/schedule_name") == BOX_SCORE_POLL_NAME
    )
    allow_pending = game_date == today or overnight_poll
    eligible = [game for game in games if not allow_pending or game["game_time_utc"] <= now]
    boxes, failures = nba_box_score_archive.capture_slate(nba_schedule.season, eligible, allow_pending=allow_pending)
    pending = sorted({game["game_id"] for game in games} - boxes.keys() - failures.keys())
    reused_count = sum(box["reused"] for box in boxes.values())
    metadata = {
        "season": nba_schedule.season,
        "game_date_pacific": game_date.isoformat(),
        "expected_game_count": len(games),
        "reused_count": reused_count,
        "captured_count": len(boxes) - reused_count,
        "metadata_created_count": sum(not box["metadata_reused"] for box in boxes.values()),
        "metadata_reused_count": sum(box["metadata_reused"] for box in boxes.values()),
        "pending_game_ids": MetadataValue.json(pending),
        "pending_count": len(pending),
        "schedule": MetadataValue.json(schedule),
        "box_scores": MetadataValue.json(boxes),
    }
    if failures:
        raise Failure(
            f"{len(failures)} of {len(games)} final NBA box scores are unavailable for {game_date}.",
            metadata={**metadata, "failed_games": MetadataValue.json(failures)},
        )
    if pending:
        yield AssetObservation(asset_key=context.asset_key, partition=context.partition_key, metadata=metadata)
        return

    yield MaterializeResult(metadata=metadata)
