"""Capture official final NBA box scores for one Pacific game date."""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

from dagster import AssetExecutionContext, AssetObservation, Failure, Field, MaterializeResult, MetadataValue, asset

from ...resources import NBAScheduleResource
from .. import BOX_SCORE_POLL_NAME
from ..partitions import box_score_game_dates
from ..resources import NBABoxScoreArchive

from ...utils import PACIFIC



@asset(group_name="box_scores", partitions_def=box_score_game_dates, output_required=False,
       config_schema={"refresh_existing": Field(bool, default_value=False, is_required=False)})
def nba_box_scores(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource, nba_box_score_archive: NBABoxScoreArchive,
) -> Iterator[AssetObservation | MaterializeResult]:
    refresh = (context.op_execution_context.op_config or {}).get("refresh_existing", False)
    game_date = date.fromisoformat(context.partition_key)
    games, schedule = nba_schedule.games_for_date(game_date, allow_empty=True)
    now = datetime.now(UTC)
    today = now.astimezone(PACIFIC).date()
    overnight_poll = (
        game_date == today - timedelta(days=1)
        and context.run.tags.get("dagster/schedule_name") == BOX_SCORE_POLL_NAME
    )
    allow_pending = not refresh and (game_date == today or overnight_poll)
    eligible = [game for game in games if not allow_pending or game["game_time_utc"] <= now]
    boxes, failures = nba_box_score_archive.capture_slate(nba_schedule.season, eligible, allow_pending=allow_pending, refresh_existing=refresh)
    pending = sorted({game["game_id"] for game in games} - boxes.keys() - failures.keys())
    reused_count = sum(box["reused"] for box in boxes.values())
    warnings = []
    for gid, box in boxes.items():
        if "previous_box_score" in box:
            event = nba_box_score_archive.save_event(
                season=nba_schedule.season, day=str(game_date), run_id=context.run.run_id,
                status="warning", message="The official NBA final box score changed during refresh.",
                schedule=schedule, game_id=gid,
                details={"previous": box["previous_box_score"],
                         "current": {k:box[k] for k in ("uri", "sha256", "metadata_uri", "metadata_sha256")}},
            )
            warnings.append(event)
            context.log.warning(f"NBA corrected box score {gid}. See {event['uri']}.")
    metadata = {
        "season": nba_schedule.season,
        "game_date_pacific": game_date.isoformat(),
        "expected_game_count": len(games),
        "refresh_existing": refresh,
        "refreshed_game_count": len(boxes) if refresh else 0,
        "changed_count": len(warnings),
        "warning_count": len(warnings),
        "warnings": MetadataValue.json(warnings),
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
        if refresh:
            metadata["box_score_failure"] = MetadataValue.json(nba_box_score_archive.save_event(
                season=nba_schedule.season, day=str(game_date), run_id=context.run.run_id,
                status="failed", message="The morning NBA box-score refresh has incomplete coverage.",
                schedule=schedule, details={"failed_games": failures, "box_scores": boxes},
            ))
        raise Failure(
            f"{len(failures)} of {len(games)} final NBA box scores are unavailable for {game_date}.",
            metadata={**metadata, "failed_games": MetadataValue.json(failures)},
        )
    if pending:
        yield AssetObservation(asset_key=context.asset_key, partition=context.partition_key, metadata=metadata)
        return

    yield MaterializeResult(metadata=metadata)
