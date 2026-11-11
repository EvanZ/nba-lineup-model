"""Archive a full Pacific slate of final NBA play-by-play."""

from datetime import date

from dagster import (
    AssetExecutionContext, DagsterEventType, DagsterRunStatus, Failure,
    MaterializeResult, MetadataValue, asset,
)

from ...resources import NBAScheduleResource
from ..partitions import play_by_play_dates
from ..resources import NBAPlayByPlayArchive


def _parent_failure(context: AssetExecutionContext, archive: NBAPlayByPlayArchive, season: str, day: str):
    """Use the selected failed parent run, without scanning unrelated run history."""
    if not context.run.parent_run_id:
        return None
    parent = context.instance.get_run_by_id(context.run.parent_run_id)
    if (
        parent is None or parent.job_name != "capture_nba_play_by_play"
        or parent.status != DagsterRunStatus.FAILURE
        or parent.tags.get("dagster/partition") != day
    ):
        return None
    records = context.instance.get_records_for_run(
        parent.run_id, of_type=DagsterEventType.STEP_FAILURE,
    ).records
    for record in reversed(records):
        event = record.event_log_entry.dagster_event
        if event.step_key != "nba_play_by_play":
            continue
        failure = event.event_specific_data.user_failure_data
        if failure and "play_by_play_failure" in failure.metadata:
            return archive.read_failure(
                failure.metadata["play_by_play_failure"].value, season=season, day=day,
            )
    return None


@asset(group_name="play_by_play", partitions_def=play_by_play_dates)
def nba_play_by_play(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
    nba_play_by_play_archive: NBAPlayByPlayArchive,
) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    season = nba_schedule.season
    schedule = None
    metadata = {"season": season, "game_date_pacific": day.isoformat()}
    failures = {}
    try:
        games, schedule = nba_schedule.games_for_date(day, allow_empty=True)
        references, failures = nba_play_by_play_archive.capture_slate(
            season, games, run_id=context.run.run_id,
        )
        reused = sum(ref["reused"] for ref in references.values())
        metadata.update({
            "expected_game_count": len(games), "captured_count": len(references) - reused,
            "reused_count": reused,
            "revised_count": sum(ref["revised"] for ref in references.values()),
            "metadata_created_count": sum(not ref["metadata_reused"] for ref in references.values()),
            "metadata_reused_count": sum(ref["metadata_reused"] for ref in references.values()),
            "schedule": MetadataValue.json(schedule),
            "play_by_play": MetadataValue.json(references),
        })
        if failures:
            raise Failure(
                f"{len(failures)} of {len(games)} final NBA play-by-play files are unavailable for {day}.",
            )
    except Exception as error:
        message = str(error)
        metadata["failed_games"] = MetadataValue.json(failures)
        try:
            reference = nba_play_by_play_archive.save_error_event(
                season=season, day=day.isoformat(), run_id=context.run.run_id,
                message=message, schedule=schedule, failed_games=failures,
            )
            metadata["play_by_play_failure"] = MetadataValue.json(reference)
        except Exception as write_error:
            message += f" Could not save the S3 error record. {write_error}"
        raise Failure(message, metadata=metadata, allow_retries=False) from error

    try:
        original = _parent_failure(context, nba_play_by_play_archive, season, day.isoformat())
        if original:
            recovery = nba_play_by_play_archive.save_error_event(
                season=season, day=day.isoformat(), run_id=context.run.run_id,
                message="All selected final NBA play-by-play files were captured and verified.",
                schedule=schedule, failed_games={}, original_failure=original,
            )
            metadata["recoveries"] = MetadataValue.json([recovery])
    except Exception as error:
        raise Failure(
            f"The play-by-play is saved. Could not verify or record recovery. {error}",
            metadata=metadata, allow_retries=False,
        ) from error
    return MaterializeResult(metadata=metadata)
