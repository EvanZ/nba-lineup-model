"""Summarize saved observed stints, without running upstream producers."""
import os
from datetime import date, datetime

from dagster import (AllPartitionMapping, AssetDep, AssetExecutionContext, AssetKey,
                     DagsterEventType, DagsterRunStatus, Failure, MaterializeResult, MetadataValue, asset)

from ...resources import NBAScheduleResource
from ...utils import PACIFIC
from ..partitions import team_rotation_dates
from ..resources import TeamRotationArchive
from ..utils import PROCESSING_VERSION


def _parent_failure(context, archive, season, day):
    parent_id = context.run.parent_run_id
    if not parent_id:
        return None
    parent = context.instance.get_run_by_id(parent_id)
    if (parent is None or parent.job_name != "build_team_rotation" or parent.status != DagsterRunStatus.FAILURE
        or parent.tags.get("dagster/partition") != day):
        return None
    entries = context.instance.get_records_for_run(parent_id, of_type=DagsterEventType.STEP_FAILURE).records
    for record in reversed(entries):
        event = record.event_log_entry.dagster_event
        if event.step_key != "team_rotation":
            continue
        data = event.event_specific_data.user_failure_data
        if data and "team_rotation_failure" in data.metadata:
            return archive.read_failure(data.metadata["team_rotation_failure"].value, season=season, day=day)
    return None


@asset(group_name="team_rotation", partitions_def=team_rotation_dates, code_version=PROCESSING_VERSION,
       deps=[AssetDep(AssetKey("nba_stints"), partition_mapping=AllPartitionMapping()),
             AssetDep(AssetKey("nba_season_schedule"))])
def team_rotation(context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
                  team_rotation_archive: TeamRotationArchive) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    first = date.fromisoformat(os.environ["GESTALT_NBA_SEASON_START_DATE"])
    season = nba_schedule.season
    metadata = {"season": season, "game_date_pacific": day.isoformat(), "processing_version": PROCESSING_VERSION}
    try:
        if not first <= day < datetime.now(PACIFIC).date():
            raise ValueError("Select a completed Pacific date within the configured season.")
        original = _parent_failure(context, team_rotation_archive, season, day.isoformat())
        records, guards, schedule = team_rotation_archive.prepare(season=season, day=day, first_day=first,
                                                                 nba_schedule=nba_schedule)
        result = team_rotation_archive.publish(records, guards)
        metadata.update({k: v for k, v in result.items() if k != "selections"})
        metadata["schedule"] = MetadataValue.json(schedule)
        metadata["selections"] = MetadataValue.json(result["selections"])
        if original:
            recovery = team_rotation_archive.save_event(season=season, day=day.isoformat(), run_id=context.run_id,
                message="All 30 team rotation summaries validated and saved.", original_failure=original)
            metadata["recoveries"] = MetadataValue.json([recovery])
    except Exception as error:
        message = str(error)
        try:
            failure = team_rotation_archive.save_event(season=season, day=day.isoformat(), run_id=context.run_id,
                                                       message=message)
            metadata["team_rotation_failure"] = MetadataValue.json(failure)
        except Exception as write_error:
            message += f" Could not record failure: {write_error}"
        raise Failure(message, metadata=metadata, allow_retries=False) from error
    return MaterializeResult(metadata=metadata)
