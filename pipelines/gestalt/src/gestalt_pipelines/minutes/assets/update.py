"""One date's postgame conditional-minutes state, ordered after its predecessor."""

from datetime import date, timedelta

from dagster import (
    AssetDep, AssetExecutionContext, AssetKey, BackfillPolicy, Failure,
    MaterializeResult, MetadataValue, TimeWindowPartitionMapping, asset,
)

from ...resources import NBAScheduleResource
from ..partitions import minutes_dates
from ..resources import MinutesArchive
from ..sensors import minutes_ready_condition
from ..utils import MinutesError, implementation_identity, update_state


@asset(
    group_name="minutes", partitions_def=minutes_dates,
    backfill_policy=BackfillPolicy.multi_run(max_partitions_per_run=1),
    automation_condition=minutes_ready_condition,
    deps=[
        AssetDep(AssetKey("player_minutes"), partition_mapping=TimeWindowPartitionMapping(
            start_offset=-1, end_offset=-1, allow_nonexistent_upstream_partitions=True,
        )),
        AssetDep(AssetKey("nba_stints")),
        AssetDep(AssetKey("nba_box_scores")),
        AssetDep(AssetKey("rosters_by_date")),
        AssetDep(AssetKey("injury_status_by_game_date")),
    ],
    code_version=implementation_identity()["source_sha256"],
    description="Postgame minutes estimates and dated three-game windows for all known players.",
)
def player_minutes(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
    minutes_archive: MinutesArchive,
) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    archive = minutes_archive
    inputs = {"priors": None, "parent": None}
    metadata = {
        "game_date_pacific": str(day), "season": archive.season,
        "model_version": archive.model_version,
    }
    try:
        if archive.season != nba_schedule.season:
            raise MinutesError("Minutes and schedule resources use different seasons.")
        priors, inputs["priors"] = archive.load_priors()
        parent = None
        if day != minutes_dates.start.date():
            parent, inputs["parent"], _ = archive.state(str(day-timedelta(days=1)))
            if parent is None:
                raise MinutesError("No accepted minutes state for the immediately preceding date.")
        _, prior, etag = archive.state(str(day))
        games, inputs["schedule"] = nba_schedule.games_for_date(day, allow_empty=True)
        data, refs = archive.load_day(str(day))
        inputs.update(refs)
        record = update_state(
            day=day, first_date=minutes_dates.start.date(), season=archive.season,
            priors=priors, parent=parent, games=games, inputs=inputs,
            version=archive.model_version, **data,
        )
        _, current_schedule = nba_schedule.load()
        if current_schedule != inputs["schedule"]:
            raise MinutesError("The official schedule selection changed during this run.")
        saved = archive.save_state(record, prior=prior, expected_etag=etag)
    except Exception as error:
        message = str(error)
        metadata["inputs"] = MetadataValue.json(inputs)
        try:
            ref = archive.save_error(
                day=str(day), run_id=context.run.run_id, message=message, inputs=inputs,
                details=error.details if isinstance(error, MinutesError) else {"error_type": type(error).__name__},
            )
            metadata["minutes_failure"] = MetadataValue.json(ref)
        except Exception as write_error:
            message += f" Could not save the S3 failure record: {write_error}"
        raise Failure(message, metadata=metadata, allow_retries=False) from error
    metadata.update({
        "minutes_uri": saved["uri"], "sha256": saved["sha256"], "reused": saved["reused"],
        "rostered_player_count": record["validation"]["rostered_player_count"],
        "retained_player_count": len(record["players"]), "game_count": len(games),
        "classifications": MetadataValue.json(record["validation"]["classifications"]),
        "inputs": MetadataValue.json(inputs),
    })
    try:
        original = archive.parent_failure(context, str(day))
        if original:
            recovered = archive.save_error(
                day=str(day), run_id=context.run.run_id, message="Conditional minutes validated and published.",
                inputs=inputs, original=original,
            )
            metadata["minutes_recovery"] = MetadataValue.json(recovered)
    except Exception as error:
        raise Failure(
            f"Minutes published, but the linked recovery record failed: {error}",
            metadata=metadata, allow_retries=False,
        ) from error
    return MaterializeResult(metadata=metadata)
