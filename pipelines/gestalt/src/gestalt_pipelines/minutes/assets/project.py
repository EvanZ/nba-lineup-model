"""Game-specific player allocations, independent of the daily learning job."""

from datetime import UTC, date, datetime, timedelta

from dagster import (
    AllPartitionMapping, AssetDep, AssetExecutionContext, AssetKey, Failure,
    BackfillPolicy, TimeWindowPartitionMapping, MaterializeResult, MetadataValue, asset,
)

from ...resources import NBAScheduleResource
from ...rosters.resources import NBARosterArchive
from ...utils import PACIFIC
from ..partitions import game_minutes_dates
from ..resources import MinutesArchive
from ..utils import MinutesError, build_game_projection, implementation_identity, validate_game_projection


@asset(
    group_name="minutes", partitions_def=game_minutes_dates,
    backfill_policy=BackfillPolicy.multi_run(max_partitions_per_run=1),
    deps=[
        AssetDep(AssetKey("player_minutes"),partition_mapping=TimeWindowPartitionMapping(
            start_offset=-2,end_offset=-1,allow_nonexistent_upstream_partitions=True,
        )),
        AssetDep(AssetKey("rosters_by_date"),partition_mapping=AllPartitionMapping()),
        AssetDep(AssetKey("injury_status_by_game_date")),
        AssetDep(AssetKey("nba_season_schedule")),
    ],
    code_version=implementation_identity()["source_sha256"],
    description="Project upcoming games or replay historical dates; preserve frozen pregame selections.",
)
def game_minutes(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
    minutes_archive: MinutesArchive, nba_roster_archive: NBARosterArchive,
) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    now = datetime.now(UTC)
    today = now.astimezone(PACIFIC).date()
    archive = minutes_archive
    inputs = {}
    current_gid = None
    metadata = {"game_date_pacific":str(day),"model_version":archive.projection_version}
    try:
        if day > today+timedelta(days=1):
            raise MinutesError("Projection date lies beyond the injury-report window.")
        historical = day < today
        if archive.season != nba_schedule.season or archive.bucket != nba_roster_archive.bucket:
            raise MinutesError("Projection resources use different seasons or data buckets.")
        games, schedule = nba_schedule.games_for_date(day,allow_empty=True)
        upcoming, frozen, missing_started = [], [], []
        for g in games:
            selected,_ = archive.game_selection(str(day),g["game_id"])
            if historical:
                if selected is not None:
                    accepted = archive.read_reference(selected,archive.game_prefix)
                    validate_game_projection(accepted,archive.season,archive.projection_version)
                    if accepted["purpose"] == "pregame_projection":
                        frozen.append(g["game_id"])
                        continue
                upcoming.append(g)
            elif g["game_time_utc"] > now:
                upcoming.append(g)
            else:
                (frozen if selected is not None else missing_started).append(g["game_id"])
        if missing_started:
            context.log.warning("Started games have no saved pregame projection: %s",missing_started)
        saved = {}
        if upcoming:
            inputs = archive.projection_selections(str(day),today=today,roster_archive=nba_roster_archive)
            inputs["schedule"] = schedule
            data = archive.projection_data(str(day),inputs)
            records = []
            for game in upcoming:
                current_gid = game["game_id"]
                _,etag = archive.game_selection(str(day),current_gid)
                record = build_game_projection(
                    game=game,day=str(day),season=archive.season,inputs=inputs,
                    version=archive.projection_version,**data,
                )
                records.append((record,etag))
            for record,etag in records:
                current_gid = record["game_id"]
                _,current_schedule = nba_schedule.load()
                if current_schedule != schedule:
                    raise MinutesError("The schedule selection changed during projection.")
                result = archive.save_game(record,expected_etag=etag,roster_archive=nba_roster_archive)
                if result["frozen"]:
                    frozen.append(current_gid)
                else:
                    saved[current_gid] = result
        metadata.update({
            "game_count":len(games),"projected_game_count":len(saved),
            "projection_mode":"historical_replay" if historical else "pregame_projection",
            "projections":MetadataValue.json(saved),
            "frozen_game_ids":MetadataValue.json(frozen),
            "unprojected_started_game_ids":MetadataValue.json(missing_started),"inputs":MetadataValue.json(inputs),
        })
    except Exception as error:
        message = str(error)
        try:
            ref = archive.save_error(
                day=str(day),run_id=context.run.run_id,message=message,inputs=inputs,
                details=error.details if isinstance(error,MinutesError) else {"error_type":type(error).__name__},
                stage="game_minutes",game_id=current_gid,
            )
            metadata["minutes_failure"] = MetadataValue.json(ref)
        except Exception as write_error:
            message += f" Could not save the S3 failure record: {write_error}"
        raise Failure(message,metadata=metadata,allow_retries=False) from error
    try:
        original = archive.parent_failure(context,str(day),stage="game_minutes")
        if original:
            ref = archive.save_error(
                day=str(day),run_id=context.run.run_id,
                message="Game projections validated; started games remain frozen.",
                inputs=inputs,original=original,stage="game_minutes",
            )
            metadata["minutes_recovery"] = MetadataValue.json(ref)
    except Exception as error:
        raise Failure(f"Projection recovery record failed: {error}",metadata=metadata,allow_retries=False) from error
    return MaterializeResult(metadata=metadata)
