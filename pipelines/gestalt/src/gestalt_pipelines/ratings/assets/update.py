"""One completed Pacific date's joint player-rating and home-court update."""

from datetime import date, datetime, time, timedelta

from dagster import (
    AssetDep, AssetExecutionContext, AssetKey, BackfillPolicy, Failure, DagsterRunStatus, RunsFilter, MaterializeResult,
    MetadataValue, TimeWindowPartitionMapping, asset,
)

from ...resources import NBAPlayerNamesResource, NBAScheduleResource
from ...utils import PACIFIC
from ..partitions import ratings_dates
from ..resources import RatingsArchive
from ..utils import implementation_identity, update_state


def _require_morning_inputs(context: AssetExecutionContext, day: str) -> None:
    morning = context.run.tags.get("ratings/morning_date")
    if morning is None:
        return
    midnight = datetime.combine(date.fromisoformat(morning), time.min, PACIFIC).timestamp()
    for job in ("parse_nba_games", "build_nba_rosters"):
        records = context.instance.get_run_records(
            filters=RunsFilter(job_name=job, tags={"dagster/partition": day}), limit=1,
        )
        latest = records[0] if records else None
        if (latest is None or latest.dagster_run.status != DagsterRunStatus.SUCCESS
                or latest.end_time is None or latest.end_time < midnight):
            raise ValueError(
                f"{job} has not completed successfully this morning for {day}. "
                "Repair that job, then re-execute update_player_ratings."
            )


@asset(
    group_name="ratings", partitions_def=ratings_dates,
    backfill_policy=BackfillPolicy.multi_run(max_partitions_per_run=1),
    deps=[
        AssetDep(AssetKey("player_rating_priors")),
        AssetDep(AssetKey("nba_stints")),
        AssetDep(AssetKey("rosters_by_date")),
        AssetDep(AssetKey("player_ratings"), partition_mapping=TimeWindowPartitionMapping(
            start_offset=-1, end_offset=-1, allow_nonexistent_upstream_partitions=True,
        )),
    ],
    code_version=implementation_identity()["source_sha256"],
    description="Ratings/HCA incorporating all completed games through this Pacific date.",
)
def player_ratings(context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
                   ratings_archive: RatingsArchive, nba_player_names: NBAPlayerNamesResource) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    archive = ratings_archive
    observed_day = day
    inputs = {"priors": None, "parent": None, "schedule": None, "stints": None, "rosters": None}
    metadata = {"season": archive.season, "model_version": archive.model_version,
                "game_date_pacific": str(day), "observed_through_game_date_pacific": str(observed_day),
                "initialization_only": False}
    try:
        _require_morning_inputs(context, str(day))
        if archive.season != nba_schedule.season:
            raise ValueError("Ratings and official schedule resources use different seasons.")
        priors, inputs["priors"] = archive.load_priors()
        parent = None
        if day != ratings_dates.start.date():
            parent, inputs["parent"], _ = archive.state(str(day-timedelta(days=1)))
            if parent is None:
                raise FileNotFoundError("No accepted ratings state for the immediately preceding date.")
        prior, etag = archive.state_selection(str(day))
        selected, inputs["schedule"] = nba_schedule.games_between(
            day-timedelta(days=1), day, allow_empty=True,
        )
        games = [g for g in selected if g["game_time_utc"].astimezone(PACIFIC).date() == day]
        previous = [g for g in selected if g["game_time_utc"].astimezone(PACIFIC).date() == day-timedelta(days=1)]
        data, references = archive.day_inputs(str(day)); inputs.update(references)
        record = update_state(
            day=day, first_date=ratings_dates.start.date(), season=archive.season,
            priors=priors, parent=parent, games=games, previous_games=previous, inputs=inputs, **data,
        )
        new_ids = set(record["new_replacement_player_ids"])
        new_players = [player for player in record["players"] if player["player_id"] in new_ids]
        if new_players:
            if nba_player_names.season != archive.season:
                raise ValueError("Ratings and player-name resources use different seasons.")
            names, inputs["player_names"] = nba_player_names.load()
            for player in new_players:
                player["player_name"] = names.get(player["player_id"])
        archive.verify_selections(str(day), inputs)
        if inputs["schedule"] is not None:
            _, current_schedule = nba_schedule.load()
            if current_schedule != inputs["schedule"]:
                raise ValueError("The official schedule selection changed during this ratings run.")
        warnings = []
        for player in new_players:
            label = player.get("player_name") or f"NBA player {player['player_id']}"
            context.log.warning(
                f"{label} (NBA ID {player['player_id']}) assigned replacement starting NAIL "
                f"{player['starting_nail']}; review this new player."
            )
            warnings.append(archive.save_replacement_warning(
                day=str(day), run_id=context.run.run_id, player=player, inputs=inputs,
            ))
        saved = archive.save_state(record, prior=prior, expected_etag=etag)
    except Exception as error:
        message = str(error)
        metadata["inputs"] = MetadataValue.json(inputs)
        try:
            failure = archive.save_event(run_id=context.run.run_id, message=message, inputs=inputs,
                                         stage="player_ratings", day=str(day))
            metadata["ratings_failure"] = MetadataValue.json(failure)
        except Exception as write_error:
            message += f" Could not save the S3 failure record: {write_error}"
        raise Failure(message, metadata=metadata, allow_retries=False) from error
    metadata.update({
        "ratings_uri": saved["uri"], "sha256": saved["sha256"], "reused": saved["reused"],
        "game_count": len(games), "included_game_count": len(record["included_game_ids"]),
        "rostered_player_count": record["validation"]["rostered_player_count"],
        "retained_player_count": len(record["players"]), "hca": record["hca"]["estimate"],
        "inputs": MetadataValue.json(inputs),
        "new_replacement_players": MetadataValue.json([
            {key: player[key] for key in ("player_id", "player_name", "starting_nail")} for player in new_players
        ]),
        "ratings_warnings": MetadataValue.json(warnings),
    })
    try:
        original = archive.parent_failure(context, stage="player_ratings", day=str(day))
        if original:
            recovery = archive.save_event(
                run_id=context.run.run_id, message="Joint ratings/HCA validated and published.",
                inputs=inputs, original=original, stage="player_ratings", day=str(day),
            )
            metadata["ratings_recovery"] = MetadataValue.json(recovery)
    except Exception as error:
        raise Failure(f"Ratings are saved, but linked recovery logging failed: {error}", metadata=metadata) from error
    return MaterializeResult(metadata=metadata)
