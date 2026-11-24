"""One forecast-date slate from published ratings and game allocations."""
from datetime import UTC, date, datetime, timedelta

from dagster import (
    AssetDep, AssetExecutionContext, AssetKey, BackfillPolicy, Failure, Field,
    MaterializeResult, MetadataValue, TimeWindowPartitionMapping, asset,
)

from ...minutes.resources import MinutesArchive
from ...minutes.utils import validate_game_projection
from ...ratings.resources import RatingsArchive
from ...resources import NBAScheduleResource
from ...utils import PACIFIC, forecast_observed_date
from ..partitions import game_prediction_dates
from ..resources import GamePredictionArchive
from ..utils import PredictionError, build_prediction, implementation_identity, prediction_fingerprint


def _reference(pointer: dict) -> dict:
    return {k: pointer[k] for k in ("uri", "sha256")}


def _report_summary(archive: MinutesArchive, allocation: dict) -> dict | None:
    """Follow the allocation's exact injury/extraction references, never latest.json."""
    injuries = archive.read_reference(allocation["inputs"]["injuries"],
        f"lake/artifacts/injury_status_by_game_date/season={archive.season}/game_date_pacific={allocation['game_date_pacific']}/")
    gid = allocation["game_id"]
    if not injuries["games"][gid]["report_used"]:
        return None
    extraction = injuries["inputs"].get("extractions", {}).get(gid) or injuries["inputs"].get("extraction")
    if extraction is None:
        raise PredictionError("The selected game allocation has no recorded injury extraction.")
    report = archive.read_reference(extraction, f"lake/artifacts/injury_report_extraction/season={archive.season}/")
    if datetime.fromisoformat(report["published_at"]) >= datetime.fromisoformat(allocation["scheduled_start_utc"]):
        raise PredictionError("The selected injury report was published after scheduled tipoff.")
    return {k: report[k] for k in ("published_at", "source_url", "pdf_uri")}


@asset(
    group_name="predictions", partitions_def=game_prediction_dates,
    config_schema={"allow_historical_correction": Field(bool, default_value=False, is_required=False)},
    backfill_policy=BackfillPolicy.multi_run(max_partitions_per_run=1),
    deps=[
        AssetDep(AssetKey("game_minutes")),
        AssetDep(AssetKey("player_ratings"), partition_mapping=TimeWindowPartitionMapping(
            start_offset=-2, end_offset=-1, allow_nonexistent_upstream_partitions=True)),
        AssetDep(AssetKey("nba_season_schedule")),
    ],
    code_version=implementation_identity()["source_sha256"],
    description="Predict a Pacific slate from accepted game minutes and ratings, at 100 possessions.",
)
def game_predictions(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
    minutes_archive: MinutesArchive, ratings_archive: RatingsArchive,
    game_prediction_archive: GamePredictionArchive,
) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    now = datetime.now(UTC)
    today = now.astimezone(PACIFIC).date()
    historical = day < today
    allow_correction = (context.op_config or {}).get("allow_historical_correction", False)
    archive = game_prediction_archive
    inputs, outputs, frozen, missing_started, warnings = {}, {}, [], [], []
    current_gid = None
    metadata = {"game_date_pacific": str(day), "model_version": archive.model_version}
    try:
        if allow_correction and not historical:
            raise PredictionError("Historical correction requires a past Pacific date partition.")
        if day > today+timedelta(days=1):
            raise PredictionError("Forecast date lies beyond tomorrow's report window.")
        if (len({archive.season, minutes_archive.season, ratings_archive.season, nba_schedule.season}) != 1
                or len({archive.bucket, minutes_archive.bucket, ratings_archive.bucket, nba_schedule.bucket}) != 1):
            raise PredictionError("Prediction resources use different seasons or buckets.")
        schedule_games, schedule = nba_schedule.games_between(day-timedelta(days=1), day, allow_empty=True)
        games = [g for g in schedule_games if g["game_time_utc"].astimezone(PACIFIC).date() == day]
        yesterday_teams = {g[side+"_team_id"] for g in schedule_games
                           if g["game_time_utc"].astimezone(PACIFIC).date() == day-timedelta(days=1)
                           for side in ("home", "away")}
        pending = []
        for game in games:
            current_gid = game["game_id"]
            selected, etag = archive.selection(str(day), current_gid)
            accepted = archive.load(selected) if selected else None
            if accepted and accepted["purpose"] == "pregame_projection" and not allow_correction and (historical or game["game_time_utc"] <= now):
                frozen.append(current_gid)
            elif not historical and game["game_time_utc"] <= now:
                missing_started.append(current_gid)
            else:
                pending.append((game, selected, etag, accepted))
        if missing_started:
            context.log.warning("Started games have no saved pregame prediction: %s", missing_started)
        if pending:
            ratings_day = str(forecast_observed_date(day, today))
            bootstrap = date.fromisoformat(ratings_day) < game_prediction_dates.start.date()
            if bootstrap:
                ratings, ratings_ref = ratings_archive.load_priors()
            else:
                ratings, ratings_ref, _ = ratings_archive.state(ratings_day)
            if ratings is None:
                raise PredictionError(f"No accepted player_ratings for {ratings_day}.")
            inputs = {"player_ratings": ratings_ref, "schedule": schedule}
            daily_ids = {p["player_id"] for p in ratings["players"]}
            priors, prior_ref = (ratings, ratings_ref) if bootstrap else (None, None)
            records = []
            for game, selected, etag, accepted in pending:
                current_gid = game["game_id"]
                minutes_pointer, _ = minutes_archive.game_selection(str(day), current_gid)
                if minutes_pointer is None:
                    raise PredictionError(f"No accepted game_minutes for {current_gid} on {day}.")
                allocation = minutes_archive.read_reference(minutes_pointer, minutes_archive.game_prefix)
                validate_game_projection(allocation, archive.season, minutes_archive.projection_version)
                missing_ids = {p["player_id"] for t in allocation["teams"].values() for p in t["players"]}-daily_ids
                if missing_ids and priors is None:
                    priors, prior_ref = ratings_archive.load_priors()
                    if prior_ref != ratings["inputs"]["priors"]:
                        raise PredictionError("Fallback priors differ from the selected ratings initialization.")
                game_inputs = {"player_ratings": ratings_ref, "game_minutes": _reference(minutes_pointer), "schedule": schedule}
                if missing_ids and not bootstrap:
                    game_inputs["rating_priors"] = prior_ref
                inputs = game_inputs
                fingerprint = prediction_fingerprint(game, game_inputs, version=archive.model_version)
                if accepted and accepted["input_fingerprint"] == fingerprint:
                    record = accepted
                else:
                    record = build_prediction(
                        game=game, day=str(day), first_date=str(game_prediction_dates.start.date()), season=archive.season, ratings=ratings, minutes=allocation,
                        priors=priors, report=_report_summary(minutes_archive, allocation),
                        b2b={side: game[side+"_team_id"] in yesterday_teams for side in ("home", "away")},
                        inputs=game_inputs, generated_at=now, version=archive.model_version,
                        purpose="historical_replay" if historical else "pregame_projection",
                    )
                existing_replacements = {
                    player["player_id"]
                    for side in ("home", "away")
                    for player in accepted[side]["players"]
                    if player["rating_source"] == "replacement"
                } if accepted else set()
                records.append((record, etag, existing_replacements))
            # Validate the whole slate before publishing any new selections.
            for record, etag, existing_replacements in records:
                current_gid = record["game_id"]
                inputs = record["inputs"]
                current_ratings = ratings_archive.selection()[0] if bootstrap else ratings_archive.state_selection(ratings_day)[0]
                if current_ratings != ratings_ref or nba_schedule.load()[1] != schedule:
                    raise PredictionError("Accepted ratings or schedule changed during prediction.")
                current_minutes, _ = minutes_archive.game_selection(str(day), current_gid)
                if current_minutes is None or _reference(current_minutes) != inputs["game_minutes"]:
                    raise PredictionError("Accepted game minutes changed during prediction.")
                if "rating_priors" in inputs and ratings_archive.selection()[0] != inputs["rating_priors"]:
                    raise PredictionError("Fallback initialization changed during prediction.")
                for side in ("home", "away"):
                    for player in record[side]["players"]:
                        if (player["rating_source"] == "replacement"
                                and player["player_id"] not in existing_replacements):
                            message = f"Player {player['player_id']} ({player['name']}) has no saved rating; using the approved replacement rating."
                            warning = archive.save_event(day=str(day), run_id=context.run.run_id,
                                message=message, inputs=inputs, status="warning", game_id=current_gid,
                                details={"player_id": player["player_id"], "name": player["name"]})
                            context.log.warning(message)
                            warnings.append(warning)
                result = archive.save(record, expected_etag=etag, allow_historical_correction=allow_correction)
                if result["frozen"]:
                    frozen.append(current_gid)
                else:
                    outputs[current_gid] = result
        metadata.update({
            "game_count": len(games), "predicted_game_count": len(outputs),
            "projection_mode": "historical_replay" if historical else "pregame_projection",
            "allow_historical_correction": allow_correction,
            "predictions": MetadataValue.json(outputs), "frozen_game_ids": MetadataValue.json(frozen),
            "unprojected_started_game_ids": MetadataValue.json(missing_started),
            "warnings": MetadataValue.json(warnings),
        })
    except Exception as error:
        message = str(error)
        try:
            ref = archive.save_event(day=str(day), run_id=context.run.run_id, message=message,
                                     inputs=inputs, game_id=current_gid,
                                     details={"error_type": type(error).__name__})
            metadata["prediction_failure"] = MetadataValue.json(ref)
        except Exception as write_error:
            message += f" Could not save the S3 failure record: {write_error}"
        raise Failure(message, metadata=metadata, allow_retries=False) from error
    try:
        original = archive.parent_failure(context, str(day))
        if original:
            recovery = archive.save_event(day=str(day), run_id=context.run.run_id,
                message="Game predictions validated; saved pregame selections preserved.",
                inputs=inputs, status="recovered", original=original)
            metadata["prediction_recovery"] = MetadataValue.json(recovery)
    except Exception as error:
        raise Failure(f"Prediction recovery record failed: {error}", metadata=metadata, allow_retries=False) from error
    return MaterializeResult(metadata=metadata)
