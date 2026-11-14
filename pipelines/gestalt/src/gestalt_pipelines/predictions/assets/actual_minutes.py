"""Manual postgame counterfactuals from saved predictions and final NBA boxes."""
import json
from datetime import UTC, date, datetime

from dagster import AssetDep, AssetExecutionContext, AssetKey, BackfillPolicy, Failure, MaterializeResult, MetadataValue, asset

from ...box_scores.resources import NBABoxScoreArchive
from ...ratings.resources import RatingsArchive
from ...resources import NBAScheduleResource
from ...utils import PACIFIC
from ..partitions import actual_minutes_prediction_dates
from ..resources import GamePredictionArchive
from ..utils import (
    PredictionError, actual_minutes_fingerprint, build_actual_minutes_prediction,
    extra_actual_player_ids, implementation_identity,
)

STAGE = "game_predictions_actual_minutes"


def _ref(pointer):
    return {k: pointer[k] for k in ("uri", "sha256")}


@asset(
    group_name="predictions", partitions_def=actual_minutes_prediction_dates,
    backfill_policy=BackfillPolicy.multi_run(max_partitions_per_run=1),
    deps=[AssetDep(AssetKey("game_predictions")), AssetDep(AssetKey("nba_box_scores")), AssetDep(AssetKey("nba_season_schedule"))],
    code_version=implementation_identity()["source_sha256"],
    description="Replace saved projected minutes with final NBA minutes, preserving that prediction's ratings and context.",
)
def game_predictions_actual_minutes(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
    nba_box_score_archive: NBABoxScoreArchive, ratings_archive: RatingsArchive,
    game_prediction_archive: GamePredictionArchive,
) -> MaterializeResult:
    day = context.partition_key
    now = datetime.now(UTC)
    archive = game_prediction_archive
    inputs, outputs, warnings = {}, {}, []
    current_gid = None
    metadata = {"game_date_pacific": day, "model_version": archive.model_version}
    try:
        if date.fromisoformat(day) >= now.astimezone(PACIFIC).date():
            raise PredictionError("Select a completed Pacific date for the daily postgame counterfactual.")
        if (len({archive.season, nba_schedule.season, ratings_archive.season}) != 1
                or len({archive.bucket, nba_schedule.bucket, ratings_archive.bucket, nba_box_score_archive.bucket}) != 1):
            raise PredictionError("Counterfactual resources use different seasons or buckets.")
        games, schedule = nba_schedule.games_for_date(date.fromisoformat(day), allow_empty=True)
        records = []
        for game in games:
            current_gid = game["game_id"]
            pointer, _ = archive.selection(day, current_gid)
            if pointer is None:
                raise PredictionError(f"No saved game prediction for {current_gid} on {day}.")
            prediction = archive.load(pointer)
            if (datetime.fromisoformat(prediction["scheduled_start_utc"]) != game["game_time_utc"]
                    or any(prediction[side]["team_id"] != game[side+"_team_id"] for side in ("home", "away"))):
                raise PredictionError("Saved prediction and official scheduled matchup differ.")
            raw_box, box_ref = nba_box_score_archive.read_saved(archive.season, game)
            box = json.loads(raw_box)["game"]
            inputs = {"prediction": _ref(pointer), "box_score": box_ref, "schedule": schedule}
            ratings, priors = None, None
            extra = extra_actual_player_ids(prediction, box)
            if extra:
                inputs["player_ratings"] = prediction["inputs"]["player_ratings"]
                ratings = ratings_archive.read_state(inputs["player_ratings"], prediction["ratings_effective_date_pacific"])
                if extra-{p["player_id"] for p in ratings["players"]}:
                    inputs["rating_priors"] = ratings["inputs"]["priors"]
                    priors = ratings_archive.read_priors(inputs["rating_priors"])
            accepted_pointer, etag = archive.selection(day, current_gid, actual_minutes=True)
            accepted = archive.load(accepted_pointer, actual_minutes=True) if accepted_pointer else None
            unchanged = accepted is not None and accepted["input_fingerprint"] == actual_minutes_fingerprint(inputs, version=archive.model_version)
            record = accepted if unchanged else build_actual_minutes_prediction(
                prediction=prediction, box=box, inputs=inputs, generated_at=now,
                ratings=ratings, priors=priors, version=archive.model_version)
            records.append((game, record, etag, unchanged))
        for game, record, etag, unchanged in records:
            current_gid = game["game_id"]
            inputs = record["inputs"]
            current, _ = archive.selection(day, current_gid)
            if current is None or _ref(current) != inputs["prediction"]:
                raise PredictionError("The baseline prediction selection changed during the counterfactual.")
            if nba_schedule.load()[1] != schedule:
                raise PredictionError("The schedule changed during the counterfactual.")
            if nba_box_score_archive.read_saved(archive.season, game)[1] != inputs["box_score"]:
                raise PredictionError("The selected final box score changed during the counterfactual.")
            if not unchanged:
                for side in ("home", "away"):
                    for player in record[side]["players"]:
                        if not player["in_baseline_prediction"] and player["rating_source"] == "replacement":
                            message = f"Actual player {player['player_id']} ({player['name']}) has no saved rating; using the approved replacement rating."
                            warnings.append(archive.save_event(day=day, run_id=context.run.run_id, stage=STAGE,
                                status="warning", message=message, game_id=current_gid, inputs=inputs,
                                details={"player_id": player["player_id"], "name": player["name"], "nail": player["nail"]}))
                            context.log.warning(message)
            outputs[current_gid] = archive.save_actual_minutes(record, expected_etag=etag)
        metadata.update({"game_count": len(games), "counterfactual_count": len(outputs),
                         "counterfactuals": MetadataValue.json(outputs), "warnings": MetadataValue.json(warnings)})
    except Exception as error:
        message = str(error)
        try:
            ref = archive.save_event(day=day, run_id=context.run.run_id, stage=STAGE,
                message=message, inputs=inputs, game_id=current_gid, details={"error_type": type(error).__name__})
            metadata["prediction_failure"] = MetadataValue.json(ref)
        except Exception as write_error:
            message += f" Could not save the S3 failure record: {write_error}"
        raise Failure(message, metadata=metadata, allow_retries=False) from error
    try:
        original = archive.parent_failure(context, day, stage=STAGE)
        if original:
            ref = archive.save_event(day=day, run_id=context.run.run_id, stage=STAGE, status="recovered",
                message="Actual-minutes counterfactuals validated; baseline predictions preserved.", inputs=inputs, original=original)
            metadata["prediction_recovery"] = MetadataValue.json(ref)
    except Exception as error:
        raise Failure(f"Counterfactual recovery record failed: {error}", metadata=metadata, allow_retries=False) from error
    return MaterializeResult(metadata=metadata)
