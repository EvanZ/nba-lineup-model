"""Refresh upcoming margins when accepted allocations, ratings or schedule change."""
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from dagster import (
    AssetKey, DagsterRunStatus, DefaultSensorStatus, RunRequest, RunsFilter,
    SensorEvaluationContext, SensorResult, SkipReason, sensor,
)

from ...utils import PACIFIC, forecast_observed_date
from ..partitions import game_prediction_dates
from ..utils import implementation_identity, json_bytes, prediction_fingerprint


def _covers_date(run, day: str) -> bool:
    if run.tags.get("dagster/partition") == day:
        return True
    first = run.tags.get("dagster/asset_partition_range_start")
    last = run.tags.get("dagster/asset_partition_range_end")
    return bool(first and last and first <= day <= last)


@sensor(
    job_name="predict_game_margins", minimum_interval_seconds=60,
    default_status=DefaultSensorStatus.STOPPED,
    required_resource_keys={"nba_schedule", "minutes_archive", "ratings_archive", "game_prediction_archive"},
    description="Refresh open today/tomorrow game margins from changed accepted inputs.",
)
def predict_game_margins_sensor(context: SensorEvaluationContext):
    now = datetime.now(UTC)
    today = now.astimezone(PACIFIC).date()
    schedule = context.resources.nba_schedule
    minutes = context.resources.minutes_archive
    ratings = context.resources.ratings_archive
    archive = context.resources.game_prediction_archive
    games, schedule_ref = schedule.games_between(today, today+timedelta(days=1), allow_empty=True)
    upcoming = {}
    for game in games:
        if game["game_time_utc"] > now:
            upcoming.setdefault(str(game["game_time_utc"].astimezone(PACIFIC).date()), []).append(game)
    if not upcoming:
        return SkipReason("No games remain before scheduled tipoff in the today/tomorrow window.")
    busy = [run for run in context.instance.get_runs(filters=RunsFilter(statuses=[
        DagsterRunStatus.QUEUED, DagsterRunStatus.STARTING, DagsterRunStatus.STARTED, DagsterRunStatus.CANCELING,
    ])) if run.job_name == "predict_game_margins" or AssetKey("game_predictions") in (run.asset_selection or set())]
    evidence_day = str(forecast_observed_date(today, today))
    bootstrap = evidence_day < str(game_prediction_dates.start.date())
    ratings_ref = ratings.selection()[0] if bootstrap else ratings.state_selection(evidence_day)[0]
    if ratings_ref is None:
        return SkipReason(f"Waiting for accepted ratings through {evidence_day} or opening-day priors.")
    implementation = implementation_identity()
    checked_ratings = False
    requests, waiting = [], []
    for day, slate in sorted(upcoming.items()):
        if any(_covers_date(run, day) for run in busy):
            continue
        identities, changed, complete = {}, False, True
        for game in slate:
            allocation, _ = minutes.game_selection(day, game["game_id"])
            if allocation is None:
                waiting.append(f"Waiting for game_minutes for {game['game_id']} on {day}.")
                complete = False
                break
            selected, _ = archive.selection(day, game["game_id"])
            inputs = {
                "player_ratings": ratings_ref, "game_minutes": {k: allocation[k] for k in ("uri", "sha256")},
                "schedule": schedule_ref,
            }
            identity = prediction_fingerprint(game, inputs, version=archive.model_version, implementation=implementation)
            if selected and selected["input_fingerprint"] != identity:
                accepted = archive.load(selected)
                if "rating_priors" in accepted["inputs"]:
                    prior = ratings.selection()[0]
                    if prior is None:
                        waiting.append("Waiting for the accepted frozen rating priors.")
                        complete = False
                        break
                    inputs["rating_priors"] = prior
                    identity = prediction_fingerprint(game, inputs, version=archive.model_version, implementation=implementation)
            identities[game["game_id"]] = identity
            changed |= selected is None or selected["input_fingerprint"] != identity
        if not complete or not changed:
            continue
        if not checked_ratings:
            # Read the larger learned state only when a forecast actually needs work.
            try:
                result = ratings.load_priors() if bootstrap else ratings.state(evidence_day)
            except FileNotFoundError as error:
                return SkipReason(str(error))
            if result[0] is None:
                return SkipReason(f"Waiting for completed-date ratings through {evidence_day}.")
            if result[1] != ratings_ref:
                return SkipReason("The ratings selection changed during this check; the next tick will reload it.")
            checked_ratings = True
        key = sha256(json_bytes({"day": day, "games": identities})).hexdigest()
        requests.append(RunRequest(run_key=key, partition_key=day, tags={
            "dagster/priority": "-1", "game_predictions/input_selection": key,
        }))
    return SensorResult(run_requests=requests) if requests else SkipReason(
        "; ".join(waiting) or "Upcoming margins already use accepted inputs, or the date is queued/running."
    )
