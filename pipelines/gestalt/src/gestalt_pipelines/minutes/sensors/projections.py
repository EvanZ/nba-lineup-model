"""Request date jobs when accepted inputs change for games still ahead."""

from datetime import UTC, datetime, timedelta
from hashlib import sha256

from dagster import (
    DagsterRunStatus, DefaultSensorStatus, RunRequest, RunsFilter,
    SensorEvaluationContext, SensorResult, SkipReason, sensor,
)

from ...utils import PACIFIC
from ..utils import json_bytes, projection_fingerprint


@sensor(
    name="project_game_minutes_sensor",job_name="project_game_minutes",
    minimum_interval_seconds=60,default_status=DefaultSensorStatus.STOPPED,
    required_resource_keys={"nba_schedule","minutes_archive","nba_roster_archive"},
    description="Project upcoming games when accepted daily minutes, rosters, or injuries change.",
)
def project_game_minutes_sensor(context: SensorEvaluationContext):
    now = datetime.now(UTC)
    today = now.astimezone(PACIFIC).date()
    schedule = context.resources.nba_schedule
    archive = context.resources.minutes_archive
    roster = context.resources.nba_roster_archive
    games,ref = schedule.games_between(today,today+timedelta(days=1),allow_empty=True)
    upcoming = {}
    for g in games:
        if g["game_time_utc"] > now:
            upcoming.setdefault(str(g["game_time_utc"].astimezone(PACIFIC).date()),[]).append(g)
    if not upcoming:
        return SkipReason("No upcoming games in the injury-report window.")
    busy = context.instance.get_runs(
        filters=RunsFilter(job_name="project_game_minutes",statuses=[
            DagsterRunStatus.QUEUED,DagsterRunStatus.STARTING,
            DagsterRunStatus.STARTED,DagsterRunStatus.CANCELING,
        ]),limit=20,
    )
    requests,waiting = [],[]
    for day,slate in sorted(upcoming.items()):
        if any(r.tags.get("dagster/partition") == day for r in busy):
            continue
        try:
            inputs = archive.projection_selections(day,today=today,roster_archive=roster)
        except FileNotFoundError as error:
            waiting.append(str(error))
            continue
        inputs["schedule"] = ref
        identities = {}
        changed = False
        for g in slate:
            identity = projection_fingerprint(g,inputs,archive.projection_version)
            identities[g["game_id"]] = identity
            selected,_ = archive.game_selection(day,g["game_id"])
            if selected is None or selected["input_fingerprint"] != identity:
                changed = True
        if changed:
            key = sha256(json_bytes({"day":day,"games":identities})).hexdigest()
            requests.append(RunRequest(
                run_key=key,partition_key=day,
                tags={"dagster/priority":"-1","game_minutes/input_selection":key},
            ))
    if not requests:
        return SkipReason("; ".join(waiting) or "Upcoming projections already use accepted inputs, or a date is running.")
    return SensorResult(run_requests=requests)


__all__ = ["project_game_minutes_sensor"]
