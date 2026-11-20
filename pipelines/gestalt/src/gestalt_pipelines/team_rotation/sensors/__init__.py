"""Refresh cumulative rotations after successful new or corrected stints."""
import os
import re
from datetime import date, datetime
from hashlib import sha256

from dagster import (AssetKey, AssetRecordsFilter, DagsterEventType, DagsterRunStatus,
                     DefaultSensorStatus, RunRequest, RunStatusSensorContext, SkipReason, run_status_sensor)

from ...resources import NBAScheduleResource
from ...utils import PACIFIC
from ..jobs import build_team_rotation
from ..utils import PROCESSING_VERSION, json_bytes


def _material(instance, day):
    records = instance.fetch_materializations(
        AssetRecordsFilter(asset_key=AssetKey("nba_stints"), asset_partitions=[day]), limit=1,
    ).records
    if not records:
        return None
    record = records[0]
    run = instance.get_run_by_id(record.run_id)
    if run is None or run.status != DagsterRunStatus.SUCCESS:
        return None
    return record.asset_materialization


def _identity(material, season):
    values = {k:v.value for k,v in material.metadata.items()}
    day = material.partition
    if values.get("season") != season or values.get("game_date_pacific") != day:
        raise ValueError("Stints materialization has the wrong season/date identity.")
    digest = values.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Stints materialization has an invalid checksum.")
    expected = f"s3://nba-gestalt-data/lake/artifacts/stints/season={season}/game_date_pacific={day}/{digest}.json"
    if values.get("stints_uri") != expected:
        raise ValueError("Stints materialization has a noncanonical source reference.")
    return {"uri": expected, "sha256": digest}


@run_status_sensor(
    run_status=DagsterRunStatus.SUCCESS, request_job=build_team_rotation,
    minimum_interval_seconds=60, default_status=DefaultSensorStatus.STOPPED,
    description="Refresh all 30 team rotations at the latest completed stints head; wait for full season coverage and include older corrections.",
)
def build_team_rotation_sensor(context: RunStatusSensorContext, nba_schedule: NBAScheduleResource):
    if context.dagster_run.status != DagsterRunStatus.SUCCESS:
        return SkipReason("The source run did not succeed.")
    events = context.instance.get_records_for_run(
        context.dagster_run.run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION,
    ).records
    matches = [r.event_log_entry.dagster_event.event_specific_data.materialization for r in events
               if r.event_log_entry.dagster_event.event_specific_data.materialization.asset_key == AssetKey("nba_stints")]
    if len(matches) != 1 or not matches[0].partition:
        return SkipReason("No single stints partition materialized in this successful run.")
    changed = matches[0]
    first = os.environ["GESTALT_NBA_SEASON_START_DATE"]
    today = str(datetime.now(PACIFIC).date())
    season = nba_schedule.season
    if not first <= changed.partition < today:
        return SkipReason("Changed stints are outside the configured completed-date season.")
    if changed.metadata.get("season") is None or changed.metadata["season"].value != season:
        return SkipReason("Changed stints belong to another season.")
    _identity(changed, season)
    dates = context.instance.get_materialized_partitions(AssetKey("nba_stints"))
    eligible = sorted(day for day in dates if first <= day < today)
    if not eligible:
        return SkipReason("Waiting for a completed stints head.")
    head = eligible[-1]
    games, schedule = nba_schedule.games_between(date.fromisoformat(first), date.fromisoformat(head), allow_empty=True)
    # Empty/off days need no fabricated source. Include the head to advance evidence metadata.
    required = {head} | {g["game_time_utc"].astimezone(PACIFIC).date().isoformat()
                         for g in games if g["game_id"].startswith("002")}
    sources = {}
    for day in sorted(required):
        material = _material(context.instance, day)
        if material is None:
            return SkipReason(f"Waiting for successful stints for {day}.")
        values = {k:v.value for k,v in material.metadata.items()}
        reference = _identity(material, season)
        expected = {g["game_id"] for g in games if g["game_time_utc"].astimezone(PACIFIC).date().isoformat() == day}
        saved = values.get("inputs", {}).get("games", {})
        if set(saved) != expected or values.get("game_count") != len(expected):
            return SkipReason(f"Waiting for complete official game coverage in stints for {day}.")
        sources[day] = reference
    identity = {"season": season, "processing_version": PROCESSING_VERSION,
                "head": head, "schedule": schedule, "stints": sources}
    return RunRequest(run_key=sha256(json_bytes(identity)).hexdigest(), partition_key=head,
                      tags={"dagster/priority": "-2"})


__all__ = ["build_team_rotation_sensor"]
