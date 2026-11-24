"""Index only after the source producer's entire run succeeds."""
import os
from hashlib import sha256
from dagster import (AssetKey, DagsterEventType, DagsterRunStatus, DefaultSensorStatus,
                     RunRequest, RunStatusSensorContext, SkipReason, run_status_sensor)
from ..jobs import publish_team_rotation_history
from ..utils import PROCESSING_VERSION, json_bytes, validate_point
from ...utils import DATA_BUCKET, NBA_TEAM_IDS


@run_status_sensor(run_status=DagsterRunStatus.SUCCESS, request_job=publish_team_rotation_history,
                   minimum_interval_seconds=60, default_status=DefaultSensorStatus.STOPPED,
                   description="Publish accepted team rotation history after successful rotation summaries, including corrections/backfills.")
def publish_team_rotation_history_sensor(context: RunStatusSensorContext):
    if context.dagster_run.status != DagsterRunStatus.SUCCESS:
        return SkipReason("The rotation source run did not succeed.")
    events = context.instance.get_records_for_run(context.dagster_run.run_id,
                                                 of_type=DagsterEventType.ASSET_MATERIALIZATION).records
    materials = [r.event_log_entry.dagster_event.event_specific_data.materialization for r in events
                 if r.event_log_entry.dagster_event.event_specific_data.materialization.asset_key == AssetKey("team_rotation")]
    if not materials:
        return SkipReason("No rotation summary materialized in this successful run.")
    season, sources = os.environ["GESTALT_NBA_SEASON"], {}
    for material in materials:
        values = {k:v.value for k,v in material.metadata.items()}
        if values.get("season") != season:
            continue
        if values.get("processing_version") != PROCESSING_VERSION or values.get("team_count") != 30:
            raise ValueError("Rotation source has inconsistent processing identity/team coverage.")
        refs = values.get("selections", [])
        if len(refs) != 30 or {r["team_id"] for r in refs} != set(NBA_TEAM_IDS):
            raise ValueError("Rotation source has incomplete team selections.")
        for ref in refs:
            validate_point(ref, season, ref["team_id"], bucket=DATA_BUCKET)
            if (ref.get("game_date_pacific") != material.partition or ref.get("season") != season
                or ref.get("processing_version") != PROCESSING_VERSION or ref.get("schema_version") != 1):
                raise ValueError("Rotation source selection has a different publication identity.")
        sources[material.partition] = refs
    if not sources:
        return SkipReason("Rotation summaries belong to another season.")
    return RunRequest(run_key=sha256(json_bytes({"season":season,"processing_version":PROCESSING_VERSION,
                                               "publications":sources})).hexdigest(), tags={"dagster/priority":"-2"})
