"""Publish game history when successful ratings/postgame materializations are ready."""
import os
import re
from datetime import UTC, date, datetime
from hashlib import sha256

from dagster import AssetKey, AssetRecordsFilter, DagsterEventType, DagsterRunStatus, DefaultSensorStatus, RunRequest, RunStatusSensorContext, SkipReason, run_status_sensor

from ...utils import PACIFIC
from ..jobs import publish_player_game_history
from ..partitions import ratings_dates
from ..utils import MODEL_VERSION, json_bytes


def _material(instance, asset, day):
    rows = instance.fetch_materializations(AssetRecordsFilter(asset_key=AssetKey(asset), asset_partitions=[day]), limit=1).records
    if not rows:
        return None
    row = rows[0]
    run = instance.get_run_by_id(row.run_id)
    if run is None or run.status != DagsterRunStatus.SUCCESS:
        return None
    return row.asset_materialization


def _identity(material):
    values = {k: v.value for k, v in material.metadata.items()}
    if values["model_version"] != MODEL_VERSION:
        raise ValueError("History source has a different configured model version.")
    day, season = material.partition, os.environ["GESTALT_NBA_SEASON"]
    prefix = f"s3://nba-gestalt-data/lake/artifacts/"
    if material.asset_key == AssetKey("player_ratings"):
        refs = {"ratings": {"uri": values["ratings_uri"], "sha256": values["sha256"]}}
        if values["season"] != season:
            raise ValueError("History source belongs to a different season.")
        root = f"{prefix}player_ratings/season={season}/model_version={MODEL_VERSION}/game_date_pacific={day}/"
    else:
        refs = {gid: {k: ref[k] for k in ("uri", "sha256")} for gid, ref in values["counterfactuals"].items()}
        root = f"{prefix}game_predictions_actual_minutes/season={season}/model_version={MODEL_VERSION}/game_date_pacific={day}/"
        if values["game_count"] != len(refs) or values["counterfactual_count"] != len(refs):
            raise ValueError("History source has incomplete postgame coverage.")
    for name, ref in refs.items():
        if not re.fullmatch("[a-f0-9]{64}", ref["sha256"]):
            raise ValueError("History materialization has an invalid source checksum.")
        expected = root+("" if name == "ratings" else f"game_id={name}/")+ref["sha256"]+".json"
        if ref["uri"] != expected:
            raise ValueError("History materialization has a noncanonical source reference.")
    return refs


@run_status_sensor(
    run_status=DagsterRunStatus.SUCCESS, request_job=publish_player_game_history,
    minimum_interval_seconds=60, default_status=DefaultSensorStatus.STOPPED,
    description="Publish complete player game history when ratings and postgame outputs are ready; older corrections refresh the latest ratings head.",
)
def publish_player_game_history_sensor(context: RunStatusSensorContext):
    if context.dagster_run.status != DagsterRunStatus.SUCCESS:
        return SkipReason("History source run did not succeed.")
    entries = context.instance.get_records_for_run(context.dagster_run.run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION).records
    matches = [e.event_log_entry.dagster_event.event_specific_data.materialization for e in entries
               if e.event_log_entry.dagster_event.event_specific_data.materialization.asset_key in
               {AssetKey("player_ratings"), AssetKey("game_predictions_actual_minutes")}]
    if len(matches) != 1 or not matches[0].partition:
        return SkipReason("No single ratings/postgame partition materialized in this successful run.")
    changed = matches[0]
    first, today = str(ratings_dates.start.date()), str(datetime.now(UTC).astimezone(PACIFIC).date())
    if not first <= changed.partition < today:
        return SkipReason("History source is outside the configured completed-date season.")
    values = {k: v.value for k,v in changed.metadata.items()}
    if values.get("season", os.environ["GESTALT_NBA_SEASON"]) != os.environ["GESTALT_NBA_SEASON"] or values.get("model_version") != MODEL_VERSION:
        return SkipReason("History source belongs to another configured season/model.")
    changed_inputs = _identity(changed)
    dates = context.instance.get_materialized_partitions(AssetKey("player_ratings"))
    eligible = sorted(d for d in dates if first <= d < today)
    if not eligible or changed.partition > eligible[-1]:
        return SkipReason("Waiting for completed ratings through the changed postgame date.")
    head = eligible[-1]
    ratings = _material(context.instance, "player_ratings", head)
    postgame = _material(context.instance, "game_predictions_actual_minutes", head)
    if ratings is None or postgame is None:
        return SkipReason("Waiting for successful ratings and postgame outputs at the latest completed ratings head.")
    identity = {"head": head, "ratings": _identity(ratings), "postgame": _identity(postgame)}
    if changed.partition < head:
        identity["correction"] = {"day": changed.partition, "asset": changed.asset_key.to_user_string(), "inputs": changed_inputs}
    return RunRequest(run_key=sha256(json_bytes(identity)).hexdigest(), partition_key=head, tags={"dagster/priority": "-2"})


__all__ = ["publish_player_game_history_sensor"]
