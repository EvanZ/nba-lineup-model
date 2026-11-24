"""Accepted rotation publications come from successful permanent run history."""
import re
from datetime import date

from dagster import AssetKey, AssetRecordsFilter, DagsterRunStatus

from ...utils import DATA_BUCKET, NBA_TEAM_IDS
from .rotation import PROCESSING_VERSION


def accepted_publications(instance, season, *, bucket=DATA_BUCKET):
    """Select the last successful publication per date; never discover by listing S3."""
    entries, cursor = [], None
    while True:
        page = instance.fetch_materializations(AssetRecordsFilter(asset_key=AssetKey("team_rotation")),
                                              limit=100, cursor=cursor)
        entries.extend(page.records)
        if not page.has_more:
            break
        cursor = page.cursor
    selected = {}
    runs = {}
    for event in sorted(entries, key=lambda e:e.storage_id):
        if event.run_id not in runs:
            runs[event.run_id] = instance.get_run_by_id(event.run_id)
        run = runs[event.run_id]
        if run is None or run.status != DagsterRunStatus.SUCCESS:
            continue
        origin = run.remote_job_origin
        if origin and (origin.repository_origin.code_location_origin.location_name != "nba_gestalt_pipelines"
                       or origin.repository_origin.repository_name != "__repository__"):
            continue
        material = event.asset_materialization
        values = {k:v.value for k,v in material.metadata.items()}
        if values.get("season") != season:
            continue
        day = material.partition
        if (not day or date.fromisoformat(day).isoformat() != day
            or values.get("game_date_pacific") != day or values.get("processing_version") != PROCESSING_VERSION):
            raise ValueError("Invalid successful rotation publication identity.")
        refs = values.get("selections", [])
        if values.get("team_count") != 30 or len(refs) != 30 or {r.get("team_id") for r in refs} != set(NBA_TEAM_IDS):
            raise ValueError(f"Successful rotation publication has incomplete team coverage for {day}.")
        for ref in refs:
            validate_point(ref, season, ref["team_id"], bucket=bucket)
            if (ref.get("schema_version") != 1 or ref.get("season") != season
                or ref.get("game_date_pacific") != day or ref.get("processing_version") != PROCESSING_VERSION):
                raise ValueError("Successful publication has inconsistent selection identities.")
        selected[day] = {"selections": refs, "dagster_run_id": event.run_id,
                         "materialization_storage_id": event.storage_id}
    if not selected:
        raise FileNotFoundError("No successful rotation publications are available for this season.")
    return selected


def validate_point(point, season, team_id, *, bucket):
    day, digest = point.get("game_date_pacific"), point.get("sha256")
    if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
        raise ValueError("Invalid rotation history point date.")
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Invalid rotation history point checksum.")
    expected = (f"s3://{bucket}/lake/artifacts/team_rotation/season={season}/team_id={team_id}/"
                f"game_date_pacific={day}/{digest}.json")
    if point.get("uri") != expected:
        raise ValueError("Rotation history reference is outside its canonical identity/path.")


def validate_history(index, season, team_id, *, bucket):
    if (index.get("schema_version") != 1 or index.get("processing_version") != PROCESSING_VERSION
        or index.get("artifact_kind") != "team_rotation_history_index"
        or index.get("season") != season or index.get("team_id") != team_id):
        raise ValueError("Invalid accepted rotation history index identity.")
    points = index.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("A rotation history index must have accepted points.")
    for point in points:
        if set(point) != {"game_date_pacific", "uri", "sha256"}:
            raise ValueError("Unexpected rotation history point fields.")
        validate_point(point, season, team_id, bucket=bucket)
    days = [p["game_date_pacific"] for p in points]
    if days != sorted(set(days)) or index.get("through_game_date_pacific") != days[-1]:
        raise ValueError("Rotation history dates are duplicated, unordered, or have the wrong head.")
