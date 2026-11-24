"""Publish accepted rotation history without calculating rotations or running upstream jobs."""
import os
from dagster import AllPartitionMapping, AssetDep, AssetExecutionContext, AssetKey, Failure, MaterializeResult, MetadataValue, asset
from ..resources import TeamRotationArchive
from ..utils import PROCESSING_VERSION, accepted_publications


@asset(group_name="team_rotation", deps=[AssetDep(AssetKey("team_rotation"), partition_mapping=AllPartitionMapping())])
def team_rotation_history(context: AssetExecutionContext, team_rotation_archive: TeamRotationArchive) -> MaterializeResult:
    season = os.environ["GESTALT_NBA_SEASON"]
    try:
        result = team_rotation_archive.publish_history(season, lambda:accepted_publications(
            context.instance, season, bucket=team_rotation_archive.bucket))
    except Exception as error:
        raise Failure(f"Could not publish accepted rotation history: {error}", metadata={"season":season}, allow_retries=False) from error
    return MaterializeResult(metadata={"season":season, "processing_version":PROCESSING_VERSION,
        **{k:v for k,v in result.items() if k not in {"source_publications", "indexes"}},
        "source_publications":MetadataValue.json(result["source_publications"]),
        "history_indexes":MetadataValue.json(result["indexes"])})
