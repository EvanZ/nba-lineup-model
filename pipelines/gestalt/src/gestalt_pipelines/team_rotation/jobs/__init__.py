"""Team rotation publication for manual launches and the stints-success trigger."""
from dagster import AssetSelection, define_asset_job, in_process_executor
from ..assets import team_rotation

build_team_rotation = define_asset_job(
    "build_team_rotation", selection=AssetSelection.assets(team_rotation), executor_def=in_process_executor,
    description="Publish all 30 team rotation summaries through the selected completed Pacific date.",
)
__all__ = ["build_team_rotation"]
