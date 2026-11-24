"""Team rotation publication for manual launches and the stints-success trigger."""
from dagster import AssetSelection, define_asset_job, in_process_executor
from ..assets import team_rotation, team_rotation_history

build_team_rotation = define_asset_job(
    "build_team_rotation", selection=AssetSelection.assets(team_rotation), executor_def=in_process_executor,
    description="Publish all 30 team rotation summaries through the selected completed Pacific date.",
)
__all__ = ["build_team_rotation"]


publish_team_rotation_history = define_asset_job(
    "publish_team_rotation_history", selection=AssetSelection.assets(team_rotation_history),
    executor_def=in_process_executor,
    description="Index successfully published rotation summaries without recalculating them.",
)
__all__ += ["publish_team_rotation_history"]
