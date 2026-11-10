"""Manual capture of official final NBA box scores."""

from dagster import AssetSelection, define_asset_job, in_process_executor

from ..assets import nba_box_scores

capture_nba_box_scores = define_asset_job(
    "capture_nba_box_scores",
    selection=AssetSelection.assets(nba_box_scores),
    executor_def=in_process_executor,
    description="Archive final official NBA box scores for a selected Pacific game date.",
)

__all__ = ["capture_nba_box_scores"]
