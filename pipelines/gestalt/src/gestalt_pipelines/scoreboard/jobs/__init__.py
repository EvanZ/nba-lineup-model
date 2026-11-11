"""Manual capture of the official NBA scoreboard."""

from dagster import AssetSelection, define_asset_job, in_process_executor

from ..assets import nba_scoreboard

capture_nba_scoreboard = define_asset_job(
    "capture_nba_scoreboard",
    selection=AssetSelection.assets(nba_scoreboard),
    executor_def=in_process_executor,
    description="Publish the latest official NBA scoreboard for its Pacific game date.",
)

__all__ = ["capture_nba_scoreboard"]
