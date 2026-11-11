"""Manual, partitioned NBA roster construction."""

from dagster import AssetSelection, define_asset_job, in_process_executor

from ..assets import rosters_by_date

build_nba_rosters = define_asset_job(
    "build_nba_rosters",
    selection=AssetSelection.assets(rosters_by_date),
    executor_def=in_process_executor,
    description="Build complete NBA rosters for selected Pacific dates from approved S3 evidence.",
)

__all__ = ["build_nba_rosters"]
