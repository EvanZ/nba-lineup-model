"""Manual capture of the official NBA season schedule."""

from dagster import AssetSelection, define_asset_job

from ..assets import nba_season_schedule

capture_nba_season_schedule = define_asset_job(
    "capture_nba_season_schedule",
    selection=AssetSelection.assets(nba_season_schedule),
    description="Fetch, validate, and publish the official NBA schedule for the configured season.",
)
