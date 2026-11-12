"""Manual daily job; ordered ranges are launched from the asset backfill UI."""

from dagster import AssetSelection, define_asset_job, in_process_executor

from ..assets import player_minutes

update_player_minutes = define_asset_job(
    "update_player_minutes", selection=AssetSelection.assets(player_minutes),
    executor_def=in_process_executor,
    description="Update one Pacific date's conditional minutes from its accepted predecessor.",
)

__all__ = ["update_player_minutes"]
