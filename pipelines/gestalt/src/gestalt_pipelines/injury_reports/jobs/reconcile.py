"""Manual job for one Pacific game date."""

from dagster import AssetSelection, define_asset_job

from ..assets.reconcile import injury_status_by_game_date

reconcile_injury_date = define_asset_job(
    "reconcile_injury_date",
    selection=AssetSelection.assets(injury_status_by_game_date),
    description="Reconcile one Pacific slate from selected S3 injury, schedule, and roster evidence.",
)
