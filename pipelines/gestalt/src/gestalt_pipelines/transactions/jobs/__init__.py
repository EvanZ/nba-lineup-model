"""Manual and scheduled capture of dated NBA transactions."""

from dagster import AssetSelection, define_asset_job, in_process_executor

from ..assets import transactions_by_date

capture_nba_transactions = define_asset_job(
    "capture_nba_transactions",
    selection=AssetSelection.assets(transactions_by_date),
    executor_def=in_process_executor,
    description="Capture official transactions for a Pacific date and preserve first-observation timestamps.",
)

__all__ = ["capture_nba_transactions"]
