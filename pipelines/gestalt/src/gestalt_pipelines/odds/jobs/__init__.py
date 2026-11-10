"""Manual capture of one live NBA market odds snapshot."""

from dagster import AssetSelection, define_asset_job, in_process_executor

from ..assets import nba_market_odds

archive_nba_market_odds = define_asset_job(
    "archive_nba_market_odds",
    selection=AssetSelection.assets(nba_market_odds),
    executor_def=in_process_executor,
    description="Capture and archive the current NBA spreads and moneylines.",
)
