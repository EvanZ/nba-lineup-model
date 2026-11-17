"""Manual capture of one live NBA market odds snapshot."""

from dagster import AssetSelection, define_asset_job, in_process_executor

from ..assets import game_odds, nba_market_odds

archive_nba_market_odds = define_asset_job(
    "archive_nba_market_odds",
    selection=AssetSelection.assets(nba_market_odds),
    executor_def=in_process_executor,
    description="Capture and archive the current NBA spreads and moneylines.",
)

publish_game_odds = define_asset_job(
    "publish_game_odds", selection=AssetSelection.assets(game_odds),
    executor_def=in_process_executor,
    description="Publish saved pregame odds references for a Pacific game date.",
)
