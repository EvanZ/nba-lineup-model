"""One-date manual job; asset partitions also support historical backfills."""
from dagster import AssetSelection, define_asset_job, in_process_executor
from ..assets import game_predictions

predict_game_margins = define_asset_job(
    "predict_game_margins", selection=AssetSelection.assets(game_predictions),
    executor_def=in_process_executor,
    description="Predict a Pacific slate from saved ratings and game minutes; preserve frozen pregame predictions.",
)

__all__ = ["predict_game_margins"]
