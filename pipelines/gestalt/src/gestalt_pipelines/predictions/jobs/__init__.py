"""One-date manual job; asset partitions also support historical backfills."""
from dagster import AssetSelection, define_asset_job, in_process_executor
from ..assets import game_predictions, game_predictions_actual_minutes

predict_game_margins = define_asset_job(
    "predict_game_margins", selection=AssetSelection.assets(game_predictions),
    executor_def=in_process_executor,
    description="Predict a Pacific slate from saved ratings and game minutes; preserve frozen pregame predictions.",
)

__all__ = ["predict_game_margins"]


predict_game_margins_actual_minutes = define_asset_job(
    "predict_game_margins_actual_minutes", selection=AssetSelection.assets(game_predictions_actual_minutes),
    executor_def=in_process_executor,
    description="Recalculate a completed Pacific slate using actual minutes and saved pregame rating inputs.",
)
__all__ += ["predict_game_margins_actual_minutes"]
