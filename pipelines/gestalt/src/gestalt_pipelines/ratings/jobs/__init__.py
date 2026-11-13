"""Manual initialization; daily updates are a separate implementation slice."""

from dagster import AssetSelection, define_asset_job, in_process_executor

from ..assets import player_rating_priors

initialize_player_ratings = define_asset_job(
    "initialize_player_ratings", selection=AssetSelection.assets(player_rating_priors),
    executor_def=in_process_executor,
    description="Export approved frozen ratings/HCA inputs to JSON; no configuration or training.",
)

__all__ = ["initialize_player_ratings"]
