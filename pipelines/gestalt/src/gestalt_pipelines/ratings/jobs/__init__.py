"""Manual initialization and one-date ratings/HCA updates."""

from dagster import AssetSelection, define_asset_job, in_process_executor

from ..assets import player_rating_priors, player_ratings

initialize_player_ratings = define_asset_job(
    "initialize_player_ratings", selection=AssetSelection.assets(player_rating_priors),
    executor_def=in_process_executor,
    description="Export approved frozen ratings/HCA inputs to JSON; no configuration or training.",
)

update_player_ratings = define_asset_job(
    "update_player_ratings", selection=AssetSelection.assets(player_ratings),
    executor_def=in_process_executor,
    description="Publish ratings for the selected Pacific date using games through the previous date.",
)

__all__ = ["initialize_player_ratings", "update_player_ratings"]
