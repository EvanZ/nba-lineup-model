"""Capture selected Pacific dates of final NBA play-by-play."""

from dagster import AssetSelection, define_asset_job, in_process_executor

from ..assets import nba_play_by_play, nba_stints

capture_nba_play_by_play = define_asset_job(
    "capture_nba_play_by_play",
    selection=AssetSelection.assets(nba_play_by_play),
    executor_def=in_process_executor,
    description="Refresh final NBA play-by-play for a selected Pacific date and preserve revisions.",
)

parse_nba_games = define_asset_job(
    "parse_nba_games",
    selection=AssetSelection.assets(nba_stints),
    executor_def=in_process_executor,
    description="Parse saved NBA play-by-play and final boxes into a complete date of stints.",
)

__all__ = ["capture_nba_play_by_play", "parse_nba_games"]
