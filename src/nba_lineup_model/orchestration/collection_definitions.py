"""Collection-only Dagster boundary for protected NBA GESTALT raw feeds.

This code location intentionally exposes only three independently executable
jobs: official injury/schedule capture, The Odds API snapshots, and official
NBA CDN live-score capture.  Model, replay, release, and publication jobs are
not loaded into this process.
"""

from dagster import Definitions

from nba_lineup_model.orchestration.definitions import (
    archive_nba_market_odds_job,
    live_score_sensor,
    nba_market_odds_schedule,
    pregame_projection_sensor,
    refresh_schedule_job,
    schedule_refresh_schedule,
    update_live_scores_job,
)


defs = Definitions(
    jobs=[
        refresh_schedule_job,
        archive_nba_market_odds_job,
        update_live_scores_job,
    ],
    schedules=[
        schedule_refresh_schedule,
        nba_market_odds_schedule,
    ],
    sensors=[
        pregame_projection_sensor,
        live_score_sensor,
    ],
)
