"""Isolated Dagster boundary for the atomic nightly GESTALT pipeline.

Collection jobs live in collection_definitions and never share this daemon.
The deprecated shared definitions module is not exposed as a code location.
"""

from dagster import Definitions

from nba_lineup_model.orchestration.definitions import (
    backfill_official_history_job,
    complete_slate_daily_release_sensor,
    prepare_daily_site_release_job,
)


defs = Definitions(
    jobs=[
        backfill_official_history_job,
        prepare_daily_site_release_job,
    ],
    sensors=[complete_slate_daily_release_sensor],
)
