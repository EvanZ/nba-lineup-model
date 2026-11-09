"""Dagster partitions for injury reports."""

from .game_date import INJURY_REPORT_SEASON, injury_game_dates

__all__ = ["INJURY_REPORT_SEASON", "injury_game_dates"]
