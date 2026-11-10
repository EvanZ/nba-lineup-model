"""Pacific game-day polling for NBA market odds."""

from dagster import DefaultScheduleStatus, ScheduleDefinition

from ..jobs import archive_nba_market_odds

nba_market_odds_game_day = ScheduleDefinition(
    job=archive_nba_market_odds,
    name="nba_market_odds_game_day",
    cron_schedule="20 8,12,15-19 * * *",
    execution_timezone="America/Los_Angeles",
    default_status=DefaultScheduleStatus.STOPPED,
    description="Capture NBA spreads and moneylines at 8:20 a.m., 12:20 p.m., and hourly from 3:20 to 7:20 p.m. Pacific.",
)

__all__ = ["nba_market_odds_game_day"]
