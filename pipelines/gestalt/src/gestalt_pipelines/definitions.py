"""Dagster definitions for NBA GESTALT pipelines."""

from dagster import Definitions, EnvVar

from .box_scores.assets import nba_box_scores
from .box_scores.jobs import capture_nba_box_scores
from .box_scores.resources import NBABoxScoreArchive
from .box_scores.schedules import box_score_poll
from .injury_reports.assets import injury_pdf_archive, injury_report_extractions, injury_status_by_game_date
from .injury_reports.jobs import capture_injury_pdfs, extract_injury_pdf, reconcile_injury_date
from .injury_reports.partitions import INJURY_REPORT_SEASON
from .injury_reports.resources import InjuryReportArchive, OfficialInjuryReports
from .injury_reports.schedules import injury_pdf_poll
from .injury_reports.sensors import injury_report_dates_ready, new_injury_pdf
from .odds.assets import nba_market_odds
from .odds.jobs import archive_nba_market_odds
from .odds.resources import OddsApi
from .odds.schedules import nba_market_odds_game_day
from .resources import NBAScheduleResource
from .scoreboard.assets import nba_scoreboard
from .scoreboard.jobs import capture_nba_scoreboard
from .scoreboard.resources import NBAScoreboardArchive
from .scoreboard.schedules import scoreboard_poll
from .season_preparation.assets import nba_season_schedule
from .season_preparation.jobs import capture_nba_season_schedule
from .transactions.assets import transactions_by_date
from .transactions.jobs import capture_nba_transactions
from .transactions.resources import NBATransactions
from .transactions.schedules import transactions_hourly

defs = Definitions(
    assets=[
        injury_pdf_archive, injury_report_extractions, injury_status_by_game_date,
        nba_season_schedule, nba_market_odds, nba_box_scores, nba_scoreboard, transactions_by_date,
    ],
    jobs=[
        capture_injury_pdfs, extract_injury_pdf, reconcile_injury_date,
        capture_nba_season_schedule, archive_nba_market_odds, capture_nba_box_scores, capture_nba_scoreboard,
        capture_nba_transactions,
    ],
    schedules=[injury_pdf_poll, nba_market_odds_game_day, box_score_poll, scoreboard_poll, transactions_hourly],
    sensors=[new_injury_pdf, injury_report_dates_ready],
    resources={
        "official_injury_reports": OfficialInjuryReports(),
        "injury_report_archive": InjuryReportArchive(season=INJURY_REPORT_SEASON),
        "nba_box_score_archive": NBABoxScoreArchive(),
        "nba_transactions": NBATransactions(season=EnvVar("GESTALT_NBA_SEASON")),
        "odds_api": OddsApi(api_key=EnvVar("GESTALT_ODDS_API_KEY")),
        "nba_scoreboard_archive": NBAScoreboardArchive(),
        "nba_schedule": NBAScheduleResource(season=EnvVar("GESTALT_NBA_SEASON")),
    },
)
