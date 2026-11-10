"""Dagster definitions for NBA GESTALT pipelines."""

from dagster import Definitions, EnvVar

from .box_scores.assets import nba_box_scores
from .box_scores.jobs import capture_nba_box_scores
from .box_scores.resources import NBABoxScoreArchive
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
from .season_preparation.assets import nba_season_schedule
from .season_preparation.jobs import capture_nba_season_schedule

defs = Definitions(
    assets=[
        injury_pdf_archive, injury_report_extractions, injury_status_by_game_date,
        nba_season_schedule, nba_market_odds, nba_box_scores,
    ],
    jobs=[
        capture_injury_pdfs, extract_injury_pdf, reconcile_injury_date,
        capture_nba_season_schedule, archive_nba_market_odds, capture_nba_box_scores,
    ],
    schedules=[injury_pdf_poll, nba_market_odds_game_day],
    sensors=[new_injury_pdf, injury_report_dates_ready],
    resources={
        "official_injury_reports": OfficialInjuryReports(),
        "injury_report_archive": InjuryReportArchive(season=INJURY_REPORT_SEASON),
        "nba_box_score_archive": NBABoxScoreArchive(),
        "odds_api": OddsApi(api_key=EnvVar("GESTALT_ODDS_API_KEY")),
        "nba_schedule": NBAScheduleResource(season=EnvVar("GESTALT_NBA_SEASON")),
    },
)
