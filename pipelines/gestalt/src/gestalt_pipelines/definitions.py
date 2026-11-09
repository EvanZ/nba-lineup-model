"""Dagster definitions for NBA GESTALT pipelines."""

from dagster import Definitions, EnvVar

from .injury_reports.assets import injury_pdf_archive, injury_report_extractions, injury_status_by_game_date
from .injury_reports.jobs import capture_injury_pdfs, extract_injury_pdf, reconcile_injury_date
from .injury_reports.partitions import INJURY_REPORT_SEASON
from .injury_reports.resources import InjuryReportArchive, OfficialInjuryReports
from .resources import NBAScheduleResource
from .season_preparation.assets import nba_season_schedule
from .season_preparation.jobs import capture_nba_season_schedule

defs = Definitions(
    assets=[injury_pdf_archive, injury_report_extractions, injury_status_by_game_date, nba_season_schedule],
    jobs=[capture_injury_pdfs, extract_injury_pdf, reconcile_injury_date, capture_nba_season_schedule],
    resources={
        "official_injury_reports": OfficialInjuryReports(),
        "injury_report_archive": InjuryReportArchive(season=INJURY_REPORT_SEASON),
        "nba_schedule": NBAScheduleResource(season=EnvVar("GESTALT_NBA_SEASON")),
    },
)
