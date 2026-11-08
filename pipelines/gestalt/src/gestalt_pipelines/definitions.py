"""Dagster definitions for NBA GESTALT pipelines."""

from dagster import Definitions

from .injury_reports.assets.capture import injury_pdf_archive
from .injury_reports.jobs.capture import capture_injury_pdfs
from .injury_reports.resources.archive import InjuryReportArchive
from .injury_reports.resources.official_nba import OfficialInjuryReports

defs = Definitions(
    assets=[injury_pdf_archive],
    jobs=[capture_injury_pdfs],
    resources={
        "official_injury_reports": OfficialInjuryReports(),
        "injury_report_archive": InjuryReportArchive(),
    },
)
