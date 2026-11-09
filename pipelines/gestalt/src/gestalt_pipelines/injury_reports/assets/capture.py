"""Capture the newest official NBA injury report for a publication date."""

from datetime import date

from dagster import Config, MaterializeResult, asset

from ..resources import InjuryReportArchive, OfficialInjuryReports


class CaptureConfig(Config):
    publication_date: str


@asset(group_name="injury_reports")
def injury_pdf_archive(
    config: CaptureConfig,
    official_injury_reports: OfficialInjuryReports,
    injury_report_archive: InjuryReportArchive,
) -> MaterializeResult:
    report = official_injury_reports.download_latest(date.fromisoformat(config.publication_date))
    return MaterializeResult(metadata=injury_report_archive.save(report))
