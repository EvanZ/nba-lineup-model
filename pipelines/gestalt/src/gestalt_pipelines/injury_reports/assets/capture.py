"""Dagster asset for a single manually selected injury report."""

from dagster import Config, MaterializeResult, asset

from ..resources.archive import InjuryReportArchive
from ..resources.official_nba import OfficialInjuryReports


class CaptureConfig(Config):
    source_url: str
    expected_sha256: str


@asset(group_name="injury_reports")
def injury_pdf_archive(
    config: CaptureConfig,
    official_injury_reports: OfficialInjuryReports,
    injury_report_archive: InjuryReportArchive,
) -> MaterializeResult:
    report = official_injury_reports.download(config.source_url)
    return MaterializeResult(metadata=injury_report_archive.save(report, config.expected_sha256))
