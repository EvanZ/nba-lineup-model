"""Capture missing official NBA PDFs for one publication-date partition."""

from datetime import date

from dagster import AssetExecutionContext, Failure, MaterializeResult, asset

from ..partitions import injury_publication_dates
from ..resources import InjuryReportArchive, OfficialInjuryReports


@asset(group_name="injury_reports", partitions_def=injury_publication_dates)
def injury_pdf_archive(
    context: AssetExecutionContext,
    official_injury_reports: OfficialInjuryReports,
    injury_report_archive: InjuryReportArchive,
) -> MaterializeResult:
    publication_date = date.fromisoformat(context.partition_key)
    captured = injury_report_archive.captured_publication_times(publication_date)
    reused_count = 0
    recovered = []
    unavailable = []
    for instants, url in official_injury_reports.publication_slots(publication_date):
        if captured.intersection(instants):
            reused_count += 1
            continue
        report = official_injury_reports.download_if_available(url)
        if report is None:
            unavailable.extend(instant.isoformat() for instant in instants)
            continue
        capture = injury_report_archive.save(report)
        if capture["created"]:
            recovered.append(capture["capture_uri"])
        else:
            reused_count += 1

    metadata = {
        "publication_date": publication_date.isoformat(),
        "reused_count": reused_count,
        "recovered_count": len(recovered),
        "recovered_capture_uris": recovered,
        "unavailable_count": len(unavailable),
        "unavailable_slots_utc": unavailable,
    }
    if not recovered and reused_count == 0:
        raise Failure(f"No official NBA injury report found for {publication_date}.", metadata=metadata)
    return MaterializeResult(metadata=metadata)
