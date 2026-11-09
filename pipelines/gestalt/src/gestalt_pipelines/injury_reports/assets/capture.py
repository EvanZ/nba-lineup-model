"""Capture missing official NBA PDFs for one publication-date partition."""

from datetime import UTC, date, datetime, timedelta
from urllib.error import HTTPError, URLError

from dagster import AssetExecutionContext, Config, Failure, MaterializeResult, RetryRequested, asset

from ..partitions import injury_publication_dates
from ..resources import InjuryReportArchive, OfficialInjuryReports


class CaptureWindow(Config):
    window_end_utc: str | None = None


@asset(group_name="injury_reports", partitions_def=injury_publication_dates)
def injury_pdf_archive(
    context: AssetExecutionContext,
    config: CaptureWindow,
    official_injury_reports: OfficialInjuryReports,
    injury_report_archive: InjuryReportArchive,
) -> MaterializeResult:
    publication_date = date.fromisoformat(context.partition_key)
    end = datetime.fromisoformat(config.window_end_utc) if config.window_end_utc else None
    if end is not None and end.utcoffset() != timedelta(0):
        raise ValueError("The capture window must end at a UTC timestamp.")
    start = end.astimezone(UTC) - timedelta(hours=1) if end is not None else None
    captured = injury_report_archive.captured_publication_times(publication_date)
    reused_count = 0
    recovered = []
    unavailable = []
    for instants, url in official_injury_reports.publication_slots(publication_date):
        if end is not None:
            instants = tuple(instant for instant in instants if start < instant <= end)
        if not instants:
            continue
        if captured.intersection(instants):
            reused_count += 1
            continue
        try:
            report = official_injury_reports.download_if_available(url)
        except (URLError, TimeoutError) as error:
            temporary = not isinstance(error, HTTPError) or error.code == 429 or error.code >= 500
            if end is None or not temporary:
                raise
            raise RetryRequested(max_retries=2, seconds_to_wait=15) from error
        if report is None or report.published_at not in instants:
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
    if end is not None:
        metadata["window_end_utc"] = end.isoformat()
    if end is None and not recovered and reused_count == 0:
        raise Failure(f"No official NBA injury report found for {publication_date}.", metadata=metadata)
    return MaterializeResult(metadata=metadata)
