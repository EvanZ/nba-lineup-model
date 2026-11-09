"""Extract the archived NBA injury reports for one publication date."""

from datetime import date

from dagster import AssetExecutionContext, Field, MaterializeResult, Selector, asset

from ..partitions import injury_publication_dates
from ..resources import InjuryReportArchive
from ..utils import PARSER_VERSION, parse_pdf
from .capture import injury_pdf_archive


def _extract_capture(
    injury_report_archive: InjuryReportArchive, capture_uri: str, publication_date: date,
) -> dict:
    if f"/publication_date={publication_date}/" not in capture_uri:
        raise ValueError(f"Select the publication-date partition for this capture URI ({publication_date} selected).")
    capture, pdf = injury_report_archive.load(capture_uri)
    report = parse_pdf(pdf)
    game_reports = report["game_team_reports"]
    if not game_reports:
        raise ValueError(f"No reported game coverage could be extracted: {capture_uri}")
    reported_dates = sorted({key.split("|", 1)[0] for key in game_reports})
    record = {
        "schema_version": 1,
        "season": injury_report_archive.season,
        "parser_version": PARSER_VERSION,
        "capture_uri": capture_uri,
        "pdf_uri": capture["immutable_uri"],
        "sha256": capture["sha256"],
        "source_url": capture["source_url"],
        "published_at": capture["published_at"],
        "reported_game_dates": reported_dates,
        **report,
    }
    extraction_uri, created = injury_report_archive.save_extraction(record)
    return {
        "capture_uri": capture_uri,
        "pdf_uri": record["pdf_uri"],
        "extraction_uri": extraction_uri,
        "sha256": record["sha256"],
        "parser_version": PARSER_VERSION,
        "reported_game_dates": reported_dates,
        "game_count": len(game_reports),
        "player_count": len(report["players"]),
        "created": created,
    }


@asset(
    deps=[injury_pdf_archive],
    group_name="injury_reports",
    partitions_def=injury_publication_dates,
    code_version=PARSER_VERSION,
    config_schema=Field(Selector({"publication_date": str, "capture_uri": str}), is_required=False),
)
def injury_report_extractions(
    context: AssetExecutionContext,
    injury_report_archive: InjuryReportArchive,
) -> MaterializeResult:
    publication_date = date.fromisoformat(context.partition_key)
    config = context.op_config or {}
    if config:
        if "publication_date" in config:
            if date.fromisoformat(config["publication_date"]) != publication_date:
                raise ValueError("Select the same Eastern publication date as the partition.")
            uri = injury_report_archive.latest_capture_uri(publication_date)
        else:
            uri = config["capture_uri"].replace("%3D", "=").replace("%3d", "=")
        return MaterializeResult(metadata=_extract_capture(injury_report_archive, uri, publication_date))

    captures = injury_report_archive.captures_for_date(publication_date)
    if not captures:
        raise FileNotFoundError(f"No saved NBA injury report capture for {publication_date}.")
    extracted = injury_report_archive.extraction_uris(publication_date, PARSER_VERSION)
    created = 0
    for _, _, uri in captures:
        if injury_report_archive.extraction_uri(uri, PARSER_VERSION) not in extracted:
            created += _extract_capture(injury_report_archive, uri, publication_date)["created"]
    return MaterializeResult(metadata={
        "publication_date": publication_date.isoformat(),
        "parser_version": PARSER_VERSION,
        "capture_count": len(captures),
        "created_count": created,
        "reused_count": len(captures) - created,
    })
