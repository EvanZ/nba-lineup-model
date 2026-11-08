"""Extract one exact archived NBA injury report."""

from dagster import Config, MaterializeResult, asset

from ..resources.archive import InjuryReportArchive
from ..utils.parse_pdf import PARSER_VERSION, parse_pdf
from .capture import injury_pdf_archive


class ExtractionConfig(Config):
    capture_uri: str


@asset(deps=[injury_pdf_archive], group_name="injury_reports", code_version=PARSER_VERSION)
def injury_report_extractions(
    config: ExtractionConfig,
    injury_report_archive: InjuryReportArchive,
) -> MaterializeResult:
    capture, pdf = injury_report_archive.load(config.capture_uri)
    report = parse_pdf(pdf)
    game_reports = report["game_team_reports"]
    if not game_reports:
        raise ValueError(f"No reported game coverage could be extracted: {config.capture_uri}")
    reported_dates = sorted({key.split("|", 1)[0] for key in game_reports})
    record = {
        "schema_version": 1,
        "parser_version": PARSER_VERSION,
        "capture_uri": config.capture_uri,
        "pdf_uri": capture["immutable_uri"],
        "sha256": capture["sha256"],
        "source_url": capture["source_url"],
        "published_at": capture["published_at"],
        "reported_game_dates": reported_dates,
        **report,
    }
    extraction_uri, created = injury_report_archive.save_extraction(record)
    return MaterializeResult(metadata={
        "capture_uri": config.capture_uri,
        "pdf_uri": record["pdf_uri"],
        "extraction_uri": extraction_uri,
        "sha256": record["sha256"],
        "parser_version": PARSER_VERSION,
        "reported_game_dates": reported_dates,
        "game_count": len(game_reports),
        "player_count": len(report["players"]),
        "created": created,
    })
