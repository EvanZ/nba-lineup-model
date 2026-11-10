"""Extract the archived NBA injury reports for one publication date."""

from datetime import date
from time import sleep

from botocore.exceptions import ClientError, ConnectionError as S3ConnectionError, ReadTimeoutError
from dagster import AssetExecutionContext, Failure, Field, MaterializeResult, Selector, asset

from ..partitions import injury_publication_dates
from ..resources import InjuryReportArchive
from ..utils import PARSER_VERSION, parse_pdf, read_failed_extraction, write_extraction_event
from .capture import injury_pdf_archive


def _temporary_error(error: Exception) -> bool:
    if isinstance(error, ClientError):
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        code = error.response.get("Error", {}).get("Code")
        return status == 429 or status >= 500 or code in {
            "InternalError", "RequestTimeout", "ServiceUnavailable", "SlowDown",
        }
    return isinstance(error, (S3ConnectionError, ReadTimeoutError, ConnectionError, TimeoutError))


def _extract_capture(
    context: AssetExecutionContext,
    archive: InjuryReportArchive,
    capture_uri: str,
    publication_date: date,
    prior_failure: tuple[dict, dict] | None = None,
) -> dict:
    if f"/publication_date={publication_date}/" not in capture_uri:
        raise ValueError(f"Select the publication-date partition for this capture URI ({publication_date} selected).")
    archive.extraction_uri(capture_uri, PARSER_VERSION)
    pdf_uri = capture_uri.removesuffix("/capture.json") + "/report.pdf"
    work = {
        "capture_uri": capture_uri,
        "pdf_uri": pdf_uri,
        "sha256": capture_uri.rsplit("/sha256=", 1)[1].split("/", 1)[0],
        "parser_version": PARSER_VERSION,
    }
    event_context = {"run_id": context.run_id, "publication_date": publication_date.isoformat(), "work": work}

    reported_dates = None
    for attempt in range(3):
        try:
            capture, pdf = archive.load(capture_uri)
            report = parse_pdf(pdf)
            game_reports = report["game_team_reports"]
            if not game_reports:
                raise ValueError("No reported game coverage could be extracted.")
            reported_dates = sorted({key.split("|", 1)[0] for key in game_reports})
            record = {
                "schema_version": 1,
                "season": archive.season,
                **work,
                "source_url": capture["source_url"],
                "published_at": capture["published_at"],
                "reported_game_dates": reported_dates,
                **report,
            }
            extraction_uri, created = archive.save_extraction(record)
        except Exception as error:
            if attempt < 2 and _temporary_error(error):
                context.log.warning(f"Temporary extraction error for {capture_uri}. Retrying the same PDF.")
                sleep(15)
                continue
            message = f"Could not extract the saved injury PDF. {error}"
            metadata = {"capture_uri": capture_uri, "parser_version": PARSER_VERSION}
            if prior_failure and prior_failure[0] == work:
                metadata["extraction_failure"] = prior_failure[1]
            else:
                try:
                    metadata["extraction_failure"] = write_extraction_event(
                        archive, **event_context, message=message, artifact_path=pdf_uri,
                        reported_game_dates=reported_dates,
                    )
                except Exception as write_error:
                    message = f"{message} Could not save the S3 error record. {write_error}"
            raise Failure(message, metadata=metadata, allow_retries=False) from error
        break

    recoveries = []
    if prior_failure:
        try:
            recoveries.append(write_extraction_event(
                archive, **event_context, message="Extracted the saved injury PDF successfully.",
                artifact_path=extraction_uri, reported_game_dates=reported_dates, original=prior_failure[1],
            ))
        except Exception as error:
            raise Failure(
                f"The extraction is saved. Could not write its recovery record. {error}",
                metadata={"extraction_failure": prior_failure[1], "extraction_uri": extraction_uri},
                allow_retries=False,
            ) from error
    return {
        **work,
        "extraction_uri": extraction_uri,
        "reported_game_dates": reported_dates,
        "game_count": len(game_reports),
        "player_count": len(report["players"]),
        "created": created,
        "recoveries": recoveries,
    }


@asset(
    deps=[injury_pdf_archive],
    group_name="injury_reports",
    partitions_def=injury_publication_dates,
    code_version=PARSER_VERSION,
    config_schema=Field(
        Selector({"publication_date": str, "capture_uri": str, "repair_failed_run_id": str}),
        is_required=False,
    ),
)
def injury_report_extractions(
    context: AssetExecutionContext,
    injury_report_archive: InjuryReportArchive,
) -> MaterializeResult:
    publication_date = date.fromisoformat(context.partition_key)
    config = context.op_config or {}
    replay = "repair_failed_run_id" in config
    prior_failure = None
    if context.run.parent_run_id:
        prior_failure = read_failed_extraction(
            context.instance, injury_report_archive, context.run.parent_run_id, publication_date.isoformat(),
        )
    if replay and prior_failure is None:
        prior_failure = read_failed_extraction(
            context.instance, injury_report_archive, config["repair_failed_run_id"], publication_date.isoformat(),
        )
        if prior_failure is None:
            raise ValueError("Select a failed extraction run with a saved error for this publication date.")

    expected_version = prior_failure[0]["parser_version"] if prior_failure and not replay else PARSER_VERSION
    requested_version = context.run.tags.get("injury_reports/parser_version", expected_version)
    if requested_version != PARSER_VERSION:
        metadata = {"extraction_failure": prior_failure[1]} if prior_failure else {}
        raise Failure(
            f"This extraction requested parser {requested_version}, but {PARSER_VERSION} is loaded.",
            metadata=metadata,
            allow_retries=False,
        )

    recovered = None
    if prior_failure:
        recovered = _extract_capture(
            context, injury_report_archive, prior_failure[0]["capture_uri"], publication_date, prior_failure,
        )
        if config:
            return MaterializeResult(metadata=recovered)
    elif config:
        if "publication_date" in config:
            if date.fromisoformat(config["publication_date"]) != publication_date:
                raise ValueError("Select the same Eastern publication date as the partition.")
            uri = injury_report_archive.latest_capture_uri(publication_date)
        else:
            uri = config["capture_uri"].replace("%3D", "=").replace("%3d", "=")
        return MaterializeResult(metadata=_extract_capture(context, injury_report_archive, uri, publication_date))

    captures = injury_report_archive.captures_for_date(publication_date)
    if not captures:
        raise FileNotFoundError(f"No saved NBA injury report capture for {publication_date}.")
    extracted = injury_report_archive.extraction_uris(publication_date, PARSER_VERSION)
    created = int(recovered["created"]) if recovered else 0
    for _, _, uri in captures:
        if injury_report_archive.extraction_uri(uri, PARSER_VERSION) not in extracted:
            created += _extract_capture(context, injury_report_archive, uri, publication_date)["created"]
    return MaterializeResult(metadata={
        "publication_date": publication_date.isoformat(),
        "parser_version": PARSER_VERSION,
        "capture_count": len(captures),
        "created_count": created,
        "reused_count": len(captures) - created,
        "recoveries": recovered["recoveries"] if recovered else [],
    })
