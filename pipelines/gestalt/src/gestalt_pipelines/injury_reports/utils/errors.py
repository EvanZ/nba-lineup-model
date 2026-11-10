"""Write immutable injury processing failures and linked recoveries."""

import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from zoneinfo import ZoneInfo

from dagster import DagsterEventType, DagsterRunStatus

from ..resources import BUCKET, InjuryReportArchive

ERROR_PREFIX = "lake/artifacts/errors/pipeline=injury_reports/"
STAGE = "injury_status_by_game_date"
EXTRACTION_STAGE = "injury_report_extractions"
PACIFIC = ZoneInfo("America/Los_Angeles")


def row_key(row: dict) -> tuple[str, str, int]:
    return row["game_id"], row["team_id"], row["source_row_index"]


def _event_key(event: dict) -> str:
    occurred = datetime.fromisoformat(event["timestamp"]).astimezone(PACIFIC).date()
    suffix = {"failed": "failure", "recovered": "recovery"}[event["status"]]
    return (
        f"{ERROR_PREFIX}season={event['season']}/occurred_date_pacific={occurred}/"
        f"{event['event_id']}.{suffix}.json"
    )


def _read_failure(
    archive: InjuryReportArchive, ref: dict, game_date: str, extractions: dict[str, dict],
) -> dict:
    prefix = f"{ERROR_PREFIX}season={archive.season}/"
    raw = archive.read_pinned(ref["uri"], ref["sha256"], prefix=prefix)
    event = json.loads(raw)
    if (
        event["schema_version"] != 1 or event["stage"] != STAGE or event["status"] != "failed"
        or event["season"] != archive.season
        or event["game_date_pacific"] != game_date
        or event["extraction"] != extractions.get(event["row"]["game_id"])
        or not re.fullmatch(r"[0-9a-f]{64}", event["event_id"])
        or datetime.fromisoformat(event["timestamp"]).tzinfo is None
        or ref["uri"] != f"s3://{BUCKET}/{_event_key(event)}"
        or "espn" in raw.decode().casefold()
    ):
        raise ValueError("The failure record does not belong to the selected date and extraction.")
    return event


def _save_event(
    archive: InjuryReportArchive, *, occurred_at: datetime, run_id: str, game_date: str,
    extraction: dict, artifact_path: str, row: dict, original: dict | None = None,
) -> dict:
    status = "recovered" if original else "failed"
    identity = [run_id, row_key(row), status, original]
    event_id = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if original:
        message = f"Resolved {row['player_name']} ({row['team']}) to NBA player ID {row['player_id']}."
    elif row.get("identity_status") == "conflicting_repair":
        message = f"The repair dictionary disagrees with the roster for {row['player_name']} ({row['team']})."
    else:
        message = f"Could not uniquely match {row['player_name']} ({row['team']}) to an NBA roster player."
    event = {
        "schema_version": 1,
        "event_id": event_id,
        "stage": STAGE,
        "season": archive.season,
        "status": status,
        "timestamp": occurred_at.isoformat(),
        "message": message,
        "dagster_run_id": run_id,
        "game_date_pacific": game_date,
        "extraction": extraction,
        "artifact_path": artifact_path,
        "row": row,
    }
    if original:
        event["original_failure"] = original
    return archive.save_json(_event_key(event), event)


def identity_rows(games: dict) -> dict[tuple[str, str, int], dict]:
    rows = {}
    for game_id, game in games.items():
        for team_id, team in game["teams"].items():
            for player_id, row in team["players"].items():
                evidence = {**row, "game_id": game_id, "team_id": team_id, "player_id": int(player_id)}
                rows[row_key(evidence)] = evidence
            for row in team["unresolved_players"]:
                evidence = {**row, "game_id": game_id, "team_id": team_id}
                rows[row_key(evidence)] = evidence
    return rows


def select_prior_failures(
    archive: InjuryReportArchive, references: list[dict], rows: dict,
    game_date: str, extractions: dict[str, dict],
) -> dict[tuple[str, str, int], dict]:
    prior = {}
    for ref in references:
        failure = _read_failure(archive, ref, game_date, extractions)
        source = failure["row"]
        key = row_key(source)
        row = rows.get(key)
        if (
            key in prior or row is None
            or any(row[field] != source[field] for field in ("player_name", "team", "pdf_uri"))
        ):
            raise ValueError("A prior failure does not uniquely identify a row in this selection.")
        prior[key] = {**ref, "event_id": failure["event_id"]}
    return prior


def write_identity_events(
    archive: InjuryReportArchive, rows: dict, prior: dict, *, occurred_at: datetime,
    run_id: str, game_date: str, extractions: dict[str, dict], artifact_path: str,
) -> tuple[list[dict], list[dict]]:
    open_failures = []
    recoveries = []
    common = {
        "occurred_at": occurred_at,
        "run_id": run_id,
        "game_date": game_date,
        "artifact_path": artifact_path,
    }
    for key, row in rows.items():
        original = prior.get(key)
        extraction = extractions[row["game_id"]]
        if "identity_status" in row:
            ref = {k: original[k] for k in ("uri", "sha256")} if original else _save_event(archive, **common, extraction=extraction, row=row)
            open_failures.append(ref)
        elif original:
            recoveries.append(_save_event(archive, **common, extraction=extraction, row=row, original=original))
    return open_failures, recoveries


def read_failed_run(instance, run_id: str, game_date: str) -> tuple[dict, list[dict]] | None:
    """Read the selected partition's pinned diagnostic status and open errors."""
    run = instance.get_run_by_id(run_id)
    if (
        run is None or run.job_name != "reconcile_injury_date"
        or run.status != DagsterRunStatus.FAILURE
        or run.tags.get("dagster/partition") != game_date
    ):
        raise ValueError("Select a failed reconciliation run for this Pacific game date.")

    records = instance.get_records_for_run(run_id, of_type=DagsterEventType.STEP_FAILURE).records
    for record in reversed(records):
        event = record.event_log_entry.dagster_event
        if event.step_key != STAGE:
            continue
        data = event.step_failure_data.user_failure_data
        if data and "statuses" in data.metadata and "open_failures" in data.metadata:
            if failures := data.metadata["open_failures"].value:
                return data.metadata["statuses"].value, failures
    return None


def write_extraction_event(
    archive: InjuryReportArchive, *, run_id: str, publication_date: str, work: dict,
    message: str, artifact_path: str, reported_game_dates: list[str] | None,
    original: dict | None = None,
) -> dict:
    status = "recovered" if original else "failed"
    identity = [run_id, EXTRACTION_STAGE, work, status, original]
    event = {
        "schema_version": 1,
        "event_id": sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "stage": EXTRACTION_STAGE,
        "season": archive.season,
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
        "message": message,
        "dagster_run_id": run_id,
        "publication_date": publication_date,
        "input": work,
        "reported_game_dates": reported_game_dates,
        "artifact_path": artifact_path,
    }
    if original:
        event["original_failure"] = original
    return archive.save_json(_event_key(event), event)


def read_failed_extraction(
    instance, archive: InjuryReportArchive, run_id: str, publication_date: str,
) -> tuple[dict, dict] | None:
    run = instance.get_run_by_id(run_id)
    if (
        run is None or run.job_name != "extract_injury_pdf"
        or run.status != DagsterRunStatus.FAILURE
        or run.tags.get("dagster/partition") != publication_date
    ):
        return None
    records = instance.get_records_for_run(run_id, of_type=DagsterEventType.STEP_FAILURE).records
    for record in reversed(records):
        event = record.event_log_entry.dagster_event
        if event.step_key != EXTRACTION_STAGE:
            continue
        data = event.step_failure_data.user_failure_data
        if not data or "extraction_failure" not in data.metadata:
            continue
        ref = data.metadata["extraction_failure"].value
        prefix = f"{ERROR_PREFIX}season={archive.season}/"
        raw = archive.read_pinned(ref["uri"], ref["sha256"], prefix=prefix)
        failure = json.loads(raw)
        if (
            failure["schema_version"] != 1 or failure["stage"] != EXTRACTION_STAGE
            or failure["status"] != "failed" or failure["season"] != archive.season
            or failure["publication_date"] != publication_date
            or not re.fullmatch(r"[0-9a-f]{64}", failure["event_id"])
            or datetime.fromisoformat(failure["timestamp"]).tzinfo is None
            or ref["uri"] != f"s3://{BUCKET}/{_event_key(failure)}"
            or b"espn" in raw.lower()
        ):
            raise ValueError("The extraction failure record does not match the selected partition.")
        return failure["input"], {**ref, "event_id": failure["event_id"]}
    return None
