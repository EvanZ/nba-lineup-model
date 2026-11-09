"""Reconcile the official games for one Pacific slate from archived evidence."""

import json
from datetime import UTC, date, datetime
from hashlib import sha256

from dagster import (
    AllPartitionMapping, AssetDep, AssetExecutionContext, Config, Failure, MaterializeResult, asset,
)

from ...resources import NBAScheduleResource
from ..partitions import injury_game_dates, injury_publication_dates
from ..resources import (
    BUCKET,
    EXTRACTION_PREFIX,
    STATUSES_PREFIX,
    InjuryReportArchive,
)
from ..utils import (
    RESOLVER_VERSION,
    identity_rows,
    read_extraction,
    read_failed_run,
    read_roster,
    reconcile_games,
    select_extractions,
    select_prior_failures,
    write_identity_events,
)
from .extract import injury_report_extractions


class S3Input(Config):
    uri: str
    sha256: str


class ReconciliationSelection(Config):
    roster_metadata: list[S3Input] | None = None
    extraction: S3Input | None = None
    schedule: S3Input | None = None


def _select_inputs(
    context: AssetExecutionContext, selection: ReconciliationSelection, archive: InjuryReportArchive,
) -> tuple[dict, list[dict]]:
    day = context.partition_key
    parent_run_id = context.run.parent_run_id
    failed_run = read_failed_run(context.instance, parent_run_id, day) if parent_run_id else None
    if failed_run is None:
        return selection.model_dump(exclude_none=True), []

    reference, errors = failed_run
    prefix = f"{STATUSES_PREFIX}/season={archive.season}/game_date_pacific={day}/"
    record = json.loads(archive.read_pinned(reference["uri"], reference["sha256"], prefix=prefix))
    if record["game_date_pacific"] != day or record["season"] != archive.season:
        raise ValueError("The failed run's saved result belongs to a different season or date.")
    inputs = record["inputs"]
    return {**inputs, "roster_metadata": [item["metadata"] for item in inputs["rosters"].values()]}, errors


@asset(
    # A Pacific slate can draw on reports from multiple Eastern publication dates.
    deps=[AssetDep(injury_report_extractions, partition_mapping=AllPartitionMapping())],
    group_name="injury_reports",
    partitions_def=injury_game_dates,
    code_version=RESOLVER_VERSION,
)
def injury_status_by_game_date(
    context: AssetExecutionContext,
    config: ReconciliationSelection,
    injury_report_archive: InjuryReportArchive,
    nba_schedule: NBAScheduleResource,
) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    selected, error_refs = _select_inputs(context, config, injury_report_archive)
    season = injury_report_archive.season
    games, schedule_ref = nba_schedule.games_for_date(day, selected.get("schedule"), allow_empty=True)
    now = datetime.now(UTC)
    game_ids = {game["game_id"] for game in games}
    legacy_extraction = selected.get("extraction")
    extraction_refs = selected.get("extractions")
    if legacy_extraction is not None:
        extraction_refs = dict.fromkeys(game_ids, legacy_extraction)
    if extraction_refs is None:
        reports = select_extractions(
            injury_report_archive, games, first_date=injury_publication_dates.start.date(), now=now,
        )
        extraction_refs = {game_id: ref for game_id, (_, ref) in reports.items()}
    else:
        if extraction_refs.keys() - game_ids:
            raise ValueError("A selected extraction belongs to a game outside this slate.")
        reports = {}
        loaded = {}
        for game_id, ref in extraction_refs.items():
            key = (ref["uri"], ref["sha256"])
            if key not in loaded:
                raw = injury_report_archive.read_pinned(
                    *key, prefix=f"{EXTRACTION_PREFIX}/season={season}/",
                )
                report = read_extraction(raw, ref["uri"], season)
                if datetime.fromisoformat(report["published_at"]) > now:
                    raise ValueError("The selected injury report has a future publication time.")
                loaded[key] = report
            reports[game_id] = (loaded[key], ref)

    required = {game[field] for game in games for field in ("away_team_id", "home_team_id")}
    rosters = {}
    roster_inputs = {}
    for ref in selected.get("roster_metadata", sorted(required)):
        team_id, players, evidence = read_roster(injury_report_archive, ref, season)
        if team_id in rosters or datetime.fromisoformat(evidence["fetched_at"]) > now:
            raise ValueError(f"Duplicate or future roster evidence for team {team_id}.")
        rosters[team_id] = players
        roster_inputs[str(team_id)] = evidence
    if required != rosters.keys():
        raise ValueError("Select exactly one official roster for each team in the slate.")

    repairs = injury_report_archive.read_player_repairs()
    statuses = {}
    for game in games:
        report, _ = reports.get(game["game_id"], (None, None))
        statuses.update(reconcile_games([game], rosters, report, repairs["entries"]))
    rows = identity_rows(statuses)
    for row in rows.values():
        name = row["player_name"]
        if row.get("identity_source") == "roster" and repairs["entries"].get(name) == row["player_id"]:
            context.log.info(f"{name} resolved directly from the roster. Its repair dictionary entry can be removed.")
    prior = select_prior_failures(
        injury_report_archive, error_refs, rows, day.isoformat(), extraction_refs,
    )
    extraction_inputs = (
        {"extraction": legacy_extraction} if legacy_extraction is not None else {"extractions": extraction_refs}
    )
    identity = {
        "schema_version": 1,
        "game_date_pacific": day.isoformat(),
        "season": season,
        "resolver_version": RESOLVER_VERSION,
        "cutoff_source": "nba_scheduled_start",
        "purpose": "manual_diagnostic",
        "inputs": {
            "schedule": schedule_ref,
            "rosters": roster_inputs,
            **extraction_inputs,
            "player_repairs": repairs,
        },
    }
    selection = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    roster_dates = sorted({evidence["roster_date_pacific"] for evidence in roster_inputs.values()})
    complete_games = sum(game["complete"] for game in statuses.values())
    status_ref = injury_report_archive.save_statuses({
        **identity,
        "selection_sha256": selection,
        "complete": complete_games == len(statuses),
        "roster_dates_pacific": roster_dates,
        "games": statuses,
    })
    failures, recoveries = write_identity_events(
        injury_report_archive, rows, prior, occurred_at=datetime.now(UTC),
        run_id=context.run_id, game_date=day.isoformat(),
        extractions=extraction_refs, artifact_path=status_ref["uri"],
    )
    missing_reports = sorted(game_id for game_id, status in statuses.items() if not status["report_used"])
    metadata = {
        "statuses": status_ref,
        "selection_sha256": selection,
        "game_count": len(statuses),
        "missing_report_game_ids": missing_reports,
        "complete_game_count_against_selected_rosters": complete_games,
        "roster_dates_pacific": roster_dates,
        "open_failures": failures,
        "recoveries": recoveries,
    }
    if failures:
        raise Failure(
            f"{len(failures)} injury player identities need reconciliation for {day}.",
            metadata=metadata,
            allow_retries=False,
        )
    if missing_reports and legacy_extraction is None:
        raise Failure(f"No eligible injury report for {len(missing_reports)} games on {day}.", metadata=metadata)
    return MaterializeResult(metadata=metadata)
