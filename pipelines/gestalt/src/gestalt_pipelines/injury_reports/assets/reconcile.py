"""Materialize a candidate injury selection from exact S3 inputs."""

import json
from datetime import UTC, date, datetime
from hashlib import sha256

from dagster import (
    AllPartitionMapping, AssetDep, AssetExecutionContext, Config, Failure, MaterializeResult, asset,
)

from ...resources import NBAScheduleResource
from ..partitions import injury_game_dates
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
    select_prior_failures,
    write_identity_events,
)
from .extract import injury_report_extractions


class S3Input(Config):
    uri: str
    sha256: str


class ReconciliationSelection(Config):
    roster_metadata: list[S3Input]
    extraction: S3Input
    schedule: S3Input | None = None


def _select_inputs(
    context: AssetExecutionContext, selection: ReconciliationSelection, archive: InjuryReportArchive,
) -> tuple[ReconciliationSelection, list[dict]]:
    parent_run_id = context.run.parent_run_id
    if parent_run_id is None:
        return selection, []

    day = context.partition_key
    reference, errors = read_failed_run(context.instance, parent_run_id, day)
    prefix = f"{STATUSES_PREFIX}/season={archive.season}/game_date_pacific={day}/"
    record = json.loads(archive.read_pinned(reference["uri"], reference["sha256"], prefix=prefix))
    if record["game_date_pacific"] != day or record["season"] != archive.season:
        raise ValueError("The failed run's saved result belongs to a different season or date.")
    inputs = record["inputs"]
    return ReconciliationSelection(
        schedule=inputs["schedule"],
        extraction=inputs["extraction"], roster_metadata=[item["metadata"] for item in inputs["rosters"].values()],
    ), errors


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
    games, schedule_ref = nba_schedule.games_for_date(
        day, selected.schedule.model_dump() if selected.schedule else None,
    )
    now = datetime.now(UTC)
    if context.run.parent_run_id is None and any(game["game_time_utc"] <= now for game in games):
        raise ValueError("This manual slice uses scheduled starts for an unstarted slate.")

    rosters = {}
    roster_inputs = {}
    for ref in selected.roster_metadata:
        team_id, players, evidence = read_roster(injury_report_archive, ref.model_dump(), season)
        if team_id in rosters or datetime.fromisoformat(evidence["fetched_at"]) > now:
            raise ValueError(f"Duplicate or future roster evidence for team {team_id}.")
        rosters[team_id] = players
        roster_inputs[str(team_id)] = evidence
    required = {game[field] for game in games for field in ("away_team_id", "home_team_id")}
    if required != rosters.keys():
        raise ValueError("Select exactly one official roster for each team in the slate.")

    extraction = injury_report_archive.read_pinned(
        selected.extraction.uri, selected.extraction.sha256, prefix=f"{EXTRACTION_PREFIX}/season={season}/",
    )
    report = read_extraction(extraction, selected.extraction.uri, season)
    if datetime.fromisoformat(report["published_at"]) > now:
        raise ValueError("The selected injury report has a future publication time.")

    repairs = injury_report_archive.read_player_repairs()
    statuses = reconcile_games(games, rosters, report, repairs["entries"])
    rows = identity_rows(statuses)
    for row in rows.values():
        name = row["player_name"]
        if row.get("identity_source") == "roster" and repairs["entries"].get(name) == row["player_id"]:
            context.log.info(f"{name} resolved directly from the roster. Its repair dictionary entry can be removed.")
    extraction_ref = selected.extraction.model_dump()
    prior = select_prior_failures(
        injury_report_archive, error_refs,
        rows, day.isoformat(), extraction_ref,
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
            "extraction": extraction_ref,
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
        extraction=extraction_ref, artifact_path=status_ref["uri"],
    )
    metadata = {
        "statuses": status_ref,
        "selection_sha256": selection,
        "game_count": len(statuses),
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
    return MaterializeResult(metadata=metadata)
