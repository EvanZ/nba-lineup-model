"""Reconcile the official games for one Pacific slate from archived evidence."""

import json
from datetime import UTC, date, datetime
from hashlib import sha256

from dagster import (
    AllPartitionMapping, AssetDep, AssetExecutionContext, AssetKey, Config, Failure, MaterializeResult, asset,
)

from ...resources import NBAPlayerNamesResource, NBAScheduleResource
from ...rosters.resources import NBARosterArchive
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
    reconcile_games,
    select_extractions,
    select_prior_failures,
    write_identity_events,
    write_identity_warnings,
)
from .extract import injury_report_extractions


class S3Input(Config):
    uri: str
    sha256: str


class ReconciliationSelection(Config):
    extraction: S3Input | None = None
    extractions: dict[str, S3Input] | None = None
    live: bool = False
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
    return {
        **{key: inputs[key] for key in ("schedule", "extraction", "extractions") if key in inputs},
        "live": record["purpose"] == "live_candidate",
    }, errors


def _advance_latest(
    archive: InjuryReportArchive,
    nba_schedule: NBAScheduleResource,
    nba_roster_archive: NBARosterArchive,
    nba_player_names: NBAPlayerNamesResource,
    games: list[dict],
    status_record: dict,
    status_ref: dict,
    extraction_refs: dict,
) -> dict:
    """Accept the full slate only while its pinned inputs are still selected."""
    day = status_record["game_date_pacific"]
    inputs = status_record["inputs"]
    candidate = {**status_ref, "selection_sha256": status_record["selection_sha256"]}
    prefix = f"{STATUSES_PREFIX}/season={archive.season}/game_date_pacific={day}/"

    for _ in range(3):
        previous, etag = archive.read_latest_status(day)
        if previous and all(previous[key] == value for key, value in candidate.items()):
            return {"latest_update": "already_current", "latest_status": previous}

        _, current_schedule = nba_schedule.load()
        if current_schedule != inputs["schedule"]:
            return {"latest_update": "schedule_changed"}
        if archive.read_player_repairs()["sha256"] != inputs["player_repairs"]["sha256"]:
            return {"latest_update": "player_repairs_changed"}
        _, current_roster = nba_roster_archive.latest_selection(archive.season, date.fromisoformat(day))
        if current_roster != inputs["rosters"]:
            return {"latest_update": "roster_changed"}
        _, current_names = nba_player_names.load()
        if current_names != inputs["player_names"]:
            return {"latest_update": "player_names_changed"}

        selected = select_extractions(
            archive, games, first_date=injury_publication_dates.start.date(),
            now=datetime.now(UTC), committed_only=True,
        )
        if {game_id: ref for game_id, (_, ref) in selected.items()} != extraction_refs:
            return {"latest_update": "reports_changed"}

        old = json.loads(archive.read_pinned(previous["uri"], previous["sha256"], prefix=prefix)) if previous else {}
        old_games = old.get("games", {})
        old_inputs = old.get("inputs", {})
        old_refs = old_inputs.get("extractions", dict.fromkeys(old_games, old_inputs.get("extraction")))
        current_games = status_record["games"]
        changed = sorted(
            game_id for game_id in old_games.keys() | current_games.keys()
            if old_games.get(game_id) != current_games.get(game_id)
            or old_refs.get(game_id) != extraction_refs.get(game_id)
        )
        pointer = {**candidate, "changed_game_ids": changed}
        if archive.save_latest_status(day, pointer, etag):
            return {"latest_update": "published", "latest_status": pointer}

    raise Failure(f"Concurrent updates prevented publishing latest injury status for {day}.", allow_retries=False)


@asset(
    # A Pacific slate can draw on reports from multiple Eastern publication dates.
    deps=[
        AssetDep(injury_report_extractions, partition_mapping=AllPartitionMapping()),
        AssetDep(AssetKey("rosters_by_date"), partition_mapping=AllPartitionMapping()),
    ],
    group_name="injury_reports",
    partitions_def=injury_game_dates,
    code_version=RESOLVER_VERSION,
)
def injury_status_by_game_date(
    context: AssetExecutionContext,
    config: ReconciliationSelection,
    injury_report_archive: InjuryReportArchive,
    nba_schedule: NBAScheduleResource,
    nba_roster_archive: NBARosterArchive,
    nba_player_names: NBAPlayerNamesResource,
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
        if extraction_refs is not None:
            raise ValueError("Select either a single report or per-game reports.")
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
    if (nba_roster_archive.bucket != BUCKET
            or nba_player_names.bucket != BUCKET
            or nba_player_names.season != season):
        raise ValueError("Injury, roster, and player-name resources must use the same bucket and season.")
    roster_day, _ = nba_roster_archive.latest_selection(season, day)
    roster_record, roster_inputs = nba_roster_archive.load(season, roster_day)
    player_names, names_input = nba_player_names.load()
    if required - set(map(int, roster_record["teams"])):
        raise ValueError("The dated roster is missing a team in the official slate.")
    rosters = {
        team: {pid: player_names[pid] for value in roster_record["teams"][str(team)]
               if (pid := int(value)) in player_names}
        for team in required
    }

    repairs = injury_report_archive.read_player_repairs()
    statuses = {}
    for game in games:
        report, _ = reports.get(game["game_id"], (None, None))
        statuses.update(reconcile_games([game], rosters, report, repairs["entries"], player_names))
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
        "purpose": "live_candidate" if selected.get("live") else "manual_diagnostic",
        "inputs": {
            "schedule": schedule_ref,
            "rosters": roster_inputs,
            "player_names": names_input,
            **extraction_inputs,
            "player_repairs": repairs,
        },
    }
    selection = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    roster_dates = [roster_record["game_date_pacific"]]
    complete_games = sum(game["complete"] for game in statuses.values())
    status_record = {
        **identity,
        "selection_sha256": selection,
        "complete": complete_games == len(statuses),
        "roster_dates_pacific": roster_dates,
        "games": statuses,
    }
    status_ref = injury_report_archive.save_statuses(status_record)
    failures, recoveries = write_identity_events(
        injury_report_archive, rows, prior, occurred_at=datetime.now(UTC),
        run_id=context.run_id, game_date=day.isoformat(),
        extractions=extraction_refs, artifact_path=status_ref["uri"],
    )
    warnings = write_identity_warnings(
        injury_report_archive, rows, occurred_at=datetime.now(UTC), run_id=context.run_id,
        game_date=day.isoformat(), extractions=extraction_refs,
        artifact_path=status_ref["uri"], inputs=status_record["inputs"],
    )
    for row in rows.values():
        if row.get("identity_warning"):
            context.log.warning(
                f"{row['player_name']} ({row['player_id']}) resolved via PlayerIndex but absent "
                f"from {row['team']}'s selected roster for game {row['game_id']}; "
                "retained the report's team/status. See the warning artifact in run metadata."
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
        "warnings": warnings,
        "warning_count": len(warnings),
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
    if missing_reports:
        metadata["latest_update"] = "incomplete_coverage"
    else:
        metadata.update(_advance_latest(
            injury_report_archive, nba_schedule, nba_roster_archive, nba_player_names,
            games, status_record, status_ref, extraction_refs,
        ))
    return MaterializeResult(metadata=metadata)
