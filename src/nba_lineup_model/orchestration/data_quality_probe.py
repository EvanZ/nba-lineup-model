"""Read-only environment probe for the daily data-quality report.

This process is intentionally outside Dagster.  It reads Dagster's persisted
run/automation state and the collector artifacts, so an orphaned Dagster run
cannot hide from the monitor that reports it.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from nba_lineup_model.season.data_quality_report import (
    build_data_quality_report,
    publish_data_quality_report,
    publish_failed_data_quality_report,
    reconcile_alert_state,
    utc_timestamp,
)
from nba_lineup_model.season.production_site_release import (
    verify_production_site_release,
)
from nba_lineup_model.season.release_publisher import read_active_activation

PACIFIC = ZoneInfo("America/Los_Angeles")
COLLECTOR_JOBS = {
    "schedule": ("refresh_schedule_window", "schedule_refresh_every_30_minutes"),
    "injury_reports": (
        "refresh_schedule_window",
        "schedule_refresh_every_30_minutes",
    ),
    "odds": ("archive_nba_market_odds", "nba_market_odds_game_day"),
    "official_scores": ("update_live_scores", "live_score_game_window"),
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def get_json(api_root: str, path: str) -> dict[str, Any]:
    url = api_root.rstrip("/") + path
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def file_observed_at(path: Path) -> str:
    if not path.exists():
        return ""
    return utc_timestamp(datetime.fromtimestamp(path.stat().st_mtime, tz=UTC))


def iso_from_sqlite(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return utc_timestamp(parsed)


def iso_from_epoch(value: object) -> str:
    if value is None:
        return ""
    return utc_timestamp(datetime.fromtimestamp(float(value), tz=UTC))


def official_schedule_snapshot(
    schedule_root: Path,
    *,
    report_date: str,
) -> dict[str, object]:
    """Derive expected coverage from official schedule files, not constants."""

    latest_dates: dict[str, dict[str, Any]] = {}
    for path in sorted((schedule_root / "latest").glob("*.json")):
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        game_date = str(payload.get("game_date") or path.stem)
        latest_dates[game_date] = payload

    completed_dates: list[str] = []
    scheduled_game_count = 0
    in_progress_game_count = 0
    today_games: list[dict[str, Any]] = []
    for game_date, payload in latest_dates.items():
        games = [game for game in payload.get("games", []) if isinstance(game, dict)]
        if games and all(bool((game.get("status") or {}).get("completed")) for game in games):
            completed_dates.append(game_date)
        if game_date == report_date:
            today_games = games
            scheduled_game_count = len(games)
            in_progress_game_count = sum(
                str((game.get("status") or {}).get("state") or "").lower() == "in"
                for game in games
            )

    final_ids: set[str] = set()
    for path in sorted((schedule_root / "final").glob("*.summary.json")):
        payload = read_json(path)
        game_id = str(payload.get("game_id") or "").strip()
        if not game_id:
            raise ValueError(f"Official final has no game ID: {path}")
        if game_id in final_ids:
            raise ValueError(f"Duplicate official final game ID: {game_id}")
        final_ids.add(game_id)

    latest_completed_date = max(completed_dates) if completed_dates else ""
    return {
        "latest_eligible_completed_date": latest_completed_date,
        "completed_game_ids": sorted(final_ids),
        "completed_games": len(final_ids),
        "scheduled_games_today": scheduled_game_count,
        "in_progress_games": in_progress_game_count,
        "zero_game_day": scheduled_game_count == 0,
        "today_game_ids": sorted(
            str(game.get("game_id"))
            for game in today_games
            if game.get("game_id") is not None
        ),
        "latest_day_count": len(latest_dates),
    }


def dagster_automation_states(schedules_db: Path) -> dict[str, str]:
    if not schedules_db.is_file():
        return {}
    connection = sqlite3.connect(f"file:{schedules_db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "select status, instigator_body from instigators"
        ).fetchall()
    finally:
        connection.close()
    result: dict[str, str] = {}
    for status, raw_body in rows:
        try:
            body = json.loads(raw_body)
            name = str((body.get("origin") or {}).get("job_name") or "")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if name:
            result[name] = str(status or "UNKNOWN")
    return result


def dagster_run_states(
    runs_db: Path,
    job_names: Sequence[str],
) -> dict[str, dict[str, object]]:
    if not runs_db.is_file():
        return {}
    connection = sqlite3.connect(f"file:{runs_db}?mode=ro", uri=True)
    try:
        output: dict[str, dict[str, object]] = {}
        for job_name in sorted(set(job_names)):
            latest = connection.execute(
                """
                select run_id, status, create_timestamp, start_time, end_time
                from runs where pipeline_name = ? order by id desc limit 1
                """,
                (job_name,),
            ).fetchone()
            success = connection.execute(
                """
                select end_time, update_timestamp
                from runs where pipeline_name = ? and status = 'SUCCESS'
                order by id desc limit 1
                """,
                (job_name,),
            ).fetchone()
            if not latest:
                continue
            output[job_name] = {
                "run_id": str(latest[0] or ""),
                "status": str(latest[1] or "MISSING"),
                "created_at": iso_from_sqlite(latest[2]),
                "started_at": iso_from_epoch(latest[3]),
                "finished_at": iso_from_epoch(latest[4]),
                "last_success_at": (
                    iso_from_epoch(success[0])
                    if success and success[0] is not None
                    else iso_from_sqlite(success[1])
                    if success
                    else ""
                ),
            }
        return output
    finally:
        connection.close()


def newest_schedule_artifact(schedule_root: Path) -> str:
    paths = list((schedule_root / "latest").glob("*.json"))
    if not paths:
        return ""
    return file_observed_at(max(paths, key=lambda path: path.stat().st_mtime))


def injury_artifact_observed_at(schedule_root: Path, report_date: str) -> str:
    candidates: list[datetime] = []
    selected = date.fromisoformat(report_date)
    for offset in range(0, 3):
        path = schedule_root / "latest" / f"{selected + timedelta(days=offset)}.json"
        if not path.is_file():
            continue
        payload = read_json(path)
        collected_at = str(payload.get("collected_at") or "")
        if collected_at:
            parsed = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
            candidates.append(parsed.astimezone(UTC))
        for game in payload.get("games", []):
            report = game.get("injury_report") if isinstance(game, dict) else None
            published_at = str((report or {}).get("published_at") or "")
            if published_at:
                parsed = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                candidates.append(parsed.astimezone(UTC))
    return utc_timestamp(max(candidates)) if candidates else ""


def odds_artifact_observed_at(schedule_root: Path) -> str:
    path = schedule_root / "odds-api" / "latest.json"
    if not path.is_file():
        return ""
    try:
        payload = read_json(path)
        if payload.get("collected_at"):
            return utc_timestamp(
                datetime.fromisoformat(
                    str(payload["collected_at"]).replace("Z", "+00:00")
                )
            )
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return file_observed_at(path)


def current_surface_snapshot(api_root: str, evidence_date: str) -> dict[str, object]:
    encoded_date = urllib.parse.quote(evidence_date)
    health = get_json(api_root, "/api/health")
    serving = get_json(api_root, "/api/serving-view")
    rankings = get_json(api_root, "/api/rankings?season=2026-27")
    teams = get_json(api_root, "/api/teams?season=2026-27")
    lineups = get_json(
        api_root,
        "/api/lineups?season=2026-27&minimum_possessions=0",
    )
    wins = get_json(api_root, "/api/win-projections")
    records = get_json(api_root, "/api/current-season-records")
    performance = get_json(api_root, "/api/schedule-performance")
    schedule = (
        get_json(api_root, f"/api/schedule?date={encoded_date}")
        if evidence_date
        else {"games": []}
    )
    season_window = next(
        (
            window
            for window in performance.get("windows", [])
            if window.get("key") == "season"
        ),
        {},
    )
    win_projection = wins.get("win_projection") or {}
    win_teams = win_projection.get("teams") or []
    return {
        "health": health,
        "serving": serving,
        "ranking_count": len(rankings.get("players") or []),
        "ranking_state": rankings.get("rating_state") or {},
        "team_count": len(teams.get("teams") or []),
        "lineup_count": len(lineups.get("lineups") or []),
        "lineup_state": lineups.get("lineup_state") or {},
        "win_team_count": len(win_teams),
        "wins_completed_team_games": sum(
            int(team.get("completed_games", 0)) for team in win_teams
        ),
        "wins_as_of": str(win_projection.get("as_of") or ""),
        "record_game_count": len(records.get("completed_game_ids") or []),
        "records_as_of": str(records.get("as_of") or records.get("through_date") or ""),
        "performance_game_count": int(season_window.get("games", -1)),
        "performance_as_of": str(performance.get("as_of") or ""),
        "schedule_game_count": len(schedule.get("games") or []),
    }


def activation_transition(
    activation: Mapping[str, object],
    bindings: Mapping[str, object],
) -> dict[str, object]:
    previous_path = str(activation.get("previous_release_path") or "")
    previous_bindings: dict[str, Any] = {}
    previous_edition = ""
    if previous_path:
        root = Path(previous_path)
        binding_path = root / "serving-bindings.json"
        activation_path = root / "activation.json"
        if binding_path.is_file():
            previous_bindings = read_json(binding_path)
        if activation_path.is_file():
            previous_edition = str(read_json(activation_path).get("edition_id") or "")
    current_components = dict(bindings.get("components") or {})
    previous_components = dict(previous_bindings.get("components") or {})
    changed = sorted(
        name
        for name in set(current_components) | set(previous_components)
        if current_components.get(name) != previous_components.get(name)
    )
    return {
        "previous_generation": activation.get("previous_generation"),
        "previous_edition_id": previous_edition,
        "current_edition_id": activation.get("edition_id"),
        "changed_components": changed,
        "forecast_selection_changed": (
            bool(previous_components)
            and current_components.get("forecast_selection_id")
            != previous_components.get("forecast_selection_id")
        ),
        "observed_data_revision_changed": (
            bool(previous_components)
            and current_components.get("observed_data_revision")
            != previous_components.get("observed_data_revision")
        ),
    }


def threshold_for(
    thresholds: Mapping[str, object],
    collector_name: str,
    field: str,
) -> float | None:
    row = thresholds.get(collector_name)
    if not isinstance(row, Mapping) or row.get(field) is None:
        return None
    return float(row[field])


def collector_rows(
    *,
    dagster_home: Path,
    schedule_root: Path,
    report_date: str,
    expected: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> list[dict[str, object]]:
    automation = dagster_automation_states(dagster_home / "schedules" / "schedules.db")
    jobs = dagster_run_states(
        dagster_home / "history" / "runs.db",
        [value[0] for value in COLLECTOR_JOBS.values()],
    )
    artifacts = {
        "schedule": newest_schedule_artifact(schedule_root),
        "injury_reports": injury_artifact_observed_at(schedule_root, report_date),
        "odds": odds_artifact_observed_at(schedule_root),
        "official_scores": file_observed_at(
            schedule_root / "pending" / "nba-live-data" / "scoreboard"
        ),
    }
    scheduled = int(expected.get("scheduled_games_today") or 0)
    in_progress = int(expected.get("in_progress_games") or 0)
    required = {
        "schedule": True,
        "injury_reports": scheduled > 0,
        "odds": scheduled > 0,
        "official_scores": in_progress > 0,
    }
    quiet_reasons = {
        "schedule": "",
        "injury_reports": "No games are scheduled today." if scheduled == 0 else "",
        "odds": "No games are scheduled today." if scheduled == 0 else "",
        "official_scores": (
            "No official NBA game is currently in progress."
            if in_progress == 0
            else ""
        ),
    }
    rows: list[dict[str, object]] = []
    for name, (job_name, automation_name) in COLLECTOR_JOBS.items():
        run = jobs.get(job_name, {})
        rows.append(
            {
                "name": name,
                "job_name": job_name,
                "automation_name": automation_name,
                "automation_status": automation.get(automation_name, "MISSING"),
                "latest_run_id": run.get("run_id", ""),
                "latest_run_status": run.get("status", "MISSING"),
                "latest_run_created_at": run.get("created_at", ""),
                "latest_run_started_at": run.get("started_at", ""),
                "latest_run_finished_at": run.get("finished_at", ""),
                "last_success_at": run.get("last_success_at", ""),
                "artifact_observed_at": artifacts[name],
                "required_now": required[name],
                "quiet_reason": quiet_reasons[name],
                "stale_after_minutes": threshold_for(
                    thresholds, name, "stale_after_minutes"
                ),
                "stall_after_minutes": threshold_for(
                    thresholds, name, "stall_after_minutes"
                ),
            }
        )
    return rows


def artifact_row(
    *,
    name: str,
    layer: str,
    artifact_id: object,
    checksum: object,
    evidence_date: str,
    revision: str,
    edition: str,
    coverage: Mapping[str, Mapping[str, int | None]],
    states: Mapping[str, bool | None],
    required_states: Sequence[str],
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "layer": layer,
        "artifact_id": str(artifact_id or ""),
        "checksum": str(checksum or ""),
        "evidence_through_date": evidence_date,
        "observed_data_revision": revision,
        "model_edition": edition,
        "coverage": dict(coverage),
        "states": dict(states),
        "required_states": list(required_states),
        "details": dict(details or {}),
    }


def collect_environment_report(
    *,
    environment: str,
    live_release: Path,
    schedule_root: Path,
    dagster_home: Path,
    api_root: str,
    report_date: str,
    thresholds: Mapping[str, object] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Collect the full report from immutable release and live serving state."""

    active = read_active_activation(live_release)
    if active is None:
        raise ValueError(f"No active release at {live_release}")
    release_root = live_release.resolve()
    runtime = verify_production_site_release(release_root)
    identity = runtime["identity"]
    approved = runtime["approved_manifest"]["identity"]
    bindings = read_json(release_root / "serving-bindings.json")
    components = dict(bindings.get("components") or {})
    edition = str(bindings.get("edition_id") or active.get("edition_id") or "")
    revision = str(components.get("observed_data_revision") or "")
    evidence_date = str(bindings.get("evidence_through_date") or "")
    official = official_schedule_snapshot(schedule_root, report_date=report_date)
    official["season"] = str(approved.get("season") or "")
    surfaces = current_surface_snapshot(api_root, evidence_date)
    release_matches = str(surfaces["serving"].get("release_id") or "") == str(
        active.get("release_id") or ""
    )
    surface_components = dict(surfaces["serving"].get("components") or {})
    component_matches = all(
        surface_components.get(name) == value for name, value in components.items()
    )
    expected_games = int(official["completed_games"])
    approved_api = dict(approved.get("api_validation") or {})
    model_quality = dict(surfaces["ranking_state"].get("data_quality") or {})
    source_exclusions = dict(model_quality.get("source_exclusions") or {})
    rating_exclusions = list(source_exclusions.get("ratings") or [])
    minute_exclusions = list(source_exclusions.get("minutes") or [])
    full_states = {
        "materialized": True,
        "validated": True,
        "activated": True,
        "served": release_matches and component_matches,
    }
    lifecycle_states = {
        "materialized": True,
        "validated": True,
        "activated": True,
        "served": None,
    }
    artifacts = [
        artifact_row(
            name="observed_daily_data",
            layer="daily_artifact",
            artifact_id=approved["data"]["bundle_sha256"],
            checksum=approved["data"]["bundle_sha256"],
            evidence_date=str(approved["evidence_through_date"]),
            revision=revision,
            edition=edition,
            coverage={
                "completed_games": {
                    "expected": expected_games,
                    "observed": int(approved["game_count"]),
                }
            },
            states=lifecycle_states,
            required_states=("materialized", "validated", "activated"),
        ),
        artifact_row(
            name="daily_model_state",
            layer="model_artifact",
            artifact_id=approved["model"]["bundle_sha256"],
            checksum=approved["model"]["bundle_sha256"],
            evidence_date=str(approved["evidence_through_date"]),
            revision=revision,
            edition=edition,
            coverage={
                "completed_games": {
                    "expected": expected_games,
                    "observed": int(approved["game_count"]),
                },
                "model_quality_issues": {
                    "expected": 0,
                    "observed": int(model_quality.get("issue_count", 0)),
                },
                "rating_exclusions": {
                    "expected": 0,
                    "observed": len(rating_exclusions),
                },
                "minutes_exclusions": {
                    "expected": 0,
                    "observed": len(minute_exclusions),
                },
            },
            states=lifecycle_states,
            required_states=("materialized", "validated", "activated"),
            details={
                "model_provenance": approved["model"].get("model_provenance"),
                "source_exclusions": source_exclusions,
            },
        ),
        artifact_row(
            name="player_pages",
            layer="served_surface",
            artifact_id=components.get("serving_bundle_id"),
            checksum=approved["serving"]["bundle_sha256"],
            evidence_date=str(
                (surfaces["health"].get("current_season_state") or {}).get(
                    "evidence_through_date", ""
                )
            ),
            revision=revision,
            edition=edition,
            coverage={
                "players": {
                    "expected": int(approved["preseason_catalog"]["player_count"]),
                    "observed": int(approved_api.get("ranking_player_count", -1)),
                },
                "player_opportunities": {
                    "expected": int(approved_api.get("verified_player_opportunities", -1)),
                    "observed": int(approved_api.get("verified_player_opportunities", -1)),
                },
                "missed_game_rows": {
                    "expected": int(approved_api.get("verified_missed_game_logs", -1)),
                    "observed": int(approved_api.get("verified_missed_game_logs", -1)),
                },
            },
            states=full_states,
            required_states=("materialized", "validated", "activated", "served"),
            details={
                "coverage_source": "activation_exhaustive_api_validation",
                "representative_player_id": approved_api.get("player_id"),
            },
        ),
        artifact_row(
            name="rankings",
            layer="served_surface",
            artifact_id=components.get("serving_bundle_id"),
            checksum=approved["serving"]["bundle_sha256"],
            evidence_date=str(surfaces["ranking_state"].get("evidence_through_date") or ""),
            revision=revision,
            edition=edition,
            coverage={
                "players": {
                    "expected": int(approved["preseason_catalog"]["player_count"]),
                    "observed": int(surfaces["ranking_count"]),
                },
                "completed_games": {
                    "expected": expected_games,
                    "observed": int(surfaces["ranking_state"].get("game_count", -1)),
                },
            },
            states=full_states,
            required_states=("materialized", "validated", "activated", "served"),
        ),
        artifact_row(
            name="teams",
            layer="served_surface",
            artifact_id=components.get("serving_bundle_id"),
            checksum=approved["serving"]["bundle_sha256"],
            evidence_date=evidence_date,
            revision=revision,
            edition=edition,
            coverage={"teams": {"expected": 30, "observed": int(surfaces["team_count"])}},
            states=full_states,
            required_states=("materialized", "validated", "activated", "served"),
        ),
        artifact_row(
            name="lineups",
            layer="served_surface",
            artifact_id=components.get("lineup_bundle_id"),
            checksum=approved["lineups"]["bundle_sha256"],
            evidence_date=str(surfaces["lineup_state"].get("evidence_through_date") or ""),
            revision=revision,
            edition=edition,
            coverage={
                "lineups": {
                    "expected": int(approved_api.get("lineup_count", -1)),
                    "observed": int(surfaces["lineup_count"]),
                },
                "completed_games": {
                    "expected": expected_games,
                    "observed": len(surfaces["lineup_state"].get("source_game_ids") or []),
                },
            },
            states=full_states,
            required_states=("materialized", "validated", "activated", "served"),
        ),
        artifact_row(
            name="wins",
            layer="served_surface",
            artifact_id=components.get("serving_bundle_id"),
            checksum=approved["serving"]["bundle_sha256"],
            evidence_date=str(surfaces["wins_as_of"]),
            revision=revision,
            edition=edition,
            coverage={
                "teams": {"expected": 30, "observed": int(surfaces["win_team_count"])},
                "completed_team_games": {
                    "expected": expected_games * 2,
                    "observed": int(surfaces["wins_completed_team_games"]),
                },
            },
            states=full_states,
            required_states=("materialized", "validated", "activated", "served"),
        ),
        artifact_row(
            name="schedule_ledgers",
            layer="served_surface",
            artifact_id=approved["schedule"]["tree_sha256"],
            checksum=approved["schedule"]["tree_sha256"],
            evidence_date=evidence_date,
            revision=revision,
            edition=edition,
            coverage={
                "pregame_ledgers": {
                    "expected": int(
                        approved["schedule"]["attribution_audit"]["served_games"]
                    ),
                    "observed": int(
                        approved["schedule"]["attribution_audit"]["pregame_ledgers"]
                    ),
                },
                "actual_minutes_ledgers": {
                    "expected": expected_games,
                    "observed": int(
                        approved["schedule"]["attribution_audit"][
                            "actual_minutes_ledgers"
                        ]
                    ),
                },
            },
            states=full_states,
            required_states=("materialized", "validated", "activated", "served"),
        ),
        artifact_row(
            name="grades_and_records",
            layer="served_surface",
            artifact_id=components.get("forecast_selection_id"),
            checksum=components.get("forecast_selection_id"),
            evidence_date=str(surfaces["records_as_of"]),
            revision=revision,
            edition=edition,
            coverage={
                "completed_games": {
                    "expected": expected_games,
                    "observed": int(surfaces["record_game_count"]),
                }
            },
            states=full_states,
            required_states=("materialized", "validated", "activated", "served"),
        ),
        artifact_row(
            name="aggregate_performance",
            layer="served_surface",
            artifact_id=components.get("forecast_selection_id"),
            checksum=components.get("forecast_selection_id"),
            evidence_date=str(surfaces["performance_as_of"]),
            revision=revision,
            edition=edition,
            coverage={
                "completed_games": {
                    "expected": expected_games,
                    "observed": int(surfaces["performance_game_count"]),
                }
            },
            states=full_states,
            required_states=("materialized", "validated", "activated", "served"),
        ),
        artifact_row(
            name="frontend",
            layer="served_surface",
            artifact_id=identity.get("frontend_sha256"),
            checksum=identity.get("frontend_sha256"),
            evidence_date=evidence_date,
            revision=revision,
            edition=edition,
            coverage={},
            states=full_states,
            required_states=("materialized", "validated", "activated", "served"),
            details={"code_revision": identity.get("code_revision")},
        ),
    ]
    transition = activation_transition(active, bindings)
    activation = {
        **dict(active),
        "state": "active",
        "observed_data_revision": revision,
        "changed_components": transition["changed_components"],
        "transition": transition,
    }
    notes = [
        "Collector thresholds are not active unless a reviewed threshold file is supplied.",
        "Player opportunity coverage comes from the exhaustive API validation sealed into "
        "the active release; the current served release/component identity is checked "
        "independently.",
    ]
    if transition["forecast_selection_changed"]:
        notes.append(
            "The active release changed the forecast selection; historical grading identity "
            "changed and is explicitly disclosed."
        )
    return build_data_quality_report(
        report_date=report_date,
        environment=environment,
        expected=official,
        activation=activation,
        artifacts=artifacts,
        collectors=collector_rows(
            dagster_home=dagster_home,
            schedule_root=schedule_root,
            report_date=report_date,
            expected=official,
            thresholds=thresholds or {},
        ),
        generated_at=generated_at,
        notes=notes,
    )


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("dev", "prod"), required=True)
    parser.add_argument("--live-release", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--dagster-home", type=Path, required=True)
    parser.add_argument("--api-root", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-date")
    parser.add_argument("--thresholds", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_date = args.report_date or datetime.now(PACIFIC).date().isoformat()
    thresholds = read_json(args.thresholds) if args.thresholds else {}
    probe_error = ""
    try:
        report = collect_environment_report(
            environment=args.environment,
            live_release=args.live_release,
            schedule_root=args.schedule_root,
            dagster_home=args.dagster_home,
            api_root=args.api_root,
            report_date=report_date,
            thresholds=thresholds,
        )
        bundle = publish_data_quality_report(report, args.output_root)
    except Exception as error:
        bundle = publish_failed_data_quality_report(
            report_date=report_date,
            environment=args.environment,
            output_root=args.output_root,
            stage="environment_probe",
            error=error,
        )
        report = read_json(bundle / "report.json")
        probe_error = f"{type(error).__name__}: {error}"
    state_path = args.output_root / "alert-state.json"
    previous = read_json(state_path) if state_path.is_file() else None
    state, notifications = reconcile_alert_state(report, previous)
    atomic_json(state_path, state)
    event_path = (
        args.output_root
        / "alert-events"
        / f"{bundle.name}.json"
    )
    atomic_json(
        event_path,
        {"events": notifications, "updated_at": state["updated_at"]},
    )
    print(
        json.dumps(
            {
                "report": (bundle / "report.html").as_posix(),
                "status": report["status"],
                "probe_error": probe_error,
                "findings": report["finding_counts"],
                "alert_events": notifications,
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "clean" else 2


if __name__ == "__main__":
    raise SystemExit(main())

