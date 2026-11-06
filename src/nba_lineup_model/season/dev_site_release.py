"""Publish one immutable development-site release identity.

The daily data and model chains, current player state, observed lineups, and
Schedule archive are independently content-addressed products.  This module
binds the exact selected versions into one small release manifest only after
their coverage and API contracts agree.  It never mutates production state.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from nba_lineup_model.season.current_serving import (
    resolve_latest_bundle,
    verify_serving_bundle,
)
from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256
from nba_lineup_model.season.daily_data_artifact import resolve_daily_data_pointer
from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
from nba_lineup_model.web_api.current_season_lineups import (
    verify_current_season_lineup_bundle,
)

SCHEMA_VERSION = 1
ARTIFACT_KIND = "dev_site_release"
REPORT_FILENAME = "report.html"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_OUTPUT_ROOT = Path("/var/lib/nba-gestalt-dev/site-releases")


def publish_dev_site_release(
    *,
    daily_data_root: Path | str,
    daily_model_root: Path | str,
    serving_root: Path | str,
    lineups_root: Path | str,
    schedule_state_dir: Path | str,
    catalog_path: Path | str,
    preseason_rankings_path: Path | str,
    preseason_profiles_path: Path | str,
    code_revision: str,
    api_validation: Mapping[str, object],
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    generated_at: datetime | None = None,
) -> Path:
    """Validate and publish a complete dev release without touching production."""

    data_root = Path(daily_data_root).resolve()
    model_root = Path(daily_model_root).resolve()
    serving_root = Path(serving_root).resolve()
    lineups_root = Path(lineups_root).resolve()
    schedule_root = Path(schedule_state_dir).resolve()
    catalog = Path(catalog_path).resolve()
    preseason_rankings = Path(preseason_rankings_path).resolve()
    preseason_profiles = Path(preseason_profiles_path).resolve()
    output = Path(output_root).resolve()

    data_bundle, data_manifest = _resolved_daily_data_tip(data_root)
    model_bundle, model_manifest = _resolved_pointer_bundle(
        model_root, verify_daily_model_artifact
    )
    data_identity = data_manifest["identity"]
    model_identity = model_manifest["identity"]
    if model_identity["input_data_sha256"] != data_manifest["bundle_sha256"]:
        raise ValueError("Daily model tip does not consume the daily data tip")
    if model_identity["game_date_pacific"] != data_identity["game_date_pacific"]:
        raise ValueError("Daily data and model tips have different Pacific dates")

    serving_bundle = resolve_latest_bundle(serving_root)
    serving_manifest = verify_serving_bundle(serving_bundle)
    serving_identity = serving_manifest["identity"]
    model_sha = str(model_manifest["bundle_sha256"])
    source_uris = serving_identity["sources"]
    if not all(model_sha in str(source["uri"]) for source in source_uris.values()):
        raise ValueError("Serving state is not wholly derived from the daily model tip")
    if serving_identity["evidence_through_date"] != model_identity["game_date_pacific"]:
        raise ValueError("Serving state and daily model have different evidence dates")

    lineup_pointer = _read_json(lineups_root / "latest.json")
    lineup_bundle = (lineups_root / str(lineup_pointer["bundle_path"])).resolve()
    _require_beneath(lineup_bundle, lineups_root, "Lineup bundle")
    lineup_manifest = verify_current_season_lineup_bundle(lineup_bundle)
    lineup_identity = lineup_manifest["identity"]
    if lineup_identity["serving_bundle_sha256"] != serving_manifest["bundle_sha256"]:
        raise ValueError("Lineups are not bound to the selected serving bundle")

    expected_game_ids = sorted(map(str, model_identity["cumulative_game_ids"]))
    if sorted(map(str, serving_identity["coverage"]["source_game_ids"])) != expected_game_ids:
        raise ValueError("Serving coverage differs from the daily model chain")
    if sorted(map(str, lineup_identity["source_game_ids"])) != expected_game_ids:
        raise ValueError("Lineup coverage differs from the daily model chain")
    final_game_ids = _schedule_final_game_ids(schedule_root)
    missing_model_finals = sorted(set(expected_game_ids) - set(final_game_ids))
    if missing_model_finals:
        raise ValueError(
            "Schedule finals omit games from the verified daily chain: "
            f"missing={missing_model_finals}"
        )
    attribution_audit = _schedule_attribution_audit(schedule_root)
    if attribution_audit["completed_games"] != len(final_game_ids):
        raise ValueError(
            "Schedule attribution coverage differs from schedule finals: "
            f"attributed={attribution_audit['completed_games']} "
            f"final={len(final_game_ids)}"
        )
    normalized_api = _normalize_api_validation(
        api_validation,
        evidence_date=str(model_identity["game_date_pacific"]),
        expected_game_count=len(expected_game_ids),
        expected_final_game_count=len(final_game_ids),
        expected_schedule_game_ids=data_identity["game_ids"],
        serving_sha=str(serving_manifest["bundle_sha256"]),
        lineup_sha=str(lineup_manifest["bundle_sha256"]),
    )
    code_revision = str(code_revision).strip()
    if len(code_revision) != 40 or any(c not in "0123456789abcdef" for c in code_revision):
        raise ValueError("Code revision must be a full lowercase Git commit SHA")
    if not catalog.is_file():
        raise FileNotFoundError(f"Game catalog is missing: {catalog}")
    if not preseason_rankings.is_file() or not preseason_profiles.is_file():
        raise FileNotFoundError("Bound preseason rankings or profiles are missing")
    rankings_frame = pd.read_parquet(preseason_rankings)
    profiles_frame = pd.read_parquet(preseason_profiles)
    ranking_ids = set(rankings_frame["player_id"].astype(int))
    profile_ids = set(profiles_frame["player_id"].astype(int))
    if ranking_ids != profile_ids:
        raise ValueError("Bound preseason rankings and profiles use different players")
    serving_state = pd.read_parquet(serving_bundle / "serving_state.parquet")
    serving_ids = set(serving_state["player_id"].astype(int))
    if ranking_ids != serving_ids:
        raise ValueError("Bound preseason catalogs differ from the serving universe")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "season": str(model_identity["season"]),
        "evidence_through_date": str(model_identity["game_date_pacific"]),
        "game_count": len(expected_game_ids),
        "code_revision": code_revision,
        "data": {
            "bundle_sha256": str(data_manifest["bundle_sha256"]),
            "bundle_path": data_bundle.as_posix(),
        },
        "model": {
            "bundle_sha256": model_sha,
            "bundle_path": model_bundle.as_posix(),
            "model_provenance": dict(model_identity["model_provenance"]),
        },
        "serving": {
            "bundle_sha256": str(serving_manifest["bundle_sha256"]),
            "bundle_path": serving_bundle.as_posix(),
        },
        "lineups": {
            "bundle_sha256": str(lineup_manifest["bundle_sha256"]),
            "bundle_path": lineup_bundle.as_posix(),
        },
        "schedule": {
            "tree_sha256": _tree_sha256(schedule_root),
            "root_path": schedule_root.as_posix(),
            "final_game_count": len(final_game_ids),
            "attribution_audit": attribution_audit,
        },
        "catalog": {
            "sha256": file_sha256(catalog),
            "path": catalog.as_posix(),
        },
        "preseason_catalog": {
            "player_count": len(ranking_ids),
            "rankings_sha256": file_sha256(preseason_rankings),
            "rankings_path": preseason_rankings.as_posix(),
            "profiles_sha256": file_sha256(preseason_profiles),
            "profiles_path": preseason_profiles.as_posix(),
        },
        "api_validation": normalized_api,
    }
    release_sha = hashlib.sha256(canonical_json(identity)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"release_sha256={release_sha}"
    if not target.exists():
        staging = output / f".site-release.tmp-{uuid4().hex}"
        staging.mkdir()
        try:
            report = _render_report(identity, release_sha)
            (staging / REPORT_FILENAME).write_text(report, encoding="utf-8")
            manifest = {
                "release_sha256": release_sha,
                "generated_at": _utc_timestamp(generated_at or datetime.now(UTC)),
                "identity": identity,
                "files": {REPORT_FILENAME: file_sha256(staging / REPORT_FILENAME)},
            }
            (staging / MANIFEST_FILENAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staging.replace(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    verify_dev_site_release(target)
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "release_sha256": release_sha,
        "release_path": target.relative_to(output).as_posix(),
        "season": identity["season"],
        "evidence_through_date": identity["evidence_through_date"],
        "game_count": identity["game_count"],
        "code_revision": identity["code_revision"],
        "promoted_at": _utc_timestamp(datetime.now(UTC)),
    }
    _atomic_write_json(output / "latest.json", pointer)
    return target


def verify_dev_site_release(path: Path | str) -> dict[str, Any]:
    bundle = Path(path).resolve()
    manifest = _read_json(bundle / MANIFEST_FILENAME)
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Dev-site release identity is missing")
    if identity.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dev-site release schema")
    if identity.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("Artifact is not a dev-site release")
    expected = hashlib.sha256(canonical_json(identity)).hexdigest()
    if manifest.get("release_sha256") != expected:
        raise ValueError("Dev-site release checksum is invalid")
    if bundle.name != f"release_sha256={expected}":
        raise ValueError("Dev-site release directory is not content-addressed")
    report_sha = manifest.get("files", {}).get(REPORT_FILENAME)
    if report_sha != file_sha256(bundle / REPORT_FILENAME):
        raise ValueError("Dev-site release report checksum changed")
    return manifest


def resolve_latest_dev_site_release(root: Path | str) -> Path:
    root_path = Path(root).resolve()
    pointer = _read_json(root_path / "latest.json")
    target = (root_path / str(pointer["release_path"])).resolve()
    _require_beneath(target, root_path, "Dev-site release")
    manifest = verify_dev_site_release(target)
    if pointer.get("release_sha256") != manifest["release_sha256"]:
        raise ValueError("Dev-site pointer checksum does not match its release")
    return target


def collect_api_validation(api_root: str, *, evidence_date: str, daily_model_root: Path | None = None) -> dict[str, object]:
    """Exercise representative site surfaces and return their release identities."""

    root = api_root.rstrip("/")
    health = _get_json(f"{root}/api/health")
    rankings = _get_json(f"{root}/api/rankings?season=2026-27")
    team = _get_json(f"{root}/api/teams/GSW?season=2026-27&minimum_possessions=0")
    player = _get_json(f"{root}/api/players/201939")
    lineups = _get_json(f"{root}/api/lineups?season=2026-27&minimum_possessions=0")
    wins = _get_json(f"{root}/api/win-projections")
    performance = _get_json(f"{root}/api/schedule-performance")
    records = _get_json(f"{root}/api/current-season-records")
    schedule = _get_json(
        f"{root}/api/schedule?{urllib.parse.urlencode({'date': evidence_date})}"
    )
    # Every actual participant on the release date must be visible in their
    # player-page API, not just the representative Curry profile above.
    checked_logs = 0
    # Full profiles include large multi-season charts and caches. Coverage uses
    # only current-season trajectories; retain those rather than every payload.
    profiles = {201939: player.get("current_season_trajectory", [])}
    for game in schedule.get("games", []):
        if not game.get("status", {}).get("completed"):
            continue
        game_id = str(game["game_id"])
        actual = game.get("retrospective", {}).get("actual_minutes", {})
        for side in ("away", "home"):
            for row in actual.get(side, []):
                minutes = float(row.get("actual_minutes", 0))
                if minutes <= 0:
                    continue
                player_id = int(row["player_id"])
                if player_id not in profiles:
                    profiles[player_id] = _get_json(f"{root}/api/players/{player_id}").get("current_season_trajectory", [])
                matches = [point for point in profiles[player_id] if str(point.get("game_id")) == game_id]
                if len(matches) != 1 or abs(float(matches[0].get("actual_minutes", -1)) - minutes) > 1e-6:
                    raise ValueError(f"Player game-log coverage mismatch: NBA game {game_id}, player {player_id}")
                checked_logs += 1
    # Validate every historical observed opportunity, not merely players who
    # logged positive minutes in the newest slate. This catches absent players,
    # whole-season absences and regressions in older dates before publication.
    checked_opportunities = 0
    checked_absences = 0
    if daily_model_root is not None:
        from nba_lineup_model.season.daily_model_replay import _read_model_chain, _component_path
        chain = _read_model_chain(Path(daily_model_root))
        if not chain:
            raise ValueError("Required daily model history is missing")
        for day, bundle, manifest in chain:
            if day.isoformat() > evidence_date:
                continue
            observations = pd.read_parquet(_component_path(bundle, manifest, "minutes_observations"))
            for row in observations.itertuples(index=False):
                player_id, game_id = int(row.player_id), str(row.game_id)
                if player_id not in profiles:
                    profiles[player_id] = _get_json(f"{root}/api/players/{player_id}").get("current_season_trajectory", [])
                matches = [p for p in profiles[player_id] if str(p.get("game_id")) == game_id]
                if (len(matches) != 1
                    or matches[0].get("played") is not bool(row.played)
                    or str(matches[0].get("team")) != str(row.team)
                    or abs(float(matches[0].get("actual_minutes", -1)) - float(row.actual_minutes)) > 1e-6
                    or (not row.played and float(matches[0].get("game_on_court_possessions", -1)) != 0)):
                    raise ValueError(f"Player opportunity coverage mismatch: NBA game {game_id}, player {player_id}")
                checked_opportunities += 1
                checked_absences += int(not row.played)
    season_window = next(
        (
            window
            for window in performance.get("windows", [])
            if window.get("key") == "season"
        ),
        {},
    )
    current_state = health.get("current_season_state") or {}
    win_projection = wins.get("win_projection", {})
    win_teams = win_projection.get("teams", [])
    wins_completed_team_games = sum(int(row.get("completed_games", 0)) for row in win_teams)
    wins_record_team_games = sum(int(row.get("actual_wins", 0)) + int(row.get("actual_losses", 0)) for row in win_teams)
    rating_state = rankings.get("rating_state") or {}
    lineup_state = lineups.get("lineup_state") or {}
    return {
        "status": "clean",
        "health": str(health.get("status", "")),
        "evidence_through_date": str(current_state.get("evidence_through_date", "")),
        "evidence_game_count": int(current_state.get("game_count", -1)),
        "serving_bundle_sha256": str(current_state.get("bundle_sha256", "")),
        "rankings_bundle_sha256": str(rating_state.get("bundle_sha256", "")),
        "lineup_serving_bundle_sha256": str(
            lineup_state.get("serving_bundle_sha256", "")
        ),
        "lineup_bundle_sha256": str(lineup_state.get("bundle_sha256", "")),
        "ranking_player_count": len(rankings.get("players", [])),
        "team_player_count": len(team.get("players", team.get("roster", []))),
        "player_id": int(player.get("player_id", -1)),
        "lineup_count": len(lineups.get("lineups", [])),
        "scheduled_game_count": int(
            win_projection.get("scheduled_game_count", -1)
        ),
        "win_projection_as_of": str(win_projection.get("as_of", "")),
        "win_projection_team_count": len(win_teams),
        "wins_completed_team_games": wins_completed_team_games,
        "wins_record_team_games": wins_record_team_games,
        "season_performance_games": int(season_window.get("games", -1)),
        "canonical_final_game_count": len(records.get("completed_game_ids", [])),
        "schedule_game_count": len(schedule.get("games", [])),
        "schedule_game_ids": [str(game["game_id"]) for game in schedule.get("games", [])],
        "verified_player_game_logs": checked_logs,
        "verified_player_opportunities": checked_opportunities,
        "verified_missed_game_logs": checked_absences,
    }


def approved_model_root(approved_identity: Mapping[str, Any], *, release: Path | None = None) -> Path:
    """Resolve and verify the release's model chain rather than a job directory."""
    reference = approved_identity["model"]
    from nba_lineup_model.season.serving_recovery import daily_path
    bundle = daily_path(Path(reference["bundle_path"]), release=release).resolve(strict=True)
    manifest = verify_daily_model_artifact(bundle)
    if (manifest["bundle_sha256"] != reference["bundle_sha256"]
        or manifest["identity"]["game_date_pacific"] != approved_identity["evidence_through_date"]):
        raise ValueError("Approved daily model identity mismatch")
    root = bundle.parents[2]
    tip, _ = _resolved_pointer_bundle(root, verify_daily_model_artifact)
    if tip != bundle:
        raise ValueError("Approved daily model chain tip differs from release")
    return root


def approved_schedule_game_ids(approved_identity: Mapping[str, Any], *, release: Path | None = None) -> list[str]:
    """Read the exact approved artifact, never an outer job's mutable pointer."""
    reference = approved_identity["data"]
    from nba_lineup_model.season.serving_recovery import daily_path
    manifest = verify_daily_data_artifact(daily_path(Path(reference["bundle_path"]), release=release))
    if manifest["bundle_sha256"] != reference["bundle_sha256"]:
        raise ValueError("Approved daily data checksum mismatch")
    identity = manifest["identity"]
    if identity["game_date_pacific"] != approved_identity["evidence_through_date"]:
        raise ValueError("Approved daily data evidence date mismatch")
    return list(identity["game_ids"])


def _normalize_api_validation(
    payload: Mapping[str, object],
    *,
    evidence_date: str,
    expected_game_count: int,
    expected_final_game_count: int,
    expected_schedule_game_ids: list[str],
    serving_sha: str,
    lineup_sha: str,
) -> dict[str, object]:
    output = dict(payload)
    required_equal = {
        "status": "clean",
        "health": "ok",
        "evidence_through_date": evidence_date,
        "evidence_game_count": expected_game_count,
        "serving_bundle_sha256": serving_sha,
        "rankings_bundle_sha256": serving_sha,
        "lineup_serving_bundle_sha256": serving_sha,
        "lineup_bundle_sha256": lineup_sha,
        "player_id": 201939,
        "win_projection_as_of": evidence_date,
        "win_projection_team_count": 30,
    }
    for key, expected in required_equal.items():
        if output.get(key) != expected:
            raise ValueError(
                f"Dev API validation mismatch for {key}: "
                f"expected {expected!r}, found {output.get(key)!r}"
            )
    canonical_final_count = int(output.get("canonical_final_game_count", -1))
    performance_count = int(output.get("season_performance_games", -1))
    if canonical_final_count < expected_final_game_count:
        raise ValueError(
            "Canonical NBA final coverage is behind the frozen schedule snapshot: "
            f"expected at least {expected_final_game_count}, found {canonical_final_count}"
        )
    if performance_count != canonical_final_count:
        raise ValueError(
            "Joined schedule performance differs from canonical NBA final coverage: "
            f"performance={performance_count} canonical={canonical_final_count}"
        )
    # Empty dates are valid only when the verified daily NBA artifact is empty.
    # Compare identities as well as counts so missing/replaced/duplicate games fail.
    schedule_ids = output.get("schedule_game_ids")
    if (not isinstance(schedule_ids, list)
        or sorted(schedule_ids) != sorted(expected_schedule_game_ids)
        or output.get("schedule_game_count") != len(expected_schedule_game_ids)):
        raise ValueError(
            f"Dev schedule coverage mismatch on {evidence_date}: "
            f"expected={sorted(expected_schedule_game_ids)!r}, "
            f"found={schedule_ids!r}, count={output.get('schedule_game_count')!r}"
        )
    for key in ("ranking_player_count", "team_player_count", "lineup_count"):
        if int(output.get(key, 0)) <= 0:
            raise ValueError(f"Dev API validation has no {key.replace('_', ' ')}")
    if int(output.get("scheduled_game_count", 0)) < expected_game_count:
        raise ValueError("Dev win projection calendar is shorter than completed coverage")
    expected_team_games = expected_game_count * 2
    for key in ("wins_completed_team_games", "wins_record_team_games"):
        if int(output.get(key, -1)) != expected_team_games:
            raise ValueError(
                f"Dev win projection {key} mismatch: "
                f"expected {expected_team_games}, found {output.get(key)!r}"
            )
    return output


def _resolved_daily_data_tip(root: Path) -> tuple[Path, dict[str, Any]]:
    pointer = _read_json(root / "latest.json")
    collection = resolve_daily_data_pointer(root, pointer)
    identity = collection["identity"]
    selected = identity["dates"][identity["through_date"]]
    bundle = (root / str(selected["bundle_path"])).resolve()
    _require_beneath(bundle, root, "Daily data artifact")
    manifest = verify_daily_data_artifact(bundle)
    if selected["bundle_sha256"] != manifest["bundle_sha256"]:
        raise ValueError("Data collection tip checksum does not match its artifact")
    return bundle, manifest


def _resolved_pointer_bundle(root: Path, verifier) -> tuple[Path, dict[str, Any]]:
    pointer = _read_json(root / "latest.json")
    bundle = (root / str(pointer["bundle_path"])).resolve()
    _require_beneath(bundle, root, "Artifact bundle")
    manifest = verifier(bundle)
    if pointer.get("bundle_sha256") != manifest["bundle_sha256"]:
        raise ValueError("Artifact pointer checksum does not match its bundle")
    return bundle, manifest


def _schedule_final_game_ids(root: Path) -> list[str]:
    game_ids: list[str] = []
    for path in sorted((root / "final").glob("*.summary.json")):
        payload = _read_json(path)
        game_id = str(payload.get("game_id", "")).strip()
        if not game_id:
            raise ValueError(f"Schedule final has no game ID: {path}")
        game_ids.append(game_id)
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("Schedule finals contain duplicate NBA game IDs")
    return sorted(game_ids)


def _schedule_attribution_audit(root: Path) -> dict[str, int]:
    """Reject a release if any served game lacks its auditable ledgers."""

    completed_ids: set[str] = set()
    for path in sorted((root / "final").glob("*.summary.json")):
        payload = _read_json(path)
        game_id = str(payload.get("game_id", "")).strip()
        pregame = payload.get("pregame")
        retrospective = payload.get("retrospective")
        if not isinstance(pregame, dict) or not isinstance(retrospective, dict):
            raise ValueError(f"Completed game {game_id or path.name} lacks both views")
        _require_schedule_ledger(
            pregame.get("projection"),
            game_id=game_id,
            actual=False,
        )
        _require_schedule_ledger(
            retrospective,
            game_id=game_id,
            actual=True,
        )
        completed_ids.add(game_id)

    served_ids: set[str] = set()
    served_completed_ids: set[str] = set()
    for path in sorted((root / "latest").glob("*.json")):
        payload = _read_json(path)
        games = payload.get("games")
        if not isinstance(games, list):
            raise ValueError(f"Schedule day has no games array: {path}")
        for game in games:
            if not isinstance(game, dict):
                raise ValueError(f"Schedule day has an invalid game: {path}")
            game_id = str(game.get("game_id", "")).strip()
            if not game_id:
                raise ValueError(f"Schedule game has no ID: {path}")
            if game_id in served_ids:
                raise ValueError(f"Schedule serves duplicate game {game_id}")
            _require_schedule_ledger(
                game.get("projection"),
                game_id=game_id,
                actual=False,
            )
            _require_schedule_projected_minutes(game, game_id=game_id)
            frozen_path = root / "frozen" / f"{game_id}.json"
            if frozen_path.is_file():
                frozen = _read_json(frozen_path)
                frozen_projection = frozen.get("projection")
                if not isinstance(frozen_projection, dict):
                    raise ValueError(f"Frozen projection is missing for {game_id}")
                if not isinstance(frozen_projection.get("ledger"), dict):
                    projection = game.get("projection")
                    provenance = (
                        projection.get("input_provenance")
                        if isinstance(projection, dict)
                        else None
                    )
                    frozen_sha = hashlib.sha256(canonical_json(frozen)).hexdigest()
                    if (
                        not isinstance(provenance, dict)
                        or provenance.get("source_frozen_pregame_sha256")
                        != frozen_sha
                    ):
                        raise ValueError(
                            "Rendered attribution is not bound to frozen pregame "
                            f"{game_id}"
                        )
            served_ids.add(game_id)
            status = game.get("status")
            if isinstance(status, dict) and bool(status.get("completed")):
                _require_schedule_ledger(
                    game.get("retrospective"),
                    game_id=game_id,
                    actual=True,
                )
                served_completed_ids.add(game_id)
    if completed_ids != served_completed_ids:
        raise ValueError(
            "Completed schedule ledgers differ between final and daily views: "
            f"final={len(completed_ids)} daily={len(served_completed_ids)}"
        )
    return {
        "served_games": len(served_ids),
        "pregame_ledgers": len(served_ids),
        "completed_games": len(completed_ids),
        "actual_minutes_ledgers": len(served_completed_ids),
    }


def _require_schedule_projected_minutes(
    game: Mapping[str, object], *, game_id: str
) -> None:
    """Validate the exact player rows selected by the Schedule UI."""

    projected = game.get("projected_minutes")
    projection = game.get("projection")
    ledger = projection.get("ledger") if isinstance(projection, dict) else None
    if not isinstance(projected, dict) or not isinstance(ledger, dict):
        raise ValueError(f"Schedule projected-minute attribution is missing for {game_id}")
    projected_ids: set[int] = set()
    for side in ("away", "home"):
        rows = projected.get(side)
        team_ledger = ledger.get(side)
        ledger_rows = team_ledger.get("players") if isinstance(team_ledger, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Schedule has no {side} projected rows for {game_id}")
        if not isinstance(ledger_rows, list):
            raise ValueError(f"Schedule has no {side} ledger rows for {game_id}")
        ledger_by_id = {
            int(row["player_id"]): row
            for row in ledger_rows
            if isinstance(row, dict) and row.get("player_id") is not None
        }
        side_ids: set[int] = set()
        for row in rows:
            if not isinstance(row, dict) or row.get("player_id") is None:
                raise ValueError(
                    f"Schedule has an unidentified {side} projected row for {game_id}"
                )
            player_id = int(row["player_id"])
            if player_id in side_ids:
                raise ValueError(
                    f"Schedule repeats player {player_id} on {side} for {game_id}"
                )
            side_ids.add(player_id)
            projected_ids.add(player_id)
            values = (
                row.get("projected_nail"),
                row.get("projected_minutes"),
                row.get("expected_contribution"),
            )
            if any(value is None for value in values):
                raise ValueError(
                    f"Schedule has incomplete displayed attribution for player "
                    f"{player_id} in {game_id}"
                )
            rating, minutes, contribution = map(float, values)
            if not math.isclose(
                contribution,
                minutes / 48.0 * rating,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    f"Schedule contribution does not reconcile for player "
                    f"{player_id} in {game_id}"
                )
            ledger_row = ledger_by_id.get(player_id)
            if ledger_row is None:
                raise ValueError(
                    f"Schedule ledger lacks player {player_id} in {game_id}"
                )
            ledger_values = (
                ledger_row.get("frozen_nail"),
                ledger_row.get("projected_minutes"),
                ledger_row.get("expected_contribution"),
            )
            if any(value is None for value in ledger_values) or any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-10)
                for left, right in zip(
                    map(float, ledger_values),
                    (rating, minutes, contribution),
                    strict=True,
                )
            ):
                raise ValueError(
                    f"Schedule displayed row differs from ledger for player "
                    f"{player_id} in {game_id}"
                )
        if side_ids != set(ledger_by_id):
            raise ValueError(
                f"Schedule displayed and ledger player sets differ on {side} for {game_id}"
            )
    injury_report = game.get("injury_report")
    assumptions = (
        injury_report.get("assumptions")
        if isinstance(injury_report, dict)
        else None
    )
    if isinstance(assumptions, list):
        missing = sorted(
            int(row["player_id"])
            for row in assumptions
            if isinstance(row, dict)
            and row.get("player_id") is not None
            and int(row["player_id"]) not in projected_ids
            and not (
                str(row.get("status")) == "Available"
                and row.get("projected_minutes") is not None
                and math.isclose(
                    float(row["projected_minutes"]), 0.0, abs_tol=1e-10
                )
            )
        )
        if missing:
            raise ValueError(
                f"Schedule injury players lack displayed attribution for {game_id}: {missing}"
            )


def _require_schedule_ledger(
    container: object,
    *,
    game_id: str,
    actual: bool,
) -> None:
    if not isinstance(container, dict):
        raise ValueError(f"Schedule projection is missing for {game_id}")
    ledger = container.get("ledger")
    if not isinstance(ledger, dict):
        kind = "actual-minutes" if actual else "pregame"
        raise ValueError(f"Schedule {kind} ledger is missing for {game_id}")
    margin = float(container["home_margin"])
    if not math.isclose(
        margin,
        float(ledger["home_margin"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError(f"Schedule ledger does not reproduce margin for {game_id}")
    minutes_key = "actual_minutes" if actual else "projected_minutes"
    contribution_key = "actual_contribution" if actual else "expected_contribution"
    expected_kind = (
        "retrospective_actual_minutes_attribution"
        if actual
        else "immutable_game_prediction_attribution"
    )
    if ledger.get("artifact_kind") != expected_kind:
        raise ValueError(f"Schedule ledger has the wrong contract for {game_id}")
    for side in ("away", "home"):
        team = ledger.get(side)
        if not isinstance(team, dict) or not isinstance(team.get("players"), list):
            raise ValueError(f"Schedule ledger lacks {side} players for {game_id}")
        players = team["players"]
        if not players:
            raise ValueError(f"Schedule ledger has no {side} players for {game_id}")
        for player in players:
            if (
                not isinstance(player, dict)
                or player.get("player_id") is None
                or player.get("frozen_nail") is None
                or player.get(minutes_key) is None
                or player.get(contribution_key) is None
            ):
                raise ValueError(
                    f"Schedule ledger has incomplete {side} player attribution for {game_id}"
                )


def _tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"Release tree is missing: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Release tree contains no files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        checksum = file_sha256(path).encode()
        digest.update(checksum)
    return digest.hexdigest()


def _render_report(identity: Mapping[str, Any], release_sha: str) -> str:
    api = identity["api_validation"]
    components = "".join(
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td><code>{html.escape(str(value))}</code></td>"
        "</tr>"
        for label, value in (
            ("Data", identity["data"]["bundle_sha256"]),
            ("Model", identity["model"]["bundle_sha256"]),
            ("Serving", identity["serving"]["bundle_sha256"]),
            ("Lineups", identity["lineups"]["bundle_sha256"]),
            ("Schedule", identity["schedule"]["tree_sha256"]),
            ("Catalog", identity["catalog"]["sha256"]),
            ("Preseason rankings", identity["preseason_catalog"]["rankings_sha256"]),
            ("Preseason profiles", identity["preseason_catalog"]["profiles_sha256"]),
            ("Code", identity["code_revision"]),
        )
    )
    checks = "".join(
        f"<li><strong>{html.escape(key.replace('_', ' ').title())}:</strong> "
        f"{html.escape(str(value))}</li>"
        for key, value in api.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>GESTALT dev release — {html.escape(identity['evidence_through_date'])}</title>
<style>
body{{font:16px system-ui;margin:2rem;max-width:1100px}}
table{{border-collapse:collapse;width:100%}}
th,td{{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}}
code{{font-size:.78rem;word-break:break-all}}
.ok{{color:#176b3a;font-weight:700}}</style></head><body>
<h1>GESTALT development release</h1>
<p class="ok">CLEAN — one immutable release identity passed all checks.</p>
<p><strong>{html.escape(identity['season'])}</strong> through Pacific date
<strong>{html.escape(identity['evidence_through_date'])}</strong> ·
{identity['game_count']} completed games · release <code>{release_sha}</code></p>
<h2>Bound artifacts</h2><table><thead><tr><th>Component</th><th>SHA-256</th></tr></thead>
<tbody>{components}</tbody></table><h2>Browser/API checks</h2><ul>{checks}</ul>
<p>Production state and pointers were not modified by this release.</p></body></html>"""


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"API response is not an object: {url}")
    return payload


def _require_beneath(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its root") from error


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-data-root", type=Path, required=True)
    parser.add_argument("--daily-model-root", type=Path, required=True)
    parser.add_argument("--serving-root", type=Path, required=True)
    parser.add_argument("--lineups-root", type=Path, required=True)
    parser.add_argument("--schedule-state-dir", type=Path, required=True)
    parser.add_argument("--catalog-path", type=Path, required=True)
    parser.add_argument("--preseason-rankings-path", type=Path, required=True)
    parser.add_argument("--preseason-profiles-path", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--api-root", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    model_pointer = _read_json(args.daily_model_root / "latest.json")
    evidence_date = str(model_pointer["game_date_pacific"])
    validation = collect_api_validation(args.api_root, evidence_date=evidence_date, daily_model_root=args.daily_model_root)
    path = publish_dev_site_release(
        daily_data_root=args.daily_data_root,
        daily_model_root=args.daily_model_root,
        serving_root=args.serving_root,
        lineups_root=args.lineups_root,
        schedule_state_dir=args.schedule_state_dir,
        catalog_path=args.catalog_path,
        preseason_rankings_path=args.preseason_rankings_path,
        preseason_profiles_path=args.preseason_profiles_path,
        code_revision=args.code_revision,
        api_validation=validation,
        output_root=args.output_root,
    )
    print(path)


if __name__ == "__main__":
    main()
