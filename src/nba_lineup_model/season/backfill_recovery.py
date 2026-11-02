"""Historical official-NBA finalization, independent of the live scoreboard.

All writes target the caller's isolated schedule candidate. Successful inputs
are checkpointed outside that candidate so a failed run can resume cheaply.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

from nba_lineup_model.ingest.nba_cdn import NbaCdnClient, RawJsonCache
from nba_lineup_model.season.cdn_lake import ingest_completed_cdn
from nba_lineup_model.season.official_schedule_runtime import (
    _apply_frozen_pregame, _build_official_actual_minutes_retrospective,
    _catalog_games, _overlay_scoreboard, _read_object,
    _require_or_attach_frozen_attribution,
)
from nba_lineup_model.season.source_policy import forbidden_references, require_clean_artifact_sources
from nba_lineup_model.web_api.win_projections import _build_final_game_archive, _write_json, _write_json_exclusive


class BackfillIncomplete(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__("Backfill incomplete: " + "; ".join(f"{item['game_id']}: {item['error']}" for item in report["failures"]))


def validate_final_box(box, game):
    event = box.get("game", {})
    if str(event.get("gameId")) != str(game["game_id"]):
        raise ValueError("Official box score game identity mismatch")
    if int(event.get("gameStatus") or 0) != 3:
        raise ValueError("Official NBA game is not final yet")
    if forbidden_references(box):
        raise ValueError("Forbidden source in official box score")
    _overlay_scoreboard(copy.deepcopy(game), event)  # validates both team identities
    for key in ("homeTeam", "awayTeam"):
        team = event.get(key, {})
        if not isinstance(team.get("score"), (int, float)) or not isinstance(team.get("players"), list):
            raise ValueError(f"Official final box lacks {key} score/players")
    return event


def finalize_official_box(game, *, state_root, boxscore):
    """Use only the frozen rating-before ledger, never today's player ratings."""
    game_id = str(game["game_id"])
    event = validate_final_box(boxscore, game)
    frozen = _read_object(state_root / "frozen" / f"{game_id}.json")
    if str(frozen.get("game_id")) != game_id:
        raise ValueError("Frozen forecast identity mismatch")
    generated = copy.deepcopy(game)
    _apply_frozen_pregame(game, frozen)
    _require_or_attach_frozen_attribution(game, frozen=frozen, generated=generated)
    _overlay_scoreboard(game, event)
    pregame = copy.deepcopy(frozen)
    for key in ("projection", "projected_minutes"):
        pregame[key] = copy.deepcopy(game[key])
    summary_path = state_root / "final" / f"{game_id}.summary.json"
    if summary_path.exists():
        summary = _read_object(summary_path)
        if str(summary.get("game_id")) != game_id:
            raise ValueError("Final summary identity mismatch")
        for side in ("home", "away"):
            if summary.get("final", {}).get(f"{side}_score") != game[f"{side}_score"]:
                raise ValueError("Existing immutable final scores disagree with official NBA box")
        retrospective = summary.get("retrospective")
        if not isinstance(retrospective, dict) or not retrospective.get("ledger"):
            raise ValueError("Existing immutable final lacks its actual-minute ledger")
    else:
        retrospective = _build_official_actual_minutes_retrospective(
            game, pregame=pregame, boxscore=boxscore, projection_payload={})
        summary = _build_final_game_archive(game, pregame=pregame, retrospective=retrospective)
        _write_json_exclusive(summary_path, summary)
    box_path = state_root / "final" / f"{game_id}.nba.json"
    if not box_path.exists():
        _write_json_exclusive(box_path, boxscore)
    game["retrospective"] = copy.deepcopy(retrospective)
    game.setdefault("final_observed_at", summary.get("archived_at") or datetime.now(UTC).isoformat())
    return game


def recover_official_slate(*, game_date: date, state_root: Path, catalog_path: Path,
                           coverage_path: Path, checkpoint_root: Path, season="2026-27",
                           fetch_box=None, ingest=None):
    """Attempt EVERY catalog game, retain good inputs, report all failures."""
    require_clean_artifact_sources(state_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    require_clean_artifact_sources(checkpoint_root)
    catalog = _catalog_games(catalog_path, start_date=game_date, end_date=game_date).get(game_date.isoformat(), [])
    latest = state_root / "latest" / f"{game_date}.json"
    board = _read_object(latest) if latest.exists() else {
        "season": season, "game_date": game_date.isoformat(), "games": [],
        "source": "NBA official schedule and liveData final box scores",
        "source_provenance": {"provider_fallbacks": [], "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest()},
    }
    by_id = {str(game["game_id"]): game for game in board["games"]}
    report = {"game_date": game_date.isoformat(), "expected_game_ids": [g["game_id"] for g in catalog], "completed_game_ids": [], "observed_final_game_ids": [], "failures": []}
    frozen_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (state_root / "frozen").glob("*.json")}
    with NbaCdnClient(cache=RawJsonCache(checkpoint_root / "requests")) as client:
        fetch = fetch_box or (lambda game_id: client.fetch_boxscore(game_id, use_cache=False).payload)
        for scheduled in catalog:
            game_id = scheduled["game_id"]
            try:
                existing = by_id.get(game_id)
                game = copy.deepcopy(existing or scheduled)
                # Identity/date always comes from the canonical NBA catalog.
                game.update(scheduled)
                game["game_date"] = game_date.isoformat()
                frozen_path = state_root / "frozen" / f"{game_id}.json"
                box_path = state_root / "final" / f"{game_id}.nba.json"
                checkpoint_box = checkpoint_root / "boxes" / f"{game_id}.json"
                box = _read_object(box_path) if box_path.exists() else (_read_object(checkpoint_box) if checkpoint_box.exists() else fetch(game_id))
                validate_final_box(box, game)
                if not checkpoint_box.exists():
                    _write_json_exclusive(checkpoint_box, box)
                _overlay_scoreboard(game, box["game"])
                by_id[game_id] = game
                report["observed_final_game_ids"].append(game_id)
                if not frozen_path.exists():
                    raise ValueError("Missing historical frozen pregame; cannot substitute current predictions")
                by_id[game_id] = finalize_official_box(game, state_root=state_root, boxscore=box)
                report["completed_game_ids"].append(game_id)
            except Exception as error:
                report["failures"].append({"game_id": game_id, "error": f"{type(error).__name__}: {error}"})
    # Preserve successes even when another game failed. No served pointer changes.
    board["games"] = [by_id[g["game_id"]] for g in catalog if g["game_id"] in by_id]
    if not latest.exists() or _read_object(latest) != board:
        _write_json(latest, board)
    if report["observed_final_game_ids"]:
        try:
            (ingest or ingest_completed_cdn)(state_dir=state_root, season=season,
                start_date=game_date, end_date=game_date, game_ids=report["observed_final_game_ids"],
                coverage_path=coverage_path, ignore_grace=True)
        except Exception as error:
            failures = getattr(error, "summary", {}).get("failures", [])
            report["failures"].extend(
                [{"game_id": str(item.get("game_id", "unknown")), "error": str(item.get("error") or item.get("message") or item)} for item in failures]
                or [{"game_id": ",".join(report["observed_final_game_ids"]), "error": f"{type(error).__name__}: {error}"}])
    frozen_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (state_root / "frozen").glob("*.json")}
    if frozen_after != frozen_before:
        raise RuntimeError("Backfill changed frozen pregame inputs")
    report["status"] = "failed" if report["failures"] else "complete"
    _write_json(checkpoint_root / "reports" / f"{game_date}.json", report)
    if report["failures"]:
        raise BackfillIncomplete(report)
    return report
