"""Explicit per-game selections and observation-only composition for serving.

Builders resolve legacy archives once. Readers use the resulting checksummed
selection map; they never search candidates, substitute forecasts, or filter
coverage by a display label.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256

observation_context: ContextVar[ObservationSnapshot | None] = ContextVar(
    "observation_snapshot", default=None
)

frozen_rating_context: ContextVar[Any] = ContextVar("frozen_game_rating_loader", default=None)


INDEX_FILENAME = "forecast-selections.json"
FORECAST_FIELDS = (
    "projection",
    "projected_minutes",
    "injury_report",
    "forecast_version",
)
MARKET_FIELDS = ("market", "prediction_market")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def forecast_inputs(game: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(game.get(key)) for key in FORECAST_FIELDS}


def build_schedule_selection(
    source: Path,
    target: Path,
    *,
    edition_id: str,
    observation_root: Path | None = None,
    forecast_registry_root: Path | None = None,
    version_references: dict[str, str] | None = None,
    environment: str = "dev",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Copy the exact existing selected history, without replay or re-selection.

    This import makes no claim about legacy publication times. Existing edition
    labels and forecast/market values are preserved byte-for-value. Disagreement
    between a displayed prediction and its grade is an explicit migration error.
    """
    from nba_lineup_model.season.source_policy import require_clean_artifact_sources
    from nba_lineup_model.web_api.win_projections import read_schedule_day

    require_clean_artifact_sources(source)
    if target.exists():
        raise ValueError("Selection output must be new; existing history is immutable")
    if bool(forecast_registry_root) != bool(version_references):
        raise ValueError("Original-live assembly requires registry and explicit version references")
    if (source / INDEX_FILENAME).is_file():
        existing = verify_schedule_selection(source)
        if existing["edition_id"] != edition_id or version_references:
            raise ValueError("An existing selection map cannot be relabeled or rewritten")
        shutil.copytree(source, target)
        return verify_schedule_selection(target)
    target.mkdir(parents=True)
    publication_ids: list[str] = []
    games: dict[str, Any] = {}
    dates: dict[str, Any] = {}
    manifest = read_object(source / "manifest.json") if (source / "manifest.json").is_file() else {}
    for path in sorted((source / "latest").glob("*.json")):
        board = read_schedule_day(
            path.stem, state_dir=source, live_score_state_dir=observation_root
        )
        for game in board["games"]:
            game_id = str(game["game_id"])
            if game_id in games:
                raise ValueError(f"Duplicate selected scheduling instance for {game_id}")
            selected = copy.deepcopy(game.get("pregame", game))
            original_selection = None
            version_id = (version_references or {}).get(game_id)
            if version_id:
                if game.get("final_audit") or game.get("pregame"):
                    raise ValueError(
                        "Cannot replace existing history through original-live assembly"
                    )
                from nba_lineup_model.season.forecast_publication import ForecastRegistry

                registry = ForecastRegistry(forecast_registry_root)
                resolved, original_selection = registry.resolve(
                    edition_id=edition_id,
                    forecast_id=version_id,
                    environment=environment,
                    now=now,
                )
                if resolved is None:
                    selected = {**selected, **{field: None for field in FORECAST_FIELDS}}
                    game["data_quality_errors"] = ["missing_original_forecast"]
                else:
                    selected = copy.deepcopy(resolved)
                if original_selection.get("state") == "open":
                    publication_ids.append(version_id)
            selected.pop("final_audit", None)
            selected.pop("retrospective", None)
            final = game.get("final_audit")
            if isinstance(final, dict):
                final_pregame = final.get("pregame", {})
                for key in (*FORECAST_FIELDS, *MARKET_FIELDS):
                    if final_pregame.get(key) != selected.get(key):
                        raise ValueError(f"Displayed/graded selection mismatch: {game_id}:{key}")
            forecast_id = digest(forecast_inputs(selected))
            market_id = digest({key: selected.get(key) for key in MARKET_FIELDS})
            scheduling_id = digest({"game_id": game_id, "cutoff": game["start_time"]})
            selection = {
                "edition_id": edition_id,
                "game_id": game_id,
                "scheduling_instance_id": scheduling_id,
                "cutoff_at": game["start_time"],
                "forecast_id": forecast_id,
                "market_selection_id": market_id,
                "provenance": "preserved_existing_selection",
                "original_publication_receipt": None,
            }
            if original_selection is not None:
                # Open resolver metadata describes the candidate directly; a
                # locked record stores the immutable selection under forecast.
                # Resumed games retain that original selection and cutoff.
                selected_forecast = (
                    original_selection if original_selection.get("state") == "open"
                    else original_selection["forecast"]
                )
                selection["scheduling_instance_id"] = selected_forecast[
                    "scheduling_instance_id"
                ]
                selection["cutoff_at"] = selected_forecast["cutoff_at"]
                selection["provenance"] = "original_live"
                selection["registry_selection"] = original_selection
                selection["version_reference"] = version_id
            selection["selection_id"] = digest(selection)
            game["forecast_selection"] = selection
            # Both visible projected fields and grading use this one selection.
            for key in (*FORECAST_FIELDS, *MARKET_FIELDS):
                if key in selected:
                    game[key] = copy.deepcopy(selected[key])
            if isinstance(game.get("pregame"), dict):
                game["pregame"] = selected
            if isinstance(final, dict):
                final["forecast_selection"] = selection
                _write(target / "final" / f"{game_id}.summary.json", final)
            for candidate_root in (source, observation_root):
                if candidate_root is None:
                    continue
                box = candidate_root / "final" / f"{game_id}.nba.json"
                if box.is_file():
                    destination = target / "final" / box.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(box.read_bytes())
                    break
            games[game_id] = {**selection, "game_date": path.stem}
        destination = target / "latest" / path.name
        _write(destination, board)
        dates[path.stem] = {"path": f"latest/{path.name}", "sha256": file_sha256(destination)}
    _write(target / "manifest.json", manifest)
    require_clean_artifact_sources(target)
    files = {
        str(path.relative_to(target)): file_sha256(path)
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    index = {
        "schema_version": 1,
        "edition_id": edition_id,
        "games": games,
        "dates": dates,
        "files": files,
    }
    if forecast_registry_root is not None:
        if set(version_references or {}) - set(games):
            raise ValueError("Forecast references contain games absent from the official schedule")
        index["forecast_registry_root"] = str(forecast_registry_root.resolve())
        index["publication_forecast_ids"] = sorted(publication_ids)
        index["environment"] = environment
    index["selection_map_sha256"] = digest(index)
    _write(target / INDEX_FILENAME, index)
    verify_schedule_selection(target)
    return index


def verify_schedule_selection(root: Path) -> dict[str, Any]:
    index = read_object(root / INDEX_FILENAME)
    unsigned = {key: value for key, value in index.items() if key != "selection_map_sha256"}
    if index.get("schema_version") != 1 or digest(unsigned) != index.get("selection_map_sha256"):
        raise ValueError("Invalid forecast selection map")
    for relative, checksum in index["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path) != checksum:
            raise ValueError(f"Selected schedule file checksum failed: {relative}")
    found: set[str] = set()
    for day in index["dates"].values():
        for game in read_object(root / day["path"])["games"]:
            gid = str(game["game_id"])
            if gid in found:
                raise ValueError(f"Duplicate game selection: {gid}")
            found.add(gid)
            selection = index["games"][gid]
            selected = game.get("pregame", game)
            if digest(forecast_inputs(selected)) != selection["forecast_id"]:
                raise ValueError(f"Forecast selection mismatch: {gid}")
            if (
                digest({key: selected.get(key) for key in MARKET_FIELDS})
                != selection["market_selection_id"]
            ):
                raise ValueError(f"Market selection mismatch: {gid}")
            if game["forecast_selection"]["selection_id"] != selection["selection_id"]:
                raise ValueError(f"Game references another selection: {gid}")
    if found != set(index["games"]):
        raise ValueError("Selected game coverage mismatch")
    return index


@dataclass(frozen=True)
class ObservationSnapshot:
    """One bounded, read-consistent collection of official observations."""

    observation_id: str
    boards: dict[str, dict[str, Any]]
    boxes: dict[str, dict[str, Any]]
    captured_files: dict[str, str]
    errors: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def capture(cls, root: Path | None, dates: list[str]) -> ObservationSnapshot:
        boards: dict[str, dict[str, Any]] = {}
        boxes: dict[str, dict[str, Any]] = {}
        hashes: dict[str, str] = {}
        errors: dict[str, list[str]] = {}
        if root is not None:
            root = root.resolve()
            for game_date in dates:
                path = root / "latest" / f"{game_date}.json"
                if not path.is_file():
                    continue
                # Only observations cross this boundary. Collector predictions,
                # frozen files, final grades, and retrospective estimates do not.
                raw = path.read_bytes()
                try:
                    board = json.loads(raw)
                    if not isinstance(board.get("games"), list):
                        raise ValueError("games must be a list")
                except (ValueError, AttributeError) as error:
                    errors[game_date] = [f"invalid_observation_board:{error}"]
                    continue
                clean = []
                for game in board["games"]:
                    clean.append(
                        {
                            key: copy.deepcopy(game.get(key))
                            for key in (
                                "game_id",
                                "away_team",
                                "home_team",
                                "start_time",
                                "away_score",
                                "home_score",
                                "status",
                                "injury_report",
                                "market",
                                "prediction_market",
                            )
                        }
                    )
                    box_path = root / "final" / f"{game['game_id']}.nba.json"
                    if box_path.is_file():
                        box_raw = box_path.read_bytes()
                        try:
                            box = json.loads(box_raw)
                            if str(box.get("game", {}).get("gameId")) != str(game["game_id"]):
                                raise ValueError("NBA box score game identity mismatch")
                            boxes[str(game["game_id"])] = box
                        except (ValueError, AttributeError) as error:
                            errors[str(game["game_id"])] = [f"invalid_official_box_score:{error}"]
                        hashes[str(box_path)] = hashlib.sha256(box_raw).hexdigest()
                boards[game_date] = {
                    "games": clean,
                    "collected_at": board.get("live_score_collected_at")
                    or board.get("collected_at"),
                }
                # Identity depends on accepted observation fields, not collector forecasts.
                hashes[f"board:{game_date}"] = digest(boards[game_date])
        return cls(digest({"files": hashes, "errors": errors}), boards, boxes, hashes, errors)


def read_selected_day(
    game_date: str,
    *,
    root: Path,
    observations: ObservationSnapshot | None = None,
) -> dict[str, Any]:
    observations = observations or observation_context.get()
    index = read_object(root / INDEX_FILENAME)
    day = index["dates"].get(game_date)
    if day is None:
        return {
            "available": False,
            "game_date": game_date,
            "games": [],
            "source": "NBA official schedule and liveData",
        }
    path = root / day["path"]
    if file_sha256(path) != day["sha256"]:
        raise ValueError("Selected schedule day changed after publication")
    board = read_object(path)
    observed_board = observations.boards.get(game_date, {}) if observations else {}
    if observations and observations.errors.get(game_date):
        board["data_quality_errors"] = observations.errors[game_date]
    if observed_board.get("collected_at"):
        board["live_score_collected_at"] = observed_board["collected_at"]
    observed_games = {str(g["game_id"]): g for g in observed_board.get("games", [])}
    for game in board["games"]:
        gid = str(game["game_id"])
        selected_ref = game["forecast_selection"]
        if selected_ref.get("provenance") == "original_live":
            from nba_lineup_model.season.forecast_publication import ForecastRegistry

            registry = ForecastRegistry(index["forecast_registry_root"])
            resolved, selection = registry.resolve(
                edition_id=index["edition_id"],
                forecast_id=selected_ref["version_reference"],
                environment=index["environment"],
            )
            # Only the registry authority can lock an original-live selection.
            # All consumers use this exact same decision, including a late worker.
            for field in (*FORECAST_FIELDS, *MARKET_FIELDS):
                game[field] = copy.deepcopy((resolved or {}).get(field))
            if selection.get("state") != "open":
                game["pregame"] = (
                    copy.deepcopy(resolved)
                    if resolved
                    else {field: None for field in FORECAST_FIELDS}
                )
            game["forecast_selection"] = {**selected_ref, "registry_selection": selection}
            if resolved is None:
                game.setdefault("data_quality_errors", []).append("missing_original_forecast")
        if observations and observations.errors.get(gid):
            game.setdefault("data_quality_errors", []).extend(observations.errors[gid])
        observed = observed_games.get(gid)
        if observed is None:
            continue
        if any(observed.get(key) != game.get(key) for key in ("home_team", "away_team")):
            game.setdefault("data_quality_errors", []).append("observation_team_identity_mismatch")
            continue
        game["latest_observations"] = {
            key: copy.deepcopy(observed.get(key))
            for key in ("injury_report", "market", "prediction_market")
        }
        game["observation_id"] = observations.observation_id
        # Completed, selected results require an explicit corrected-data revision.
        # A collector update alone cannot silently regrade historical results.
        final = game.get("final_audit")
        if isinstance(final, dict):
            for key in ("home_score", "away_score"):
                if observed.get(key) is not None and observed[key] != final["final"].get(key):
                    game.setdefault("data_quality_errors", []).append(
                        f"unapproved_result_revision:{key}"
                    )
            if (
                selected_ref.get("provenance") == "original_live"
                and not (final.get("retrospective") or {}).get("ledger")
                and not game.get("data_quality_errors")
            ):
                retrospective = _actual_retrospective(
                    game,
                    final.get("pregame", game),
                    observations.boxes.get(gid),
                    observed_board.get("collected_at"),
                )
                if retrospective:
                    final["retrospective"] = retrospective
                    game["retrospective"] = retrospective
            continue
        for key in ("home_score", "away_score", "status"):
            if observed.get(key) is not None:
                game[key] = copy.deepcopy(observed[key])
        if game.get("status", {}).get("state") != "post":
            continue
        selected = game.get("pregame", game)
        if not isinstance(selected.get("projection"), dict):
            game["data_quality_errors"] = ["missing_original_forecast"]
            continue
        from nba_lineup_model.web_api.win_projections import _build_final_game_archive

        retrospective = _actual_retrospective(
            game,
            selected,
            observations.boxes.get(gid),
            observed_board.get("collected_at"),
        )
        stamp = observed_board.get("collected_at")
        if not stamp:
            game.setdefault("data_quality_errors", []).append("missing_observation_timestamp")
            continue
        archived_at = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        try:
            audit = _build_final_game_archive(
                game,
                pregame=selected,
                retrospective=retrospective,
                archived_at=archived_at,
            )
        except (ValueError, KeyError, TypeError) as error:
            game.setdefault("data_quality_errors", []).append(f"grading_unavailable:{error}")
            continue
        audit["forecast_selection"] = game["forecast_selection"]
        audit["observation_id"] = observations.observation_id
        game["final_audit"] = audit
        # A final score can arrive before its official box score. An empty
        # object is not a valid retrospective: consumers require actual_minutes.
        # Keep the final score, grading and explicit data-quality error visible.
        if retrospective:
            game["retrospective"] = retrospective
        else:
            game.pop("retrospective", None)
    board["serving_selection"] = {
        "edition_id": index["edition_id"],
        "selection_map_sha256": index["selection_map_sha256"],
        "observation_id": observations.observation_id if observations else None,
    }
    return board


def selected_result_records(root: Path, *, season: str) -> list[dict[str, Any]]:
    """Observed W-L records never depend on forecast/market availability."""
    index = read_object(root / INDEX_FILENAME)
    records = []
    for date in sorted(index["dates"]):
        board = read_selected_day(date, root=root)
        for game in board["games"]:
            if isinstance(game.get("final_audit"), dict):
                records.append(game["final_audit"])
            elif game.get("status", {}).get("completed") is True:
                records.append(
                    {
                        "season": season,
                        "game_id": game["game_id"],
                        "start_time": game["start_time"],
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "archived_at": board.get("live_score_collected_at")
                        or board.get("collected_at"),
                        "final": {
                            key: game.get(key) for key in ("status", "home_score", "away_score")
                        },
                    }
                )
    return records


def _actual_retrospective(game, selected, box, collected_at) -> dict[str, Any]:
    if box is None:
        game.setdefault("data_quality_errors", []).append("missing_official_box_score")
        return {}
    from nba_lineup_model.season.official_schedule_runtime import (
        _build_official_actual_minutes_retrospective,
        _duration_minutes,
    )

    try:
        # The official box, not the projected rotation, determines participation.
        actual_ids = {
            int(player["personId"])
            for side in ("homeTeam", "awayTeam")
            for player in box.get("game", {}).get(side, {}).get("players", [])
            if player.get("personId") is not None
            and _duration_minutes((player.get("statistics") or {}).get("minutes")) > 0
        }
        ledger = selected.get("projection", {}).get("ledger", {})
        saved_ids = {
            int(row["player_id"]) for side in ("home", "away")
            for row in ledger.get(side, {}).get("players", [])
            if row.get("player_id") is not None
        }
        missing = actual_ids - saved_ids
        payload = {}
        if missing:
            loader = frozen_rating_context.get()
            if loader is None:
                raise ValueError("Exact frozen model resolver is unavailable")
            ratings = loader(selected)
            payload = {"players": [{"player_id": pid, "projected_nail": ratings[pid]}
                                   for pid in sorted(missing) if pid in ratings]}
        result = _build_official_actual_minutes_retrospective(
            game,
            pregame=selected,
            boxscore=box,
            projection_payload=payload,
        )
        result["computed_at"] = collected_at
        return result
    except (ValueError, KeyError, TypeError, OSError) as error:
        game.setdefault("data_quality_errors", []).append(f"retrospective_unavailable:{error}")
        return {}
