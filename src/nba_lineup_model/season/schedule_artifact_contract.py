"""Fail-closed validation for the complete NBA GESTALT schedule lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class ScheduleArtifactContractError(RuntimeError):
    """Raised when a schedule tree cannot be promoted or served safely."""


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ScheduleArtifactContractError(f"Expected a JSON object: {path}")
    return payload


def _espn_references(value: object, *, path: str = "$") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if "espn" in str(key).casefold():
                matches.append(child_path)
            matches.extend(_espn_references(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(_espn_references(child, path=f"{path}[{index}]"))
    elif isinstance(value, str) and "espn" in value.casefold():
        matches.append(path)
    return matches


def validate_schedule_artifacts(root: Path | str) -> dict[str, object]:
    state_root = Path(root).resolve()
    selections = None
    if (state_root / "forecast-selections.json").is_file():
        from nba_lineup_model.season.serving_selection import verify_schedule_selection
        try:
            selections = verify_schedule_selection(state_root)
        except (OSError, ValueError, KeyError) as error:
            raise ScheduleArtifactContractError(f"Invalid forecast selection: {error}") from error
    failures: list[str] = []
    latest_paths = sorted((state_root / "latest").glob("*.json"))
    if not latest_paths:
        failures.append("latest:no schedule boards")

    tree_digest = hashlib.sha256()
    artifact_paths = sorted(path for path in state_root.rglob("*") if path.is_file())
    for path in artifact_paths:
        if not path.is_file():
            continue
        relative = path.relative_to(state_root).as_posix()
        tree_digest.update(relative.encode())
        tree_digest.update(hashlib.sha256(path.read_bytes()).digest())
        if "espn" in relative.casefold():
            failures.append(f"forbidden-filename:{relative}")
        if path.suffix != ".json":
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"invalid-json:{relative}:{exc}")
            continue
        for reference in _espn_references(payload):
            failures.append(f"forbidden-espn-reference:{relative}:{reference}")

    game_ids: set[str] = set()
    completed_ids: set[str] = set()
    started_ids: set[str] = set()
    game_count = 0
    legacy_frozen_attribution_count = 0
    for latest_path in latest_paths:
        payload = _read_object(latest_path)
        source = str(payload.get("source") or "")
        if "NBA official" not in source:
            failures.append(f"unapproved-board-source:{latest_path.name}:{source}")
        provenance = payload.get("source_provenance")
        fallbacks = provenance.get("provider_fallbacks") if isinstance(provenance, dict) else None
        if fallbacks not in (None, []):
            failures.append(f"provider-fallbacks:{latest_path.name}:{fallbacks}")
        games = payload.get("games")
        if not isinstance(games, list):
            failures.append(f"invalid-games:{latest_path.name}")
            continue
        for game in games:
            if not isinstance(game, dict):
                failures.append(f"invalid-game:{latest_path.name}")
                continue
            game_count += 1
            game_id = str(game.get("game_id") or "")
            if not game_id:
                failures.append(f"missing-game-id:{latest_path.name}")
                continue
            if game_id in game_ids:
                failures.append(f"duplicate-game-id:{game_id}")
            game_ids.add(game_id)
            projection = game.get("projection")
            if not isinstance(projection, dict) or not isinstance(
                projection.get("ledger"), dict
            ):
                failures.append(f"missing-latest-ledger:{game_id}")
            status = game.get("status") if isinstance(game.get("status"), dict) else {}
            state = str(status.get("state") or "pre")
            completed = bool(status.get("completed")) or state == "post"
            started = completed or state == "in"
            if started:
                started_ids.add(game_id)
                frozen_path = state_root / "frozen" / f"{game_id}.json"
                frozen = None
                if selections is not None:
                    # The verified #55 map pins the exact forecast and market.
                    # Never synthesize or require a second legacy frozen copy.
                    frozen = game.get("pregame", game)
                elif frozen_path.is_file():
                    frozen = _read_object(frozen_path)
                if frozen is None:
                    failures.append(f"missing-frozen:{game_id}")
                else:
                    frozen_projection = frozen.get("projection")
                    if not isinstance(frozen_projection, dict):
                        failures.append(f"missing-frozen-projection:{game_id}")
                    elif not isinstance(frozen_projection.get("ledger"), dict):
                        frozen_bundle = str(
                            frozen_projection.get("serving_bundle_sha256") or ""
                        )
                        latest_bundle = str(
                            projection.get("serving_bundle_sha256") or ""
                        ) if isinstance(projection, dict) else ""
                        if not frozen_bundle or frozen_bundle != latest_bundle:
                            failures.append(
                                f"unreconciled-legacy-frozen-ledger:{game_id}"
                            )
                        else:
                            legacy_frozen_attribution_count += 1
            if not completed:
                continue
            completed_ids.add(game_id)
            summary_path = state_root / "final" / f"{game_id}.summary.json"
            boxscore_path = state_root / "final" / f"{game_id}.nba.json"
            if not summary_path.is_file():
                failures.append(f"missing-final-summary:{game_id}")
                continue
            if not boxscore_path.is_file():
                failures.append(f"missing-official-boxscore:{game_id}")
            summary = _read_object(summary_path)
            retrospective = summary.get("retrospective")
            if not isinstance(retrospective, dict):
                failures.append(f"missing-final-retrospective:{game_id}")
            else:
                if not isinstance(retrospective.get("actual_minutes"), dict):
                    failures.append(f"missing-actual-minutes:{game_id}")
                if not isinstance(retrospective.get("ledger"), dict):
                    failures.append(f"missing-retrospective-ledger:{game_id}")
            grading = summary.get("grading")
            if not isinstance(grading, dict) or not isinstance(
                grading.get("model_favorite_won"), bool
            ):
                failures.append(f"missing-winner-grading:{game_id}")
            if not isinstance(game.get("retrospective"), dict):
                failures.append(f"missing-latest-retrospective:{game_id}")

    summary_ids = {
        path.name.removesuffix(".summary.json")
        for path in (state_root / "final").glob("*.summary.json")
    }
    boxscore_ids = {
        path.name.removesuffix(".nba.json")
        for path in (state_root / "final").glob("*.nba.json")
    }
    for game_id in sorted(summary_ids - game_ids):
        failures.append(f"orphan-final-summary:{game_id}")
    for game_id in sorted(boxscore_ids - game_ids):
        failures.append(f"orphan-official-boxscore:{game_id}")

    if failures:
        rendered = "\n".join(f"- {failure}" for failure in sorted(set(failures)))
        raise ScheduleArtifactContractError(
            f"Schedule artifact contract failed for {state_root}:\n{rendered}"
        )
    return {
        "state_root": str(state_root),
        "latest_day_count": len(latest_paths),
        "game_count": game_count,
        "started_game_count": len(started_ids),
        "completed_game_count": len(completed_ids),
        "legacy_frozen_attribution_count": legacy_frozen_attribution_count,
        "tree_sha256": tree_digest.hexdigest(),
        "espn_reference_count": 0,
        "status": "clean",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_schedule_artifacts(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
