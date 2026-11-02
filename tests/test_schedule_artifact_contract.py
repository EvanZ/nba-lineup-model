from __future__ import annotations

import copy
import json

import pytest

from nba_lineup_model.season.schedule_artifact_contract import (
    ScheduleArtifactContractError,
    validate_schedule_artifacts,
)


def _complete_tree(tmp_path):
    game_id = "0022600001"
    projection = {"ledger": {"away": {}, "home": {}}}
    retrospective = {"actual_minutes": {"away": [], "home": []}, "ledger": {}}
    game = {
        "game_id": game_id,
        "projection": projection,
        "retrospective": retrospective,
        "status": {"state": "post", "completed": True},
    }
    latest = tmp_path / "latest"
    frozen = tmp_path / "frozen"
    final = tmp_path / "final"
    latest.mkdir()
    frozen.mkdir()
    final.mkdir()
    (latest / "2026-10-21.json").write_text(
        json.dumps(
            {
                "source": "NBA official schedule and liveData",
                "source_provenance": {"provider_fallbacks": []},
                "games": [game],
            }
        )
    )
    (frozen / f"{game_id}.json").write_text(
        json.dumps({"game_id": game_id, "projection": projection})
    )
    (final / f"{game_id}.summary.json").write_text(
        json.dumps(
            {
                "game_id": game_id,
                "retrospective": retrospective,
                "grading": {"model_favorite_won": True},
            }
        )
    )
    (final / f"{game_id}.nba.json").write_text(
        json.dumps({"game": {"gameId": game_id}})
    )
    return game_id


def test_complete_official_lifecycle_passes(tmp_path) -> None:
    _complete_tree(tmp_path)
    result = validate_schedule_artifacts(tmp_path)
    assert result["status"] == "clean"
    assert result["completed_game_count"] == 1
    assert result["espn_reference_count"] == 0


@pytest.mark.parametrize("contamination", ["filename", "field", "value"])
def test_any_espn_contamination_fails_closed(tmp_path, contamination) -> None:
    game_id = _complete_tree(tmp_path)
    latest_path = tmp_path / "latest" / "2026-10-21.json"
    payload = json.loads(latest_path.read_text())
    if contamination == "filename":
        (tmp_path / "final" / f"{game_id}.espn.json").write_text("{}")
    elif contamination == "field":
        payload["games"][0]["espn_game_id"] = "401000001"
        latest_path.write_text(json.dumps(payload))
    else:
        payload["source"] = "ESPN scoreboard"
        latest_path.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactContractError, match="forbidden"):
        validate_schedule_artifacts(tmp_path)
