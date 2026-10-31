from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_lineup_model.players.in_season_rosters import (
    _merge_minutes_catalog,
    normalize_player_name,
    reconcile_current_roster,
)


def test_reconcile_current_roster_resolves_suffixes_and_historical_players() -> None:
    espn = pd.DataFrame(
        [
            {
                "team": "MIA",
                "espn_player_id": "3064482",
                "player_name": "Bobby Portis",
                "listed_position": "F",
                "height_inches": 82,
                "height_raw": "6' 10\"",
                "weight_pounds": 250,
                "birth_date": "1995-02-10T08:00Z",
                "age": 31,
                "experience": "11",
                "school": "Arkansas",
                "espn_status": "active",
            },
            {
                "team": "LAC",
                "espn_player_id": "4683935",
                "player_name": "Blake Wesley",
                "listed_position": "G",
                "height_inches": 76,
                "height_raw": "6' 4\"",
                "weight_pounds": 190,
                "birth_date": "2003-03-16T08:00Z",
                "age": 23,
                "experience": "5",
                "school": "Notre Dame",
                "espn_status": "active",
            },
            {
                "team": "LAC",
                "espn_player_id": "999",
                "player_name": "Unmapped Player",
                "listed_position": "G",
                "height_inches": 75,
                "height_raw": "6' 3\"",
                "weight_pounds": 185,
                "birth_date": "",
                "age": 22,
                "experience": "0",
                "school": "",
                "espn_status": "active",
            },
        ]
    )
    frozen = pd.DataFrame(
        [
            {
                "season": "2026-27",
                "team_id": "1610612749",
                "team_abbreviation": "MIL",
                "player_id": "1626171",
                "player_name": "Bobby Portis Jr.",
                "listed_position": "F",
                "height_raw": "6-10",
                "height_inches": 82,
                "weight_pounds": 250.0,
                "birth_date": "FEB 10, 1995",
                "age": 31.0,
                "experience": "10",
                "school": "Arkansas",
                "how_acquired": "Signed",
                "supplemental_status": "0",
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "player_id": 1631104,
                "player_name": "Blake Wesley",
                "season_start_year": 2025,
                "listed_position": "G",
                "height_inches": 76,
                "weight_pounds": 190.0,
                "age": 23.0,
                "nba_experience_years": 3,
                "college": "Notre Dame",
            }
        ]
    )

    current, unresolved = reconcile_current_roster(
        espn,
        frozen_roster=frozen,
        historical_panel=history,
        movement_name_to_id={},
    )

    assert normalize_player_name("Bobby Portis Jr.") == normalize_player_name(
        "Bobby Portis"
    )
    assert set(current["player_id"]) == {"1626171", "1631104"}
    assert current.set_index("player_id").loc["1631104", "team_abbreviation"] == "LAC"
    assert unresolved["player_name"].tolist() == ["Unmapped Player"]


def test_minutes_overlay_appends_new_player_without_rewriting_frozen_prior(
    tmp_path: Path,
) -> None:
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(
        json.dumps(
            {
                "minutes": {
                    "players": [
                        {
                            "player_id": 1,
                            "player_name": "Incumbent",
                            "team": "OLD",
                            "team_id": 1,
                            "conditional_minutes_per_game": 30.0,
                        }
                    ]
                }
            }
        )
    )
    current = pd.DataFrame(
        [
            {
                "player_id": "1",
                "player_name": "Incumbent",
                "team_abbreviation": "LAC",
                "team_id": "1610612746",
            },
            {
                "player_id": "2",
                "player_name": "New Player",
                "team_abbreviation": "LAC",
                "team_id": "1610612746",
            },
        ]
    )
    output = tmp_path / "overlay.json"

    additions = _merge_minutes_catalog(
        base_cache_path=frozen_path,
        prior_overlay_path=None,
        candidate={
            "players": [
                {
                    "player_id": 1,
                    "player_name": "Incumbent",
                    "team": "LAC",
                    "team_id": 1610612746,
                    "conditional_minutes_per_game": 12.0,
                },
                {
                    "player_id": 2,
                    "player_name": "New Player",
                    "team": "LAC",
                    "team_id": 1610612746,
                    "conditional_minutes_per_game": 18.0,
                },
            ]
        },
        current_roster=current,
        output_path=output,
    )

    rows = {
        int(row["player_id"]): row
        for row in json.loads(output.read_text())["minutes"]["players"]
    }
    assert additions == [2]
    assert rows[1]["conditional_minutes_per_game"] == 30.0
    assert rows[1]["team"] == "LAC"
    assert rows[2]["conditional_minutes_per_game"] == 18.0
