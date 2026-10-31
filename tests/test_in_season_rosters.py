from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_lineup_model.players.in_season_rosters import (
    _merge_minutes_catalog,
    normalize_player_name,
    reconcile_current_roster,
)
from nba_lineup_model.web_api.preseason_rankings_cache import append_target_bios


def test_runtime_target_bios_can_reuse_history_without_packaged_draft_file(
    tmp_path: Path,
) -> None:
    panel = pd.DataFrame(
        [
            {
                "season": "2025-26",
                "season_start_year": 2025,
                "player_id": 1631104,
                "player_name": "Blake Wesley",
                "age": 22.0,
                "listed_position": "G",
                "height_inches": 76.0,
                "weight_pounds": 190.0,
                "nba_experience_years": 3,
                "is_rookie": False,
                "rapm_seconds": 100.0,
                "rapm_exposure_eligible": True,
                "rapm_possessions": 50.0,
                "draft_year": 2022.0,
                "draft_round": 1.0,
                "draft_number": 25.0,
                "is_undrafted": False,
            }
        ]
    )
    roster = pd.DataFrame(
        [
            {
                "player_id": 1631104,
                "player_name": "Blake Wesley",
                "age": 23.0,
                "listed_position": "G",
                "height_inches": 76,
                "weight_pounds": 190,
                # Runtime roster sources can report zero even for a player who
                # has a completed prior-season panel row.
                "experience": "0",
            }
        ]
    )

    output = append_target_bios(
        panel,
        roster,
        target_season="2026-27",
        draft_history_path=tmp_path / "not-packaged.parquet",
        allow_missing_draft_history=True,
    )

    target = output.loc[output["season"].eq("2026-27")].iloc[0]
    assert int(target["player_id"]) == 1631104
    assert int(target["draft_year"]) == 2022
    assert int(target["draft_number"]) == 25
    assert int(target["nba_experience_years"]) == 4
    assert not bool(target["is_rookie"])


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
