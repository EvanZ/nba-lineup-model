from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_lineup_model.players.in_season_rosters import (
    _extend_runtime_minutes_catalog,
    _merge_minutes_catalog,
    _validate_coupled_preseason_prior_catalogs,
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


def test_reconcile_current_roster_uses_nba_ids_even_when_names_or_teams_change():
    from nba_lineup_model.players.in_season_rosters import ROSTER_COLUMNS
    rows = []
    for player_id, name, team_id in [(1626171, "Bobby Portis Jr.", "1610612748"), (1631104, "Blake Wesley", "1610612746")]:
        row = dict.fromkeys(ROSTER_COLUMNS, "")
        row.update(player_id=str(player_id), player_name=name, team_id=team_id, season="2026-27")
        rows.append(row)
    current, unresolved = reconcile_current_roster(
        pd.DataFrame(rows), frozen_roster=pd.DataFrame(),
        historical_panel=pd.DataFrame(), movement_name_to_id={},
    )
    assert unresolved.empty
    assert set(current.player_id) == {1626171, 1631104}
    assert set(current.identity_source) == {"nba_commonteamroster"}
    by_id = current.set_index("player_id")
    assert by_id.loc[1626171, "team_abbreviation"] == "MIA"
    assert by_id.loc[1631104, "team_abbreviation"] == "LAC"
    duplicate = pd.concat([pd.DataFrame(rows), pd.DataFrame(rows[:1])])
    with pytest.raises(ValueError, match="invalid player identities"):
        reconcile_current_roster(duplicate, frozen_roster=pd.DataFrame(), historical_panel=pd.DataFrame(), movement_name_to_id={})


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


def test_runtime_minutes_append_uses_last_observed_available_game_role(
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
                            "team": "LAC",
                            "position": "G",
                            "conditional_minutes_per_game": 30.0,
                            "has_prior_minutes_state": True,
                            "uses_draft_cold_start_prior": False,
                        },
                        {
                            "player_id": 3,
                            "player_name": "Cold Start",
                            "team": "LAC",
                            "position": "F",
                            "conditional_minutes_per_game": 8.0,
                            "has_prior_minutes_state": False,
                            "uses_draft_cold_start_prior": True,
                        },
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
                "listed_position": "G",
                "age": 27,
            },
            {
                "player_id": "2",
                "player_name": "New Player",
                "team_abbreviation": "LAC",
                "listed_position": "G",
                "age": 23,
            },
        ]
    )
    rankings = pd.DataFrame([{"player_id": 2, "rapm": -1.25}])
    rotation_path = tmp_path / "rotation.parquet"
    pd.DataFrame(
        [
            {
                "season": "2025-26",
                "season_start_year": 2025,
                "player_id": 2,
                "actual_minutes_per_available_game": 12.5,
                "actual_total_minutes": 250.0,
            }
        ]
    ).to_parquet(rotation_path, index=False)
    output = tmp_path / "overlay.json"

    additions = _extend_runtime_minutes_catalog(
        base_cache_path=frozen_path,
        prior_overlay_path=None,
        rankings=rankings,
        current_roster=current,
        rotation_history_path=rotation_path,
        output_path=output,
    )

    rows = {
        int(row["player_id"]): row
        for row in json.loads(output.read_text())["minutes"]["players"]
    }
    assert additions == [2]
    assert rows[1]["conditional_minutes_per_game"] == 30.0
    assert rows[2]["conditional_minutes_per_game"] == 12.5
    assert rows[2]["baseline_minutes_per_game"] == 0.0
    assert rows[2]["projected_nail"] == -1.25


def test_coupled_preseason_prior_catalogs_require_identical_player_ids(
    tmp_path: Path,
) -> None:
    rankings = tmp_path / "rankings.parquet"
    pd.DataFrame([{"player_id": 1}, {"player_id": 2}]).to_parquet(
        rankings, index=False
    )
    minutes = tmp_path / "minutes.json"
    minutes.write_text(
        json.dumps({"minutes": {"players": [{"player_id": 1}]}})
    )

    with pytest.raises(
        ValueError,
        match=r"ratings_only=\[2\]; minutes_only=\[\]",
    ):
        _validate_coupled_preseason_prior_catalogs(
            rankings_path=rankings,
            minutes_path=minutes,
        )

    minutes.write_text(
        json.dumps(
            {"minutes": {"players": [{"player_id": 1}, {"player_id": 2}]}}
        )
    )
    _validate_coupled_preseason_prior_catalogs(
        rankings_path=rankings,
        minutes_path=minutes,
    )
