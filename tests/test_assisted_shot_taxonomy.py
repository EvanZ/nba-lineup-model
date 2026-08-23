from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.assisted_shot_taxonomy import (
    build_season_assisted_shot_profiles,
    classify_assisted_shot_events,
    reconcile_season_assisted_shots,
)


def test_assisted_shot_classification_tracks_all_three_families() -> None:
    events = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1", "g1"],
            "team_id": [1, 1, 1, 1],
            "team_tricode": ["ONE"] * 4,
            "event_type": ["2pt", "2pt", "3pt", "2pt"],
            "event_subtype": ["Layup", "Hook Shot", "Jump Shot", "DUNK"],
            "player_id": [11, 12, 13, 14],
            "shot_result": ["Made", "Made", "Made", "Missed"],
            "description": [
                "A. One Layup (B. Two 1 AST)",
                "C. Three Hook Shot (2 PTS)",
                None,
                "D. Four DUNK (E. Five 1 AST)",
            ],
        }
    )

    classified = classify_assisted_shot_events(events)

    assert classified["shot_family"].tolist() == ["rim", "non_rim_two", "three", "rim"]
    assert classified["assist_status"].tolist() == ["assisted", "unassisted", "unknown", "assisted"]
    assert classified["is_assisted"].tolist() == [True, False, False, True]


def test_assisted_shot_classification_keeps_metadata_aligned_after_non_shots() -> None:
    events = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "team_id": [1, 2, 1],
            "team_tricode": ["ONE", "TWO", "ONE"],
            "event_type": ["timeout", "2pt", "3pt"],
            "event_subtype": [None, "Layup", "Jump Shot"],
            "player_id": [None, 11, 12],
            "shot_result": [None, "Made", "Made"],
            "description": ["Timeout", "Two Layup", "Three 3PT (Two 1 AST)"],
        }
    )

    classified = classify_assisted_shot_events(events)

    assert classified["team_tricode"].tolist() == ["TWO", "ONE"]
    assert classified["assist_status"].tolist() == ["unassisted", "assisted"]


def test_assisted_shot_profiles_keep_unknown_separate_from_unassisted() -> None:
    panel = pd.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "season_start_year": [2025, 2025],
            "player_id": [1, 2],
            "player_name": ["One", "Two"],
            "rapm_possessions": [100.0, 200.0],
        }
    )
    events = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "team_id": [1, 1, 1],
            "team_tricode": ["ONE"] * 3,
            "event_type": ["2pt", "2pt", "3pt"],
            "event_subtype": ["Layup", "Hook", "Jump Shot"],
            "player_id": [1, 1, 2],
            "shot_result": ["Made", "Made", "Made"],
            "description": ["One Layup (Two 1 AST)", "One Hook", None],
        }
    )

    profiles, coverage = build_season_assisted_shot_profiles(
        "2025-26", panel, classify_assisted_shot_events(events)
    )
    one = profiles.loc[profiles["player_id"].eq(1)].iloc[0]
    two = profiles.loc[profiles["player_id"].eq(2)].iloc[0]

    assert one["assisted_rim_makes"] == 1
    assert one["unassisted_non_rim_two_makes"] == 1
    assert one["unknown_assist_status_makes"] == 0
    assert two["unknown_three_makes"] == 1
    assert two["unassisted_makes"] == 0
    assert two["assist_status_coverage"] == 0.0
    assert coverage["unknown_assist_status_made_shot_count"] == 1


def test_reconciliation_matches_player_box_totals(tmp_path) -> None:
    players = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "team_id": [1, 1, 2],
            "team_tricode": ["ONE", "ONE", "TWO"],
            "statistics_fieldGoalsMade": [1, 1, 1],
            "statistics_twoPointersMade": [1, 0, 1],
            "statistics_threePointersMade": [0, 1, 0],
            "statistics_assists": [1, 0, 0],
        }
    )
    players.to_parquet(tmp_path / "g1.parquet", index=False)
    events = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "team_id": [1, 1, 2],
            "team_tricode": ["ONE", "ONE", "TWO"],
            "event_type": ["2pt", "3pt", "2pt"],
            "event_subtype": ["Layup", "Jump Shot", "Hook"],
            "player_id": [1, 2, 3],
            "shot_result": ["Made", "Made", "Made"],
            "description": ["One Layup (Two 1 AST)", "Two 3PT", "Three Hook"],
        }
    )

    result = reconcile_season_assisted_shots(
        "2025-26", classify_assisted_shot_events(events), tmp_path
    )
    one = result.loc[result["team_id"].eq(1)].iloc[0]

    assert one["fgm_exact"] == True
    assert one["two_pm_exact"] == True
    assert one["three_pm_exact"] == True
    assert one["assists_exact"] == True
