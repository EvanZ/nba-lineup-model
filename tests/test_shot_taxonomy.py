from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.shot_taxonomy import (
    SHOT_FAMILIES,
    build_season_shot_profiles,
    classify_shot_events,
)


def test_shot_taxonomy_coarsens_historical_labels_without_dropping_twos() -> None:
    events = pd.DataFrame(
        {
            "event_type": ["2pt", "2pt", "2pt", "3pt", "2pt"],
            "event_subtype": ["Driving Layup Shot", "Slam Dunk Shot", "Hook Shot", "Jump Shot", None],
            "player_id": [1, 1, 2, 2, 3],
            "shot_result": ["Made", "Missed", "Made", "Missed", "Made"],
        }
    )

    classified = classify_shot_events(events)

    assert classified["shot_family"].tolist() == ["rim", "rim", "non_rim_two", "three", "non_rim_two"]
    assert classified["has_final_result"].all()


def test_shot_profiles_use_possessions_and_shrink_low_volume_finishing() -> None:
    panel = pd.DataFrame(
        {
            "season": ["2025-26", "2025-26", "2025-26"],
            "season_start_year": [2025, 2025, 2025],
            "player_id": [1, 2, 3],
            "player_name": ["Rim Player", "Three Player", "No Attempts"],
            "rapm_possessions": [100.0, 100.0, 100.0],
        }
    )
    events = pd.DataFrame(
        {
            "event_type": ["2pt", "2pt", "2pt", "2pt", "3pt", "3pt"],
            "event_subtype": ["Layup", "DUNK", "Layup", "Hook Shot", "Jump Shot", "Jump Shot"],
            "player_id": [1, 1, 2, 2, 2, 2],
            "shot_result": ["Made", "Missed", "Made", "Made", "Made", "Missed"],
        }
    )

    profiles, references, coverage = build_season_shot_profiles(
        "2025-26", panel, classify_shot_events(events)
    )
    rim = profiles.loc[profiles["player_id"].eq(1)].iloc[0]
    no_attempts = profiles.loc[profiles["player_id"].eq(3)].iloc[0]

    assert tuple(references["shot_family"]) == SHOT_FAMILIES
    assert rim["rim_attempts"] == 2
    assert rim["rim_attempts_per_100"] > 0.0
    assert rim["rim_fg_pct"] == 0.5
    assert rim["rim_fg_pct"] < rim["rim_fg_pct_shrunk"] < 1.0
    assert no_attempts["shot_profile_available"] == False
    assert no_attempts["three_attempts_per_100"] > 0.0
    assert coverage["panel_matched_shot_event_count"] == 6
