from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.schedule_controls import (
    BACK_TO_BACK_COLUMN,
    attach_back_to_back_feature,
    build_back_to_back_game_features,
    fit_back_to_back_schedule_model,
)


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["001", "002", "003", "004"],
            "season": ["2024-25"] * 4,
            "game_date": ["2024-10-20", "2024-10-21", "2024-10-22", "2024-10-22"],
            "home_team_id": [1, 3, 1, 4],
            "away_team_id": [2, 1, 3, 1],
        }
    )


def test_back_to_back_flags_follow_the_full_team_calendar() -> None:
    features = build_back_to_back_game_features(_catalog()).set_index("game_id")

    assert features.loc["001", ["home_back_to_back", "away_back_to_back"]].tolist() == [0, 0]
    assert features.loc["002", ["home_back_to_back", "away_back_to_back"]].tolist() == [0, 1]
    assert features.loc["003", ["home_back_to_back", "away_back_to_back"]].tolist() == [1, 1]
    # A second game on one calendar date is not another back-to-back.
    assert features.loc["004", ["home_back_to_back", "away_back_to_back"]].tolist() == [0, 0]
    assert features.loc["002", BACK_TO_BACK_COLUMN] == -1.0


def test_attach_back_to_back_feature_preserves_one_control_per_game() -> None:
    features = build_back_to_back_game_features(_catalog())
    rows = pd.DataFrame({"game_id": ["002", "002", "003"], "value": [1, 2, 3]})

    attached = attach_back_to_back_feature(rows, features)

    assert attached[BACK_TO_BACK_COLUMN].tolist() == [-1.0, -1.0, 0.0]


def test_back_to_back_model_is_antisymmetric_by_construction() -> None:
    stints = pd.DataFrame(
        {
            BACK_TO_BACK_COLUMN: [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
            "possessions": [10.0] * 6,
        }
    )
    model = fit_back_to_back_schedule_model(
        stints,
        np.array([2.0, 2.0, 0.0, 0.0, -2.0, -2.0]),
        alpha=1.0,
        source_season="2024-25",
    )
    schedule = pd.DataFrame(
        {"game_id": ["home", "neutral", "away"], BACK_TO_BACK_COLUMN: [1.0, 0.0, -1.0]}
    )
    predicted = model.predict_games(
        pd.DataFrame({"game_id": ["home", "neutral", "away"]}),
        schedule,
    )

    assert predicted[0] == -predicted[2]
    assert abs(predicted[1]) < 1e-12
