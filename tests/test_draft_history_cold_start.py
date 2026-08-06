from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.draft_history_cold_start import (
    draft_history_profiles,
    score_draft_history_profiles,
)


class _DraftModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.repeat(-1.0, len(features))


class _ExposureModel:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([0.75, 0.25]), (len(features), 1))


def test_draft_history_profiles_and_blend_preserve_string_identifiers() -> None:
    source = pd.DataFrame(
        {
            "season": ["2026-27"],
            "draft_year": [2026],
            "player_id": pd.Series(["1649999"], dtype="string"),
            "player_name": ["First Pick"],
            "draft_round": [1],
            "draft_round_pick": [1],
            "draft_number": [1],
            "draft_team_abbreviation": ["BOS"],
            "affiliation": ["School One"],
        }
    )
    reference = pd.DataFrame(
        {
            "draft_number": [1, 30],
            "is_undrafted": [False, False],
            "draft_year": [2020, 2020],
            "season_start_year": [2020, 2020],
            "age": [20.0, 22.0],
            "height_inches": [78.0, 80.0],
            "weight_pounds": [210.0, 230.0],
            "draft_pick": [1.0, 30.0],
            "draft_age": [20.0, 22.0],
            "body_mass_index": [24.3, 25.3],
        }
    )

    rankings = score_draft_history_profiles(
        draft_history_profiles(
            source,
            season="2026-27",
            roster_profiles=pd.DataFrame(
                {
                    "player_id": pd.Series(["1649999"], dtype="string"),
                    "team_id": pd.Series(["1610612738"], dtype="string"),
                    "team_abbreviation": ["BOS"],
                    "listed_position": ["F"],
                    "age": [20.0],
                    "height_inches": [80.0],
                    "weight_pounds": [220.0],
                }
            ),
        ),
        draft_model=_DraftModel(),
        exposure_model=_ExposureModel(),
        draft_training=reference,
        exposure_training=reference,
        replacement_rapm=-5.0,
    )

    row = rankings.iloc[0]
    assert rankings["player_id"].dtype.name == "string"
    assert row["player_id"] == "1649999"
    assert row["draft_rate_rapm"] == -1.0
    assert row["low_exposure_probability"] == 0.25
    assert row["cold_start_rapm_prior"] == -2.0
    assert row["listed_position"] == "F"
    assert row["height_inches"] == 80.0
    assert row["roster_team_abbreviation"] == "BOS"
