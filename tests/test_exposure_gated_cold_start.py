from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.modeling.exposure_gated_cold_start import (
    blend_cold_start_prior_components,
)


def test_exposure_gate_continuously_blends_draft_and_replacement_rates() -> None:
    rankings = blend_cold_start_prior_components(
        _draft_rankings(),
        _exposure_predictions(),
        replacement_rapm=-4.0,
    ).set_index("player_id")

    assert rankings.loc[1, "blended_cold_start_prior"] == pytest.approx(1.4)
    assert rankings.loc[2, "blended_cold_start_prior"] == pytest.approx(-3.4)
    assert rankings.loc[1, "rank"] == 1
    assert rankings.loc[2, "rank"] == 2


def test_exposure_gate_requires_matching_target_cohorts() -> None:
    exposure = _exposure_predictions().iloc[:1]

    with pytest.raises(ValueError, match="do not match"):
        blend_cold_start_prior_components(
            _draft_rankings(), exposure, replacement_rapm=-4.0
        )


def _draft_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["One", "Two"],
            "listed_position": ["G", "F"],
            "draft_status": ["drafted_1_60", "undrafted"],
            "draft_number": [1.0, None],
            "draft_prior": [2.0, -1.0],
        }
    )


def _exposure_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2],
            "draft_age": [19.0, 22.0],
            "predicted_replacement_probability": [0.1, 0.8],
            "predicted_rotation_probability": [0.9, 0.2],
        }
    )
