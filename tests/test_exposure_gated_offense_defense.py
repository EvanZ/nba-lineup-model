from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.modeling.exposure_gated_offense_defense import (
    blend_od_cold_start_priors,
    combine_frozen_od_coefficients,
)


def test_od_exposure_gate_blends_each_side_independently() -> None:
    priors = blend_od_cold_start_priors(
        pd.DataFrame(
            {
                "player_id": [1, 2],
                "draft_offense_rapm": [2.0, -1.0],
                "draft_defense_rapm": [1.0, 3.0],
            }
        ),
        pd.DataFrame(
            {
                "player_id": [1, 2],
                "predicted_replacement_probability": [0.25, 0.75],
            }
        ),
        replacement_offense_rapm=-4.0,
        replacement_defense_rapm=-2.0,
    ).set_index("player_id")

    assert priors.loc[1, "offense_rapm"] == pytest.approx(0.5)
    assert priors.loc[1, "defense_rapm"] == pytest.approx(0.25)
    assert priors.loc[2, "offense_rapm"] == pytest.approx(-3.25)
    assert priors.loc[2, "defense_rapm"] == pytest.approx(-0.75)
    assert priors.loc[1, "net_rapm"] == pytest.approx(0.75)
    assert priors.loc[2, "net_rapm"] == pytest.approx(-4.0)


def test_od_cold_start_priors_cannot_overlap_returning_state() -> None:
    returning = pd.DataFrame(
        {"player_id": [1], "offense_rapm": [1.0], "defense_rapm": [2.0]}
    )
    rookies = pd.DataFrame(
        {"player_id": [1], "offense_rapm": [0.0], "defense_rapm": [0.0]}
    )

    with pytest.raises(ValueError, match="overlap"):
        combine_frozen_od_coefficients(returning, rookies)
