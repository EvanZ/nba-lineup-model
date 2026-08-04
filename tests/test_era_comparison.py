from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.era_comparison import (
    _model_comparison,
    _player_season_exposure,
    _standardize_player_seasons,
)


def test_player_exposure_sums_multiple_teams_and_keeps_primary_team() -> None:
    stints = pd.DataFrame(
        {
            "season": ["2020-21", "2020-21"],
            "home_team_id": [1, 3],
            "away_team_id": [2, 2],
            "home_team_tricode": ["AAA", "CCC"],
            "away_team_tricode": ["BBB", "BBB"],
            "home_player_ids": [[10, 11, 12, 13, 14], [10, 31, 32, 33, 34]],
            "away_player_ids": [[20, 21, 22, 23, 24], [20, 21, 22, 23, 24]],
            "duration_seconds": [100.0, 200.0],
        }
    )

    exposure = _player_season_exposure(stints)

    player = exposure.loc[exposure["player_id"].eq(10)].iloc[0]
    assert player.seconds == pytest.approx(300.0)
    assert player.team_count == 2
    assert player.team_tricode == "CCC"


def test_standardization_uses_exposure_weighted_season_reference() -> None:
    coefficients = pd.DataFrame(
        {
            "season": ["2020-21"] * 3,
            "player_id": [1, 2, 3],
            "rapm": [-1.0, 0.0, 2.0],
        }
    )
    exposure = pd.DataFrame(
        {
            "season": ["2020-21"] * 3,
            "player_id": [1, 2, 3],
            "team_id": [1, 1, 1],
            "team_tricode": ["AAA"] * 3,
            "team_count": [1] * 3,
            "seconds": [1.0, 1.0, 2.0],
            "minutes": [1 / 60, 1 / 60, 2 / 60],
        }
    )
    catalog = pd.DataFrame({"player_id": [1, 2, 3], "display_name": ["One", "Two", "Three"]})

    result = _standardize_player_seasons(coefficients, exposure, catalog)

    expected_mean = 0.75
    expected_scale = np.sqrt(1.6875)
    assert result["season_rapm_mean"].tolist() == pytest.approx([expected_mean] * 3)
    assert result["season_rapm_scale"].tolist() == pytest.approx([expected_scale] * 3)
    assert np.average(result["era_standardized_rapm"], weights=result["seconds"]) == pytest.approx(
        0.0
    )
    assert np.average(
        result["era_standardized_rapm"] ** 2, weights=result["seconds"]
    ) == pytest.approx(1.0)


def test_model_comparison_aligns_shared_player_seasons() -> None:
    base = pd.DataFrame(
        {
            "season": ["2020-21"],
            "player_id": [1],
            "player_name": ["One"],
            "minutes": [2_000.0],
            "rapm": [4.0],
            "era_standardized_rapm": [2.0],
            "wins_above_average_per_reference_minutes": [5.0],
            "qualified": [True],
        }
    )
    canonical = base.assign(rapm=3.0, era_standardized_rapm=1.5).copy()
    canonical["wins_above_average_per_reference_minutes"] = 3.75

    result = _model_comparison(base, canonical)

    assert len(result) == 1
    assert result.loc[0, "era_standardized_rapm_difference"] == pytest.approx(0.5)
    assert result.loc[0, "wins_per_reference_minutes_difference"] == pytest.approx(1.25)
