from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.rotation.forward_nail_roster_squash import (
    _season_allocation_targets,
    apply_forward_nail_tilt,
    assert_full_regular_season_minute_coverage,
)


def _raw_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "team": ["ONE", "ONE", "ONE"],
            "player_id": [1, 2, 3],
            "player_name": ["One", "Two", "Three"],
            "raw_expected_total_minutes": [100.0, 80.0, 70.0],
        }
    )


def test_zero_nail_beta_exactly_recovers_raw_weight_and_rotation_selection() -> None:
    output = apply_forward_nail_tilt(
        _raw_predictions(),
        preseason_nail=pd.Series({1: -3.0, 2: 0.0, 3: 3.0}),
        nail_beta=0.0,
        initial_rotation_size=2,
    ).set_index("player_id")

    assert output["nail_tilt"].to_dict() == pytest.approx({1: 1.0, 2: 1.0, 3: 1.0})
    assert output["tilted_raw_expected_total_minutes"].to_dict() == pytest.approx(
        {1: 100.0, 2: 80.0, 3: 70.0}
    )
    assert output["is_baseline_rotation_candidate"].to_dict() == {1: True, 2: True, 3: False}
    assert output["raw_expected_total_minutes"].to_dict() == pytest.approx(
        {1: 100.0, 2: 80.0, 3: 0.0}
    )


def test_positive_nail_beta_can_change_pre_normalization_rotation_selection() -> None:
    output = apply_forward_nail_tilt(
        _raw_predictions(),
        preseason_nail=pd.Series({1: -5.0, 2: 0.0, 3: 5.0}),
        nail_beta=1.0,
        initial_rotation_size=2,
    ).set_index("player_id")

    assert output.loc[3, "nail_tilt"] > output.loc[1, "nail_tilt"]
    assert output.loc[3, "is_baseline_rotation_candidate"]
    assert not output.loc[1, "is_baseline_rotation_candidate"]
    assert np.isclose(output.loc[1, "raw_expected_total_minutes"], 0.0)


def test_default_target_uses_the_entire_regular_season() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "team_id": [1, 1, 1, 1],
            "team_tricode": ["ONE", "ONE", "ONE", "ONE"],
            "player_id": [1, 2, 1, 3],
            "minutes": [120.0, 120.0, 60.0, 180.0],
            "game_time_utc": [
                "2025-10-01T00:00:00Z",
                "2025-10-01T00:00:00Z",
                "2025-10-03T00:00:00Z",
                "2025-10-03T00:00:00Z",
            ],
        }
    )

    target = _season_allocation_targets(games, team_games=None)[0]

    assert target.target_game_count == 2
    assert target.player_shares == pytest.approx({1: 0.375, 2: 0.25, 3: 0.375})


def test_full_season_gate_rejects_a_missing_catalog_game(tmp_path) -> None:
    catalog_path = tmp_path / "games.parquet"
    pd.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "season_type": ["regular", "regular"],
            "game_id": ["g1", "g2"],
            "home_team_id": [1, 1],
            "away_team_id": [2, 2],
        }
    ).to_parquet(catalog_path, index=False)
    incomplete_minutes = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "team_id": [1, 2],
            "team_tricode": ["ONE", "TWO"],
            "player_id": [10, 20],
            "minutes": [240.0, 240.0],
            "game_time_utc": ["2025-10-01T00:00:00Z", "2025-10-01T00:00:00Z"],
        }
    )

    with pytest.raises(ValueError, match="missing_games=1"):
        assert_full_regular_season_minute_coverage(
            "2025-26", incomplete_minutes, catalog_path=catalog_path
        )
