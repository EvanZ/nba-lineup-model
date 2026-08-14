from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.profile_token_mart import (
    SHOT_TOKEN_COLUMNS,
    _attach_shot_profiles,
    _build_target_tokens,
)


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season, age_offset in (("2023-24", 0), ("2024-25", 1)):
        for player_id in (1, 2, 3):
            rows.append(
                {
                    "season": season,
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "rapm_possessions": 500.0 + player_id,
                    "rapm_exposure_eligible": player_id == 1,
                    "field_goals_attempted": 100.0 + 10.0 * player_id,
                    "free_throws_attempted": 20.0 + 5.0 * player_id,
                    "free_throws_made": 15.0 + 4.0 * player_id,
                    "age": 22.0 + age_offset + player_id,
                    "nba_experience_years": age_offset,
                    "is_rookie": (season == "2023-24" and player_id == 3)
                    or (season == "2024-25" and player_id == 2),
                    "height_inches": 76.0 + player_id,
                    "weight_pounds": 200.0 + player_id,
                    "draft_number": 10.0 if player_id == 1 else None,
                    "is_undrafted": player_id == 2,
                }
            )
    return pd.DataFrame(rows)


def _profiles() -> pd.DataFrame:
    profile_columns = {
        "three_pa_per_100": [7.0, 5.0],
        "three_pm_per_100": [3.0, 2.0],
        "assists_per_100": [5.0, 2.0],
        "turnovers_per_100": [2.0, 1.0],
        "usage_per_100": [20.0, 12.0],
        "offensive_rebounds_per_100": [1.0, 2.0],
        "defensive_rebounds_per_100": [4.0, 5.0],
        "steals_per_100": [1.0, 0.5],
        "blocks_per_100": [0.2, 0.7],
        "offensive_rebound_pct": [3.0, 7.0],
        "defensive_rebound_pct": [12.0, 18.0],
    }
    return pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["Player 1", "Player 2"],
            "is_rookie": [False, True],
            "profile_source": ["prior_season", "exposure_gated_rookie"],
            "profile_imputed": [0, 1],
            "profile_replacement_weight": [0.0, 0.8],
            **profile_columns,
        }
    )


def _shots() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in ("2023-24", "2024-25"):
        rows.append(
            {
                "season": season,
                "player_id": 1,
                "on_court_possessions": 500.0,
                "rim_attempts_per_100": 4.0,
                "rim_fg_pct_shrunk": 0.6,
                "rim_attempts": 20,
                "rim_makes": 12,
                "non_rim_two_attempts_per_100": 7.0,
                "non_rim_two_fg_pct_shrunk": 0.4,
                "non_rim_two_attempts": 35,
                "non_rim_two_makes": 14,
                "three_attempts_per_100": 8.0,
                "three_fg_pct_shrunk": 0.37,
                "three_attempts": 40,
                "three_makes": 15,
            }
        )
        rows.append(
            {
                "season": season,
                "player_id": 3,
                "on_court_possessions": 20.0,
                "rim_attempts_per_100": 2.0,
                "rim_fg_pct_shrunk": 0.5,
                "rim_attempts": 1,
                "rim_makes": 1,
                "non_rim_two_attempts_per_100": 3.0,
                "non_rim_two_fg_pct_shrunk": 0.35,
                "non_rim_two_attempts": 2,
                "non_rim_two_makes": 1,
                "three_attempts_per_100": 4.0,
                "three_fg_pct_shrunk": 0.33,
                "three_attempts": 3,
                "three_makes": 1,
            }
        )
    return pd.DataFrame(rows)


def test_target_tokens_use_only_prior_season_profile_rows() -> None:
    tokens = _build_target_tokens(
        target="2024-25",
        target_priors=pd.DataFrame({"player_id": [1, 2], "prior_rapm": [1.5, -0.5]}),
        panel=_panel(),
        profiles=_profiles(),
        shots=_shots(),
    )

    assert tokens["source_season"].tolist() == ["2023-24", "2023-24"]
    assert tokens.loc[tokens["player_id"].eq(1), "three_attempts_per_100"].item() == 8.0
    cold = tokens.loc[tokens["player_id"].eq(2)].iloc[0]
    assert cold["shot_profile_imputed"] == 1
    assert cold["shot_profile_source"] == "exposure_gated_rookie"
    assert cold.loc[list(SHOT_TOKEN_COLUMNS)].notna().all()
    assert cold["free_throw_attempts_per_100"] > 0.0
    assert 0.0 < cold["free_throw_pct_hierarchical"] < 1.0


def test_shot_cold_start_uses_profile_replacement_weight() -> None:
    output = pd.DataFrame(
        {
            "player_id": [2],
            "is_rookie": [1],
            "profile_replacement_weight": [1.0],
            "draft_number": [None],
            "is_undrafted": [True],
        }
    )
    result = _attach_shot_profiles(output, target="2024-25", panel=_panel(), shots=_shots())

    assert result["shot_profile_imputed"].item() == 1
    assert result["shot_profile_replacement_weight"].item() == 1.0
    assert result["three_attempts_per_100"].item() == 4.0
