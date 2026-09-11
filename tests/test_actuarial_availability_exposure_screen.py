from __future__ import annotations

import pandas as pd

from nba_lineup_model.rotation.actuarial_availability_exposure_screen import (
    build_forward_exposure_pairs,
    concentration_gini,
    poisson_deviance,
    run_frozen_exposure_screen,
)


def _summary() -> pd.DataFrame:
    rows = []
    for season, episodes in (("2022-23", 1), ("2023-24", 0), ("2024-25", 2)):
        rows.append(
            {
                "season": season,
                "player_id": 1,
                "player_name": "Carryover",
                "availability_episode_count": episodes,
                "unavailability_loss_cost": 0.1,
                "available_rostered_games": 70,
                "rostered_team_games": 82,
                "panel_gp": 65,
                "panel_gs": 60,
                "panel_minutes": 2100.0,
            }
        )
        rows.append(
            {
                "season": season,
                "player_id": 2,
                "player_name": "Independent",
                "availability_episode_count": episodes,
                "unavailability_loss_cost": 0.0,
                "available_rostered_games": 82,
                "rostered_team_games": 82,
                "panel_gp": 75,
                "panel_gs": 70,
                "panel_minutes": 2400.0,
            }
        )
    return pd.DataFrame(rows)


def test_cross_season_episode_is_excluded_from_new_frequency_target() -> None:
    episodes = pd.DataFrame(
        {
            "player_id": [1],
            "episode_start_season": ["2022-23"],
            "episode_end_season": ["2023-24"],
        }
    )

    pairs = build_forward_exposure_pairs(_summary(), episodes)

    carryover = pairs.loc[
        pairs["player_id"].eq(1) & pairs["target_season"].eq("2023-24")
    ].iloc[0]
    independent = pairs.loc[
        pairs["player_id"].eq(2) & pairs["target_season"].eq("2023-24")
    ].iloc[0]
    assert carryover["has_cross_season_unavailability_carryover"]
    assert not independent["has_cross_season_unavailability_carryover"]


def test_count_metrics_reward_correct_episode_predictions() -> None:
    assert poisson_deviance([0.0, 2.0], [0.01, 1.99]) < poisson_deviance(
        [0.0, 2.0], [1.0, 0.1]
    )
    assert concentration_gini([0.0, 0.0, 2.0], [0.1, 0.2, 0.9]) > 0.0


def test_frequency_screen_uses_target_minutes_and_available_games_as_offsets() -> None:
    rows = []
    for year in range(2016, 2024):
        season = f"{year}-{str(year + 1)[-2:]}"
        for player_id in range(1, 5):
            rows.append(
                {
                    "target_season": season,
                    "target_season_start_year": year,
                    "player_id": player_id,
                    "target_availability_episode_count": float((player_id + year) % 4),
                    "target_available_rostered_games": 20.0 + 15.0 * player_id,
                    "target_panel_minutes": 450.0 + 500.0 * player_id,
                    "has_cross_season_unavailability_carryover": False,
                    "prior_unavailability_loss_cost": 0.02 * player_id,
                    "prior_availability_episode_count": float(player_id % 3),
                    "target_age": 21.0 + player_id,
                }
            )
    pairs = pd.DataFrame(rows)

    metrics, predictions, deciles = run_frozen_exposure_screen(
        pairs,
        frozen_seasons=("2023-24",),
    )

    assert set(metrics["frequency_offset"]) == {
        "Player minutes",
        "Medically available games",
    }
    minutes = predictions.loc[predictions["frequency_offset"].eq("Player minutes")]
    available_games = predictions.loc[
        predictions["frequency_offset"].eq("Medically available games")
    ]
    assert minutes["exposure_value"].tolist() == [950.0, 1450.0, 1950.0, 2450.0]
    assert available_games["exposure_value"].tolist() == [35.0, 50.0, 65.0, 80.0]
    assert set(deciles["rate_label"]) == {
        "Episodes per 1,000 player minutes",
        "Episodes per 100 medically available games",
    }
