import pandas as pd

from nba_lineup_model.modeling.team_change_rating_stability import build_player_transitions


def test_build_player_transitions_keeps_only_clean_consecutive_seasons() -> None:
    ratings = pd.DataFrame(
        {
            "season": ["2023-24", "2024-25", "2023-24", "2025-26"],
            "player_id": [1, 1, 2, 2],
            "player_name": ["Same", "Same", "Gap", "Gap"],
            "rapm": [1.0, 1.5, 0.0, 2.0],
            "prior_rapm": [0.5, 0.7, 0.0, 0.1],
            "rapm_adjustment_from_prior": [0.5, 0.8, 0.0, 1.9],
        }
    )
    panel = pd.DataFrame(
        {
            "season": ["2023-24", "2024-25", "2023-24", "2025-26"],
            "season_start_year": [2023, 2024, 2023, 2025],
            "player_id": [1, 1, 2, 2],
            "primary_team_tricode": ["AAA", "BBB", "CCC", "DDD"],
            "team_count": [1, 1, 1, 1],
            "rapm_possessions": [1200.0, 1400.0, 1400.0, 1400.0],
            "age": [25.0, 26.0, 24.0, 26.0],
        }
    )

    transitions = build_player_transitions(ratings, panel, minimum_possessions=1000.0)

    assert len(transitions) == 1
    row = transitions.iloc[0]
    assert row.player_id == 1
    assert row.team_changed
    assert row.transition_group == "changed_team"
    assert row.rating_change == 0.5


def test_build_player_transitions_excludes_multi_team_seasons() -> None:
    ratings = pd.DataFrame(
        {
            "season": ["2023-24", "2024-25"],
            "player_id": [1, 1],
            "player_name": ["Traded", "Traded"],
            "rapm": [1.0, 2.0],
            "prior_rapm": [0.0, 0.0],
            "rapm_adjustment_from_prior": [1.0, 2.0],
        }
    )
    panel = pd.DataFrame(
        {
            "season": ["2023-24", "2024-25"],
            "season_start_year": [2023, 2024],
            "player_id": [1, 1],
            "primary_team_tricode": ["AAA", "BBB"],
            "team_count": [2, 1],
            "rapm_possessions": [1200.0, 1400.0],
            "age": [25.0, 26.0],
        }
    )

    transitions = build_player_transitions(ratings, panel, minimum_possessions=1000.0)

    assert transitions.empty
