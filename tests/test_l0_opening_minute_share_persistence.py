from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.rotation.l0_opening_minute_share_persistence import (
    evaluate_l0_opening_minute_share_persistence,
    select_epsilon,
)


def _minutes(
    *,
    game_id: str,
    team_id: int,
    shares: dict[int, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id] * len(shares),
            "team_id": [team_id] * len(shares),
            "team_tricode": ["MIN"] * len(shares),
            "player_id": list(shares),
            "player_name": [f"Player {player_id}" for player_id in shares],
            "game_time_utc": ["2025-10-21T00:00:00Z"] * len(shares),
            "minute_share": list(shares.values()),
        }
    )


def test_opening_baseline_restricts_prior_share_to_candidates_and_smooths() -> None:
    prior = _minutes(game_id="prior", team_id=1, shares={1: 0.6, 2: 0.4})
    target = _minutes(game_id="target", team_id=1, shares={1: 0.5, 3: 0.5})
    candidates = pd.DataFrame(
        {
            "team_id": [1, 1],
            "team": ["MIN", "MIN"],
            "player_id": [1, 3],
            "player_name": ["Player 1", "Player 3"],
        }
    )

    predictions, metrics = evaluate_l0_opening_minute_share_persistence(
        prior,
        target,
        candidates,
        epsilon=0.2,
    )

    predicted = predictions.set_index("player_id")["predicted_minute_share"].to_dict()
    assert predicted == pytest.approx({1: 0.9, 3: 0.1})
    assert metrics.loc[0, "outside_candidate_actual_share"] == 0.0
    assert metrics.loc[0, "allocation_total_variation"] == pytest.approx(0.4)


def test_epsilon_selection_prefers_smallest_tie_on_source_tv() -> None:
    prior = _minutes(game_id="prior", team_id=1, shares={1: 0.5, 2: 0.5})
    target = _minutes(game_id="target", team_id=1, shares={1: 0.5, 2: 0.5})
    candidates = pd.DataFrame(
        {
            "team_id": [1, 1],
            "team": ["MIN", "MIN"],
            "player_id": [1, 2],
            "player_name": ["Player 1", "Player 2"],
        }
    )

    epsilon, grid = select_epsilon(prior, target, candidates, epsilon_grid=(0.0, 0.2))

    assert epsilon == 0.0
    assert grid.loc[0, "mean_allocation_total_variation"] == 0.0
