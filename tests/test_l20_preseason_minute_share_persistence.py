from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.rotation.l20_preseason_minute_share_persistence import (
    evaluate_l20_preseason_minute_share_persistence,
)


def _games(allocations: list[dict[int, float]], *, team_id: int = 1) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game_number, shares in enumerate(allocations, start=1):
        for player_id, share in shares.items():
            rows.append(
                {
                    "game_id": f"g{game_number}",
                    "team_id": team_id,
                    "team_tricode": "MIN",
                    "player_id": player_id,
                    "minutes": share * 240.0,
                    "game_time_utc": f"2025-10-{game_number:02d}T00:00:00Z",
                }
            )
    return pd.DataFrame(rows)


def test_l20_predicts_one_static_allocation_for_cumulative_target() -> None:
    prior = _games([{1: 0.8, 2: 0.2}, {1: 0.8, 2: 0.2}])
    target = _games([{1: 0.5, 3: 0.5}, {1: 0.3, 3: 0.7}])
    candidates = pd.DataFrame(
        {
            "team_id": [1, 1],
            "team": ["MIN", "MIN"],
            "player_id": [1, 3],
            "player_name": ["Player 1", "Player 3"],
        }
    )

    predictions, metrics = evaluate_l20_preseason_minute_share_persistence(
        prior,
        target,
        candidates,
        epsilon=0.0,
        team_games=2,
    )

    predicted = predictions.set_index("player_id")["predicted_minute_share"].to_dict()
    actual = predictions.set_index("player_id")["actual_minute_share"].to_dict()
    assert predicted == pytest.approx({1: 1.0, 3: 0.0})
    assert actual == pytest.approx({1: 0.4, 3: 0.6})
    assert metrics.loc[0, "target_game_count"] == 2
