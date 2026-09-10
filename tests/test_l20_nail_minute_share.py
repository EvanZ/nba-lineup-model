from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.rotation.l20_nail_minute_share import (
    RatingAwareSeasonInputs,
    build_rating_aware_shares,
    evaluate_l20_nail_minute_share,
)


def _games(allocations: list[dict[int, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game_number, shares in enumerate(allocations, start=1):
        for player_id, share in shares.items():
            rows.append(
                {
                    "game_id": f"g{game_number}",
                    "team_id": 1,
                    "team_tricode": "MIN",
                    "player_id": player_id,
                    "minutes": share * 240.0,
                    "game_time_utc": f"2025-10-{game_number:02d}T00:00:00Z",
                }
            )
    return pd.DataFrame(rows)


def test_zero_nail_weight_exactly_recovers_smoothed_minute_allocation() -> None:
    minutes = pd.Series({1: 80.0, 2: 20.0})
    shares, _z_scores = build_rating_aware_shares(
        [1, 2],
        minutes,
        pd.Series({1: -10.0, 2: 10.0}),
        epsilon=0.32,
        nail_weight=0.0,
        rating_scale=5.0,
    )
    assert shares == pytest.approx({1: 0.704, 2: 0.296})


def test_positive_nail_weight_rebalances_minutes_toward_better_rated_player() -> None:
    minutes = pd.Series({1: 80.0, 2: 20.0})
    shares, _z_scores = build_rating_aware_shares(
        [1, 2],
        minutes,
        pd.Series({1: 0.0, 2: 10.0}),
        epsilon=0.0,
        nail_weight=1.0,
        rating_scale=5.0,
    )
    assert shares[2] > 0.2
    assert sum(shares.values()) == pytest.approx(1.0)


def test_evaluation_uses_completed_ratings_without_target_minutes() -> None:
    inputs = RatingAwareSeasonInputs(
        season="2025-26",
        prior_game_minutes=_games([{1: 0.8, 2: 0.2}]),
        target_game_minutes=_games([{1: 0.4, 2: 0.6}]),
        opening_candidates=pd.DataFrame(
            {
                "team_id": [1, 1],
                "team": ["MIN", "MIN"],
                "player_id": [1, 2],
                "player_name": ["Player 1", "Player 2"],
            }
        ),
        completed_nail_ratings=pd.Series({1: 0.0, 2: 10.0}),
    )
    predictions, metrics = evaluate_l20_nail_minute_share(
        inputs, epsilon=0.0, nail_weight=1.0, team_games=1
    )
    assert predictions.set_index("player_id").loc[2, "predicted_minute_share"] > 0.2
    assert metrics.loc[0, "season"] == "2025-26"
