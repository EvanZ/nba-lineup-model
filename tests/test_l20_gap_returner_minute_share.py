from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.rotation.l20_gap_returner_minute_share import (
    build_gap_returner_scores,
)


def _minutes(values: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": list(values),
            "minutes": list(values.values()),
        }
    )


def test_gap_returner_uses_most_recent_active_season_only() -> None:
    candidates = pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "team": ["MIN", "MIN", "MIN"],
            "player_id": [1, 2, 3],
            "player_name": ["Returning", "Incumbent", "Rookie"],
        }
    )

    scores, returners = build_gap_returner_scores(
        _minutes({2: 100.0}),
        [_minutes({1: 80.0}), _minutes({1: 240.0})],
        candidates,
        returner_weight=0.5,
    )

    assert scores.to_dict() == pytest.approx({1: 40.0, 2: 100.0})
    assert returners.to_dict() == {1: True, 2: False, 3: False}
