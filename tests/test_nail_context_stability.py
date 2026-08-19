from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.nail_context_stability import (
    _lineup_ledger,
    _player_ledger,
    _weighted_pearson,
)


def _stints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "source_context_season": ["2024-25", "2024-25"],
            "game_id": ["001", "001"],
            "possessions": [10.0, 20.0],
            "home_team_tricode": ["HOM", "HOM"],
            "away_team_tricode": ["AWY", "AWY"],
            "home_player_ids": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            "away_player_ids": [[6, 7, 8, 9, 10], [6, 7, 8, 9, 10]],
            "frozen_nale": [1.0, 2.0],
            "completed_nale": [2.0, 4.0],
            "nale_revision": [1.0, 2.0],
        }
    )


def test_lineup_and_player_ledgers_preserve_home_away_orientation() -> None:
    stints = _stints()
    lineups = _lineup_ledger(stints, minimum_possessions=1.0)
    home = lineups.loc[lineups["team"].eq("HOM")].iloc[0]
    away = lineups.loc[lineups["team"].eq("AWY")].iloc[0]
    assert home["possessions"] == 30.0
    assert home["frozen_nale"] == 5.0 / 3.0
    assert away["frozen_nale"] == -5.0 / 3.0

    profiles = pd.DataFrame(
        {"player_id": list(range(1, 11)), "player_name": [f"P{value}" for value in range(1, 11)]}
    )
    players = _player_ledger(stints, profiles, minimum_possessions=1.0)
    assert players.loc[players["player_id"].eq(1), "frozen_nale"].item() == 5.0 / 3.0
    assert players.loc[players["player_id"].eq(6), "frozen_nale"].item() == -5.0 / 3.0


def test_weighted_pearson_uses_possessions() -> None:
    left = pd.Series([0.0, 1.0, 3.0])
    right = pd.Series([0.0, 2.0, 6.0])
    weights = pd.Series([1.0, 10.0, 100.0])
    assert _weighted_pearson(left, right, weights) == 1.0
    assert np.isnan(
        _weighted_pearson(pd.Series([1.0, 1.0]), pd.Series([1.0, 2.0]), pd.Series([1.0, 1.0]))
    )
