from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.flows.process_season import _eligible_audit_game_ids


def test_selects_only_eligible_regular_games_for_requested_season(tmp_path) -> None:
    path = tmp_path / "games.parquet"
    pd.DataFrame(
        {
            "game_id": ["0021900001", "0021900002", "0021900003", "0022000001"],
            "season": ["2019-20", "2019-20", "2019-20", "2020-21"],
            "season_type": ["regular", "regular", "playoffs", "regular"],
            "status": ["pass", "fail", "warning", "warning"],
        }
    ).to_parquet(path, index=False)

    assert _eligible_audit_game_ids(str(path), "2019-20") == ["0021900001"]


def test_rejects_duplicate_eligible_audit_games(tmp_path) -> None:
    path = tmp_path / "games.parquet"
    pd.DataFrame(
        {
            "game_id": ["0021900001", "0021900001"],
            "season": ["2019-20", "2019-20"],
            "season_type": ["regular", "regular"],
            "status": ["pass", "warning"],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        _eligible_audit_game_ids(str(path), "2019-20")
