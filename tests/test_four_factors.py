from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.modeling.four_factors import build_four_factor_stint_frame


def test_four_factor_stints_use_event_boundaries_and_oliver_ftr() -> None:
    stints = pd.DataFrame(
        {
            "game_id": ["001", "001"],
            "stint_index": [0, 1],
            "start_event_index": [0, 5],
            "end_event_index": [4, 9],
            "points_home": [3, 0],
            "points_away": [0, 3],
            "catalog_home_team_id": [1, 1],
            "catalog_away_team_id": [2, 2],
        }
    )
    events = pd.DataFrame(
        {
            "game_id": ["001"] * 8,
            "event_index": [0, 1, 2, 3, 4, 5, 6, 7],
            "event_type": ["2pt", "3pt", "freethrow", "rebound", "turnover", "2pt", "rebound", "freethrow"],
            "event_subtype": [None, None, "1 of 1", "defensive", "lost ball", None, "offensive", "1 of 1"],
            "shot_result": ["Made", "Missed", "Made", None, None, "Made", None, "Made"],
            "team_id": [1, 2, 1, 2, 1, 2, 2, 2],
        }
    )

    frame, audit = build_four_factor_stint_frame(stints, events)

    first = frame.iloc[0]
    second = frame.iloc[1]
    assert first["home_efg_pct"] == pytest.approx(1.0)
    assert first["home_tov_pct"] == pytest.approx(1.0 / 2.44)
    assert first["home_orb_pct"] == pytest.approx(0.0)
    assert first["home_ftr"] == pytest.approx(1.0)
    assert second["away_efg_pct"] == pytest.approx(1.0)
    assert second["away_orb_pct"] == pytest.approx(1.0)
    assert second["away_ftr"] == pytest.approx(1.0)
    assert audit.loc[0, "unassigned_factor_event_count"] == 0
    assert audit.loc[0, "unknown_team_factor_event_count"] == 0
    assert audit.loc[0, "is_four_factor_eligible"]
    assert set(frame["four_factor_quality_status"]) == {"clean"}
