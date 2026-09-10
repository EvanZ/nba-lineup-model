from __future__ import annotations

from datetime import date

import pandas as pd

from nba_lineup_model.players.opening_rosters import (
    _apply_movements,
    _reconcile_opening_game_roster,
)


def test_apply_movements_moves_adds_and_removes_players_before_opening_day() -> None:
    base_state = {
        "1": {
            "team": "DAL",
            "player_name": "One Player",
            "last_regular_season_game_id": "0022400001",
            "last_regular_season_game_date": date(2025, 4, 13),
        },
        "2": {
            "team": "BOS",
            "player_name": "Two Player",
            "last_regular_season_game_id": "0022400001",
            "last_regular_season_game_date": date(2025, 4, 13),
        },
    }
    movements = pd.DataFrame(
        [
            {
                "transaction_date": date(2025, 7, 1),
                "transaction_type": "Trade",
                "player_id": "1",
                "team": "LAL",
                "player_slug": "one-player",
                "additional_sort": 0,
            },
            {
                "transaction_date": date(2025, 7, 2),
                "transaction_type": "Signing",
                "player_id": "3",
                "team": "LAL",
                "player_slug": "three-player",
                "additional_sort": 0,
            },
            {
                "transaction_date": date(2025, 7, 3),
                "transaction_type": "Waive",
                "player_id": "2",
                "team": "BOS",
                "player_slug": "two-player",
                "additional_sort": 0,
            },
            {
                "transaction_date": date(2025, 10, 21),
                "transaction_type": "Signing",
                "player_id": "4",
                "team": "LAL",
                "player_slug": "opening-day-player",
                "additional_sort": 0,
            },
        ]
    )

    actual = _apply_movements(
        base_state,
        movements,
        start_after=date(2025, 4, 13),
        cutoff_before=date(2025, 10, 21),
    )

    assert set(actual) == {"1", "3"}
    assert actual["1"]["team"] == "LAL"
    assert actual["1"]["player_name"] == "One Player"
    assert actual["3"]["player_name"] == "Three Player"
    assert "4" not in actual


def test_reconciliation_records_opening_game_membership_gap() -> None:
    state = {
        "1": {
            "team": "MIN",
            "player_name": "Known Player",
            "last_regular_season_game_id": "0022400001",
            "last_regular_season_game_date": date(2025, 4, 13),
        }
    }

    reconciliation = _reconcile_opening_game_roster(
        state,
        {"1": "Known Player", "2": "Recovered Player"},
        team="MIN",
    )

    assert reconciliation == [
        {
            "player_id": "2",
            "player_name": "Recovered Player",
            "reconciliation_reason": "missing_from_transaction_state",
        }
    ]
    assert state["2"]["membership_source"] == "opening_game_reconciliation"
