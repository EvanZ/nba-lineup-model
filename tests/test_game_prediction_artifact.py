from __future__ import annotations

import pytest

from nba_lineup_model.season.game_prediction_artifact import (
    build_actual_minutes_ledger,
    build_spread_ledger,
    complete_projected_minutes_attribution,
    enrich_projected_minutes,
)


def _pregame_ledger() -> dict[str, object]:
    projected = enrich_projected_minutes(
        {
            "away": [
                {"player_id": 1, "player_name": "Away One", "projected_minutes": 120.0},
                {"player_id": 2, "player_name": "Away Two", "projected_minutes": 120.0},
            ],
            "home": [
                {"player_id": 3, "player_name": "Home One", "projected_minutes": 120.0},
                {"player_id": 4, "player_name": "Home Two", "projected_minutes": 120.0},
            ],
        },
        ratings={1: 1.0, 2: -1.0, 3: 2.0, 4: 0.0},
        game_id="game",
    )
    return build_spread_ledger(
        game_id="game",
        away_team="AWY",
        home_team="HME",
        projected_minutes=projected,
        home_court_adjustment=1.5,
        back_to_back_coefficient=-0.75,
        home_back_to_back=True,
        away_back_to_back=False,
        provenance={"source": "test"},
    )


def test_injury_listed_player_gets_zero_minute_attribution() -> None:
    projected = complete_projected_minutes_attribution(
        {
            "away": [
                {"player_id": 1, "player_name": "Away One", "projected_minutes": 240.0}
            ],
            "home": [
                {"player_id": 2, "player_name": "Home One", "projected_minutes": 240.0}
            ],
        },
        injury_report={
            "assumptions": [
                {
                    "player_id": 3,
                    "player_name": "Listed Reserve",
                    "team": "HME",
                    "status": "Available",
                    "play_probability": 1.0,
                    "reason": "-",
                }
            ]
        },
        away_team="AWY",
        home_team="HME",
        ratings={1: 1.0, 2: 2.0, 3: -0.5},
        game_id="game",
    )

    reserve = next(row for row in projected["home"] if row["player_id"] == 3)
    assert reserve["projected_nail"] == pytest.approx(-0.5)
    assert reserve["projected_minutes"] == 0.0
    assert reserve["expected_contribution"] == 0.0


def test_actual_minutes_ledger_uses_frozen_ratings_and_reconciles_margin() -> None:
    ledger = build_actual_minutes_ledger(
        game_id="game",
        away_team="AWY",
        home_team="HME",
        actual_minutes={
            "away": [
                {"player_id": 1, "player_name": "Away One", "actual_minutes": 48.0},
                {"player_id": 2, "player_name": "Away Two", "actual_minutes": 192.0},
            ],
            "home": [
                {"player_id": 3, "player_name": "Home One", "actual_minutes": 192.0},
                {"player_id": 4, "player_name": "Home Two", "actual_minutes": 48.0},
            ],
        },
        ratings={1: 1.0, 2: -1.0, 3: 2.0, 4: 0.0},
        pregame_ledger=_pregame_ledger(),
        provenance={"minutes": "official"},
    )

    assert ledger["away"]["team_strength"] == pytest.approx(-3.0)
    assert ledger["home"]["team_strength"] == pytest.approx(8.0)
    assert ledger["team_strength_difference"] == pytest.approx(11.0)
    assert ledger["home_court_adjustment"] == pytest.approx(1.5)
    assert ledger["back_to_back"]["adjustment"] == pytest.approx(-0.75)
    assert ledger["home_margin"] == pytest.approx(11.75)
    assert sum(
        row["actual_contribution"] for row in ledger["home"]["players"]
    ) == pytest.approx(ledger["home"]["team_strength"])


def test_actual_minutes_ledger_normalizes_overtime_minutes_to_five_players() -> None:
    ledger = build_actual_minutes_ledger(
        game_id="game",
        away_team="AWY",
        home_team="HME",
        actual_minutes={
            "away": [{"player_id": 1, "player_name": "Away One", "actual_minutes": 265.0}],
            "home": [{"player_id": 3, "player_name": "Home One", "actual_minutes": 265.0}],
        },
        ratings={1: 1.0, 3: 2.0},
        pregame_ledger=_pregame_ledger(),
        provenance={},
    )

    assert ledger["away"]["team_strength"] == pytest.approx(5.0)
    assert ledger["home"]["team_strength"] == pytest.approx(10.0)
