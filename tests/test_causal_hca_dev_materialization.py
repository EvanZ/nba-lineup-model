from __future__ import annotations

from datetime import date

import pytest

from nba_lineup_model.season.causal_hca_dev_materialization import (
    PregameState,
    _candidate_pregame,
    _retrospective_rating_payload,
)


def _rotation(start_player_id: int) -> list[dict[str, object]]:
    return [
        {
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "projected_minutes": 48.0,
            "status": "Available",
            "play_probability": 1.0,
        }
        for player_id in range(start_player_id, start_player_id + 5)
    ]


def test_candidate_game_ledger_uses_exact_causal_home_court_state() -> None:
    state = PregameState(
        as_of=date(2026, 11, 1),
        ratings={player_id: 0.0 for player_id in range(1, 11)},
        home_court=1.75,
        bundle_sha256="a" * 64,
        rating_source_uri=f"artifact://daily-model/{'a' * 64}/player_ratings",
        home_court_source_uri=(
            f"artifact://daily-model/{'a' * 64}/home_court_state"
        ),
        home_court_sha256="b" * 64,
        cumulative_game_count=92,
    )
    game = {
        "game_id": "0022600158",
        "away_team": "AAA",
        "home_team": "BBB",
        "projected_minutes": {
            "away": _rotation(1),
            "home": _rotation(6),
        },
        "injury_report": {"assumptions": []},
        "projection": {
            "home_margin": 2.836661,
            "fair_home_spread": -2.836661,
            "home_win_probability": 0.57,
            "input_provenance": {"minutes_source_uri": "artifact://minutes/frozen"},
            "ledger": {
                "back_to_back": {
                    "coefficient": -0.75,
                    "home_team_is_back_to_back": False,
                    "away_team_is_back_to_back": False,
                    "adjustment": 0.0,
                }
            },
        },
    }

    candidate = _candidate_pregame(game, state=state, probability_scale=0.1)

    ledger = candidate["projection"]["ledger"]
    assert ledger["home_court_adjustment"] == pytest.approx(1.75)
    assert ledger["team_strength_difference"] == pytest.approx(0.0)
    assert ledger["back_to_back"]["adjustment"] == pytest.approx(0.0)
    assert ledger["home_margin"] == pytest.approx(1.75)
    assert candidate["projection"]["home_margin"] == pytest.approx(1.75)
    assert candidate["projection"]["fair_home_spread"] == pytest.approx(-1.75)
    assert ledger["provenance"]["home_court_source_uri"] == (
        f"artifact://daily-model/{'a' * 64}/home_court_state"
    )
    assert ledger["provenance"]["home_court_state_sha256"] == "b" * 64


def test_retrospective_payload_includes_players_outside_projected_rotation() -> None:
    state = PregameState(
        as_of=date(2026, 10, 20),
        ratings={1: 2.5, 2: -1.25, 1643052: -1.37},
        home_court=1.69,
        bundle_sha256="a" * 64,
        rating_source_uri=f"artifact://daily-model/{'a' * 64}/player_ratings",
        home_court_source_uri=(
            f"artifact://daily-model/{'a' * 64}/home_court_state"
        ),
        home_court_sha256="b" * 64,
        cumulative_game_count=1,
    )

    payload = _retrospective_rating_payload(state)

    assert payload == {
        "players": [
            {"player_id": 1, "projected_nail": 2.5},
            {"player_id": 2, "projected_nail": -1.25},
            {"player_id": 1643052, "projected_nail": -1.37},
        ]
    }
