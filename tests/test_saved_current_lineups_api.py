from datetime import date, datetime
from unittest.mock import Mock

from fastapi.testclient import TestClient

from nba_lineup_model.web_api import saved_current_lineups as current
from nba_lineup_model.web_api.gateway import MarketGateway


def test_current_lineups_use_one_latest_rating_state_for_all_observed_games(monkeypatch):
    keys = ("one", "two", "three")
    day = "2026-11-19"
    stints = [
        {
            "game_id": game,
            "possessions": possessions,
            "margin": margin,
            "home_team_id": 1,
            "away_team_id": 2,
            "home_team": "HOM",
            "away_team": "AWY",
            "home_player_ids": [5, 2, 3, 4, 1],
            "away_player_ids": [10, 9, 8, 7, 6],
        }
        for game, possessions, margin in zip(keys, (10, 30, 60), (3, -2, 12), strict=True)
    ]
    schedule = [{"game_id": f"002{i:07}", "game_date": day} for i in range(3)]
    reader = Mock()
    reader.official_schedule.return_value = (schedule, [], {})
    monkeypatch.setattr(current, "_reader", lambda: reader)
    monkeypatch.setattr(current, "_date_stints", lambda *_: (day, stints))
    clock = Mock()
    clock.now.return_value = datetime(2026, 11, 21, 12, tzinfo=current.PACIFIC)
    monkeypatch.setattr(current, "datetime", clock)

    latest = Mock(return_value=date(2026, 11, 20))
    snapshot = {
        "effective_from_game_date_pacific": "2026-11-21",
        "players": [
            {
                "player_id": pid,
                "player_name": f"Player {pid}",
                "nail": pid / 10,
                "starting_nail": pid / 10,
                "residual": 0,
                "standard_error": 1,
                "on_court_possessions": 10,
            }
            for pid in range(1, 10)
        ],
    }
    rating_ref = {"uri": "saved-ratings", "sha256": "test-source"}
    state = Mock(return_value=(snapshot, rating_ref))
    monkeypatch.setattr(current.profiles, "_latest_day", latest)
    monkeypatch.setattr(current.profiles, "_state", state)
    current._load.cache_clear()

    try:
        legacy = Mock(side_effect=AssertionError("Current Lineups must bypass the evaluator"))
        client = TestClient(MarketGateway(legacy))
        response = client.get("/api/lineups?season=2026-27&minimum_possessions=0")
        assert response.status_code == 200
        payload = response.json()
        assert payload["kind"] == "saved_current_lineups"
        assert payload["coverage"]["games"] == 3
        assert payload["coverage"]["lineups"] == 2
        assert payload["coverage"]["rated_lineups"] == 1
        assert payload["rating_state"]["effective_from_date"] == "2026-11-21"
        assert payload["rating_state"]["evidence_through_date"] == "2026-11-20"
        assert payload["sources"]["ratings"] == rating_ref
        by_team = {row["team"]: row for row in payload["lineups"]}
        home, away = by_team["HOM"], by_team["AWY"]
        assert home["player_ids"] == [1, 2, 3, 4, 5]
        assert (home["games"], home["possessions"], home["actual_net_rating"]) == (3, 100, 13)
        assert home["lineup_nail"] == 1.5
        assert away["lineup_nail"] is None and away["actual_net_rating"] == -13
        assert not {"context_edge", "gestalt_score", "player_edge"} & home.keys()
        assert (
            client.get("/api/lineups?season=2026-27&minimum_possessions=101").json()["lineups"]
            == []
        )
        latest.assert_called_once_with(
            reader.client,
            "lake/artifacts/player_ratings/season=2026-27/model_version=v1/",
            date(2026, 11, 20),
        )
        state.assert_called_once_with(
            reader.client, "player_ratings", "2026-27", date(2026, 11, 20)
        )
        legacy.assert_not_called()
    finally:
        current._load.cache_clear()
