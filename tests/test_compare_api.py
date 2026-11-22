"""Compare search reads saved catalogs even when the legacy API is disabled."""

from fastapi.testclient import TestClient

from nba_lineup_model.web_api import compare
from nba_lineup_model.web_api.gateway import MarketGateway


def test_search_includes_archived_active_and_unnamed_players(monkeypatch):
    def archive(player_id, name, seasons):
        return {
            "player_id": player_id,
            "player_name": name,
            "history_seasons": seasons,
            "latest_season": "2025-26",
            "latest_team": "DEN",
        }

    historical = {
        1495: archive(1495, "Tim Duncan", 19),
        203999: archive(203999, "Nikola Jokić", 11),
        888: archive(888, None, 1),
    }
    current = [
        {"player_id": 203999, "player_name": "Nikola Jokić", "team": "DEN"},
        {"player_id": 3000000, "player_name": "Current Rookie", "team": "SAS"},
    ]
    monkeypatch.setattr(compare, "historical_catalog", lambda _: historical)
    monkeypatch.setattr(compare, "current_rankings", lambda _: {
        "players": current,
        "metadata_warning": None,
    })
    client = TestClient(MarketGateway(None))

    def search(query):
        response = client.get("/api/compare/players", params={"q": query})
        assert response.status_code == 200
        assert response.json()["unavailable"] == []
        result, = response.json()["players"]
        return result

    assert search("Duncan")["history_seasons"] == 19
    assert search("Rookie")["history_seasons"] == 0
    assert search("888")["player_id"] == 888
    jokic = search("Nikola Jokic")
    assert jokic["latest_season"] == "2026-27"
    assert jokic["history_seasons"] == 11
