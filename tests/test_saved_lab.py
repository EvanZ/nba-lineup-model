"""Lab reads can serve predictions without enabling the legacy application."""

import pytest
from fastapi.testclient import TestClient
from test_web_api import _evaluator

from nba_lineup_model.web_api import lab
from nba_lineup_model.web_api.gateway import MarketGateway


def test_isolated_lab_routes_preserve_the_compiled_matchup(monkeypatch):
    reference = _evaluator(compiled_linear=True)
    state = reference._season_state(reference.season)
    monkeypatch.setattr(lab, "historical_seasons", lambda: (reference.season,))
    monkeypatch.setattr(lab, "preseason", lambda: {})
    monkeypatch.setattr(lab, "player_pool", lambda _season: state.players)
    monkeypatch.setattr(lab, "lineup_state", lambda _season: state)
    monkeypatch.setattr(lab, "rating_paths", lambda ids, _season: {pid: [] for pid in ids})
    monkeypatch.setattr(
        lab,
        "model_data",
        lambda: ({reference.season: reference.context_model}, reference.schedule_controls),
    )
    client = TestClient(MarketGateway(None))

    seasons = client.get("/api/lab/seasons").json()["seasons"]
    assert seasons == [lab.PRESEASON_PREVIEW_SEASON, reference.season]
    results = client.get("/api/lab/players", params={"q": "Jokic", "team": "TST"})
    assert results.json()["players"][0]["player_name"] == "Nikola Jokić"
    assert client.get("/api/players/by-id?player_id=1").status_code == 503

    request = {
        "unit_player_ids": [1, 2, 3, 4, 5],
        "opponent_player_ids": [6, 7, 8, 9, 10],
        "unit_season": reference.season,
        "opponent_season": reference.season,
        "court": "unit_home",
        "opponent_back_to_back": True,
    }
    evaluated = client.post("/api/lab/matchup", json=request)
    assert evaluated.status_code == 200
    expected = reference.evaluate(**request)
    for field in (
        "additive_margin",
        "contextual_adjustment",
        "predicted_net_rating",
        "predicted_win_pct",
    ):
        assert evaluated.json()[field] == pytest.approx(expected[field])
    assert (
        evaluated.json()["composition_feature_contributions"]
        == expected["composition_feature_contributions"]
    )

    request["opponent_player_ids"] = [1, 7, 8, 9, 10]
    assert client.post("/api/lab/matchup", json=request).status_code == 422
