"""Search and official media work without the legacy evaluator or source writes."""

from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from test_player_summary_api import DAY, NEW_PID, PID, fixture

from nba_lineup_model.web_api import player_discovery as api
from nba_lineup_model.web_api import player_summary as profiles
from nba_lineup_model.web_api.gateway import MarketGateway

PNG = b"\x89PNG\r\n\x1a\n" + b"test"


@pytest.fixture
def inputs():
    s3, _, _, _ = fixture()
    return profiles.read_player_inputs(s3, season=profiles.SEASON, cutoff=DAY)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("curry", PID),
        ("STEPHEN CURRY", PID),
        ("curry stephen", PID),
        (str(PID), PID),
        ("Otega", NEW_PID),
        ("Retired", 23),
    ],
)
def test_search_uses_current_and_historical_ids(inputs, query, expected):
    result = api.search_players(inputs, query, 8)
    assert result[0]["player_id"] == expected
    assert "history_seasons" not in result[0] and "latest_season" not in result[0]
    if expected == 23:
        assert result[0]["summary_available"] is False
    else:
        assert result[0]["summary_available"] is True


def test_search_accents_empty_queries_and_limits(inputs):
    inputs["biographies"][42] = {"name": "José Player"}
    inputs["biographies"][43] = {"name": "Jose Player Two"}
    assert api.search_players(inputs, "jose", 1)[0]["player_id"] == 42
    assert len(api.search_players(inputs, "player", 1)) == 1
    assert api.search_players(inputs, "....", 8) == []
    assert api.search_players(inputs, "not-a-real-name", 8) == []
    assert api.search_players(inputs, "   ", 8) == []


def test_search_and_headshot_routes_bypass_legacy(inputs, monkeypatch):
    monkeypatch.setattr(profiles, "shared_inputs", lambda: (inputs, "private, max-age=123"))
    upstream = Mock(
        return_value=httpx.Response(
            200,
            content=PNG,
            headers={"content-type": "image/png"},
            request=httpx.Request(
                "GET", "https://cdn.nba.com/headshots/nba/latest/260x190/201939.png"
            ),
        )
    )
    monkeypatch.setattr(api.httpx, "get", upstream)
    legacy = Mock(side_effect=AssertionError("Legacy evaluator must not initialize"))
    client = TestClient(MarketGateway(legacy))
    api._headshot.cache_clear()
    try:
        result = client.get("/api/search", params={"q": "curry"})
        assert result.status_code == 200 and result.json()["players"][0]["player_id"] == PID
        assert result.json()["scope"] == "players" and result.json()["teams"] == []
        assert result.headers["cache-control"] == "private, max-age=123"
        assert (
            client.get("/api/search", params={"q": "Retired"}).json()["players"][0]["player_id"]
            == 23
        )
        assert client.get("/api/search", params={"q": ""}).status_code == 422
        assert client.get("/api/search", params={"q": "x" * 129}).status_code == 422
        assert client.get("/api/search", params={"q": "curry", "limit": 26}).status_code == 422
        photo = client.get(f"/api/headshots/{PID}.png")
        assert photo.status_code == 200 and photo.content == PNG
        assert photo.headers["content-type"] == "image/png"
        assert photo.headers["cache-control"] == "public, max-age=43200"
        assert client.get(f"/api/headshots/{PID}.png").status_code == 200
        assert upstream.call_count == 1
        assert client.get("/api/headshots/999999999.png").status_code == 404
        assert client.get("/api/headshots/0.png").status_code == 422
        assert upstream.call_count == 1
        legacy.assert_not_called()
        assert api._headshot.cache_info().maxsize == 64
    finally:
        api._headshot.cache_clear()


@pytest.mark.parametrize(
    "damage", ["content_type", "signature", "too_large", "not_found", "timeout"]
)
def test_invalid_or_unavailable_photos_fail_cleanly(inputs, monkeypatch, damage):
    monkeypatch.setattr(profiles, "shared_inputs", lambda: (inputs, "private, max-age=123"))
    raw = b"not a PNG" if damage == "signature" else PNG
    if damage == "too_large":
        raw = PNG + b"0" * 2_000_000
    reply = httpx.Response(
        404 if damage == "not_found" else 200,
        content=raw,
        headers={"content-type": "image/svg+xml" if damage == "content_type" else "image/png"},
        request=httpx.Request("GET", "https://cdn.nba.com/headshots/nba/latest/260x190/201939.png"),
    )
    upstream = Mock(return_value=reply)
    if damage == "timeout":
        upstream.side_effect = httpx.ReadTimeout("timeout")
    monkeypatch.setattr(api.httpx, "get", upstream)
    api._headshot.cache_clear()
    try:
        response = TestClient(MarketGateway(None)).get(f"/api/headshots/{PID}.png")
        assert response.status_code == 404 and response.headers["cache-control"] == "no-store"
    finally:
        api._headshot.cache_clear()


def test_search_does_not_hide_invalid_required_inputs(monkeypatch):
    monkeypatch.setattr(profiles, "shared_inputs", Mock(side_effect=ValueError("bad checksum")))
    response = TestClient(MarketGateway(None)).get("/api/search", params={"q": "curry"})
    assert response.status_code == 503 and response.headers["cache-control"] == "no-store"
