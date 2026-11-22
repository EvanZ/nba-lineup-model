from unittest.mock import Mock

import duckdb
import pytest
from fastapi.testclient import TestClient

from nba_lineup_model.web_api import historical_lineups as history
from nba_lineup_model.web_api.gateway import MarketGateway


@pytest.fixture
def archive(tmp_path, monkeypatch):
    file = str(tmp_path / "lineups.parquet")
    legacy = Mock(side_effect=AssertionError("Historical lineups must bypass the evaluator"))
    reader = Mock()
    reader.connection = duckdb.connect()
    reader.connection.execute(
        """
        COPY (
            SELECT 1 AS team_id, 'AAA' AS team, ids AS player_ids,
                   ['One','Two','Three','Four','Five'] AS player_names, label AS lineup_label,
                   possessions, 5 AS games, 7.4E0 AS player_rating, 2.125E0 AS player_edge,
                   2.25E0 AS offensive_edge, context_edge - 0.125E0 AS defensive_edge,
                   0.1E0 AS composition_rating, 0.2E0 AS composition_edge, 0.3E0 AS matchup_bonus,
                   context_edge, context_edge + 2.125E0 AS gestalt_score,
                   -3.45E0 AS actual_net_rating, 100.5E0 AS actual_offensive_rating,
                   103.95E0 AS actual_defensive_rating
            FROM (VALUES
                ([1,2,3,4,5], 'regular', 600, 2.0E0),
                ([1,6,7,8,9], 'larger sample', 800, 2.0E0),
                ([1,2,3,4,6], 'small sample', 125, 5.0E0)
            ) AS t(ids, label, possessions, context_edge)
        ) TO ? (FORMAT PARQUET)
        """,
        [file],
    )
    monkeypatch.setattr(history, "_reader", lambda: reader)
    monkeypatch.setattr(history, "source_for", lambda season: {"uri": file})
    history._query.cache_clear()
    yield TestClient(MarketGateway(legacy)), reader, legacy
    history._query.cache_clear()
    reader.connection.close()


def test_archive_filters_before_ranking_and_reuses_cached_query(archive):
    client, reader, legacy = archive
    response = client.get("/api/lineups")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available_seasons"][:2] == ["2026-27", "2025-26"]
    assert payload["season"] == "2025-26"
    assert [(r["rank"], r["lineup_label"]) for r in payload["lineups"]] == [
        (1, "larger sample"),
        (2, "regular"),
    ]
    response = client.get("/api/lineups", params={"minimum_possessions": 0, "player_id": [4, 1]})
    assert [(r["rank"], r["lineup_label"]) for r in response.json()["lineups"]] == [
        (1, "small sample"),
        (2, "regular"),
    ]
    assert response.json()["lineups"][0]["actual_net_rating"] == -3.45
    repeat = client.get("/api/lineups", params={"minimum_possessions": 0, "player_id": [1, 4, 4]})
    assert repeat.content == response.content
    assert reader.authenticate.call_count == 2
    assert reader.check_sources.call_count == 4
    legacy.assert_not_called()


def test_absent_season_and_archive_replacement_are_explicit(archive):
    client, reader, _ = archive
    assert client.get("/api/lineups?season=1996-97").status_code == 404
    reader.check_sources.side_effect = ValueError("Archive object changed")
    response = client.get("/api/lineups")
    assert response.status_code == 503
    assert response.headers["Cache-Control"] == "no-store"
    reader.authenticate.assert_not_called()
