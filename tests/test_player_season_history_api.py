"""Per-player DuckDB queries, missing observations and result-only caching."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from nba_lineup_model.web_api import player_season_history as api
from nba_lineup_model.web_api.gateway import MarketGateway

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def tables(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    first = {
        "season": "2024-25",
        "player_id": 1,
        "player_name": "Player One",
        "age": 25,
        "rapm": 2.53125,
        "prior_rapm": 2,
        "rapm_adjustment_from_prior": 0.43125,
        "additive_profile_adjustment": 0.1,
        "rank_all_players": 10,
    }
    second = {**first, "season": "2025-26", "rapm": 3.25, "rank_all_players": 8}
    data = {
        "ratings": [first, second, {**first, "player_id": 2, "player_name": "Other Player"}],
        "panel": [
            {
                "season": "2024-25",
                "player_id": 1,
                "games": 70,
                "games_started": 60,
                "rapm_possessions": 4000,
            }
        ],
        "splits": [
            {
                "season": "2024-25",
                "player_id": 1,
                "total_nail": 2.53125,
                "offense_rating": 3.03125,
                "defense_rating": -0.5,
            },
            {
                "season": "2025-26",
                "player_id": 1,
                "total_nail": 3.25,
                "offense_rating": 4,
                "defense_rating": -0.75,
            },
        ],
        "context": [
            {"season": "2024-25", "player_id": 1, "observed_context_exposure": 0.123456789}
        ],
        "teams": [
            {
                "season": "2024-25",
                "player_id": 1,
                "team_id": 1610612747,
                "team": "LAL",
                "possessions": 2500,
                "games": 45,
                "last_game_time_utc": "2025-01-01",
                "is_primary_team": True,
                "is_latest_team": False,
            },
            {
                "season": "2024-25",
                "player_id": 1,
                "team_id": 1610612742,
                "team": "DAL",
                "possessions": 1500,
                "games": 25,
                "last_game_time_utc": "2025-04-01",
                "is_primary_team": False,
                "is_latest_team": True,
            },
        ],
    }
    paths = {name: str(tmp_path / (name + ".parquet")) for name in data}
    for name, rows in data.items():
        pq.write_table(pa.Table.from_pylist(rows), paths[name])
    connection = duckdb.connect(config={"enable_external_file_cache": "false"})
    yield connection, paths, data
    connection.close()


def test_queries_requested_player_and_preserves_trades_precision_and_missing_data(tables):
    connection, paths, _ = tables
    seasons = api.query_player(connection, paths, 1)
    assert [s["season"] for s in seasons] == ["2024-25", "2025-26"]
    assert all(s["player_id"] == 1 for s in seasons)
    assert seasons[0]["season_max_player_id"] == 2
    assert seasons[0]["season_max_player_name"] == "Other Player"
    assert seasons[0]["team"] == "DAL"
    assert seasons[0]["rating"] == 2.53125 and seasons[0]["nail_rank"] == 10
    assert seasons[0]["observed_context_exposure"] == 0.123456789
    assert [s["team"] for s in seasons[0]["team_splits"]] == ["LAL", "DAL"]
    assert seasons[1]["games"] is None and seasons[1]["possessions"] is None
    assert seasons[1]["observed_context_exposure"] is None
    assert seasons[1]["team_splits"] == []
    assert api.query_player(connection, paths, 999) == []
    assert connection.execute("SELECT count(*) FROM duckdb_external_file_cache()").fetchone() == (
        0,
    )


@pytest.mark.parametrize("problem", ["duplicate", "split_mismatch", "future"])
def test_fails_invalid_saved_history_instead_of_partial_result(tables, problem):
    import pyarrow as pa
    import pyarrow.parquet as pq

    connection, paths, data = tables
    if problem == "duplicate":
        data["ratings"].append(data["ratings"][0])
        name = "ratings"
    elif problem == "future":
        data["ratings"][0]["season"] = "2026-27"
        name = "ratings"
    else:
        data["splits"][0]["offense_rating"] = 999
        name = "splits"
    pq.write_table(pa.Table.from_pylist(data[name]), paths[name])
    with pytest.raises(ValueError):
        api.query_player(connection, paths, 1)


def test_registered_object_replacement_is_rejected_before_query():
    reader = api.SeasonReader.__new__(api.SeasonReader)
    reader.client = Mock()
    source = next(iter(api.SOURCES.values()))
    reader.client.head_object.return_value = {
        "VersionId": "changed",
        "ETag": source["etag"],
        "ContentLength": source["size_bytes"],
    }
    with pytest.raises(ValueError, match="source object changed"):
        reader.check_sources()


def test_endpoint_caches_query_result_per_player_and_does_not_cache_errors(monkeypatch):
    reader = Mock()
    reader.history.side_effect = lambda pid: {
        "kind": "saved_player_season_history",
        "player_id": pid,
        "seasons": [],
    }
    monkeypatch.setattr(api, "_reader", lambda: reader)
    api._history.cache_clear()
    with TestClient(MarketGateway(None)) as client:
        for pid in (1, 1, 2):
            response = client.get(f"/api/players/{pid}/season-history")
            assert response.status_code == 200 and response.json()["player_id"] == pid
            assert response.headers["cache-control"] == "private, max-age=3600"
        assert reader.history.call_count == 2
        reader.history.side_effect = RuntimeError("private query details")
        for _ in range(2):
            response = client.get("/api/players/3/season-history")
            assert response.status_code == 503 and response.headers["cache-control"] == "no-store"
            assert "private query details" not in response.text
        assert reader.history.call_count == 4
        assert client.get("/api/players/0/season-history").status_code == 422
    api._history.cache_clear()


def test_source_identity_is_part_of_result_cache():
    with pytest.raises(ValueError, match="source selection"):
        api._history(1, "another-source")
