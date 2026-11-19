"""Saved rotation records preserve observations, preseason forecasts and result caching."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from nba_lineup_model.web_api import player_rotation_history_api as api
from nba_lineup_model.web_api.gateway import MarketGateway

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def saved(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    past = {
        "season": "2025-26",
        "season_start_year": 2025,
        "player_id": 1,
        "player_name": "Player One",
        "age": 25,
        "actual_available_games": 70.0,
        "known_roster_games": 82.0,
        "actual_availability_share": 70 / 82,
        "injury_or_illness_games": 8.0,
        "rest_games": 4.0,
        "actual_minutes_per_available_game": 31.23456789,
        "actual_total_minutes": 2186.4197523,
        "predicted_available_share": 0.8,
        "predicted_minutes_per_available_game": 30.125,
        "projected_total_minutes": 1976.2,
        "is_preseason_forecast": False,
    }
    forecast = {
        **past,
        "season": "2026-27",
        "season_start_year": 2026,
        "is_preseason_forecast": True,
        **{
            k: None
            for k in (
                "actual_available_games",
                "known_roster_games",
                "actual_availability_share",
                "injury_or_illness_games",
                "rest_games",
                "actual_minutes_per_available_game",
                "actual_total_minutes",
            )
        },
    }
    rows = [past, forecast, {**past, "player_id": 2}]
    path = tmp_path / "rotation.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path), rows


def test_only_requested_player_with_saved_forecast_and_null_observations(saved):
    path, _ = saved
    with duckdb.connect(config={"enable_external_file_cache": "false"}) as connection:
        rows = api.query_player(connection, path, 1)
        assert [r["season"] for r in rows] == ["2025-26", "2026-27"]
        assert rows[0]["actual_minutes_per_available_game"] == 31.23456789
        assert rows[1]["is_preseason_forecast"] is True
        assert rows[1]["actual_total_minutes"] is None
        assert rows[1]["predicted_minutes_per_available_game"] == 30.125
        assert api.query_player(connection, path, 999) == []
        assert connection.execute(
            "SELECT count(*) FROM duckdb_external_file_cache()"
        ).fetchone() == (0,)


@pytest.mark.parametrize(
    "invalid", ["duplicate", "bad_year", "current_observed", "share", "negative"]
)
def test_bad_saved_records_fail_instead_of_becoming_partial_history(saved, invalid):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path, rows = saved
    if invalid == "duplicate":
        rows.append(rows[0])
    elif invalid == "bad_year":
        rows[0]["season_start_year"] = 2024
    elif invalid == "current_observed":
        rows[1]["is_preseason_forecast"] = False
    elif invalid == "share":
        rows[0]["predicted_available_share"] = 2
    else:
        rows[0]["actual_minutes_per_available_game"] = -1
    pq.write_table(pa.Table.from_pylist(rows), path)
    with duckdb.connect() as connection, pytest.raises(ValueError):
        api.query_player(connection, path, 1)


def test_endpoint_shares_one_cached_query_for_chart_and_table_and_does_not_cache_failure(
    monkeypatch,
):
    reader = Mock()
    reader.connection = object()
    query = Mock(
        side_effect=lambda conn, uri, pid: [
            {"player_id": pid, "player_name": "Player", "season": "2025-26"}
        ]
    )
    monkeypatch.setattr(api, "_reader", lambda: reader)
    monkeypatch.setattr(api, "query_player", query)
    api._history.cache_clear()
    with TestClient(MarketGateway(None)) as client:
        for pid in (1, 1, 2):
            response = client.get(f"/api/players/{pid}/rotation-history")
            assert response.status_code == 200
            assert response.json()["player_id"] == pid
        assert query.call_count == 2
        query.side_effect = RuntimeError("private query arguments")
        for _ in range(2):
            response = client.get("/api/players/3/rotation-history")
            assert response.status_code == 503 and response.headers["cache-control"] == "no-store"
            assert "private query arguments" not in response.text
        assert query.call_count == 4
    api._history.cache_clear()
