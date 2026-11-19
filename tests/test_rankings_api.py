"""Rank the full saved population with cached direct queries and no legacy bootstrap."""

from unittest.mock import Mock

import duckdb
import pytest
from fastapi.testclient import TestClient
from test_player_summary_api import DAY, INDEX, NEW_PID, PID, fixture, publish

from nba_lineup_model.web_api import rankings as api
from nba_lineup_model.web_api.gateway import MarketGateway


@pytest.fixture(autouse=True)
def clear_cache():
    for f in (api.current_rankings, api.historical_rankings, api.historical_seasons):
        f.cache_clear()
    yield
    for f in (api.current_rankings, api.historical_rankings, api.historical_seasons):
        f.cache_clear()


@pytest.fixture
def current(monkeypatch):
    s3, ratings, _, _ = fixture()
    s3.objects = {k: v for k, v in s3.objects.items() if "player_minutes/" not in k}
    monkeypatch.setattr(api, "_reader", lambda: Mock(client=s3))
    monkeypatch.setattr(api, "historical_seasons", lambda source: ["2025-26", "1996-97"])
    monkeypatch.setattr(api, "time", lambda: 3601)
    return s3, ratings


def test_full_current_population_precision_nullable_metadata_and_cache(current):
    s3, ratings = current
    legacy = Mock(side_effect=AssertionError("Do not bootstrap legacy"))
    client = TestClient(MarketGateway(legacy))
    response = client.get("/api/rankings")
    assert response.status_code == 200
    body = response.json()
    assert [p["player_id"] for p in body["players"]] == [NEW_PID, PID]
    assert [p["rank"] for p in body["players"]] == [1, 2]
    curry = next(p for p in body["players"] if p["player_id"] == PID)
    assert curry["rapm"] == ratings["players"][0]["nail"]
    assert curry["prior_rating"] == ratings["players"][0]["starting_nail"]
    assert curry["season_update"] == ratings["players"][0]["residual"]
    assert curry["possessions"] == ratings["players"][0]["on_court_possessions"]
    newcomer = body["players"][0]
    assert (
        newcomer["player_name"] == ratings["players"][1]["player_name"]
        and newcomer["position"] is None
    )
    assert newcomer["games"] is None and newcomer["offense_rating"] is None
    assert body["field_labels"]["prior_rating"] == "Starting NAIL"
    assert "games" not in body["available_columns"] and "ffapr" not in body["available_columns"]
    assert body["rating_state"]["evidence_through_date"] == str(DAY)
    assert len(body["available_seasons"]) == 3
    calls = len(s3.calls)
    assert client.get("/api/rankings").json() == body and len(s3.calls) == calls
    assert not any(
        any(bad in key for bad in ["player_minutes/", "rosters/", "transactions/"])
        for _, key in s3.calls
    )
    legacy.assert_not_called()


def test_changed_current_pointer_revalidated_hourly(current, monkeypatch):
    s3, ratings = current
    client = TestClient(MarketGateway(None))
    before = client.get("/api/rankings").json()
    ratings["players"][0]["nail"] += 0.25
    ratings["players"][0]["residual"] += 0.25
    publish(s3, "player_ratings", ratings)
    assert client.get("/api/rankings").json() == before
    monkeypatch.setattr(api, "time", lambda: 7201)
    after = client.get("/api/rankings").json()
    assert after["sources"]["ratings"]["sha256"] != before["sources"]["ratings"]["sha256"]
    assert (
        next(p for p in after["players"] if p["player_id"] == PID)["rapm"]
        == ratings["players"][0]["nail"]
    )


def test_invalid_metadata_keeps_ratings_and_is_retried(current):
    s3, _ = current
    raw = s3.objects[INDEX + ".json"]
    s3.objects[INDEX + ".json"] += b" "
    client = TestClient(MarketGateway(None))
    response = client.get("/api/rankings")
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert response.json()["metadata_warning"] and len(response.json()["players"]) == 2
    assert all(p["position"] is None for p in response.json()["players"])
    s3.objects[INDEX + ".json"] = raw
    assert client.get("/api/rankings").json()["metadata_warning"] is None


def test_bad_ratings_fail_and_failure_is_not_cached(current):
    s3, ratings = current
    key = next(k for k in s3.objects if "player_ratings/" in k and not k.endswith("latest.json"))
    raw = s3.objects[key]
    s3.objects[key] += b" "
    client = TestClient(MarketGateway(None))
    response = client.get("/api/rankings")
    assert response.status_code == 503 and response.headers["cache-control"] == "no-store"
    s3.objects[key] = raw
    assert client.get("/api/rankings").status_code == 200
    assert client.get("/api/rankings?season=bad").status_code == 422
    assert client.get("/api/rankings?season=1900-01").status_code == 404


@pytest.fixture
def historical(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    def rating(pid, nail, rank):
        return {
            "season": "2025-26",
            "player_id": pid,
            "player_name": f"Player {pid}",
            "age": 25.0,
            "rapm": nail,
            "rank_all_players": rank,
            "prior_rapm": 2.0,
            "rapm_adjustment_from_prior": nail - 2.125,
            "additive_profile_adjustment": 0.125,
        }

    def team(pid, code, latest):
        return {
            "season": "2025-26",
            "player_id": pid,
            "team": code,
            "team_id": 1 if code == "LAL" else 2,
            "is_latest_team": latest,
        }

    data = {
        "ratings": [rating(1, 3.125, 10), rating(2, 0.0, 500), rating(3, 9.0, 1)],
        "teams": [team(1, "BOS", False), team(1, "LAL", True), team(2, "BOS", True)],
        "panel": [
            {
                "season": "2025-26",
                "player_id": 1,
                "player_name": "Panel Player",
                "listed_position": "G",
                "draft_year": 2010,
                "draft_round": 1,
                "draft_number": 5,
                "is_undrafted": False,
                "games": 80,
                "rapm_possessions": 999.125,
            }
        ],
        "splits": [
            {
                "season": "2025-26",
                "player_id": 1,
                "total_nail": 3.125,
                "offense_rating": 2.0,
                "defense_rating": 1.125,
            }
        ],
        "context": [{"season": "2025-26", "player_id": 1, "observed_context_exposure": 0.25}],
    }
    sources = {}
    for key, rows in data.items():
        file = tmp_path / f"{key}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), file)
        sources[key] = {"uri": str(file)}
    connection = duckdb.connect(
        config={"threads": 1, "memory_limit": "64MB", "enable_external_file_cache": False}
    )
    reader = Mock(connection=connection)
    monkeypatch.setattr(api, "SOURCES", sources)
    monkeypatch.setattr(api, "_reader", lambda: reader)
    yield data, sources, reader
    connection.close()


def test_historical_full_population_trade_assignment_and_saved_ranks(historical):
    _, _, reader = historical
    result = api.historical_rankings("2025-26", api.SOURCE_ID)
    assert [p["player_id"] for p in result["players"]] == [3, 1, 2]
    assert [p["rank"] for p in result["players"]] == [1, 10, 500]
    unassigned, first, missing = result["players"]
    assert unassigned["team"] == "-" and unassigned["possessions"] is None
    assert first["team"] == "LAL" and first["possessions"] == 999.125 and first["games"] == 80
    assert first["offense_rating"] + first["defense_rating"] == first["rapm"] == 3.125
    assert first["draft_class_year"] == 2010 and first["observed_context_exposure"] == 0.25
    assert missing["games"] is None and missing["rapm"] == 0.0
    assert api.historical_rankings("2025-26", api.SOURCE_ID) == result
    assert reader.check_sources.call_count == 2 and reader.authenticate.call_count == 1
    assert api.historical_rankings.cache_info().maxsize == 8
    assert api.historical_seasons(api.SOURCE_ID) == ["2025-26"]
    with pytest.raises(ValueError):
        api.historical_rankings("2025-26", "changed-selection")
    client = TestClient(MarketGateway(None))
    assert client.get("/api/rankings?season=2025-26").status_code == 200
    assert client.get("/api/rankings?season=2024-25").status_code == 404


@pytest.mark.parametrize("damage", ["split", "duplicate", "nonfinite"])
def test_bad_historical_inputs_are_not_cached(historical, damage):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data, sources, _ = historical
    key = "splits" if damage == "split" else "ratings"
    if damage == "split":
        data[key][0]["total_nail"] = 99.0
    elif damage == "duplicate":
        data[key].append(data[key][0])
    else:
        data[key][0]["rapm"] = float("nan")
    pq.write_table(pa.Table.from_pylist(data[key]), sources[key]["uri"])
    with pytest.raises(ValueError):
        api.historical_rankings("2025-26", api.SOURCE_ID)
    assert api.historical_rankings.cache_info().currsize == 0


def test_zero_exposure_introduction_stays_in_population(current):
    s3, ratings = current
    ratings["players"][1].update(
        on_court_possessions=0.0,
        initialized_on="2026-10-29",
        initialization={"kind": "introduced_prior"},
    )
    publish(s3, "player_ratings", ratings)
    data = TestClient(MarketGateway(None)).get("/api/rankings").json()
    newcomer = next(p for p in data["players"] if p["player_id"] == NEW_PID)
    assert newcomer["possessions"] == 0.0 and newcomer["initialized_on"] == "2026-10-29"
    assert newcomer["initialization"]["kind"] == "introduced_prior"
    assert len(data["players"]) == 2 and data["field_labels"]["prior_rating"] == "Starting NAIL"


def test_historical_navigation_failure_does_not_hide_current_ratings(current, monkeypatch):
    lookup = Mock(side_effect=ValueError("Historical source unavailable"))
    monkeypatch.setattr(api, "historical_seasons", lookup)
    client = TestClient(MarketGateway(None))
    response = client.get("/api/rankings")
    assert response.status_code == 200 and len(response.json()["players"]) == 2
    assert response.json()["available_seasons"] == ["2026-27"]
    assert response.json()["navigation_warning"] and response.headers["cache-control"] == "no-store"
    assert client.get("/api/rankings?season=2025-26").status_code == 503
    lookup.side_effect = None
    lookup.return_value = ["2025-26"]
    assert client.get("/api/rankings").json()["navigation_warning"] is None


def test_missing_historical_names_do_not_hide_saved_ratings(historical):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data, sources, _ = historical
    data["ratings"][0]["player_name"] = None
    data["ratings"][2]["player_name"] = None
    pq.write_table(pa.Table.from_pylist(data["ratings"]), sources["ratings"]["uri"])
    players = api.historical_rankings("2025-26", api.SOURCE_ID)["players"]
    rows = {p["player_id"]: p for p in players}
    assert len(rows) == 3 and rows[3]["rapm"] == 9.0 and rows[3]["rank"] == 1
    assert rows[3]["name_missing"] and rows[3]["player_name"] == "NBA player 3"
    assert rows[1]["player_name"] == "Panel Player" and not rows[1]["name_missing"]
