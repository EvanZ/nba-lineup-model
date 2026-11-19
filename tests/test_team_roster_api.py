"""Roster coverage, saved statistics, provenance, and bounded result caching."""

import hashlib
import json
from unittest.mock import Mock

import duckdb
import pytest
from fastapi.testclient import TestClient
from test_player_summary_api import DAY, NEW_PID, PID, fixture, publish

from nba_lineup_model.web_api import team_navigation as nav
from nba_lineup_model.web_api import team_roster as api
from nba_lineup_model.web_api.gateway import MarketGateway


@pytest.fixture(autouse=True)
def clear_caches():
    api.supplemental_names.cache_clear()
    api.current_inputs.cache_clear()
    api.historical_roster.cache_clear()
    yield
    api.supplemental_names.cache_clear()
    api.current_inputs.cache_clear()
    api.historical_roster.cache_clear()


@pytest.fixture
def current(monkeypatch):
    s3, ratings, _, _ = fixture()
    # The roster must work without conditional minutes or any per-player requests.
    s3.objects = {k: v for k, v in s3.objects.items() if "player_minutes/" not in k}
    teams = {str(i): [] for i in range(1, 31)}
    teams["1"] = [PID, NEW_PID, 999]
    day = "2026-11-18"
    doc = {
        "schema_version": 1,
        "season": api.SEASON,
        "game_date_pacific": day,
        "teams": teams,
        "inputs": {"transactions": {}},
    }
    root = f"lake/artifacts/rosters/season={api.SEASON}/game_date_pacific={day}/"
    raw = json.dumps(doc).encode()
    sha = hashlib.sha256(raw).hexdigest()
    uri = f"s3://{api.BUCKET}/{root}{sha}.json"
    s3.objects[root + sha + ".json"] = raw
    monkeypatch.setattr(api, "_reader", lambda: Mock(client=s3))
    return s3, ratings, doc, day, {"roster": {"uri": uri, "sha256": sha}}


def test_current_membership_precision_and_missing_data(current):
    s3, ratings, _, day, sources = current
    result = api.current_roster({"team_id": 1}, day, sources)
    assert result["status"] == "available"
    rows = {p["player_id"]: p for p in result["players"]}
    assert set(rows) == {PID, NEW_PID, 999}
    assert rows[PID]["nail"] == ratings["players"][0]["nail"]
    assert rows[PID]["possessions"] == ratings["players"][0]["on_court_possessions"]
    assert rows[NEW_PID]["player_name"] == ratings["players"][1]["player_name"]
    assert rows[NEW_PID]["position"] is None
    assert rows[999]["nail"] is None and rows[999]["rank"] is None
    assert rows[999]["name_missing"] and rows[999]["player_name"] == "NBA player 999"
    assert result["players"][-1]["player_id"] == 999
    assert result["rating_state"]["evidence_through_date"] == str(DAY)
    assert result["roster_as_of"] == day
    assert all(p["games"] is None and p["offense_rating"] is None for p in rows.values())
    assert result["rating_state"]["rank_population_count"] == len(ratings["players"])
    assert not any("player_minutes/" in key for _, key in s3.calls)
    calls = len(s3.calls)
    api.current_roster({"team_id": 1}, day, sources)
    assert len(s3.calls) == calls


def test_current_bad_checksum_not_cached(current):
    s3, _, _, day, sources = current
    key = sources["roster"]["uri"].split(api.BUCKET + "/", 1)[1]
    raw = s3.objects[key]
    s3.objects[key] += b" "
    with pytest.raises(ValueError, match="checksum"):
        api.current_roster({"team_id": 1}, day, sources)
    s3.objects[key] = raw
    assert api.current_roster({"team_id": 1}, day, sources)["status"] == "available"


def test_corrected_selection_invalidates_current_cache(current, monkeypatch):
    s3, ratings, _, day, sources = current
    monkeypatch.setattr(api, "time", lambda: 3601)
    first = api.current_roster({"team_id": 1}, day, sources)
    ratings["players"][0]["nail"] += 0.125
    ratings["players"][0]["residual"] += 0.125
    publish(s3, "player_ratings", ratings)
    # Mutable selections revalidate at the next hourly key; prior bytes stay unchanged.
    monkeypatch.setattr(api, "time", lambda: 7201)
    updated = api.current_roster({"team_id": 1}, day, sources)
    assert first["sources"]["ratings"]["sha256"] != updated["sources"]["ratings"]["sha256"]
    assert (
        next(p for p in updated["players"] if p["player_id"] == PID)["nail"]
        == ratings["players"][0]["nail"]
    )


@pytest.fixture
def historical(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    def member(pid, team, latest):
        return {
            "season": "2025-26",
            "player_id": pid,
            "team_id": 1 if team == "LAL" else 2,
            "team": team,
            "possessions": 20.0,
            "games": 2,
            "is_latest_team": latest,
            "is_primary_team": latest,
            "last_game_time_utc": "2026-04-01" if latest else "2026-01-01",
        }

    data = {
        "teams": [
            member(1, "BOS", False),
            member(1, "LAL", True),
            member(2, "LAL", True),
            member(3, "LAL", False),
            member(3, "BOS", True),
        ],
        "ratings": [
            {
                "season": "2025-26",
                "player_id": 1,
                "player_name": "Traded Player",
                "rapm": 3.125,
                "rank_all_players": 100,
            },
            {
                "season": "2025-26",
                "player_id": 3,
                "player_name": "Former Player",
                "rapm": 9.0,
                "rank_all_players": 1,
            },
        ],
        "panel": [
            {
                "season": "2025-26",
                "player_id": 1,
                "player_name": "Traded Player",
                "listed_position": "G",
                "games": 80,
                "rapm_possessions": 999.125,
            },
            {
                "season": "2025-26",
                "player_id": 2,
                "player_name": "Unrated Player",
                "listed_position": None,
                "games": 1,
                "rapm_possessions": 10.0,
            },
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
    monkeypatch.setattr(api, "_reader", lambda: reader)
    monkeypatch.setattr(api, "HISTORICAL", sources)
    yield data, sources, reader
    connection.close()


def test_historical_season_end_membership_and_saved_statistics(historical):
    _, _, reader = historical
    result = api.historical_roster("LAL", "2025-26", api.SOURCE_ID)
    assert [p["player_id"] for p in result["players"]] == [1, 2]
    first, unrated = result["players"]
    assert first["rank"] == 100 and first["nail"] == 3.125
    assert first["possessions"] == 999.125 and first["games"] == 80
    assert first["offense_rating"] + first["defense_rating"] == first["nail"]
    assert [s["team"] for s in first["team_splits"]] == ["BOS", "LAL"]
    assert unrated["player_name"] == "Unrated Player" and unrated["nail"] is None
    assert unrated["position"] is None and unrated["rank"] is None
    assert api.historical_roster("LAL", "2025-26", api.SOURCE_ID) == result
    assert reader.check_sources.call_count == 2 and reader.authenticate.call_count == 1
    assert api.historical_roster.cache_info().maxsize == 128
    with pytest.raises(ValueError):
        api.historical_roster("LAL", "2025-26", "changed-source")


@pytest.mark.parametrize("damage", ["split", "duplicate", "nonfinite"])
def test_historical_corrupt_saved_data_fails_without_hiding_players(historical, damage):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data, sources, _ = historical
    key = "splits" if damage == "split" else "panel"
    if damage == "split":
        data[key][0]["total_nail"] = 999.0
    elif damage == "duplicate":
        data[key].append(data[key][0])
    else:
        data[key][0]["rapm_possessions"] = float("nan")
    pq.write_table(pa.Table.from_pylist(data[key]), sources[key]["uri"])
    with pytest.raises(ValueError):
        api.historical_roster("LAL", "2025-26", api.SOURCE_ID)
    assert api.historical_roster.cache_info().currsize == 0


def test_roster_failure_does_not_blank_navigation(monkeypatch):
    entry = {
        "team": "LAL",
        "team_id": 1,
        "display_name": "Los Angeles Lakers",
        "seasons": ["2026-27"],
        "latest_season": "2026-27",
        "season_count": 1,
    }
    monkeypatch.setattr(
        nav, "catalog", lambda: (({"LAL": entry}, "2026-11-18", {}), "private, max-age=60")
    )
    monkeypatch.setattr(api, "current_roster", Mock(side_effect=ValueError("bad hash")))
    result = TestClient(MarketGateway(None)).get("/api/teams/LAL")
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    assert result.json()["display_name"] == "Los Angeles Lakers"
    assert result.json()["roster"]["status"] == "unavailable"
    assert "roster" not in result.json()["deferred_sections"]


def test_current_rank_population_is_not_team_local(current):
    s3, ratings, _, day, sources = current
    outside = {
        **ratings["players"][0],
        "player_id": 555,
        "player_name": "Outside roster",
        "nail": 100.0,
        "starting_nail": 100.0,
        "residual": 0.0,
    }
    ratings["players"].append(outside)
    publish(s3, "player_ratings", ratings)
    result = api.current_roster({"team_id": 1}, day, sources)
    assert next(p for p in result["players"] if p["player_id"] == PID)["rank"] == 3
    assert result["rating_state"]["rank_population_count"] == 3
    assert 555 not in [p["player_id"] for p in result["players"]]


@pytest.mark.parametrize("damage", [None, "checksum", "source", "identity", "timestamp"])
def test_transaction_names_use_verified_text_not_invented_slugs(current, damage):
    s3, _, roster, _, _ = current
    day = "2026-11-17"
    root = f"lake/raw/nba/transactions/season={api.SEASON}/game_date_pacific={day}/"
    doc = {
        "season": api.SEASON,
        "game_date_pacific": day,
        "source_url": api.TRANSACTIONS_URL,
        "transactions": [
            {
                "PLAYER_ID": "999",
                "PLAYER_SLUG": "luka-doncic",
                "TRANSACTION_DESCRIPTION": "The club signed Luka Dončić.",
                "TRANSACTION_DATE": day + "T12:00:00",
                "first_observed_at": day + "T20:00:00+00:00",
            }
        ],
    }
    if damage == "source":
        doc["source_url"] = "https://example.com/transactions"
    if damage == "identity":
        doc["season"] = "2025-26"
    if damage == "timestamp":
        doc["transactions"][0]["first_observed_at"] = day + "T20:00:00"
    sha = s3.save(root + "latest.json", doc)
    ref = {"uri": f"s3://{api.BUCKET}/{root}latest.json", "sha256": sha}
    roster["inputs"]["transactions"] = {day: ref}
    if damage == "checksum":
        s3.objects[root + "latest.json"] += b" "
    if damage:
        with pytest.raises(ValueError):
            api.transaction_names(s3, roster, {999})
    else:
        names, sources = api.transaction_names(s3, roster, {999})
        assert names == {999: "Luka Dončić"} and sources == {day: ref}
    assert (
        api.transaction_name(
            {"PLAYER_SLUG": "invented-name", "TRANSACTION_DESCRIPTION": "No actual name here."}
        )
        is None
    )


def test_failed_newcomer_name_source_cannot_hide_other_teams(current, monkeypatch):
    s3, _, doc, day, _ = current
    doc["teams"]["1"].remove(999)
    doc["teams"]["2"] = [999]
    raw = json.dumps(doc).encode()
    sha = hashlib.sha256(raw).hexdigest()
    root = f"lake/artifacts/rosters/season={api.SEASON}/game_date_pacific={day}/"
    s3.objects[root + sha + ".json"] = raw
    sources = {"roster": {"uri": f"s3://{api.BUCKET}/{root}{sha}.json", "sha256": sha}}
    entries = {
        code: {
            "team": code,
            "team_id": tid,
            "display_name": code,
            "seasons": [api.SEASON],
            "latest_season": api.SEASON,
            "season_count": 1,
        }
        for code, tid in [("LAL", 1), ("CLE", 2)]
    }
    monkeypatch.setattr(nav, "catalog", lambda: ((entries, day, sources), "private, max-age=60"))
    lookup = Mock(side_effect=ValueError("Pinned transaction checksum changed"))
    monkeypatch.setattr(api, "transaction_names", lookup)
    client = TestClient(MarketGateway(None))
    healthy = client.get("/api/teams/LAL").json()["roster"]
    assert healthy["status"] == "available" and len(healthy["players"]) == 2
    assert not healthy["name_source_errors"]
    lookup.assert_not_called()
    affected = client.get("/api/teams/CLE")
    assert affected.headers["cache-control"] == "no-store"
    roster = affected.json()["roster"]
    assert roster["status"] == "available"
    assert roster["players"][0]["player_id"] == 999
    assert roster["players"][0]["name_missing"]
    assert roster["identity_warning"] and roster["name_source_errors"][0]["player_ids"] == [999]
    # Rejected bytes never supply a name, and the failure is retried, not cached.
    lookup.side_effect = None
    lookup.return_value = ({999: "Verified official name"}, {})
    repaired = client.get("/api/teams/CLE")
    assert repaired.headers["cache-control"] == "private, max-age=60"
    assert repaired.json()["roster"]["players"][0]["player_name"] == "Verified official name"
    assert not repaired.json()["roster"]["name_source_errors"]
    again = client.get("/api/teams/LAL")
    assert again.headers["cache-control"] == "private, max-age=60"
    assert len(again.json()["roster"]["players"]) == 2
