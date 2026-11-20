"""Rotation consumers read compact accepted summaries and preserve their observed values."""

from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_saved_schedule_api import SavedS3

from nba_lineup_model.web_api import team_rotation as api
from nba_lineup_model.web_api.gateway import MarketGateway
from nba_lineup_model.web_api.schedule import ScheduleReader

DAY = "2026-11-19"
ROOT = "lake/artifacts/team_rotation/season=2026-27/team_id=1/"


def document():
    active = np.array(
        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 1], [1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 1]]
    )
    weights = np.array([2.5, 3, 4.25, 1.5])
    exposure = active.T @ weights
    minutes = active.T @ (np.array([100, 120, 140, 160]) / 60)
    shared = active.T @ (active * weights[:, None])
    similarity = shared / (exposure[:, None] + exposure[None, :] - shared)
    correlation = np.corrcoef(active * weights[:, None], rowvar=False)
    order = sorted(range(6), key=lambda i: (-exposure[i], f"Player {i + 1:02}", i + 1))
    gids = ["0022600001", "0022600002"]
    return {
        "schema_version": 1,
        "processing_version": "v1",
        "artifact_kind": "team_rotation",
        "season": "2026-27",
        "team_id": 1,
        "team": "HOM",
        "game_date_pacific": DAY,
        "last_team_game_date_pacific": "2026-11-18",
        "calculation": {
            "population": "team_regular_season_stints_only",
            "possessions": "positive_only",
            "floor_signal": "stint_possessions_when_on_court_else_zero",
            "correlation": "ordinary_pearson",
            "similarity": "shared_over_union",
        },
        "player_order": [i + 1 for i in order],
        "players": [
            {
                "player_id": i + 1,
                "player_name": f"Player {i + 1:02}",
                "on_court_possessions": float(exposure[i]),
                "on_court_minutes": float(minutes[i]),
                "games": 2,
            }
            for i in order
        ],
        "shared_possessions": shared[np.ix_(order, order)].tolist(),
        "shared_court_similarity": similarity[np.ix_(order, order)].tolist(),
        "floor_correlation": correlation[np.ix_(order, order)].tolist(),
        "coverage": {"complete": True, "game_count": 2, "stint_count": 4, "game_ids": gids},
        "inputs": {"box_scores": {gid: {} for gid in gids}, "stints": {}, "schedule": {}},
    }


def publish(s3, doc):
    import hashlib

    from test_saved_schedule_api import encoded

    digest = hashlib.sha256(encoded(doc)).hexdigest()
    key = ROOT + f"game_date_pacific={doc['game_date_pacific']}/{digest}.json"
    s3.put_fixture(key, doc)
    pointer = {
        key: doc[key]
        for key in [
            "schema_version",
            "processing_version",
            "season",
            "team_id",
            "game_date_pacific",
        ]
    }
    pointer.update(uri=f"s3://nba-gestalt-data/{key}", sha256=digest)
    s3.put_fixture(ROOT + "latest.json", pointer)
    return key, pointer


@pytest.fixture
def source(monkeypatch):
    s3 = SavedS3()
    doc = document()
    publish(s3, doc)
    reader = ScheduleReader(s3)
    monkeypatch.setattr(api, "_reader", lambda: reader)
    monkeypatch.setattr(
        api,
        "catalog",
        lambda: (
            ({"HOM": {"team_id": 1, "seasons": ["2026-27"]}}, DAY, {}),
            "private, max-age=600",
        ),
    )
    monkeypatch.setattr(api, "time", lambda: 3601)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 20, 12, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(api, "datetime", Clock)
    api.cached_summary.cache_clear()
    legacy = Mock(side_effect=AssertionError("Legacy must not initialize"))
    yield s3, reader, TestClient(MarketGateway(legacy)), doc, legacy
    api.cached_summary.cache_clear()


@pytest.mark.parametrize("ordering", ["hclust", "aoe", "fpc"])
@pytest.mark.parametrize("metric", ["floor", "overlap"])
def test_ordering_preserves_every_saved_player_and_matrix_value(source, ordering, metric):
    s3, _, client, doc, legacy = source
    response = client.get(
        "/api/teams/hom/rotation", params={"ordering": ordering, "order_metric": metric}
    )
    assert (
        response.status_code == 200 and response.headers["cache-control"] == "private, max-age=599"
    )
    data = response.json()
    assert data["kind"] == "saved_team_rotation" and data["game_date_pacific"] == DAY
    assert data["game_count"] == 2 and len(data["players"]) == 6
    indices = [doc["player_order"].index(p["player_id"]) for p in data["players"]]
    for field in ["shared_possessions", "shared_court_similarity", "floor_correlation"]:
        assert np.array_equal(np.array(data[field]), np.array(doc[field])[np.ix_(indices, indices)])
    assert data["players"] == [doc["players"][i] for i in indices]
    assert len(data["dendrogram"]["branches"]) == (5 if ordering == "hclust" else 0)
    for branch in data["dendrogram"]["branches"]:
        assert len(branch["x"]) == len(branch["height"]) == 4
        assert all(0 <= value <= 1 for value in branch["x"] + branch["height"])
    assert len(s3.calls) == 2 and all(key.startswith(ROOT) for _, key in s3.calls)
    legacy.assert_not_called()


def test_control_changes_reuse_one_cached_summary_and_limit_before_ordering(source):
    s3, _, client, doc, _ = source
    original = deepcopy(doc)
    for ordering in ["hclust", "aoe", "fpc"]:
        for metric in ["floor", "overlap"]:
            for limit in [2, 3, 5, 30]:
                r = client.get(
                    "/api/teams/HOM/rotation",
                    params={"ordering": ordering, "order_metric": metric, "max_players": limit},
                )
                assert r.status_code == 200
                assert {p["player_id"] for p in r.json()["players"]} == set(
                    doc["player_order"][:limit]
                )
    assert len(s3.calls) == 2 and doc == original


def test_corrected_pointer_refreshes_without_requiring_latest_upstream_inputs(source, monkeypatch):
    s3, _, client, doc, _ = source
    before = client.get("/api/teams/HOM/rotation?ordering=fpc").json()
    corrected = deepcopy(doc)
    corrected["floor_correlation"][0][1] = corrected["floor_correlation"][1][0] = 0.9
    publish(s3, corrected)
    assert client.get("/api/teams/HOM/rotation?ordering=fpc").json() == before
    monkeypatch.setattr(api, "time", lambda: 4201)
    after = client.get("/api/teams/HOM/rotation?ordering=fpc").json()
    assert after["source"] != before["source"]
    assert after["players"] and all(key.startswith(ROOT) for _, key in s3.calls)


@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "pointer",
        "identity",
        "population",
        "coverage",
        "shape",
        "nan",
        "diagonal",
        "players",
        "similarity",
    ],
)
def test_invalid_summary_fails_explicitly_and_is_not_cached(source, damage):
    s3, _, client, doc, _ = source
    if damage == "identity":
        doc["team"] = "AWY"
    elif damage == "population":
        doc["calculation"]["population"] = "league_wide"
    elif damage == "coverage":
        doc["coverage"]["complete"] = False
    elif damage == "shape":
        doc["floor_correlation"][0].pop()
    elif damage == "nan":
        doc["floor_correlation"][0][1] = float("nan")
    elif damage == "diagonal":
        doc["floor_correlation"][0][0] = 0.9
    elif damage == "players":
        doc["player_order"][1] = doc["player_order"][0]
    elif damage == "similarity":
        doc["shared_court_similarity"][0][1] = doc["shared_court_similarity"][1][0] = 0.5
    key, pointer = publish(s3, doc)
    if damage == "checksum":
        s3.objects[key] = (b"changed", {})
    elif damage == "pointer":
        pointer["uri"] = pointer["uri"].replace("team_id=1", "team_id=2")
        s3.put_fixture(ROOT + "latest.json", pointer)
    response = client.get("/api/teams/HOM/rotation")
    assert response.status_code == 503 and response.headers["cache-control"] == "no-store"
    assert api.cached_summary.cache_info().currsize == 0


def test_missing_unknown_and_unpublished_history_do_not_fall_back_to_legacy(source):
    s3, _, client, _, legacy = source
    assert client.get("/api/teams/ZZZ/rotation").status_code == 404
    assert client.get("/api/teams/HOM/rotation?season=2025-26").status_code == 404
    assert s3.calls == []
    del s3.objects[ROOT + "latest.json"]
    response = client.get("/api/teams/HOM/rotation")
    assert response.status_code == 404 and response.headers["cache-control"] == "no-store"
    legacy.assert_not_called()


@pytest.mark.parametrize(
    "params",
    [{"max_players": 1}, {"max_players": 31}, {"ordering": "other"}, {"order_metric": "ratings"}],
)
def test_invalid_controls_are_rejected_before_s3_reads(source, params):
    s3, _, client, _, legacy = source
    assert client.get("/api/teams/HOM/rotation", params=params).status_code == 422
    assert s3.calls == []
    legacy.assert_not_called()


def test_empty_accepted_summary_is_explicitly_unavailable(source):
    s3, _, client, doc, _ = source
    doc.update(
        player_order=[],
        players=[],
        shared_possessions=[],
        shared_court_similarity=[],
        floor_correlation=[],
        last_team_game_date_pacific=None,
    )
    doc["coverage"].update(game_ids=[], game_count=0, stint_count=0)
    doc["inputs"]["box_scores"] = {}
    publish(s3, doc)
    response = client.get("/api/teams/HOM/rotation")
    assert response.status_code == 404 and "Not enough" in response.json()["detail"]


def test_historical_retired_team_uses_query_adapter_without_current_identity(source, monkeypatch):
    s3, _, client, doc, legacy = source
    doc.update(
        team="SEA",
        season="2007-08",
        game_date_pacific="2008-04-16",
        last_team_game_date_pacific="2008-04-16",
    )
    doc.pop("inputs")
    doc["coverage"]["scope"] = "saved_historical_regular_season_dataset"
    monkeypatch.setattr(
        api,
        "catalog",
        lambda: (({"SEA": {"seasons": ["2007-08"]}}, DAY, {}), "private, max-age=600"),
    )
    query = Mock(
        return_value=(
            doc,
            {
                "kind": "historical_parquet_query",
                "uri": "s3://nba-gestalt-data/lake/analytical/rapm_stints/2007-08/regular/part-00000.parquet",
                "sha256": "a" * 64,
            },
        )
    )
    monkeypatch.setattr(api, "historical_summary", query)
    for ordering in ["hclust", "aoe", "fpc"]:
        response = client.get(
            "/api/teams/SEA/rotation", params={"season": "2007-08", "ordering": ordering}
        )
        assert response.status_code == 200
        assert (
            response.json()["team"] == "SEA"
            and response.json()["source"]["kind"] == "historical_parquet_query"
        )
    query.assert_called_once()
    assert s3.calls == []
    legacy.assert_not_called()


def test_published_summary_still_requires_its_pinned_box_coverage(source):
    s3, _, client, doc, _ = source
    doc["inputs"]["box_scores"].pop("0022600001")
    publish(s3, doc)
    assert client.get("/api/teams/HOM/rotation").status_code == 503
