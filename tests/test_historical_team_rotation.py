"""Historical rotations query only the selected club and cache compact results."""

import base64
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import Mock

import duckdb
import numpy as np
import pytest

from nba_lineup_model.web_api import historical_team_rotation as api
from nba_lineup_model.web_api.team_rotation import validate_summary


def seed(connection, path, rows):
    connection.execute("DROP TABLE IF EXISTS s")
    connection.execute(
        """CREATE TABLE s (
            season VARCHAR, season_type VARCHAR, game_id VARCHAR, game_time_utc TIMESTAMPTZ,
            home_team_id BIGINT, away_team_id BIGINT, home_team_tricode VARCHAR,
            away_team_tricode VARCHAR, home_player_ids BIGINT[], away_player_ids BIGINT[],
            possessions DOUBLE, duration_seconds DOUBLE, source_build_run_id VARCHAR,
            quality_issue_codes_json VARCHAR)"""
    )
    connection.executemany("INSERT INTO s VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    connection.execute("COPY s TO ? (FORMAT PARQUET)", [str(path)])


class Connection:
    def __init__(self):
        self.raw = duckdb.connect(
            config={"enable_external_file_cache": "false", "threads": 1, "temp_directory": ""}
        )
        self.calls = []

    def execute(self, query, params):
        self.calls.append((query, params))
        return self.raw.execute(query, params)


@pytest.fixture
def source(tmp_path, monkeypatch):
    connection = Connection()
    lineups = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 5, 6], [1, 2, 4, 5, 6]]
    rows = []
    for i, lineup in enumerate(lineups):
        rows.append(
            [
                "2025-26",
                "regular",
                "0022500001" if i < 2 else "0022500002",
                datetime(2026, 4, 10 + i // 2, 1, tzinfo=UTC),
                1,
                2,
                "SEA",
                "AWY",
                lineup,
                [7, 8, 9, 10, 11],
                [2.5, 3, 4.25, 1.5][i],
                100 + i * 20,
                "official_nba",
                "[]",
            ]
        )
    # A traded player appears for another club. This must not enter SEA's vectors.
    rows.append(
        [
            "2025-26",
            "regular",
            "0022500003",
            datetime(2026, 4, 12, 1, tzinfo=UTC),
            3,
            4,
            "OTH",
            "ALT",
            [1, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            99,
            100,
            "official_nba",
            "[]",
        ]
    )
    path = tmp_path / "stints.parquet"
    names_path = tmp_path / "names.parquet"
    seed(connection.raw, path, rows)
    connection.raw.execute(
        "CREATE TABLE names (season VARCHAR, player_id BIGINT, player_name VARCHAR)"
    )
    connection.raw.executemany(
        "INSERT INTO names VALUES (?, ?, ?)",
        [["2025-26", i, f"Player {i:02}"] for i in range(1, 21)],
    )
    connection.raw.execute("COPY names TO ? (FORMAT PARQUET)", [str(names_path)])
    descriptors = {
        "season": "2025-26",
        "stints": {"uri": str(path)},
        "names": {"uri": str(names_path)},
        "manifest": {"uri": "manifest"},
        "row_count": len(rows),
        "source_game_count": 3,
    }
    reader = Mock(connection=connection)
    monkeypatch.setattr(api, "_reader", lambda: reader)
    monkeypatch.setattr(
        api, "_sources", lambda season, period: json.dumps(descriptors, sort_keys=True)
    )
    api._query_team.cache_clear()
    api._dataset_stats.cache_clear()
    yield reader, descriptors, rows, path, names_path
    api._query_team.cache_clear()
    api._dataset_stats.cache_clear()
    connection.raw.close()


def test_team_only_math_names_and_retired_identity_match_saved_stints(source):
    reader, _, rows, _, _ = source
    doc, ref = api.historical_summary("SEA", "2025-26", 0)
    validate_summary(doc, "SEA")
    assert (
        doc["team_id"] == 1
        and doc["coverage"]["game_count"] == 2
        and doc["coverage"]["stint_count"] == 4
    )
    assert len(doc["players"]) == 6 and set(doc["player_order"]) == set(range(1, 7))
    assert doc["coverage"]["scope"] == "saved_historical_regular_season_dataset"
    assert ref["kind"] == "historical_parquet_query"
    ids = doc["player_order"]
    active = np.zeros((4, 6))
    weights = np.array([row[10] for row in rows[:4]])
    for i, row in enumerate(rows[:4]):
        for pid in row[8]:
            active[i, ids.index(pid)] = 1
    assert np.allclose(
        doc["floor_correlation"], np.corrcoef(active * weights[:, None], rowvar=False)
    )
    assert np.allclose(doc["shared_possessions"], active.T @ (active * weights[:, None]))
    assert doc["players"][ids.index(1)]["on_court_possessions"] == sum(weights)
    assert all(p["player_name"] == f"Player {p['player_id']:02}" for p in doc["players"])
    assert len(reader.connection.calls) == 2
    assert all("SELECT" in query for query, _ in reader.connection.calls)
    assert reader.check_sources.call_count == 2


def test_query_result_is_cached_even_when_metadata_period_changes(source):
    reader, _, _, _, _ = source
    first = api.historical_summary("SEA", "2025-26", 0)
    assert api.historical_summary("SEA", "2025-26", 1) == first
    assert len(reader.connection.calls) == 2
    api.historical_summary("AWY", "2025-26", 1)
    assert len(reader.connection.calls) == 3  # Dataset checks are cached separately.


def test_version_change_requeries_instead_of_reusing_old_result(source):
    reader, descriptors, _, _, _ = source
    api.historical_summary("SEA", "2025-26", 0)
    descriptors["stints"]["version_id"] = "new-version"
    api.historical_summary("SEA", "2025-26", 1)
    assert len(reader.connection.calls) == 4


def test_missing_team_is_explicit(source):
    with pytest.raises(Exception) as error:
        api.historical_summary("ZZZ", "2025-26", 0)
    assert getattr(error.value, "status_code", None) == 404


@pytest.mark.parametrize("damage", ["game_id", "season", "zero", "source", "timestamp", "lineup"])
def test_invalid_historical_inputs_are_not_cached(source, damage):
    reader, descriptors, rows, path, _ = source
    changed = deepcopy(rows)
    if damage == "game_id":
        changed[0][2] = "bad"
    elif damage == "season":
        changed[0][0] = "2024-25"
    elif damage == "zero":
        changed[0][10] = 0
    elif damage == "source":
        changed[0][12] = "espn"
    elif damage == "timestamp":
        changed[0][3] = None
    elif damage == "lineup":
        changed[0][8] = [1, 1, 3, 4, 5]
    path.unlink()
    seed(reader.connection.raw, path, changed)
    with pytest.raises(ValueError):
        api.historical_summary("SEA", "2025-26", 0)
    assert api._query_team.cache_info().currsize == 0


def test_missing_names_are_not_replaced_with_invented_labels(source):
    reader, _, _, _, names_path = source
    names_path.unlink()
    reader.connection.raw.execute("DELETE FROM names WHERE player_id=1")
    reader.connection.raw.execute("COPY names TO ? (FORMAT PARQUET)", [str(names_path)])
    with pytest.raises(ValueError, match="lack official names"):
        api.historical_summary("SEA", "2025-26", 0)


def test_ambiguous_names_fail_explicitly(source):
    reader, _, _, _, names_path = source
    names_path.unlink()
    reader.connection.raw.execute("INSERT INTO names VALUES ('2025-26',1,'Another name')")
    reader.connection.raw.execute("COPY names TO ? (FORMAT PARQUET)", [str(names_path)])
    with pytest.raises(ValueError, match="ambiguous"):
        api.historical_summary("SEA", "2025-26", 0)


def test_full_object_checksum_is_verified_without_downloading_parquet():
    digest = hashlib.sha256(b"parquet").hexdigest()
    client = Mock()
    client.head_object.return_value = {
        "ChecksumType": "FULL_OBJECT",
        "ChecksumSHA256": base64.b64encode(bytes.fromhex(digest)).decode(),
        "ContentLength": 7,
        "VersionId": "v1",
        "ETag": "etag",
    }
    ref = api._head(
        client,
        "s3://nba-gestalt-data/lake/analytical/rapm_stints/2025-26/regular/part-00000.parquet",
        digest,
        7,
    )
    assert ref["sha256"] == digest and ref["version_id"] == "v1"
    client.get_object.assert_not_called()
    client.head_object.return_value["ChecksumSHA256"] = base64.b64encode(
        b"wrong" * 6 + b"!!"
    ).decode()
    with pytest.raises(ValueError, match="checksum"):
        api._head(client, ref["uri"], digest, 7)
    client.get_object.assert_not_called()
