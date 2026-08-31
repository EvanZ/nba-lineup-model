from __future__ import annotations

import json
from pathlib import Path

import httpx

from nba_lineup_model.players.rosters import collect_team_rosters, team_roster_frame
from nba_lineup_model.players.source import (
    PlayerStatsCache,
    PlayerStatsClient,
    PlayerStatsEndpoint,
)


def test_team_roster_frame_preserves_player_identifiers_and_measurements() -> None:
    frame = team_roster_frame(_payload(), season="2026-27", team_id=1610612738)

    assert frame["player_id"].dtype.name == "string"
    assert frame.loc[0, "player_id"] == "1649999"
    assert frame.loc[0, "team_abbreviation"] == "BOS"
    assert frame.loc[0, "height_inches"] == 80
    assert frame.loc[0, "weight_pounds"] == 220


def test_team_roster_collection_caches_by_team_and_writes_combined_table(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=json.dumps(_payload()).encode(), request=request)

    cache = PlayerStatsCache(tmp_path / "raw")
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = PlayerStatsClient(cache=cache, http_client=http_client)
    try:
        output = collect_team_rosters(
            "2026-27",
            raw_dir=tmp_path / "raw",
            curated_dir=tmp_path / "curated",
            team_ids=(1610612738,),
            request_delay_seconds=0,
            client=client,
        )
    finally:
        client.close()
        http_client.close()

    assert dict(requests[0].url.params) == {
        "LeagueID": "00",
        "Season": "2026-27",
        "TeamID": "1610612738",
    }
    assert output.is_file()
    cached = client.fetch_team_roster("2026-27", 1610612738)
    assert cache.path_for(cached.endpoint, "2026-27", None, 1610612738).is_file()


def test_refresh_archives_previous_roster_snapshot(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        payload = _payload()
        payload["resultSets"][0]["rowSet"][0][3] = f"Player {requests}"
        return httpx.Response(200, content=json.dumps(payload).encode(), request=request)

    cache = PlayerStatsCache(tmp_path / "raw")
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = PlayerStatsClient(cache=cache, http_client=http_client)
    try:
        collect_team_rosters(
            "2026-27",
            raw_dir=tmp_path / "raw",
            curated_dir=tmp_path / "curated",
            team_ids=(1610612738,),
            request_delay_seconds=0,
            client=client,
        )
        output = collect_team_rosters(
            "2026-27",
            raw_dir=tmp_path / "raw",
            curated_dir=tmp_path / "curated",
            team_ids=(1610612738,),
            request_delay_seconds=0,
            refresh=True,
            client=client,
        )
    finally:
        client.close()
        http_client.close()

    snapshot_files = list(
        (tmp_path / "raw" / "commonteamroster" / "2026-27" / "snapshots").glob("*/1610612738.json")
    )
    assert len(snapshot_files) == 1
    assert b"Player 1" in snapshot_files[0].read_bytes()
    assert (
        b"Player 2"
        in cache.path_for(PlayerStatsEndpoint.TEAM_ROSTER, "2026-27", None, 1610612738).read_bytes()
    )
    manifest = json.loads((output.parent / "_manifest.json").read_text())
    assert manifest["previous_snapshot"] == str(snapshot_files[0].parent)


def _payload() -> dict[str, object]:
    return {
        "resultSets": [
            {
                "name": "CommonTeamRoster",
                "headers": [
                    "TeamID",
                    "SEASON",
                    "LeagueID",
                    "PLAYER",
                    "NICKNAME",
                    "PLAYER_SLUG",
                    "NUM",
                    "POSITION",
                    "HEIGHT",
                    "WEIGHT",
                    "BIRTH_DATE",
                    "AGE",
                    "EXP",
                    "SCHOOL",
                    "PLAYER_ID",
                    "HOW_ACQUIRED",
                    "SUPPLEMENTAL_STATUS",
                ],
                "rowSet": [
                    [
                        1610612738,
                        "2026",
                        "00",
                        "First Pick",
                        "First",
                        "first-pick",
                        "1",
                        "F",
                        "6-8",
                        "220",
                        "JAN 01, 2006",
                        20.0,
                        "R",
                        "School One",
                        "1649999",
                        "#1 Pick in 2026 Draft",
                        0,
                    ]
                ],
            }
        ]
    }
