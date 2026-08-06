from __future__ import annotations

import json
from pathlib import Path

import httpx

from nba_lineup_model.players.draft_history import collect_draft_history, draft_history_frame
from nba_lineup_model.players.source import PlayerStatsCache, PlayerStatsClient


def test_draft_history_preserves_player_identifiers_and_slots() -> None:
    frame = draft_history_frame(_payload(), season="2026-27")

    assert frame["player_id"].dtype.name == "string"
    assert frame["player_id"].tolist() == ["1649999", "1650000"]
    assert frame["draft_number"].tolist() == [1, 2]


def test_draft_history_collects_raw_response_and_normalized_table(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=json.dumps(_payload()).encode(), request=request)

    cache = PlayerStatsCache(tmp_path / "raw")
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = PlayerStatsClient(cache=cache, http_client=http_client)
    try:
        output = collect_draft_history(
            "2026-27",
            raw_dir=tmp_path / "raw",
            curated_dir=tmp_path / "curated",
            client=client,
        )
    finally:
        client.close()
        http_client.close()

    assert dict(requests[0].url.params)["Season"] == "2026"
    assert cache.path_for(client.fetch_draft_history("2026-27").endpoint, "2026-27").is_file()
    assert output.is_file()


def _payload() -> dict[str, object]:
    return {
        "resultSets": [
            {
                "name": "DraftHistory",
                "headers": [
                    "PERSON_ID",
                    "PLAYER_NAME",
                    "ROUND_NUMBER",
                    "ROUND_PICK",
                    "OVERALL_PICK",
                    "TEAM_ID",
                    "TEAM_ABBREVIATION",
                    "ORGANIZATION",
                ],
                "rowSet": [
                    ["1649999", "First Pick", 1, 1, 1, "1610612738", "BOS", "School One"],
                    ["1650000", "Second Pick", 1, 2, 2, "1610612744", "GSW", "School Two"],
                ],
            }
        ]
    }
