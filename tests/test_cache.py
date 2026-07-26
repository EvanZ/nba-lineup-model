from __future__ import annotations

import json

from nba_lineup_model.ingest.nba_cdn import CachedResponse, NbaCdnEndpoint, RawJsonCache


def test_raw_json_cache_round_trip(tmp_path):
    cache = RawJsonCache(tmp_path)
    raw_body = (
        b'{ "game": { "gameId": "0022000180", '
        b'"teamId": 1610612738, "score": "03", "actions": [] } }\n'
    )
    response = CachedResponse(
        endpoint=NbaCdnEndpoint.PLAY_BY_PLAY,
        game_id="0022000180",
        url="https://example.test/playbyplay_0022000180.json",
        payload=json.loads(raw_body),
        raw_body=raw_body,
    )

    path = cache.write(response)
    restored = cache.read(NbaCdnEndpoint.PLAY_BY_PLAY, "0022000180")

    assert path.exists()
    assert path.read_bytes() == raw_body
    assert cache.metadata_path_for(
        NbaCdnEndpoint.PLAY_BY_PLAY,
        "0022000180",
    ).exists()
    assert restored is not None
    assert restored.endpoint == NbaCdnEndpoint.PLAY_BY_PLAY
    assert restored.payload["game"]["gameId"] == "0022000180"
    assert restored.payload["game"]["teamId"] == 1610612738
    assert restored.payload["game"]["score"] == "03"
    assert restored.raw_body == raw_body
