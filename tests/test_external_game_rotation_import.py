from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nba_lineup_model.audit.external_game_rotation_import import (
    ExternalRotationImport,
    external_rotation_payload,
)
from nba_lineup_model.ingest.nba_stats import (
    NbaStatsEndpoint,
    NbaStatsRawCache,
    StatsCachedResponse,
)
from nba_lineup_model.season.schema import CatalogGame


def test_external_rotation_payload_has_compatible_team_intervals() -> None:
    game = CatalogGame(
        game_id="0020000001",
        season="2000-01",
        season_type="regular",
        game_date=datetime(2000, 10, 31, tzinfo=UTC).date(),
        game_time_utc=None,
        game_status="final",
        home_team_id=2,
        home_team_tricode="HOM",
        away_team_id=1,
        away_team_tricode="AWY",
        period_count=4,
        is_overtime=False,
        source_status_code=3,
        source_status_text="Final",
        source_url="https://example.test/game",
        source_fetched_at=datetime(2000, 10, 31, tzinfo=UTC),
    )
    intervals = pd.DataFrame(
        [
            {
                "game_id": game.game_id,
                "team_id": team,
                "player_id": player,
                "in_time_real": 0,
                "out_time_real": 28800,
            }
            for team, players in ((1, range(11, 16)), (2, range(21, 26)))
            for player in players
        ]
    )
    payload = external_rotation_payload(
        ExternalRotationImport(
            game=game,
            intervals=intervals,
            discarded_nonpositive_intervals=0,
            exact_period_starts={"period_2": True, "period_3": True, "period_4": True},
        )
    )

    assert payload["parameters"] == {"GameID": "0020000001", "LeagueID": "00"}
    assert [result["name"] for result in payload["resultSets"]] == ["AwayTeam", "HomeTeam"]
    assert payload["resultSets"][0]["rowSet"][0] == [1, 11, 0, 28800]


def test_imported_payload_is_readable_from_stats_cache(tmp_path: Path) -> None:
    game = CatalogGame(
        game_id="0020000001",
        season="2000-01",
        season_type="regular",
        game_date=datetime(2000, 10, 31, tzinfo=UTC).date(),
        game_time_utc=None,
        game_status="final",
        home_team_id=2,
        home_team_tricode="HOM",
        away_team_id=1,
        away_team_tricode="AWY",
        period_count=4,
        is_overtime=False,
        source_status_code=3,
        source_status_text="Final",
        source_url="https://example.test/game",
        source_fetched_at=datetime(2000, 10, 31, tzinfo=UTC),
    )
    intervals = pd.DataFrame(
        [
            {
                "game_id": game.game_id,
                "team_id": team,
                "player_id": player,
                "in_time_real": 0,
                "out_time_real": 28800,
            }
            for team, players in ((1, range(11, 16)), (2, range(21, 26)))
            for player in players
        ]
    )
    imported = ExternalRotationImport(
        game,
        intervals,
        0,
        {"period_2": True, "period_3": True, "period_4": True},
    )
    payload = external_rotation_payload(imported)

    cache = NbaStatsRawCache(tmp_path)
    cache.write(
        StatsCachedResponse(
            endpoint=NbaStatsEndpoint.GAME_ROTATION,
            game_id=game.game_id,
            url="file:///tmp/external.csv",
            fetched_at=datetime.now(UTC),
            payload=payload,
            raw_body=json.dumps(payload).encode(),
        )
    )

    assert cache.read(NbaStatsEndpoint.GAME_ROTATION, game.game_id) is not None
