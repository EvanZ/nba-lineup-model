"""Tests for raw-box-score rotation minute labels."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from nba_lineup_model.rotation import l1_minute_share_persistence as minute_source


def test_regular_game_minutes_uses_raw_boxscore_not_quality_filtered_curated_rows(
    monkeypatch,
    tmp_path,
) -> None:
    game = SimpleNamespace(
        game_id="0022500001",
        game_time_utc=datetime(2025, 10, 1, tzinfo=UTC),
        game_date=datetime(2025, 10, 1, tzinfo=UTC).date(),
    )
    monkeypatch.setattr(minute_source, "read_game_catalog", lambda _path: object())
    monkeypatch.setattr(
        minute_source,
        "select_catalog_games",
        lambda *_args, **_kwargs: [game],
    )
    monkeypatch.setattr(
        minute_source,
        "_read_cached_boxscore",
        lambda *_args, **_kwargs: {
            "game": {
                "homeTeam": {
                    "teamId": 1,
                    "teamTricode": "HME",
                    "players": [
                        {
                            "personId": 10,
                            "nameI": "A. Home",
                            "statistics": {"minutes": "PT24M00.00S"},
                        },
                        {
                            "personId": 11,
                            "nameI": "B. Home",
                            "statistics": {"minutes": ""},
                        },
                    ],
                },
                "awayTeam": {
                    "teamId": 2,
                    "teamTricode": "AWY",
                    "players": [
                        {
                            "personId": 20,
                            "nameI": "A. Away",
                            "statistics": {"minutes": "PT24M00.00S"},
                        },
                    ],
                },
            }
        },
    )

    result = minute_source.read_regular_game_minutes(
        "2025-26",
        curated_dir=tmp_path / "intentionally-absent-curated-players",
    )

    assert result.loc[result["player_id"].eq(10), "minutes"].item() == 24.0
    assert result.loc[result["player_id"].eq(11), "minutes"].item() == 0.0
    assert result.loc[result["player_id"].eq(11), "minute_share"].item() == 0.0
    assert set(result["game_id"]) == {"0022500001"}


def test_regular_game_minutes_requires_a_raw_boxscore_for_every_catalog_game(
    monkeypatch,
) -> None:
    game = SimpleNamespace(
        game_id="0022500001",
        game_time_utc=datetime(2025, 10, 1, tzinfo=UTC),
        game_date=datetime(2025, 10, 1, tzinfo=UTC).date(),
    )
    monkeypatch.setattr(minute_source, "read_game_catalog", lambda _path: object())
    monkeypatch.setattr(
        minute_source,
        "select_catalog_games",
        lambda *_args, **_kwargs: [game],
    )
    monkeypatch.setattr(minute_source, "_read_cached_boxscore", lambda *_args, **_kwargs: None)

    with pytest.raises(FileNotFoundError, match="1 catalog games"):
        minute_source.read_regular_game_minutes("2025-26")
