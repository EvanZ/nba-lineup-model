from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from nba_lineup_model.season.schedule import (
    NbaScheduleClient,
    NbaScheduleError,
    ScheduleResponse,
    SeasonScheduleCache,
    catalog_from_schedule,
    replace_catalog_season,
)
from nba_lineup_model.season.schema import CatalogGame, GameCatalog
from nba_lineup_model.season.storage import read_game_catalog

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scheduleleaguev2_minimal.json"
SOURCE_URL = "https://stats.nba.com/stats/scheduleleaguev2?LeagueID=00&Season=2025-26"
FETCHED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def schedule_response() -> ScheduleResponse:
    raw_body = FIXTURE_PATH.read_bytes()
    return ScheduleResponse(
        season="2025-26",
        url=SOURCE_URL,
        fetched_at=FETCHED_AT,
        payload=json.loads(raw_body),
        raw_body=raw_body,
    )


def test_schedule_cache_preserves_source_bytes_and_provenance(tmp_path: Path):
    cache = SeasonScheduleCache(tmp_path)
    response = schedule_response()

    path = cache.write(response)
    restored = cache.read("2025-26")

    assert path.read_bytes() == FIXTURE_PATH.read_bytes()
    assert restored == response
    assert restored is not None
    assert restored.raw_body == response.raw_body
    assert restored.url == SOURCE_URL


def test_schedule_cache_rejects_hash_mismatch(tmp_path: Path):
    cache = SeasonScheduleCache(tmp_path)
    path = cache.write(schedule_response())
    path.write_bytes(b"{}")

    with pytest.raises(NbaScheduleError, match="hash mismatch"):
        cache.read("2025-26")


def test_schedule_client_uses_season_parameter_and_populates_cache(tmp_path: Path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=FIXTURE_PATH.read_bytes(), request=request)

    cache = SeasonScheduleCache(tmp_path)
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NbaScheduleClient(cache=cache, http_client=http_client)

    response = client.fetch("2025-26", use_cache=False)
    cached = client.fetch("2025-26")

    assert len(requests) == 1
    assert requests[0].url.params["LeagueID"] == "00"
    assert requests[0].url.params["Season"] == "2025-26"
    assert response.raw_body == FIXTURE_PATH.read_bytes()
    assert cached.raw_body == response.raw_body


def test_schedule_normalizes_game_identity_status_and_overtime():
    catalog = catalog_from_schedule(schedule_response())
    games = {game.game_id: game for game in catalog.games}

    regulation = games["0022500001"]
    assert regulation.game_id == "0022500001"
    assert regulation.season_type == "regular"
    assert regulation.game_date == date(2025, 10, 21)
    assert regulation.game_time_utc == datetime(2025, 10, 22, tzinfo=UTC)
    assert regulation.home_team_id == 1610612745
    assert isinstance(regulation.home_team_id, int)
    assert regulation.period_count == 4
    assert regulation.is_overtime is False

    double_overtime = games["0022500002"]
    assert double_overtime.period_count == 6
    assert double_overtime.is_overtime is True
    assert games["0052500101"].season_type == "play_in"
    assert games["0042500401"].season_type == "playoffs"


def test_schedule_rejects_source_season_mismatch():
    response = schedule_response()
    payload = response.model_copy(deep=True).payload
    payload["leagueSchedule"]["seasonYear"] = "2024-25"

    with pytest.raises(NbaScheduleError, match="does not match requested season"):
        catalog_from_schedule(response.model_copy(update={"payload": payload}))


def test_schedule_rejects_nested_contract_drift():
    response = schedule_response()
    payload = response.model_copy(deep=True).payload
    del payload["leagueSchedule"]["gameDates"][0]["games"][0]["homeTeam"]

    with pytest.raises(NbaScheduleError, match="missing required fields"):
        catalog_from_schedule(response.model_copy(update={"payload": payload}))


def test_schedule_skips_unidentifiable_historical_preseason_placeholder():
    response = schedule_response()
    payload = response.model_copy(deep=True).payload
    placeholder = payload["leagueSchedule"]["gameDates"][0]["games"][0].copy()
    placeholder.update(
        {
            "gameId": "0012500001",
            "gameStatus": 0,
            "gameStatusText": "",
            "homeTeam": {"teamId": 0, "teamTricode": None},
            "awayTeam": {"teamId": 0, "teamTricode": None},
        }
    )
    payload["leagueSchedule"]["gameDates"][0]["games"].insert(0, placeholder)

    catalog = catalog_from_schedule(response.model_copy(update={"payload": payload}))

    assert {game.game_id for game in catalog.games} == {
        "0022500001",
        "0022500002",
        "0052500101",
        "0042500401",
    }


def test_schedule_skips_unidentifiable_historical_all_star_placeholder():
    response = schedule_response()
    payload = response.model_copy(deep=True).payload
    placeholder = payload["leagueSchedule"]["gameDates"][0]["games"][0].copy()
    placeholder.update(
        {
            "gameId": "0032500011",
            "gameLabel": "All-Star Game",
            "gameStatus": 0,
            "gameStatusText": "",
            "homeTeam": {"teamId": 0, "teamTricode": None},
            "awayTeam": {"teamId": 0, "teamTricode": None},
        }
    )
    payload["leagueSchedule"]["gameDates"][0]["games"].insert(0, placeholder)

    catalog = catalog_from_schedule(response.model_copy(update={"payload": payload}))

    assert "0032500011" not in {game.game_id for game in catalog.games}


def test_schedule_skips_unidentifiable_legacy_non_game_placeholder():
    response = schedule_response()
    payload = response.model_copy(deep=True).payload
    placeholder = payload["leagueSchedule"]["gameDates"][0]["games"][0].copy()
    placeholder.update(
        {
            "gameId": "0092500001",
            "gameStatus": 0,
            "gameStatusText": "",
            "homeTeam": {"teamId": 0, "teamTricode": None},
            "awayTeam": {"teamId": 0, "teamTricode": None},
        }
    )
    payload["leagueSchedule"]["gameDates"][0]["games"].insert(0, placeholder)

    catalog = catalog_from_schedule(response.model_copy(update={"payload": payload}))

    assert "0092500001" not in {game.game_id for game in catalog.games}


def test_schedule_rejects_invalid_regular_season_team_identity():
    response = schedule_response()
    payload = response.model_copy(deep=True).payload
    payload["leagueSchedule"]["gameDates"][0]["games"][0]["homeTeam"] = {
        "teamId": 0,
        "teamTricode": None,
    }

    with pytest.raises(NbaScheduleError, match="invalid identity"):
        catalog_from_schedule(response.model_copy(update={"payload": payload}))


def test_schedule_rejects_unidentifiable_final_playoff_game():
    response = schedule_response()
    payload = response.model_copy(deep=True).payload
    game = payload["leagueSchedule"]["gameDates"][1]["games"][0]
    game["homeTeam"] = {"teamId": 0, "teamTricode": None}

    with pytest.raises(NbaScheduleError, match="invalid identity"):
        catalog_from_schedule(response.model_copy(update={"payload": payload}))


def test_nba_cup_knockout_games_follow_source_game_type():
    response = schedule_response()
    payload = response.model_copy(deep=True).payload
    game = payload["leagueSchedule"]["gameDates"][0]["games"][0]
    game["gameLabel"] = "Emirates NBA Cup"
    game["gameSubLabel"] = "Quarterfinals"

    quarterfinal = catalog_from_schedule(response.model_copy(update={"payload": payload})).games[0]
    game["gameId"] = "0062500001"
    game["gameSubLabel"] = "Championship"
    championship = catalog_from_schedule(response.model_copy(update={"payload": payload})).games[0]

    assert quarterfinal.season_type == "regular"
    assert championship.season_type == "nba_cup_final"


def test_schedule_preserves_source_status_text_lexically():
    response = schedule_response()
    payload = response.model_copy(deep=True).payload
    payload["leagueSchedule"]["gameDates"][0]["games"][0]["gameStatusText"] = "Final               "

    game = catalog_from_schedule(response.model_copy(update={"payload": payload})).games[0]

    assert game.game_status == "final"
    assert game.period_count == 4
    assert game.source_status_text == "Final               "


def test_replace_catalog_season_preserves_other_seasons():
    discovered = catalog_from_schedule(schedule_response())
    prior_game = CatalogGame(
        game_id="0022400001",
        season="2024-25",
        season_type="regular",
        game_date=date(2024, 10, 22),
        game_status="final",
        home_team_id=1610612745,
        home_team_tricode="HOU",
        away_team_id=1610612760,
        away_team_tricode="OKC",
        period_count=4,
        is_overtime=False,
        source_status_code=3,
        source_status_text="Final",
        source_url=SOURCE_URL,
        source_fetched_at=FETCHED_AT,
    )
    stale_current_game = discovered.games[0].model_copy(
        update={"source_fetched_at": datetime(2026, 7, 26, tzinfo=UTC)}
    )
    existing = GameCatalog(games=[prior_game, stale_current_game])

    merged = replace_catalog_season(existing, discovered)

    assert {game.season for game in merged.games} == {"2024-25", "2025-26"}
    assert len(merged.games) == len(discovered.games) + 1
    assert (
        next(game for game in merged.games if game.game_id == "0022500001") == (discovered.games[0])
    )


def test_discovery_cli_builds_catalog_from_cached_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    from nba_lineup_model.season.discover import main

    raw_dir = tmp_path / "raw"
    output_path = tmp_path / "catalog" / "games.parquet"
    SeasonScheduleCache(raw_dir).write(schedule_response())
    monkeypatch.setattr(
        "sys.argv",
        [
            "nba-discover-season",
            "2025-26",
            "--raw-dir",
            str(raw_dir),
            "--output",
            str(output_path),
        ],
    )

    main()

    catalog = read_game_catalog(output_path)
    assert len(catalog.games) == 4
    assert "Discovered 4 games for 2025-26" in capsys.readouterr().out
