from __future__ import annotations

import json
from datetime import UTC, date, datetime

from nba_lineup_model.audit.game_rotation_fetch import (
    ordered_candidate_seasons,
    unresolved_period_lineup_candidates,
)
from nba_lineup_model.audit.game_rotation_probe import (
    game_rotation_evidence,
    latest_failed_builds,
    select_probe_games,
)
from nba_lineup_model.audit.game_rotation_recovery import cached_rotation_candidates
from nba_lineup_model.ingest.nba_stats import (
    NbaStatsEndpoint,
    NbaStatsRawCache,
    StatsCachedResponse,
)
from nba_lineup_model.season.schema import CatalogGame, GameBuildRecord


def build_record(*, game_id: str, status: str, minute: int) -> GameBuildRecord:
    timestamp = datetime(2026, 8, 2, 15, minute, tzinfo=UTC)
    kwargs = {
        "run_id": "test",
        "attempt_id": f"attempt-{game_id}-{minute}",
        "attempt_number": 1,
        "game_id": game_id,
        "season": "1996-97",
        "season_type": "regular",
        "started_at": timestamp,
        "finished_at": timestamp,
        "duration_seconds": 0.0,
        "status": status,
        "use_cache": True,
    }
    if status == "failed":
        return GameBuildRecord(
            **kwargs,
            terminal_stage="reconstruct",
            error_type="ValueError",
            error_message="ambiguous lineup",
        )
    return GameBuildRecord(
        **kwargs,
        terminal_stage="complete",
        play_by_play_sha256="a" * 64,
        boxscore_sha256="b" * 64,
        event_count=1,
        lineup_stint_count=1,
        possession_count=1,
        possession_segment_count=1,
        validation_issue_count=0,
        output_table_count=6,
    )


def test_selects_only_latest_failed_builds_with_catalog_entries():
    failed = build_record(game_id="0029600001", status="failed", minute=1)
    succeeded = build_record(game_id="0029600001", status="succeeded", minute=2)
    remaining = build_record(game_id="0029600002", status="failed", minute=3)
    catalog_game = CatalogGame(
        game_id="0029600002",
        season="1996-97",
        season_type="regular",
        game_date=date(1996, 11, 1),
        game_status="final",
        home_team_id=1610612738,
        home_team_tricode="BOS",
        away_team_id=1610612741,
        away_team_tricode="CHI",
        source_url="https://stats.nba.com/stats/scheduleleaguev2",
        source_fetched_at=datetime(2026, 8, 2, tzinfo=UTC),
    )

    selected = select_probe_games(
        [catalog_game],
        latest_failed_builds([failed, succeeded, remaining]),
        per_season=3,
    )

    assert [(game.game_id, record.game_id) for game, record in selected] == [
        ("0029600002", "0029600002")
    ]


def test_validates_two_team_rotation_intervals():
    payload = {
        "resultSets": [
            {
                "name": "AwayTeam",
                "headers": ["PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"],
                "rowSet": [[1, 0.0, 120.0]],
            },
            {
                "name": "HomeTeam",
                "headers": ["PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"],
                "rowSet": [[2, 10.0, 130.0]],
            },
        ]
    }

    assert game_rotation_evidence(payload).available is True
    assert game_rotation_evidence({"resultSets": []}).reason == "missing_team_rotation"


def test_selects_only_failed_games_with_valid_cached_rotation(tmp_path):
    game = CatalogGame(
        game_id="0029600002",
        season="1996-97",
        season_type="regular",
        game_date=date(1996, 11, 1),
        game_status="final",
        home_team_id=1610612738,
        home_team_tricode="BOS",
        away_team_id=1610612741,
        away_team_tricode="CHI",
        source_url="https://stats.nba.com/stats/scheduleleaguev2",
        source_fetched_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    payload = {
        "parameters": {"GameID": game.game_id, "LeagueID": "00"},
        "resultSets": [
            {
                "name": "AwayTeam",
                "headers": ["PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"],
                "rowSet": [[1, 0.0, 120.0]],
            },
            {
                "name": "HomeTeam",
                "headers": ["PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"],
                "rowSet": [[2, 0.0, 120.0]],
            },
        ],
    }
    cache = NbaStatsRawCache(tmp_path)
    cache.write(
        StatsCachedResponse(
            endpoint=NbaStatsEndpoint.GAME_ROTATION,
            game_id=game.game_id,
            url="https://stats.nba.test/gamerotation",
            fetched_at=datetime(2026, 8, 2, tzinfo=UTC),
            payload=payload,
            raw_body=json.dumps(payload, separators=(",", ":")).encode(),
        )
    )

    candidates = cached_rotation_candidates(
        [game],
        latest_failed_builds([build_record(game_id=game.game_id, status="failed", minute=1)]),
        cache,
    )

    selected = [
        (candidate.game_id, evidence.available)
        for candidate, _record, evidence in candidates
    ]
    assert selected == [(game.game_id, True)]


def test_selects_only_unresolved_period_lineup_failures():
    game = CatalogGame(
        game_id="0029600002",
        season="1996-97",
        season_type="regular",
        game_date=date(1996, 11, 1),
        game_status="final",
        home_team_id=1610612738,
        home_team_tricode="BOS",
        away_team_id=1610612741,
        away_team_tricode="CHI",
        source_url="https://stats.nba.com/stats/scheduleleaguev2",
        source_fetched_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    ambiguous = build_record(game_id=game.game_id, status="failed", minute=1).model_copy(
        update={"error_message": "Period lineup remains ambiguous for team 1: 4 legal states"}
    )
    quality_failure = build_record(game_id="0029600003", status="failed", minute=2).model_copy(
        update={"error_message": "Quality gate failed: possession:score_conservation_failed"}
    )

    candidates = unresolved_period_lineup_candidates(
        [game],
        latest_failed_builds([ambiguous, quality_failure]),
    )

    assert [(selected.game_id, record.game_id) for selected, record in candidates] == [
        (game.game_id, game.game_id)
    ]


def test_orders_candidate_seasons_descending_with_a_latest_bound():
    record = build_record(game_id="0029600002", status="failed", minute=1)
    candidates = []
    for season in ("1996-97", "2017-18", "2018-19", "2020-21"):
        start_year = int(season[:4])
        candidates.append(
            (
                CatalogGame(
                    game_id=record.game_id,
                    season=season,
                    season_type="regular",
                    game_date=date(start_year, 11, 1),
                    game_status="final",
                    home_team_id=1610612738,
                    home_team_tricode="BOS",
                    away_team_id=1610612741,
                    away_team_tricode="CHI",
                    source_url="https://stats.nba.com/stats/scheduleleaguev2",
                    source_fetched_at=datetime(2026, 8, 2, tzinfo=UTC),
                ),
                record.model_copy(update={"season": season}),
            )
        )

    assert ordered_candidate_seasons(
        candidates,
        max_season="2018-19",
        reverse=True,
    ) == ["2018-19", "2017-18", "1996-97"]
