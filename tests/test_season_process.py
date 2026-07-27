from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nba_lineup_model.ingest.nba_cdn import (
    CachedResponse,
    NbaCdnEndpoint,
    RawJsonCache,
)
from nba_lineup_model.season.process import (
    process_catalog_game,
    processing_code_fingerprint,
    quality_record_for_outcome,
    sample_processing_games,
)
from nba_lineup_model.season.schema import CatalogGame
from nba_lineup_model.season.storage import (
    merge_quality_records,
    read_quality_report,
)

FIXTURES = Path(__file__).parent / "fixtures"


def processing_game(
    *,
    game_id: str = "0020000001",
    season_type: str = "regular",
    game_date: date = date(2020, 12, 22),
    is_overtime: bool = False,
    period_count: int | None = None,
) -> CatalogGame:
    return CatalogGame(
        game_id=game_id,
        season="2020-21",
        season_type=season_type,
        game_date=game_date,
        game_status="final",
        home_team_id=100,
        home_team_tricode="HOM",
        away_team_id=200,
        away_team_tricode="AWY",
        period_count=period_count if period_count is not None else 5 if is_overtime else 2,
        is_overtime=is_overtime,
        source_status_code=3,
        source_status_text="Final",
        source_url="https://stats.nba.com/stats/scheduleleaguev2",
        source_fetched_at=datetime(2026, 7, 27, tzinfo=UTC),
    )


def seed_processing_raw(raw_dir: Path) -> None:
    play_by_play = json.loads(
        (FIXTURES / "playbyplay_lineup_scenario.json").read_text()
    )
    boxscore = json.loads((FIXTURES / "boxscore_lineup_scenario.json").read_text())
    game = boxscore["game"]
    game.update(
        {
            "gameStatus": 3,
            "gameTimeUTC": "2020-12-22T00:00:00Z",
        }
    )
    game["homeTeam"].update(
        {
            "score": 2,
            "statistics": {
                "fieldGoalsAttempted": 0,
                "freeThrowsAttempted": 2,
                "reboundsOffensive": 0,
                "turnovers": 0,
            },
        }
    )
    game["awayTeam"].update(
        {
            "score": 2,
            "statistics": {
                "fieldGoalsAttempted": 1,
                "freeThrowsAttempted": 0,
                "reboundsOffensive": 0,
                "turnovers": 0,
            },
        }
    )
    cache = RawJsonCache(raw_dir)
    for endpoint, payload in (
        (NbaCdnEndpoint.PLAY_BY_PLAY, play_by_play),
        (NbaCdnEndpoint.BOXSCORE, boxscore),
    ):
        raw_body = json.dumps(payload, separators=(",", ":")).encode()
        cache.write(
            CachedResponse(
                endpoint=endpoint,
                game_id="0020000001",
                url=f"https://cdn.nba.com/{endpoint.value}/0020000001",
                payload=payload,
                raw_body=raw_body,
            )
        )


def test_process_catalog_game_builds_quality_and_resumes(tmp_path: Path):
    game = processing_game()
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    seed_processing_raw(raw_dir)

    built = process_catalog_game(
        game,
        run_id="build",
        code_version="code-v1",
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    assert built.record.status == "succeeded"
    assert built.record.output_table_count == 6
    assert built.quality is not None
    assert built.quality.status in {"pass", "warning"}
    quality = quality_record_for_outcome(built)
    assert quality is not None

    resumed = process_catalog_game(
        game,
        run_id="resume",
        code_version="code-v1",
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        prior_success=built.record,
        prior_quality=quality,
    )

    assert resumed.record.status == "skipped"
    assert resumed.record.skip_reason == "matching_source_code_quality_and_outputs"
    assert resumed.quality is None


def test_process_catalog_game_rebuilds_when_an_output_is_missing(tmp_path: Path):
    game = processing_game()
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    seed_processing_raw(raw_dir)
    built = process_catalog_game(
        game,
        run_id="build",
        code_version="code-v1",
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )
    quality = quality_record_for_outcome(built)
    assert quality is not None
    (processed_dir / "events" / f"{game.game_id}.parquet").unlink()

    rebuilt = process_catalog_game(
        game,
        run_id="rebuild",
        code_version="code-v1",
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        prior_success=built.record,
        prior_quality=quality,
    )

    assert rebuilt.record.status == "succeeded"
    assert (processed_dir / "events" / f"{game.game_id}.parquet").exists()


def test_process_catalog_game_records_missing_raw_preflight(tmp_path: Path):
    outcome = process_catalog_game(
        processing_game(),
        run_id="missing",
        code_version="code-v1",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )

    assert outcome.record.status == "failed"
    assert outcome.record.terminal_stage == "preflight"
    assert outcome.record.error_type == "NbaCdnError"
    assert outcome.quality is not None
    assert outcome.quality.status == "error"
    assert quality_record_for_outcome(outcome) is not None
    assert outcome.retryable is False


def test_all_star_processing_does_not_apply_standard_period_expectations(
    tmp_path: Path,
):
    game = processing_game(
        season_type="all_star",
        period_count=4,
    )
    seed_processing_raw(tmp_path / "raw")

    outcome = process_catalog_game(
        game,
        run_id="all-star",
        code_version="code-v1",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )

    assert outcome.record.status == "succeeded"
    assert outcome.quality is not None
    assert "catalog:period_count_mismatch" not in outcome.quality.issue_codes
    assert "audit:overtime_expectation_mismatch" not in outcome.quality.issue_codes


def test_quality_report_round_trips_and_preserves_game_id(tmp_path: Path):
    game = processing_game()
    seed_processing_raw(tmp_path / "raw")
    outcome = process_catalog_game(
        game,
        run_id="quality",
        code_version="code-v1",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )
    quality = quality_record_for_outcome(outcome)
    assert quality is not None
    games_path = tmp_path / "quality" / "games.parquet"
    summary_path = tmp_path / "quality" / "summary.parquet"

    report = merge_quality_records(
        [quality],
        games_path,
        summary_path=summary_path,
    )

    assert read_quality_report(games_path) == report
    assert summary_path.exists()
    schema = pq.read_schema(games_path)
    assert pa.types.is_string(schema.field("game_id").type) or pa.types.is_large_string(
        schema.field("game_id").type
    )


def test_processing_code_fingerprint_changes_with_source(tmp_path: Path):
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    first = processing_code_fingerprint(tmp_path)
    source.write_text("VALUE = 2\n")

    assert processing_code_fingerprint(tmp_path) != first


def test_processing_sample_covers_season_type_and_overtime_strata():
    games = [
        processing_game(game_id="0020000001"),
        processing_game(game_id="0020000002"),
        processing_game(game_id="0020000003", is_overtime=True),
        processing_game(game_id="0020000004", is_overtime=True),
        processing_game(game_id="0040000001", season_type="playoffs"),
        processing_game(game_id="0040000002", season_type="playoffs"),
        processing_game(
            game_id="0040000003",
            season_type="playoffs",
            is_overtime=True,
        ),
        processing_game(
            game_id="0040000004",
            season_type="playoffs",
            is_overtime=True,
        ),
    ]

    sampled = sample_processing_games(
        games,
        games_per_stratum=1,
        random_seed=7,
    )

    assert len(sampled) == 4
    assert {
        (game.season_type, game.is_overtime) for game in sampled
    } == {
        ("regular", False),
        ("regular", True),
        ("playoffs", False),
        ("playoffs", True),
    }
    assert sampled == sample_processing_games(
        games,
        games_per_stratum=1,
        random_seed=7,
    )
