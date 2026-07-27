from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from nba_lineup_model.audit import sample_audit_manifest
from nba_lineup_model.season import (
    BuildLedger,
    CatalogGame,
    CuratedDatasetLayout,
    CuratedPartition,
    GameBuildRecord,
    GameCatalog,
    append_build_record,
    read_build_ledger,
    read_game_catalog,
    write_build_ledger,
    write_game_catalog,
)
from nba_lineup_model.season.catalog import read_catalog_source
from nba_lineup_model.season.storage import catalog_frame, catalog_from_frame


def catalog_game(
    *,
    game_id: str = "0022500001",
    game_date: date = date(2025, 10, 21),
) -> CatalogGame:
    return CatalogGame(
        game_id=game_id,
        season="2025-26",
        season_type="regular",
        game_date=game_date,
        game_time_utc=datetime(2025, 10, 22, 0, 0, tzinfo=UTC),
        game_status="final",
        home_team_id=1610612745,
        home_team_tricode="HOU",
        away_team_id=1610612760,
        away_team_tricode="OKC",
        period_count=6,
        is_overtime=True,
        source_status_code=3,
        source_status_text="Final",
        source_url="https://cdn.nba.com/example/schedule.json",
        source_fetched_at=datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
    )


def successful_build_record() -> GameBuildRecord:
    started_at = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
    return GameBuildRecord(
        run_id="2025-26-backfill",
        attempt_id="2025-26-backfill:0022500001:1",
        attempt_number=1,
        game_id="0022500001",
        season="2025-26",
        season_type="regular",
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=2.5),
        duration_seconds=2.5,
        status="succeeded",
        terminal_stage="complete",
        use_cache=True,
        code_version="abc123",
        play_by_play_sha256="a" * 64,
        boxscore_sha256="b" * 64,
        event_count=515,
        lineup_stint_count=42,
        possession_count=201,
        possession_segment_count=205,
        validation_issue_count=0,
        output_table_count=6,
    )


def test_catalog_round_trips_with_stable_parquet_types(tmp_path: Path):
    catalog = GameCatalog(games=[catalog_game()])
    path = write_game_catalog(catalog, tmp_path / "games.parquet")

    assert read_game_catalog(path) == catalog

    schema = pq.read_schema(path)
    assert pa.types.is_string(schema.field("game_id").type) or pa.types.is_large_string(
        schema.field("game_id").type
    )
    assert schema.field("home_team_id").type == pa.int64()
    assert schema.field("away_team_id").type == pa.int64()
    assert schema.field("period_count").type == pa.int64()
    assert schema.field("game_date").type == pa.date32()
    assert schema.field("source_fetched_at").type.tz == "UTC"


def test_catalog_csv_import_preserves_game_id_lexical_value(tmp_path: Path):
    source_path = tmp_path / "games.csv"
    catalog_frame([catalog_game()]).to_csv(source_path, index=False)

    imported = catalog_from_frame(read_catalog_source(source_path))

    assert imported.games[0].game_id == "0022500001"
    assert isinstance(imported.games[0].home_team_id, int)


def test_catalog_drives_audit_overtime_expectation():
    manifest = sample_audit_manifest(
        catalog_frame([catalog_game()]),
        games_per_stratum=1,
    )

    assert manifest.games[0].expected_overtime is True


def test_catalog_rejects_duplicate_games_and_inconsistent_overtime():
    game = catalog_game()
    with pytest.raises(ValidationError, match="duplicate game IDs"):
        GameCatalog(games=[game, game])

    invalid = game.model_dump()
    invalid["is_overtime"] = False
    with pytest.raises(ValidationError, match="Overtime flag"):
        CatalogGame.model_validate(invalid)


def test_catalog_rejects_dates_outside_season():
    with pytest.raises(ValidationError, match="does not fall within"):
        catalog_game(game_date=date(2024, 10, 21))


def test_build_ledger_round_trips_success_and_failure(tmp_path: Path):
    success = successful_build_record()
    failure_start = datetime(2026, 7, 26, 12, 1, tzinfo=UTC)
    failure = GameBuildRecord(
        run_id="2025-26-backfill",
        attempt_id="2025-26-backfill:0022500002:1",
        attempt_number=1,
        game_id="0022500002",
        season="2025-26",
        season_type="regular",
        started_at=failure_start,
        finished_at=failure_start + timedelta(seconds=1),
        duration_seconds=1.0,
        status="failed",
        terminal_stage="fetch",
        use_cache=True,
        error_type="NbaCdnError",
        error_message="NBA CDN request failed",
    )
    ledger = BuildLedger(records=[success, failure])
    path = write_build_ledger(ledger, tmp_path / "builds.parquet")

    assert read_build_ledger(path) == ledger
    schema = pq.read_schema(path)
    assert pa.types.is_string(schema.field("game_id").type) or pa.types.is_large_string(
        schema.field("game_id").type
    )
    assert schema.field("attempt_number").type == pa.int64()
    assert schema.field("duration_seconds").type == pa.float64()
    assert schema.field("started_at").type.tz == "UTC"


def test_successful_build_requires_hashes_and_counts():
    record = successful_build_record().model_dump()
    record["play_by_play_sha256"] = None

    with pytest.raises(ValidationError, match="source hashes"):
        GameBuildRecord.model_validate(record)


def test_build_ledger_append_is_atomic_and_rejects_duplicate_attempts(tmp_path: Path):
    path = tmp_path / "builds.parquet"
    record = successful_build_record()

    ledger = append_build_record(record, path)

    assert ledger.records == [record]
    assert read_build_ledger(path).records == [record]
    with pytest.raises(ValidationError, match="duplicate attempt IDs"):
        append_build_record(record, path)


def test_curated_layout_is_deterministic_and_rejects_invalid_partitions():
    layout = CuratedDatasetLayout(Path("warehouse"))
    partition = CuratedPartition(
        table="possession_segments",
        season="2025-26",
        season_type="playoffs",
    )

    assert layout.part_path(partition, 3) == Path(
        "warehouse/possession_segments/season=2025-26/"
        "season_type=playoffs/part-00003.parquet"
    )

    with pytest.raises(ValidationError):
        CuratedPartition(
            table="possession_segments",
            season="2025-26",
            season_type="../playoffs",
        )
    with pytest.raises(ValueError, match="non-negative"):
        layout.part_path(partition, -1)
