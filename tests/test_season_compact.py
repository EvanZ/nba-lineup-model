from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from nba_lineup_model.season.compact import (
    CURATED_METADATA_COLUMNS,
    CuratedGameSource,
    compact_curated_partition,
    curation_code_fingerprint,
    read_curated_partition_manifest,
    validate_curated_partition,
)
from nba_lineup_model.season.layout import CuratedPartition
from nba_lineup_model.season.schema import (
    CatalogGame,
    GameBuildRecord,
    GameQualityRecord,
)


def curated_game(
    game_id: str,
    *,
    game_date: date,
    game_time_utc: datetime | None,
) -> CatalogGame:
    return CatalogGame(
        game_id=game_id,
        season="2020-21",
        season_type="regular",
        game_date=game_date,
        game_time_utc=game_time_utc,
        game_status="final",
        home_team_id=100,
        home_team_tricode="HOM",
        away_team_id=200,
        away_team_tricode="AWY",
        period_count=4,
        is_overtime=False,
        source_status_code=3,
        source_status_text="Final",
        source_url="https://cdn.nba.com/example/schedule.json",
        source_fetched_at=datetime(2026, 7, 27, tzinfo=UTC),
    )


def curated_source(
    game_id: str,
    *,
    game_date: date,
    game_time_utc: datetime | None,
    quality_status: str = "pass",
) -> CuratedGameSource:
    game = curated_game(
        game_id,
        game_date=game_date,
        game_time_utc=game_time_utc,
    )
    started_at = datetime(2026, 7, 27, 1, 0, tzinfo=UTC)
    build = GameBuildRecord(
        run_id="process-run",
        attempt_id=f"process-run:{game_id}:1",
        attempt_number=1,
        game_id=game_id,
        season=game.season,
        season_type=game.season_type,
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=1),
        duration_seconds=1.0,
        status="succeeded",
        terminal_stage="complete",
        use_cache=True,
        code_version="sha256:processing",
        play_by_play_sha256="a" * 64,
        boxscore_sha256="b" * 64,
        event_count=2,
        lineup_stint_count=1,
        possession_count=1,
        possession_segment_count=1,
        validation_issue_count=1 if quality_status == "warning" else 0,
        output_table_count=6,
    )
    quality = GameQualityRecord(
        game_id=game_id,
        season=game.season,
        season_type=game.season_type,
        sample_group="season_process",
        status=quality_status,
        issue_codes=("audit:warning",) if quality_status == "warning" else (),
        run_id=build.run_id,
        attempt_number=1,
        code_version=build.code_version,
        play_by_play_sha256=build.play_by_play_sha256,
        boxscore_sha256=build.boxscore_sha256,
        recorded_at=build.finished_at,
    )
    return CuratedGameSource(game=game, build=build, quality=quality)


def seed_event_sources(
    processed_dir: Path,
    sources: list[CuratedGameSource],
) -> None:
    output_dir = processed_dir / "events"
    output_dir.mkdir(parents=True)
    for number, source in enumerate(sources, start=1):
        pd.DataFrame(
            {
                "game_id": pd.Series(
                    [source.game.game_id] * number,
                    dtype="string",
                ),
                "event_index": pd.Series(range(number), dtype="Int64"),
            }
        ).to_parquet(output_dir / f"{source.game.game_id}.parquet", index=False)


def test_compacts_partition_with_lossless_dataset_read_and_provenance(
    tmp_path: Path,
):
    sources = [
        curated_source(
            "0020000001",
            game_date=date(2020, 12, 22),
            game_time_utc=datetime(2020, 12, 23, tzinfo=UTC),
        ),
        curated_source(
            "0020000002",
            game_date=date(2020, 12, 23),
            game_time_utc=datetime(2020, 12, 24, tzinfo=UTC),
            quality_status="warning",
        ),
        curated_source(
            "0020000003",
            game_date=date(2020, 12, 24),
            game_time_utc=None,
        ),
    ]
    processed_dir = tmp_path / "processed"
    curated_dir = tmp_path / "curated"
    seed_event_sources(processed_dir, sources)
    partition = CuratedPartition(
        table="events",
        season="2020-21",
        season_type="regular",
    )

    outcome = compact_curated_partition(
        partition,
        list(reversed(sources)),
        run_id="compact-run",
        processed_dir=processed_dir,
        curated_dir=curated_dir,
        games_per_part=2,
        curation_code_version="sha256:curation",
    )

    assert outcome.status == "succeeded"
    assert outcome.game_count == 3
    assert outcome.input_row_count == outcome.output_row_count == 6
    assert outcome.part_count == 2
    manifest = read_curated_partition_manifest(partition, curated_dir)
    assert manifest.game_ids == (
        "0020000001",
        "0020000002",
        "0020000003",
    )
    assert manifest.game_row_counts == {
        "0020000001": 1,
        "0020000002": 2,
        "0020000003": 3,
    }
    assert manifest.quality_pass_count == 2
    assert manifest.quality_warning_count == 1
    validate_curated_partition(manifest, curated_dir)

    direct_part = pd.read_parquet(
        curated_dir
        / "events"
        / "2020-21"
        / "regular"
        / "part-00000.parquet"
    )
    assert set(CURATED_METADATA_COLUMNS) <= set(direct_part.columns)
    assert set(direct_part["season"].astype(str)) == {"2020-21"}
    assert set(direct_part["season_type"].astype(str)) == {"regular"}
    assert str(direct_part["game_id"].dtype).startswith("string")

    dataset_frame = pd.read_parquet(curated_dir / "events")
    assert len(dataset_frame) == 6
    assert set(dataset_frame["season"].astype(str)) == {"2020-21"}
    assert set(dataset_frame["season_type"].astype(str)) == {"regular"}
    assert set(dataset_frame["game_id"].astype(str)) == {
        "0020000001",
        "0020000002",
        "0020000003",
    }
    assert dataset_frame["catalog_home_team_id"].dtype == "Int64"
    assert str(dataset_frame["game_time_utc"].dtype) == "datetime64[ns, UTC]"


def test_valid_partition_resumes_and_changed_source_rebuilds(tmp_path: Path):
    sources = [
        curated_source(
            "0020000001",
            game_date=date(2020, 12, 22),
            game_time_utc=None,
        )
    ]
    processed_dir = tmp_path / "processed"
    curated_dir = tmp_path / "curated"
    seed_event_sources(processed_dir, sources)
    partition = CuratedPartition(
        table="events",
        season="2020-21",
        season_type="regular",
    )
    kwargs = {
        "run_id": "compact-run",
        "processed_dir": processed_dir,
        "curated_dir": curated_dir,
        "curation_code_version": "sha256:curation",
    }

    built = compact_curated_partition(partition, sources, **kwargs)
    resumed = compact_curated_partition(partition, sources, **kwargs)
    source_path = processed_dir / "events" / "0020000001.parquet"
    changed = pd.read_parquet(source_path)
    changed.loc[0, "event_index"] = 99
    changed.to_parquet(source_path, index=False)
    rebuilt = compact_curated_partition(partition, sources, **kwargs)

    assert built.status == "succeeded"
    assert resumed.status == "skipped"
    assert rebuilt.status == "succeeded"
    assert pd.read_parquet(
        curated_dir
        / "events"
        / "2020-21"
        / "regular"
        / "part-00000.parquet"
    ).loc[0, "event_index"] == 99


def test_corrupt_curated_part_is_rebuilt(tmp_path: Path):
    sources = [
        curated_source(
            "0020000001",
            game_date=date(2020, 12, 22),
            game_time_utc=None,
        )
    ]
    processed_dir = tmp_path / "processed"
    curated_dir = tmp_path / "curated"
    seed_event_sources(processed_dir, sources)
    partition = CuratedPartition(
        table="events",
        season="2020-21",
        season_type="regular",
    )
    kwargs = {
        "run_id": "compact-run",
        "processed_dir": processed_dir,
        "curated_dir": curated_dir,
        "curation_code_version": "sha256:curation",
    }
    compact_curated_partition(partition, sources, **kwargs)
    part_path = (
        curated_dir
        / "events"
        / "2020-21"
        / "regular"
        / "part-00000.parquet"
    )
    part_path.write_bytes(b"corrupt")

    rebuilt = compact_curated_partition(partition, sources, **kwargs)

    assert rebuilt.status == "succeeded"
    assert len(pd.read_parquet(part_path)) == 1


def test_rejects_mismatched_build_and_quality_provenance():
    source = curated_source(
        "0020000001",
        game_date=date(2020, 12, 22),
        game_time_utc=None,
    )
    quality = source.quality.model_copy(update={"code_version": "different"})

    with pytest.raises(ValidationError, match="code versions must match"):
        CuratedGameSource(
            game=source.game,
            build=source.build,
            quality=quality,
        )


def test_curation_code_fingerprint_changes_with_source(tmp_path: Path):
    source = tmp_path / "compact.py"
    source.write_text("VALUE = 1\n")
    first = curation_code_fingerprint([source])
    source.write_text("VALUE = 2\n")

    assert curation_code_fingerprint([source]) != first
