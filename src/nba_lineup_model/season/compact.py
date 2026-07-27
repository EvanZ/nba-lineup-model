from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nba_lineup_model.season.layout import (
    CURATED_TABLES,
    CuratedDatasetLayout,
    CuratedPartition,
)
from nba_lineup_model.season.schema import (
    SHA256_PATTERN,
    CatalogGame,
    GameBuildRecord,
    GameQualityRecord,
)

CURATED_METADATA_COLUMNS = (
    "season",
    "season_type",
    "game_date",
    "game_time_utc",
    "catalog_home_team_id",
    "catalog_home_team_tricode",
    "catalog_away_team_id",
    "catalog_away_team_tricode",
    "quality_status",
    "quality_issue_codes_json",
    "quality_recorded_at",
    "quality_run_id",
    "source_build_run_id",
    "source_build_attempt_id",
    "processing_code_version",
    "play_by_play_sha256",
    "boxscore_sha256",
)


class CuratedGameSource(BaseModel):
    """Catalog, successful-build, and quality evidence for one curated game."""

    model_config = ConfigDict(strict=True, extra="forbid")

    game: CatalogGame
    build: GameBuildRecord
    quality: GameQualityRecord

    @model_validator(mode="after")
    def validate_provenance(self) -> CuratedGameSource:
        game_keys = (self.game.game_id, self.game.season, self.game.season_type)
        if (
            (self.build.game_id, self.build.season, self.build.season_type) != game_keys
            or (
                self.quality.game_id,
                self.quality.season,
                self.quality.season_type,
            )
            != game_keys
        ):
            raise ValueError("Catalog, build, and quality game keys must match")
        if self.game.game_status != "final":
            raise ValueError("Curated games must be final")
        if self.build.status != "succeeded":
            raise ValueError("Curated games require a successful build")
        if self.build.output_table_count != len(CURATED_TABLES):
            raise ValueError("Curated games require all processed output tables")
        if self.quality.status not in {"pass", "warning"}:
            raise ValueError("Curated games require pass or warning quality")
        if not self.build.code_version:
            raise ValueError("Curated games require a processing code version")
        if self.build.code_version != self.quality.code_version:
            raise ValueError("Build and quality processing code versions must match")
        for endpoint in ("play_by_play", "boxscore"):
            build_hash = getattr(self.build, f"{endpoint}_sha256")
            quality_hash = getattr(self.quality, f"{endpoint}_sha256")
            if build_hash is None or build_hash != quality_hash:
                raise ValueError(f"Build and quality {endpoint} hashes must match")
        return self


class CuratedPartRecord(BaseModel):
    """Integrity evidence for one curated Parquet shard."""

    model_config = ConfigDict(strict=True, extra="forbid")

    filename: str = Field(pattern=r"^part-\d{5}\.parquet$")
    part_number: int = Field(ge=0)
    game_count: int = Field(ge=1)
    row_count: int = Field(ge=1)
    byte_count: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class CuratedPartitionManifest(BaseModel):
    """Versioned, lossless-compaction contract for one curated partition."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    prefect_flow_run_id: str | None = None
    prefect_task_run_id: str | None = None
    partition: CuratedPartition
    created_at: datetime
    curation_code_version: str = Field(min_length=1)
    input_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    games_per_part: int = Field(ge=1)
    quality_pass_count: int = Field(ge=0)
    quality_warning_count: int = Field(ge=0)
    source_processing_code_versions: tuple[str, ...] = Field(min_length=1)
    game_count: int = Field(ge=1)
    game_ids: tuple[str, ...] = Field(min_length=1)
    game_row_counts: dict[str, int]
    input_row_count: int = Field(ge=1)
    output_row_count: int = Field(ge=1)
    part_count: int = Field(ge=1)
    parts: tuple[CuratedPartRecord, ...] = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Datetime values must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_counts(self) -> CuratedPartitionManifest:
        if len(self.game_ids) != self.game_count:
            raise ValueError("Manifest game count does not match game IDs")
        if len(set(self.game_ids)) != self.game_count:
            raise ValueError("Manifest game IDs must be unique")
        if set(self.game_row_counts) != set(self.game_ids):
            raise ValueError("Manifest row-count game IDs do not match")
        if any(row_count < 1 for row_count in self.game_row_counts.values()):
            raise ValueError("Manifest game row counts must be positive")
        if sum(self.game_row_counts.values()) != self.input_row_count:
            raise ValueError("Manifest game rows do not match input rows")
        if self.input_row_count != self.output_row_count:
            raise ValueError("Curated compaction must conserve rows")
        if len(self.parts) != self.part_count:
            raise ValueError("Manifest part count does not match parts")
        if sum(part.row_count for part in self.parts) != self.output_row_count:
            raise ValueError("Manifest part rows do not match output rows")
        if len({part.filename for part in self.parts}) != self.part_count:
            raise ValueError("Manifest part filenames must be unique")
        if tuple(part.part_number for part in self.parts) != tuple(
            range(self.part_count)
        ):
            raise ValueError("Manifest part numbers must be contiguous")
        if (
            self.quality_pass_count + self.quality_warning_count
            != self.game_count
        ):
            raise ValueError("Manifest quality counts do not match games")
        return self

    @classmethod
    def read(cls, path: Path | str) -> CuratedPartitionManifest:
        return cls.model_validate_json(Path(path).read_text())

    def write(self, path: Path | str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(self.model_dump_json(indent=2) + "\n")
        temporary_path.replace(output_path)
        return output_path


class PartitionCompactionOutcome(BaseModel):
    """Terminal outcome of compacting one curated table partition."""

    model_config = ConfigDict(strict=True, extra="forbid")

    partition: CuratedPartition
    status: Literal["succeeded", "skipped", "failed"]
    manifest_path: str | None = None
    game_count: int = Field(default=0, ge=0)
    input_row_count: int = Field(default=0, ge=0)
    output_row_count: int = Field(default=0, ge=0)
    part_count: int = Field(default=0, ge=0)
    error_type: str | None = None
    error_message: str | None = None


def curation_code_fingerprint(
    source_paths: Sequence[Path | str] | None = None,
) -> str:
    """Hash curation sources so implementation changes invalidate resume."""

    paths = (
        sorted(Path(path) for path in source_paths)
        if source_paths is not None
        else sorted((Path(__file__), Path(__file__).with_name("layout.py")))
    )
    if not paths:
        raise ValueError("At least one curation source path is required")
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Curation source does not exist: {path}")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def curated_input_fingerprint(
    partition: CuratedPartition,
    sources: Sequence[CuratedGameSource],
    processed_dir: Path | str = Path("data/processed"),
) -> str:
    """Hash exact source files and metadata selected for one partition."""

    ordered_sources = _validate_partition_sources(partition, sources)
    digest = hashlib.sha256()
    digest.update(b"curated-input-v1\0")
    digest.update(partition.model_dump_json().encode())
    digest.update(b"\0")
    for source in ordered_sources:
        path = _processed_path(partition, source.game.game_id, processed_dir)
        if not path.is_file():
            raise ValueError(f"Missing processed {partition.table} output: {path}")
        evidence = {
            "game": source.game.model_dump(mode="json"),
            "build": source.build.model_dump(mode="json"),
            "quality": source.quality.model_dump(mode="json"),
            "processed_byte_count": path.stat().st_size,
            "processed_sha256": _sha256_file(path),
        }
        digest.update(
            json.dumps(
                evidence,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def compact_curated_partition(
    partition: CuratedPartition,
    sources: Sequence[CuratedGameSource],
    *,
    run_id: str,
    processed_dir: Path | str = Path("data/processed"),
    curated_dir: Path | str = Path("data/curated"),
    games_per_part: int = 100,
    force: bool = False,
    curation_code_version: str | None = None,
    prefect_flow_run_id: str | None = None,
    prefect_task_run_id: str | None = None,
) -> PartitionCompactionOutcome:
    """Compact validated per-game Parquet into one atomic curated partition."""

    if games_per_part < 1:
        raise ValueError("games_per_part must be positive")
    ordered_sources = _validate_partition_sources(partition, sources)
    curation_code_version = curation_code_version or curation_code_fingerprint()
    input_fingerprint = curated_input_fingerprint(
        partition,
        ordered_sources,
        processed_dir,
    )
    layout = CuratedDatasetLayout(Path(curated_dir))
    target_dir = layout.partition_dir(partition)
    manifest_path = layout.manifest_path(partition)

    if not force and manifest_path.is_file():
        try:
            existing = CuratedPartitionManifest.read(manifest_path)
            if (
                existing.input_fingerprint == input_fingerprint
                and existing.curation_code_version == curation_code_version
                and existing.games_per_part == games_per_part
            ):
                validate_curated_partition(existing, curated_dir)
                return _outcome("skipped", existing, manifest_path)
        except (OSError, ValueError):
            pass

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = target_dir.parent / f".{target_dir.name}.tmp-{uuid4().hex}"
    backup_dir = target_dir.parent / f".{target_dir.name}.bak-{uuid4().hex}"
    temporary_dir.mkdir()
    try:
        parts: list[CuratedPartRecord] = []
        game_row_counts: dict[str, int] = {}
        for part_number, start in enumerate(
            range(0, len(ordered_sources), games_per_part)
        ):
            chunk = ordered_sources[start : start + games_per_part]
            frames: list[pd.DataFrame] = []
            for source in chunk:
                frame = _read_curated_source(partition, source, processed_dir)
                game_row_counts[source.game.game_id] = len(frame)
                frames.append(frame)
            part_frame = pd.concat(frames, ignore_index=True, sort=False)
            part_path = temporary_dir / f"part-{part_number:05d}.parquet"
            part_frame.to_parquet(part_path, index=False)
            parts.append(
                CuratedPartRecord(
                    filename=part_path.name,
                    part_number=part_number,
                    game_count=len(chunk),
                    row_count=len(part_frame),
                    byte_count=part_path.stat().st_size,
                    sha256=_sha256_file(part_path),
                )
            )

        quality_counts = Counter(source.quality.status for source in ordered_sources)
        row_count = sum(game_row_counts.values())
        manifest = CuratedPartitionManifest(
            run_id=run_id,
            prefect_flow_run_id=prefect_flow_run_id,
            prefect_task_run_id=prefect_task_run_id,
            partition=partition,
            created_at=datetime.now(UTC),
            curation_code_version=curation_code_version,
            input_fingerprint=input_fingerprint,
            games_per_part=games_per_part,
            quality_pass_count=quality_counts["pass"],
            quality_warning_count=quality_counts["warning"],
            source_processing_code_versions=tuple(
                sorted({source.quality.code_version for source in ordered_sources})
            ),
            game_count=len(ordered_sources),
            game_ids=tuple(source.game.game_id for source in ordered_sources),
            game_row_counts=game_row_counts,
            input_row_count=row_count,
            output_row_count=sum(part.row_count for part in parts),
            part_count=len(parts),
            parts=tuple(parts),
        )
        manifest.write(temporary_dir / "_manifest.json")
        _validate_manifest_files(manifest, temporary_dir)
        _publish_partition(temporary_dir, target_dir, backup_dir)
        validate_curated_partition(manifest, curated_dir)
        return _outcome("succeeded", manifest, manifest_path)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        if backup_dir.exists() and not target_dir.exists():
            backup_dir.replace(target_dir)
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def read_curated_partition_manifest(
    partition: CuratedPartition,
    curated_dir: Path | str = Path("data/curated"),
) -> CuratedPartitionManifest:
    """Read the manifest for one curated partition."""

    return CuratedPartitionManifest.read(
        CuratedDatasetLayout(Path(curated_dir)).manifest_path(partition)
    )


def validate_curated_partition(
    manifest: CuratedPartitionManifest,
    curated_dir: Path | str = Path("data/curated"),
) -> None:
    """Require exact files, hashes, schemas, game IDs, and row conservation."""

    partition_dir = CuratedDatasetLayout(Path(curated_dir)).partition_dir(
        manifest.partition
    )
    _validate_manifest_files(manifest, partition_dir)


def _validate_manifest_files(
    manifest: CuratedPartitionManifest,
    partition_dir: Path,
) -> None:
    expected_files = {part.filename for part in manifest.parts}
    actual_files = {path.name for path in partition_dir.glob("part-*.parquet")}
    if actual_files != expected_files:
        raise ValueError("Curated partition files do not match its manifest")

    game_row_counts: Counter[str] = Counter()
    for part in manifest.parts:
        path = partition_dir / part.filename
        if not path.is_file():
            raise ValueError(f"Missing curated part: {path}")
        if path.stat().st_size != part.byte_count:
            raise ValueError(f"Curated part byte count changed: {path}")
        if _sha256_file(path) != part.sha256:
            raise ValueError(f"Curated part hash changed: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != part.row_count:
            raise ValueError(f"Curated part row count changed: {path}")
        missing_columns = (
            {"game_id", *CURATED_METADATA_COLUMNS} - set(frame.columns)
        )
        if missing_columns:
            raise ValueError(
                f"Curated part is missing columns {sorted(missing_columns)}: {path}"
            )
        identifiers = frame["game_id"].astype("string")
        if identifiers.isna().any():
            raise ValueError(f"Curated part contains null game IDs: {path}")
        if identifiers.nunique() != part.game_count:
            raise ValueError(f"Curated part game count changed: {path}")
        game_row_counts.update(
            {
                str(game_id): int(count)
                for game_id, count in identifiers.value_counts().items()
            }
        )

    if dict(game_row_counts) != manifest.game_row_counts:
        raise ValueError("Curated game row counts do not match the manifest")
    if sum(game_row_counts.values()) != manifest.output_row_count:
        raise ValueError("Curated output rows do not match the manifest")


def _validate_partition_sources(
    partition: CuratedPartition,
    sources: Sequence[CuratedGameSource],
) -> list[CuratedGameSource]:
    if not sources:
        raise ValueError("At least one curated game source is required")
    ordered = sorted(
        sources,
        key=lambda source: (source.game.game_date, source.game.game_id),
    )
    game_ids = [source.game.game_id for source in ordered]
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("Curated partition sources contain duplicate game IDs")
    for source in ordered:
        if (
            source.game.season != partition.season
            or source.game.season_type != partition.season_type
        ):
            raise ValueError("Curated game source does not match its partition")
    return ordered


def _read_curated_source(
    partition: CuratedPartition,
    source: CuratedGameSource,
    processed_dir: Path | str,
) -> pd.DataFrame:
    path = _processed_path(partition, source.game.game_id, processed_dir)
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Processed {partition.table} output is empty: {path}")
    collisions = set(frame.columns) & set(CURATED_METADATA_COLUMNS)
    if collisions:
        raise ValueError(
            f"Processed output collides with curated columns {sorted(collisions)}: {path}"
        )
    identifiers = frame["game_id"].astype("string")
    if identifiers.isna().any() or set(identifiers) != {source.game.game_id}:
        raise ValueError(f"Processed {partition.table} output has wrong game_id: {path}")
    frame["game_id"] = identifiers
    _add_metadata(frame, source)
    return frame


def _add_metadata(frame: pd.DataFrame, source: CuratedGameSource) -> None:
    row_count = len(frame)
    index = frame.index
    game = source.game
    build = source.build
    quality = source.quality

    for column, value in (
        ("season", game.season),
        ("season_type", game.season_type),
    ):
        frame[column] = pd.Series([value] * row_count, index=index, dtype="string")
    frame["game_date"] = pd.Series([game.game_date] * row_count, index=index)
    frame["game_time_utc"] = pd.Series(
        pd.array([game.game_time_utc] * row_count, dtype="datetime64[ns, UTC]"),
        index=index,
    )
    for column, value in (
        ("catalog_home_team_id", game.home_team_id),
        ("catalog_away_team_id", game.away_team_id),
    ):
        frame[column] = pd.Series([value] * row_count, index=index, dtype="Int64")
    for column, value in (
        ("catalog_home_team_tricode", game.home_team_tricode),
        ("catalog_away_team_tricode", game.away_team_tricode),
        ("quality_status", quality.status),
        (
            "quality_issue_codes_json",
            json.dumps(quality.issue_codes, separators=(",", ":")),
        ),
        ("quality_run_id", quality.run_id),
        ("source_build_run_id", build.run_id),
        ("source_build_attempt_id", build.attempt_id),
        ("processing_code_version", quality.code_version),
        ("play_by_play_sha256", quality.play_by_play_sha256),
        ("boxscore_sha256", quality.boxscore_sha256),
    ):
        frame[column] = pd.Series([value] * row_count, index=index, dtype="string")
    frame["quality_recorded_at"] = pd.Series(
        pd.array([quality.recorded_at] * row_count, dtype="datetime64[ns, UTC]"),
        index=index,
    )


def _processed_path(
    partition: CuratedPartition,
    game_id: str,
    processed_dir: Path | str,
) -> Path:
    return Path(processed_dir) / partition.table / f"{game_id}.parquet"


def _publish_partition(
    temporary_dir: Path,
    target_dir: Path,
    backup_dir: Path,
) -> None:
    if target_dir.exists():
        target_dir.replace(backup_dir)
    try:
        temporary_dir.replace(target_dir)
    except Exception:
        if backup_dir.exists() and not target_dir.exists():
            backup_dir.replace(target_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _outcome(
    status: Literal["succeeded", "skipped"],
    manifest: CuratedPartitionManifest,
    manifest_path: Path,
) -> PartitionCompactionOutcome:
    return PartitionCompactionOutcome(
        partition=manifest.partition,
        status=status,
        manifest_path=str(manifest_path),
        game_count=manifest.game_count,
        input_row_count=manifest.input_row_count,
        output_row_count=manifest.output_row_count,
        part_count=manifest.part_count,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
