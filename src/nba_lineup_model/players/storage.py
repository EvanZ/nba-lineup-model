from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from nba_lineup_model.players.schema import (
    PlayerCatalog,
    PlayerIdentity,
    PlayerSeasonBio,
    PlayerSeasonBioDataset,
    PlayerSeasonBioManifest,
)

PLAYER_CATALOG_COLUMNS = tuple(PlayerIdentity.model_fields)
PLAYER_SEASON_BIO_COLUMNS = tuple(PlayerSeasonBio.model_fields)

_CATALOG_STRING_COLUMNS = (
    "first_name",
    "last_name",
    "display_name",
    "player_slug",
    "listed_position",
    "height_raw",
    "college",
    "country",
    "latest_team_abbreviation",
    "latest_team_slug",
    "latest_jersey_number",
    "supplemental_status",
    "source_season",
    "source_url",
    "source_sha256",
)
_CATALOG_INTEGER_COLUMNS = (
    "schema_version",
    "player_id",
    "height_inches",
    "weight_pounds",
    "draft_year",
    "draft_round",
    "draft_number",
    "from_year",
    "to_year",
    "latest_team_id",
)
_BIO_STRING_COLUMNS = (
    "player_name",
    "season",
    "season_type",
    "team_abbreviation",
    "listed_position",
    "height_raw",
    "college",
    "country",
    "player_index_source_sha256",
    "bio_source_url",
    "bio_source_sha256",
)
_BIO_INTEGER_COLUMNS = (
    "schema_version",
    "player_id",
    "team_id",
    "height_inches",
    "weight_pounds",
    "draft_year",
    "draft_round",
    "draft_number",
)


def player_catalog_frame(
    catalog: PlayerCatalog | Sequence[PlayerIdentity],
) -> pd.DataFrame:
    """Return a typed historical player catalog ordered by numeric ID."""

    players = catalog.players if isinstance(catalog, PlayerCatalog) else list(catalog)
    validated = PlayerCatalog(players=players)
    rows = [player.model_dump(mode="python") for player in validated.players]
    frame = pd.DataFrame(rows, columns=PLAYER_CATALOG_COLUMNS)
    frame = frame.sort_values("player_id", kind="stable").reset_index(drop=True)
    for column in _CATALOG_STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in _CATALOG_INTEGER_COLUMNS:
        frame[column] = frame[column].astype("Int64")
    for column in ("is_undrafted", "roster_status", "is_defunct_team"):
        frame[column] = frame[column].astype("boolean")
    frame["source_fetched_at"] = pd.to_datetime(
        frame["source_fetched_at"],
        utc=True,
    )
    return frame


def player_catalog_from_frame(frame: pd.DataFrame) -> PlayerCatalog:
    """Validate a historical player catalog frame."""

    return PlayerCatalog(
        players=[
            PlayerIdentity.model_validate(_python_record(row))
            for row in frame.to_dict(orient="records")
        ]
    )


def write_player_catalog(
    catalog: PlayerCatalog | Sequence[PlayerIdentity],
    path: Path | str = Path("data/catalog/players.parquet"),
) -> Path:
    """Atomically write the canonical historical player catalog."""

    output_path = Path(path)
    _atomic_write_parquet(player_catalog_frame(catalog), output_path)
    return output_path


def read_player_catalog(
    path: Path | str = Path("data/catalog/players.parquet"),
) -> PlayerCatalog:
    """Read and validate the historical player catalog."""

    return player_catalog_from_frame(pd.read_parquet(Path(path)))


def merge_player_catalogs(
    existing: PlayerCatalog,
    incoming: PlayerCatalog,
) -> PlayerCatalog:
    """Preserve the union of players while preferring the latest source row."""

    by_player_id = {player.player_id: player for player in existing.players}
    for player in incoming.players:
        current = by_player_id.get(player.player_id)
        if current is None or int(player.source_season[:4]) >= int(current.source_season[:4]):
            by_player_id[player.player_id] = player
    return PlayerCatalog(players=[by_player_id[player_id] for player_id in sorted(by_player_id)])


def player_season_bio_frame(
    dataset: PlayerSeasonBioDataset | Sequence[PlayerSeasonBio],
) -> pd.DataFrame:
    """Return a typed player-season bio frame ordered by player ID."""

    if isinstance(dataset, PlayerSeasonBioDataset):
        validated = dataset
    else:
        players = list(dataset)
        if not players:
            raise ValueError("At least one player-season bio row is required")
        validated = PlayerSeasonBioDataset(
            season=players[0].season,
            season_type=players[0].season_type,
            players=players,
        )
    rows = [player.model_dump(mode="python") for player in validated.players]
    frame = pd.DataFrame(rows, columns=PLAYER_SEASON_BIO_COLUMNS)
    frame = frame.sort_values("player_id", kind="stable").reset_index(drop=True)
    for column in _BIO_STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in _BIO_INTEGER_COLUMNS:
        frame[column] = frame[column].astype("Int64")
    frame["age"] = frame["age"].astype("Float64")
    frame["is_undrafted"] = frame["is_undrafted"].astype("boolean")
    frame["bio_source_fetched_at"] = pd.to_datetime(
        frame["bio_source_fetched_at"],
        utc=True,
    )
    return frame


def player_season_bios_from_frame(
    frame: pd.DataFrame,
) -> PlayerSeasonBioDataset:
    """Validate one player-season bio frame."""

    players = [
        PlayerSeasonBio.model_validate(_python_record(row))
        for row in frame.to_dict(orient="records")
    ]
    if not players:
        raise ValueError("Player-season bio frame is empty")
    return PlayerSeasonBioDataset(
        season=players[0].season,
        season_type=players[0].season_type,
        players=players,
    )


def player_season_partition_dir(
    season: str,
    season_type: str,
    curated_dir: Path | str = Path("data/curated"),
) -> Path:
    """Return the plain-directory player-season partition path."""

    return Path(curated_dir) / "player_seasons" / season / season_type


def write_player_season_bios(
    dataset: PlayerSeasonBioDataset,
    curated_dir: Path | str = Path("data/curated"),
) -> PlayerSeasonBioManifest:
    """Atomically publish one self-contained player-season bio partition."""

    target_dir = player_season_partition_dir(
        dataset.season,
        dataset.season_type,
        curated_dir,
    )
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = target_dir.parent / f".{target_dir.name}.tmp-{uuid4().hex}"
    backup_dir = target_dir.parent / f".{target_dir.name}.bak-{uuid4().hex}"
    temporary_dir.mkdir()
    try:
        part_path = temporary_dir / "part-00000.parquet"
        frame = player_season_bio_frame(dataset)
        frame.to_parquet(part_path, index=False)
        index_hashes = {player.player_index_source_sha256 for player in dataset.players}
        bio_hashes = {player.bio_source_sha256 for player in dataset.players}
        if len(index_hashes) != 1 or len(bio_hashes) != 1:
            raise ValueError("Player-season rows must share exact source identities")
        manifest = PlayerSeasonBioManifest(
            season=dataset.season,
            season_type=dataset.season_type,
            created_at=datetime.now(UTC),
            player_count=len(dataset.players),
            player_ids=tuple(player.player_id for player in dataset.players),
            player_index_source_sha256=next(iter(index_hashes)),
            bio_source_sha256=next(iter(bio_hashes)),
            row_count=len(frame),
            byte_count=part_path.stat().st_size,
            part_sha256=_sha256_file(part_path),
        )
        _write_manifest(manifest, temporary_dir / "_manifest.json")
        _validate_manifest_files(manifest, temporary_dir)
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
        validate_player_season_partition(
            dataset.season,
            dataset.season_type,
            curated_dir,
        )
        return manifest
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        if backup_dir.exists() and not target_dir.exists():
            backup_dir.replace(target_dir)
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def read_player_season_bios(
    season: str,
    season_type: str = "regular",
    curated_dir: Path | str = Path("data/curated"),
) -> PlayerSeasonBioDataset:
    """Read and validate one player-season bio partition."""

    part_path = player_season_partition_dir(season, season_type, curated_dir) / "part-00000.parquet"
    return player_season_bios_from_frame(pd.read_parquet(part_path))


def read_player_season_manifest(
    season: str,
    season_type: str = "regular",
    curated_dir: Path | str = Path("data/curated"),
) -> PlayerSeasonBioManifest:
    """Read one player-season bio integrity manifest."""

    path = player_season_partition_dir(season, season_type, curated_dir) / "_manifest.json"
    return PlayerSeasonBioManifest.model_validate_json(path.read_text())


def validate_player_season_partition(
    season: str,
    season_type: str = "regular",
    curated_dir: Path | str = Path("data/curated"),
) -> PlayerSeasonBioManifest:
    """Require exact file integrity, schema, IDs, and row counts."""

    partition_dir = player_season_partition_dir(
        season,
        season_type,
        curated_dir,
    )
    manifest = read_player_season_manifest(season, season_type, curated_dir)
    if manifest.season != season or manifest.season_type != season_type:
        raise ValueError("Player-season manifest does not match its partition")
    _validate_manifest_files(manifest, partition_dir)
    return manifest


def _validate_manifest_files(
    manifest: PlayerSeasonBioManifest,
    partition_dir: Path,
) -> None:
    part_path = partition_dir / manifest.part_filename
    actual_parts = {path.name for path in partition_dir.glob("part-*.parquet")}
    if actual_parts != {manifest.part_filename}:
        raise ValueError("Player-season part files do not match the manifest")
    if part_path.stat().st_size != manifest.byte_count:
        raise ValueError("Player-season part byte count changed")
    if _sha256_file(part_path) != manifest.part_sha256:
        raise ValueError("Player-season part hash changed")
    dataset = player_season_bios_from_frame(pd.read_parquet(part_path))
    player_ids = tuple(player.player_id for player in dataset.players)
    if player_ids != manifest.player_ids:
        raise ValueError("Player-season IDs do not match the manifest")
    if len(dataset.players) != manifest.row_count:
        raise ValueError("Player-season row count does not match the manifest")


def _write_manifest(manifest: PlayerSeasonBioManifest, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    temporary_path.replace(path)


def _python_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _python_value(value) for key, value in record.items()}


def _python_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if pd.api.types.is_scalar(value) and bool(pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime | date | str | int | float | bool):
        return value
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
