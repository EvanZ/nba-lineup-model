from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nba_lineup_model.modeling.schema import NeuralPossessionManifest
from nba_lineup_model.season.compact import (
    read_curated_partition_manifest,
    validate_curated_partition,
)
from nba_lineup_model.season.layout import CuratedDatasetLayout, CuratedPartition

NEURAL_POSSESSION_COLUMNS = (
    "schema_version",
    "season",
    "season_type",
    "game_id",
    "game_date",
    "game_time_utc",
    "possession_id",
    "possession_index",
    "period",
    "offense_team_id",
    "defense_team_id",
    "offense_team_tricode",
    "defense_team_tricode",
    "offense_player_ids",
    "defense_player_ids",
    "home_offense",
    "home_offense_sign",
    "offense_points",
    "defense_points",
    "target_offense_margin",
    "target_home_margin",
    "quality_status",
    "source_build_run_id",
    "processing_code_version",
    "play_by_play_sha256",
    "boxscore_sha256",
)

_REQUIRED_SEGMENT_COLUMNS = {
    "season",
    "season_type",
    "game_id",
    "game_date",
    "game_time_utc",
    "possession_id",
    "possession_index",
    "period",
    "offense_team_id",
    "defense_team_id",
    "home_player_ids",
    "away_player_ids",
    "points_home",
    "points_away",
    "offense_points",
    "catalog_home_team_id",
    "catalog_away_team_id",
    "catalog_home_team_tricode",
    "catalog_away_team_tricode",
    "quality_status",
    "source_build_run_id",
    "processing_code_version",
    "play_by_play_sha256",
    "boxscore_sha256",
}


class PossessionTensorDataset(Dataset[dict[str, torch.Tensor]]):
    """Contiguous tensor representation of fixed-lineup possessions."""

    def __init__(
        self,
        possessions: pd.DataFrame,
        player_columns: Mapping[int, int],
    ) -> None:
        if possessions.empty:
            raise ValueError("Tensor possession dataset cannot be empty")
        mapping = {int(player_id): int(column) for player_id, column in player_columns.items()}
        if not mapping or 0 in mapping.values():
            raise ValueError("Player columns must use positive indices and reserve zero")
        self.offense_player_indices = torch.as_tensor(
            _encode_lineups(possessions["offense_player_ids"], mapping),
            dtype=torch.long,
        )
        self.defense_player_indices = torch.as_tensor(
            _encode_lineups(possessions["defense_player_ids"], mapping),
            dtype=torch.long,
        )
        self.home_offense_sign = torch.as_tensor(
            possessions["home_offense_sign"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.target = torch.as_tensor(
            possessions["target_offense_margin"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.row_index = torch.as_tensor(
            possessions.index.to_numpy(dtype=np.int64, copy=True),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "offense_player_indices": self.offense_player_indices[index],
            "defense_player_indices": self.defense_player_indices[index],
            "home_offense_sign": self.home_offense_sign[index],
            "target": self.target[index],
            "row_index": self.row_index[index],
        }


def neural_code_fingerprint(
    source_paths: Sequence[Path | str] | None = None,
) -> str:
    """Hash neural-owned sources for reproducible datasets and runs."""

    if source_paths is None:
        package_root = Path(__file__).parents[1]
        candidates = (
            package_root / "evaluation" / "metrics.py",
            package_root / "modeling" / "neural.py",
            package_root / "modeling" / "neural_data.py",
            package_root / "modeling" / "schema.py",
            package_root / "modeling" / "train.py",
            package_root / "models" / "neural.py",
        )
        paths = tuple(path for path in candidates if path.is_file())
    else:
        paths = tuple(Path(path) for path in source_paths)
    ordered = sorted(paths)
    if not ordered:
        raise ValueError("At least one neural modeling source path is required")
    digest = hashlib.sha256()
    for path in ordered:
        if not path.is_file():
            raise ValueError(f"Neural modeling source does not exist: {path}")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def neural_possessions_frame(possession_segments: pd.DataFrame) -> pd.DataFrame:
    """Orient single-lineup possessions by offense and exclude ambiguous rows."""

    missing = _REQUIRED_SEGMENT_COLUMNS - set(possession_segments.columns)
    if missing:
        raise ValueError(f"Possession segments missing columns: {sorted(missing)}")
    if possession_segments.empty:
        raise ValueError("Possession segments cannot be empty")
    if possession_segments["season_type"].astype(str).nunique() != 1:
        raise ValueError("Neural possessions must come from one season type")

    segments = possession_segments.copy()
    keys = ["game_id", "possession_id"]
    segment_counts = segments.groupby(keys, sort=False)["possession_id"].transform("size")
    output = segments.loc[segment_counts.eq(1)].copy()
    if output.empty:
        raise ValueError("No single-lineup possessions are available")
    if output.duplicated(keys).any():
        raise ValueError("Neural possession keys must be unique")

    home_offense = output["offense_team_id"].eq(output["catalog_home_team_id"])
    away_offense = output["offense_team_id"].eq(output["catalog_away_team_id"])
    if not (home_offense | away_offense).all() or (home_offense & away_offense).any():
        raise ValueError("Possession offense team must match exactly one game team")
    expected_defense = output["catalog_away_team_id"].where(
        home_offense,
        output["catalog_home_team_id"],
    )
    if not output["defense_team_id"].eq(expected_defense).all():
        raise ValueError("Possession defense team does not match its opponent")

    output["schema_version"] = 1
    output["home_offense"] = home_offense
    output["home_offense_sign"] = np.where(home_offense, 1.0, -1.0)
    output["offense_player_ids"] = [
        list(home_players if is_home else away_players)
        for home_players, away_players, is_home in zip(
            output["home_player_ids"],
            output["away_player_ids"],
            home_offense,
            strict=True,
        )
    ]
    output["defense_player_ids"] = [
        list(away_players if is_home else home_players)
        for home_players, away_players, is_home in zip(
            output["home_player_ids"],
            output["away_player_ids"],
            home_offense,
            strict=True,
        )
    ]
    _validate_lineups(output)
    output["offense_team_tricode"] = output["catalog_home_team_tricode"].where(
        home_offense,
        output["catalog_away_team_tricode"],
    )
    output["defense_team_tricode"] = output["catalog_away_team_tricode"].where(
        home_offense,
        output["catalog_home_team_tricode"],
    )
    offense_points = output["points_home"].where(home_offense, output["points_away"])
    defense_points = output["points_away"].where(home_offense, output["points_home"])
    if not offense_points.eq(output["offense_points"]).all():
        raise ValueError("Source offense points do not match game-relative points")
    output["offense_points"] = offense_points
    output["defense_points"] = defense_points
    output["target_offense_margin"] = offense_points - defense_points
    output["target_home_margin"] = (
        output["target_offense_margin"] * output["home_offense_sign"]
    )

    output = output.loc[:, NEURAL_POSSESSION_COLUMNS].sort_values(
        ["game_time_utc", "game_id", "possession_index"],
        kind="stable",
    )
    return _type_neural_possessions(output.reset_index(drop=True))


def build_neural_possession_dataset(
    season: str,
    *,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
) -> NeuralPossessionManifest:
    """Validate curated segments and publish single-lineup possessions atomically."""

    partition = CuratedPartition(
        table="possession_segments",
        season=season,
        season_type="regular",
    )
    curated_root = Path(curated_dir)
    curated_manifest = read_curated_partition_manifest(partition, curated_root)
    validate_curated_partition(curated_manifest, curated_root)
    source_dir = CuratedDatasetLayout(curated_root).partition_dir(partition)
    segments = pd.read_parquet(source_dir)
    output = neural_possessions_frame(segments)
    possession_counts = segments.groupby(["game_id", "possession_id"], sort=False).size()

    target_dir = Path(analytical_dir) / "neural_possessions" / season / "regular"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = target_dir.parent / f".{target_dir.name}.tmp-{uuid4().hex}"
    backup_dir = target_dir.parent / f".{target_dir.name}.bak-{uuid4().hex}"
    temporary_dir.mkdir()
    try:
        part_path = temporary_dir / "part-00000.parquet"
        output.to_parquet(part_path, index=False)
        manifest = NeuralPossessionManifest(
            season=season,
            created_at=datetime.now(UTC),
            builder_code_version=neural_code_fingerprint(),
            possession_segments_manifest_sha256=_sha256_file(source_dir / "_manifest.json"),
            possession_segments_input_fingerprint=curated_manifest.input_fingerprint,
            source_game_count=curated_manifest.game_count,
            source_segment_count=len(segments),
            source_possession_count=len(possession_counts),
            included_possession_count=len(output),
            excluded_multi_segment_possession_count=int(possession_counts.gt(1).sum()),
            player_count=len(
                set().union(*output["offense_player_ids"], *output["defense_player_ids"])
            ),
            home_offense_possession_count=int(output["home_offense"].sum()),
            away_offense_possession_count=int((~output["home_offense"]).sum()),
            row_count=len(output),
            byte_count=part_path.stat().st_size,
            part_sha256=_sha256_file(part_path),
        )
        (temporary_dir / "_manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n"
        )
        validate_neural_possession_partition(temporary_dir)
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
        validate_neural_possession_partition(target_dir)
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


def read_neural_possessions(
    season: str,
    analytical_dir: Path | str = Path("data/analytical"),
) -> pd.DataFrame:
    """Read and validate one regular-season neural possession dataset."""

    partition_dir = Path(analytical_dir) / "neural_possessions" / season / "regular"
    validate_neural_possession_partition(partition_dir)
    return pd.read_parquet(partition_dir / "part-00000.parquet")


def validate_neural_possession_partition(
    partition_dir: Path | str,
) -> NeuralPossessionManifest:
    """Require exact neural possession files, hashes, and row invariants."""

    root = Path(partition_dir)
    manifest = NeuralPossessionManifest.model_validate_json(
        (root / "_manifest.json").read_text()
    )
    part_path = root / manifest.part_filename
    actual_parts = {path.name for path in root.glob("part-*.parquet")}
    if actual_parts != {manifest.part_filename}:
        raise ValueError("Neural possession part files do not match the manifest")
    if part_path.stat().st_size != manifest.byte_count:
        raise ValueError("Neural possession part byte count changed")
    if _sha256_file(part_path) != manifest.part_sha256:
        raise ValueError("Neural possession part hash changed")
    frame = pd.read_parquet(part_path)
    if len(frame) != manifest.row_count:
        raise ValueError("Neural possession row count changed")
    missing = set(NEURAL_POSSESSION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Neural possessions missing columns: {sorted(missing)}")
    if frame.duplicated(["game_id", "possession_id"]).any():
        raise ValueError("Neural possession keys must be unique")
    if set(frame["season_type"].astype(str)) != {"regular"}:
        raise ValueError("Neural possessions must contain regular-season games only")
    if not frame["home_offense_sign"].isin((-1.0, 1.0)).all():
        raise ValueError("Neural home-offense signs must be negative or positive one")
    _validate_lineups(frame)
    return manifest


def player_vocabulary(possessions: pd.DataFrame) -> dict[int, int]:
    """Map stable NBA player IDs to embedding rows, reserving zero for unknown."""

    players = sorted(
        set().union(
            *possessions["offense_player_ids"],
            *possessions["defense_player_ids"],
        )
    )
    return {int(player_id): index for index, player_id in enumerate(players, start=1)}


def _encode_lineups(
    lineups: pd.Series,
    player_columns: Mapping[int, int],
) -> np.ndarray:
    encoded = np.empty((len(lineups), 5), dtype=np.int64)
    for row_index, lineup in enumerate(lineups):
        if len(lineup) != 5:
            raise ValueError("Tensor lineups must contain exactly five players")
        encoded[row_index] = [player_columns.get(int(player_id), 0) for player_id in lineup]
    return encoded


def _validate_lineups(frame: pd.DataFrame) -> None:
    for side in ("offense", "defense"):
        lineups = frame[f"{side}_player_ids"]
        if not lineups.map(len).eq(5).all():
            raise ValueError("Neural possessions require exactly five players per team")
        if not lineups.map(lambda players: len(set(players)) == 5).all():
            raise ValueError("Neural possession lineups cannot contain duplicate players")
    overlaps = [
        bool(set(offense) & set(defense))
        for offense, defense in zip(
            frame["offense_player_ids"],
            frame["defense_player_ids"],
            strict=True,
        )
    ]
    if any(overlaps):
        raise ValueError("A player cannot appear on both sides of one possession")


def _type_neural_possessions(frame: pd.DataFrame) -> pd.DataFrame:
    for column in (
        "season",
        "season_type",
        "game_id",
        "possession_id",
        "offense_team_tricode",
        "defense_team_tricode",
        "quality_status",
        "source_build_run_id",
        "processing_code_version",
        "play_by_play_sha256",
        "boxscore_sha256",
    ):
        frame[column] = frame[column].astype("string")
    for column in (
        "schema_version",
        "possession_index",
        "period",
        "offense_team_id",
        "defense_team_id",
        "offense_points",
        "defense_points",
        "target_offense_margin",
        "target_home_margin",
    ):
        frame[column] = frame[column].astype("int64")
    frame["home_offense"] = frame["home_offense"].astype(bool)
    frame["home_offense_sign"] = frame["home_offense_sign"].astype("float64")
    frame["game_time_utc"] = pd.to_datetime(frame["game_time_utc"], utc=True)
    return frame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
