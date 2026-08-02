from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.schema import RapmStintManifest
from nba_lineup_model.season.compact import (
    read_curated_partition_manifest,
    validate_curated_partition,
)
from nba_lineup_model.season.layout import CuratedDatasetLayout, CuratedPartition

RAPM_STINT_COLUMNS = (
    "schema_version",
    "season",
    "season_type",
    "game_id",
    "game_date",
    "game_time_utc",
    "stint_index",
    "period",
    "home_team_id",
    "away_team_id",
    "home_team_tricode",
    "away_team_tricode",
    "home_player_ids",
    "away_player_ids",
    "start_elapsed_game_seconds",
    "end_elapsed_game_seconds",
    "duration_seconds",
    "points_home",
    "points_away",
    "home_margin",
    "home_offensive_possessions",
    "away_offensive_possessions",
    "possessions",
    "target_home_net_rating",
    "quality_status",
    "quality_issue_codes_json",
    "source_build_run_id",
    "processing_code_version",
    "play_by_play_sha256",
    "boxscore_sha256",
)

_REQUIRED_STINT_COLUMNS = {
    "season",
    "season_type",
    "game_id",
    "game_date",
    "game_time_utc",
    "stint_index",
    "period",
    "start_event_index",
    "end_event_index",
    "start_elapsed_game_seconds",
    "end_elapsed_game_seconds",
    "duration_seconds",
    "points_home",
    "points_away",
    "home_player_ids",
    "away_player_ids",
    "catalog_home_team_id",
    "catalog_away_team_id",
    "catalog_home_team_tricode",
    "catalog_away_team_tricode",
    "quality_status",
    "quality_issue_codes_json",
    "source_build_run_id",
    "processing_code_version",
    "play_by_play_sha256",
    "boxscore_sha256",
}
_REQUIRED_SEGMENT_COLUMNS = {
    "game_id",
    "possession_id",
    "start_event_index",
    "end_event_index",
    "offense_team_id",
    "home_player_ids",
    "away_player_ids",
    "points_home",
    "points_away",
    "catalog_home_team_id",
    "catalog_away_team_id",
}


def modeling_code_fingerprint(
    source_paths: Sequence[Path | str] | None = None,
) -> str:
    """Hash modeling-owned source files for reproducible analytical artifacts."""

    if source_paths is None:
        package_root = Path(__file__).parents[1]
        entries = (
            package_root / "evaluation" / "metrics.py",
            package_root / "modeling" / "schema.py",
            package_root / "modeling" / "stints.py",
            package_root / "modeling" / "train.py",
            package_root / "models" / "baselines.py",
        )
    else:
        entries = tuple(Path(path) for path in source_paths)
    paths = sorted(entries)
    if not paths:
        raise ValueError("At least one modeling source path is required")
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Modeling source does not exist: {path}")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def rapm_stints_frame(
    lineup_stints: pd.DataFrame,
    possession_segments: pd.DataFrame,
) -> pd.DataFrame:
    """Build positive-exposure RAPM stints with conserved possession shares."""

    _require_columns(lineup_stints, _REQUIRED_STINT_COLUMNS, "lineup stints")
    _require_columns(
        possession_segments,
        _REQUIRED_SEGMENT_COLUMNS,
        "possession segments",
    )
    stints = lineup_stints.copy()
    segments = possession_segments.copy()
    _validate_source_frames(stints, segments)

    segment_counts = segments.groupby(["game_id", "possession_id"], sort=False)[
        "possession_id"
    ].transform("size")
    segments["_possession_share"] = 1.0 / segment_counts.astype(float)
    segments["_stint_index"] = _assign_segments_to_stints(stints, segments)
    home_offense = segments["offense_team_id"].eq(segments["catalog_home_team_id"])
    away_offense = segments["offense_team_id"].eq(segments["catalog_away_team_id"])
    if not (home_offense | away_offense).all():
        raise ValueError("Possession segment offense team is not a game team")
    segments["_home_possessions"] = segments["_possession_share"].where(
        home_offense,
        0.0,
    )
    segments["_away_possessions"] = segments["_possession_share"].where(
        away_offense,
        0.0,
    )

    exposure = (
        segments.groupby(["game_id", "_stint_index"], sort=False)
        .agg(
            home_offensive_possessions=("_home_possessions", "sum"),
            away_offensive_possessions=("_away_possessions", "sum"),
            segment_points_home=("points_home", "sum"),
            segment_points_away=("points_away", "sum"),
        )
        .reset_index()
        .rename(columns={"_stint_index": "stint_index"})
    )
    output = stints.merge(
        exposure,
        on=["game_id", "stint_index"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "home_offensive_possessions",
        "away_offensive_possessions",
        "segment_points_home",
        "segment_points_away",
    ):
        output[column] = output[column].fillna(0.0)
    if not output["points_home"].eq(output["segment_points_home"]).all():
        raise ValueError("Possession segments do not conserve home stint points")
    if not output["points_away"].eq(output["segment_points_away"]).all():
        raise ValueError("Possession segments do not conserve away stint points")

    output["possessions"] = (
        output["home_offensive_possessions"] + output["away_offensive_possessions"]
    ) / 2.0
    output = output.loc[output["possessions"].gt(0)].copy()
    output["home_margin"] = output["points_home"] - output["points_away"]
    output["target_home_net_rating"] = 100.0 * output["home_margin"] / output["possessions"]
    output["schema_version"] = 1
    output["home_team_id"] = output["catalog_home_team_id"]
    output["away_team_id"] = output["catalog_away_team_id"]
    output["home_team_tricode"] = output["catalog_home_team_tricode"]
    output["away_team_tricode"] = output["catalog_away_team_tricode"]
    output = output.loc[:, RAPM_STINT_COLUMNS]
    return _type_rapm_stints(output)


def build_rapm_stint_dataset(
    season: str,
    *,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
) -> RapmStintManifest:
    """Validate curated regular-season inputs and publish RAPM stints atomically."""

    season_type = "regular"
    curated_root = Path(curated_dir)
    analytical_root = Path(analytical_dir)
    lineup_partition = CuratedPartition(
        table="lineup_stints",
        season=season,
        season_type=season_type,
    )
    segment_partition = CuratedPartition(
        table="possession_segments",
        season=season,
        season_type=season_type,
    )
    lineup_manifest = read_curated_partition_manifest(lineup_partition, curated_root)
    segment_manifest = read_curated_partition_manifest(segment_partition, curated_root)
    validate_curated_partition(lineup_manifest, curated_root)
    validate_curated_partition(segment_manifest, curated_root)
    if lineup_manifest.game_ids != segment_manifest.game_ids:
        raise ValueError("Lineup stint and possession segment games do not match")

    layout = CuratedDatasetLayout(curated_root)
    lineup_dir = layout.partition_dir(lineup_partition)
    segment_dir = layout.partition_dir(segment_partition)
    lineup_stints = pd.read_parquet(lineup_dir)
    segments = pd.read_parquet(segment_dir)
    output, excluded_game_ids = build_rapm_stints_from_curated_games(
        season,
        curated_dir=curated_root,
    )
    excluded_stint_count = int(
        lineup_stints["game_id"].astype(str).isin(excluded_game_ids).sum()
    )

    target_dir = analytical_root / "rapm_stints" / season / season_type
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = target_dir.parent / f".{target_dir.name}.tmp-{uuid4().hex}"
    backup_dir = target_dir.parent / f".{target_dir.name}.bak-{uuid4().hex}"
    temporary_dir.mkdir()
    try:
        part_path = temporary_dir / "part-00000.parquet"
        output.to_parquet(part_path, index=False)
        possession_sizes = segments.groupby(["game_id", "possession_id"]).size()
        manifest = RapmStintManifest(
            schema_version=2,
            season=season,
            created_at=datetime.now(UTC),
            builder_code_version=modeling_code_fingerprint(),
            lineup_stints_manifest_sha256=_sha256_file(lineup_dir / "_manifest.json"),
            possession_segments_manifest_sha256=_sha256_file(segment_dir / "_manifest.json"),
            lineup_stints_input_fingerprint=lineup_manifest.input_fingerprint,
            possession_segments_input_fingerprint=segment_manifest.input_fingerprint,
            source_game_count=lineup_manifest.game_count,
            source_stint_count=len(lineup_stints),
            source_possession_count=len(possession_sizes),
            multi_segment_possession_count=int(possession_sizes.gt(1).sum()),
            included_stint_count=len(output),
            excluded_zero_exposure_stint_count=(
                len(lineup_stints) - excluded_stint_count - len(output)
            ),
            excluded_nonconserving_stint_count=excluded_stint_count,
            excluded_nonconserving_game_ids=excluded_game_ids,
            player_count=len(set().union(*output["home_player_ids"], *output["away_player_ids"])),
            home_offensive_possessions=float(output["home_offensive_possessions"].sum()),
            away_offensive_possessions=float(output["away_offensive_possessions"].sum()),
            row_count=len(output),
            byte_count=part_path.stat().st_size,
            part_sha256=_sha256_file(part_path),
        )
        (temporary_dir / "_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        validate_rapm_stint_partition(temporary_dir)
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
        validate_rapm_stint_partition(target_dir)
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


def read_rapm_stints(
    season: str,
    analytical_dir: Path | str = Path("data/analytical"),
) -> pd.DataFrame:
    """Read and validate one regular-season RAPM stint dataset."""

    partition_dir = Path(analytical_dir) / "rapm_stints" / season / "regular"
    validate_rapm_stint_partition(partition_dir)
    return pd.read_parquet(partition_dir / "part-00000.parquet")


def validate_rapm_stint_partition(
    partition_dir: Path | str,
) -> RapmStintManifest:
    """Require exact RAPM part integrity and row-level invariants."""

    root = Path(partition_dir)
    manifest = RapmStintManifest.model_validate_json((root / "_manifest.json").read_text())
    part_path = root / manifest.part_filename
    actual_parts = {path.name for path in root.glob("part-*.parquet")}
    if actual_parts != {manifest.part_filename}:
        raise ValueError("RAPM stint part files do not match the manifest")
    if part_path.stat().st_size != manifest.byte_count:
        raise ValueError("RAPM stint part byte count changed")
    if _sha256_file(part_path) != manifest.part_sha256:
        raise ValueError("RAPM stint part hash changed")
    frame = pd.read_parquet(part_path)
    if len(frame) != manifest.row_count:
        raise ValueError("RAPM stint row count changed")
    _require_columns(frame, set(RAPM_STINT_COLUMNS), "RAPM stints")
    if not frame["possessions"].gt(0).all():
        raise ValueError("RAPM stints must have positive possession exposure")
    if set(frame["season_type"].astype(str)) != {"regular"}:
        raise ValueError("RAPM stints must contain regular-season games only")
    return manifest


def build_rapm_stints_from_processed_games(
    game_ids: Sequence[str],
    *,
    processed_dir: Path | str = Path("data/processed"),
) -> pd.DataFrame:
    """Build RAPM stints directly from selected persisted per-game tables."""

    root = Path(processed_dir)
    selected = tuple(str(game_id) for game_id in game_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Selected game IDs must be non-empty and unique")
    lineup_paths = [root / "lineup_stints" / f"{game_id}.parquet" for game_id in selected]
    segment_paths = [
        root / "possession_segments" / f"{game_id}.parquet" for game_id in selected
    ]
    missing = [str(path) for path in (*lineup_paths, *segment_paths) if not path.is_file()]
    if missing:
        raise ValueError(f"Selected processed tables are missing: {missing[:3]}")
    return rapm_stints_frame(
        pd.concat([pd.read_parquet(path) for path in lineup_paths], ignore_index=True),
        pd.concat([pd.read_parquet(path) for path in segment_paths], ignore_index=True),
    )


def build_rapm_stints_from_legacy_processed_games(
    game_ids: Sequence[str],
    *,
    catalog_path: Path | str = Path("data/catalog/games.parquet"),
    processed_dir: Path | str = Path("data/processed"),
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Adapt cached pre-envelope per-game tables into valid RAPM source rows."""

    catalog = pd.read_parquet(catalog_path).set_index("game_id")
    root = Path(processed_dir)
    outputs: list[pd.DataFrame] = []
    excluded: list[str] = []
    for game_id in game_ids:
        key = str(game_id)
        try:
            game = catalog.loc[key]
            stints = pd.read_parquet(root / "lineup_stints" / f"{key}.parquet")
            segments = pd.read_parquet(root / "possession_segments" / f"{key}.parquet")
            common = {
                "season": str(game.season),
                "season_type": str(game.season_type),
                "game_date": game.game_date,
                "game_time_utc": game.game_time_utc,
                "catalog_home_team_id": int(game.home_team_id),
                "catalog_away_team_id": int(game.away_team_id),
                "catalog_home_team_tricode": str(game.home_team_tricode),
                "catalog_away_team_tricode": str(game.away_team_tricode),
            }
            for column, value in common.items():
                stints[column] = value
                segments[column] = value
            stints["quality_status"] = "warning"
            stints["quality_issue_codes_json"] = "[]"
            stints["source_build_run_id"] = "legacy_processed_playoff"
            stints["processing_code_version"] = "legacy_processed_playoff"
            stints["play_by_play_sha256"] = ""
            stints["boxscore_sha256"] = ""
            outputs.append(rapm_stints_frame(stints, segments))
        except (KeyError, OSError, ValueError):
            excluded.append(key)
    if not outputs:
        raise ValueError("No valid legacy processed RAPM stints were produced")
    return pd.concat(outputs, ignore_index=True), tuple(sorted(excluded))


def build_rapm_stints_from_curated_games(
    season: str,
    *,
    curated_dir: Path | str = Path("data/curated"),
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build historical RAPM stints while excluding invalid game-level inputs.

    Historical curated partitions can contain a small number of games produced
    before a lineup or possession reconstruction repair.  RAPM requires exact
    point conservation, so an invalid game is excluded rather than allowing it
    to contaminate a season-wide design matrix.  The caller persists the
    returned game IDs as part of its run provenance.
    """

    layout = CuratedDatasetLayout(Path(curated_dir))
    partition_kwargs = {"season": season, "season_type": "regular"}
    lineup_dir = layout.partition_dir(CuratedPartition(table="lineup_stints", **partition_kwargs))
    segment_dir = layout.partition_dir(
        CuratedPartition(table="possession_segments", **partition_kwargs)
    )
    lineup_stints = pd.read_parquet(lineup_dir)
    segments = pd.read_parquet(segment_dir)
    outputs: list[pd.DataFrame] = []
    excluded: list[str] = []
    segment_games = {
        str(game_id): frame for game_id, frame in segments.groupby("game_id", sort=False)
    }
    for game_id, game_stints in lineup_stints.groupby("game_id", sort=False):
        key = str(game_id)
        game_segments = segment_games.get(key)
        if game_segments is None:
            excluded.append(key)
            continue
        try:
            outputs.append(rapm_stints_frame(game_stints, game_segments))
        except ValueError:
            excluded.append(key)
    if not outputs:
        raise ValueError(f"No valid RAPM stints were produced for {season}")
    return (
        pd.concat(outputs, ignore_index=True).sort_values(
            ["game_time_utc", "game_id", "stint_index"], kind="stable"
        ).reset_index(drop=True),
        tuple(sorted(excluded)),
    )


def _assign_segments_to_stints(
    stints: pd.DataFrame,
    segments: pd.DataFrame,
) -> pd.Series:
    assignments = pd.Series(index=segments.index, dtype="int64")
    stint_games = {game_id: frame for game_id, frame in stints.groupby("game_id")}
    for game_id, game_segments in segments.groupby("game_id", sort=False):
        if game_id not in stint_games:
            raise ValueError(f"Possession segments have no lineup stints: {game_id}")
        game_stints = stint_games[game_id].sort_values(
            ["start_event_index", "stint_index"],
            kind="stable",
        )
        starts = game_stints["start_event_index"].to_numpy()
        positions = (
            np.searchsorted(
                starts,
                game_segments["start_event_index"].to_numpy(),
                side="right",
            )
            - 1
        )
        if (positions < 0).any():
            raise ValueError(f"Possession segment precedes the first stint: {game_id}")
        selected = game_stints.iloc[positions]
        if not (
            game_segments["end_event_index"].to_numpy() <= selected["end_event_index"].to_numpy()
        ).all():
            raise ValueError(f"Possession segment crosses a lineup stint: {game_id}")
        for segment_row, stint_row in zip(
            game_segments.itertuples(),
            selected.itertuples(),
            strict=True,
        ):
            if tuple(segment_row.home_player_ids) != tuple(stint_row.home_player_ids) or tuple(
                segment_row.away_player_ids
            ) != tuple(stint_row.away_player_ids):
                raise ValueError(f"Possession segment lineup does not match its stint: {game_id}")
        assignments.loc[game_segments.index] = selected["stint_index"].to_numpy()
    if assignments.isna().any():
        raise ValueError("Not every possession segment was assigned to a stint")
    return assignments.astype("int64")


def _validate_source_frames(
    stints: pd.DataFrame,
    segments: pd.DataFrame,
) -> None:
    if stints.empty or segments.empty:
        raise ValueError("RAPM source frames cannot be empty")
    season_types = set(stints["season_type"].astype(str))
    if season_types not in ({"regular"}, {"playoffs"}):
        raise ValueError("RAPM source frames must contain one supported season type")
    if set(stints["game_id"].astype(str)) != set(segments["game_id"].astype(str)):
        raise ValueError("Lineup stint and possession segment games do not match")
    keys = stints[["game_id", "stint_index"]]
    if keys.duplicated().any():
        raise ValueError("Lineup stint keys must be unique")
    for side in ("home", "away"):
        sizes = stints[f"{side}_player_ids"].map(len)
        if not sizes.eq(5).all():
            raise ValueError("RAPM requires exactly five players per team")
        if not stints[f"{side}_player_ids"].map(lambda ids: len(set(ids)) == 5).all():
            raise ValueError("RAPM lineups cannot contain duplicate players")
    overlaps = [
        bool(set(home) & set(away))
        for home, away in zip(
            stints["home_player_ids"],
            stints["away_player_ids"],
            strict=True,
        )
    ]
    if any(overlaps):
        raise ValueError("A player cannot appear for both teams in one stint")


def _type_rapm_stints(frame: pd.DataFrame) -> pd.DataFrame:
    for column in (
        "season",
        "season_type",
        "game_id",
        "home_team_tricode",
        "away_team_tricode",
        "quality_status",
        "quality_issue_codes_json",
        "source_build_run_id",
        "processing_code_version",
        "play_by_play_sha256",
        "boxscore_sha256",
    ):
        frame[column] = frame[column].astype("string")
    for column in (
        "schema_version",
        "stint_index",
        "period",
        "home_team_id",
        "away_team_id",
        "points_home",
        "points_away",
        "home_margin",
    ):
        frame[column] = frame[column].astype("int64")
    for column in (
        "start_elapsed_game_seconds",
        "end_elapsed_game_seconds",
        "duration_seconds",
        "home_offensive_possessions",
        "away_offensive_possessions",
        "possessions",
        "target_home_net_rating",
    ):
        frame[column] = frame[column].astype("float64")
    frame["game_time_utc"] = pd.to_datetime(frame["game_time_utc"], utc=True)
    return frame.sort_values(
        ["game_time_utc", "game_id", "stint_index"],
        kind="stable",
    ).reset_index(drop=True)


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
