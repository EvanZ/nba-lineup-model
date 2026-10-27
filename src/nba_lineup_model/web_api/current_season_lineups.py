"""Immutable current-season observed-lineup serving artifacts.

The player serving bundle and the observed lineup materialization are separate
products: ratings/minutes can be promoted even when play-by-play processing is
temporarily delayed.  This module binds a lineup snapshot to the exact promoted
player-state checksum and rejects a mismatched or partially written snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import numpy as np
import pandas as pd

LINEUPS_FILENAME = "lineups.parquet"
STINTS_FILENAME = "stints.parquet"
EXPOSURE_FILENAME = "player_team_exposure.parquet"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_CURRENT_SEASON_LINEUPS_ROOT = Path(
    "/var/lib/nba-gestalt/current-season-lineups"
)


@dataclass(frozen=True)
class CurrentSeasonLineupCatalog:
    """One verified current-season lineup snapshot."""

    season: str
    lineups: pd.DataFrame
    stints: pd.DataFrame
    player_team_exposure: pd.DataFrame
    metadata: dict[str, Any]
    bundle_path: Path


def load_current_season_lineup_catalog(
    *,
    serving_metadata: dict[str, Any],
    root: Path | str | None = None,
    bundle_path: Path | str | None = None,
) -> CurrentSeasonLineupCatalog | None:
    """Load the lineup artifact matching the promoted player serving state."""

    if not serving_metadata:
        return None
    configured = bundle_path or os.environ.get("GESTALT_CURRENT_SEASON_LINEUPS_BUNDLE")
    root_path = Path(
        root
        or os.environ.get(
            "GESTALT_CURRENT_SEASON_LINEUPS_ROOT",
            DEFAULT_CURRENT_SEASON_LINEUPS_ROOT,
        )
    ).resolve()
    if configured:
        selected = Path(configured).resolve()
    else:
        pointer_path = root_path / "latest.json"
        if not pointer_path.is_file():
            return None
        pointer = _read_json(pointer_path)
        selected = (root_path / str(pointer["bundle_path"])).resolve()
    manifest = verify_current_season_lineup_bundle(selected)
    identity = manifest["identity"]
    expected_sha = str(serving_metadata.get("bundle_sha256", ""))
    if str(identity["serving_bundle_sha256"]) != expected_sha:
        raise ValueError(
            "Current-season lineups do not match the promoted serving bundle"
        )
    if str(identity["season"]) != str(serving_metadata.get("season")):
        raise ValueError("Current-season lineup and serving seasons differ")
    if str(identity["as_of_cutoff_time"]) != str(
        serving_metadata.get("as_of_cutoff_time")
    ):
        raise ValueError("Current-season lineup and serving cutoffs differ")
    return CurrentSeasonLineupCatalog(
        season=str(identity["season"]),
        lineups=pd.read_parquet(selected / LINEUPS_FILENAME),
        stints=pd.read_parquet(selected / STINTS_FILENAME),
        player_team_exposure=pd.read_parquet(selected / EXPOSURE_FILENAME),
        metadata={**identity, "bundle_sha256": str(manifest["bundle_sha256"])},
        bundle_path=selected,
    )


def publish_current_season_lineup_bundle(
    *,
    lineups: pd.DataFrame,
    stints: pd.DataFrame,
    player_team_exposure: pd.DataFrame,
    serving_metadata: dict[str, Any],
    source_game_checksums: dict[str, str],
    root: Path | str,
) -> Path:
    """Publish and atomically promote one content-addressed lineup snapshot."""

    _validate_frames(lineups, stints, player_team_exposure)
    source_game_ids = sorted(stints["game_id"].astype(str).unique().tolist())
    if source_game_ids != sorted(source_game_checksums):
        raise ValueError("Lineup source-game checksums do not cover the exact stint games")
    root_path = Path(root).resolve()
    root_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=".lineups-", dir=root_path.parent))
    try:
        staging.mkdir(parents=True, exist_ok=True)
        lineups.to_parquet(staging / LINEUPS_FILENAME, index=False)
        stints.to_parquet(staging / STINTS_FILENAME, index=False)
        player_team_exposure.to_parquet(staging / EXPOSURE_FILENAME, index=False)
        artifact_hashes = {
            filename: _file_sha256(staging / filename)
            for filename in (LINEUPS_FILENAME, STINTS_FILENAME, EXPOSURE_FILENAME)
        }
        identity = {
            "schema_version": 1,
            "season": str(serving_metadata["season"]),
            "evidence_through_date": str(serving_metadata["evidence_through_date"]),
            "as_of_cutoff_time": str(serving_metadata["as_of_cutoff_time"]),
            "serving_bundle_sha256": str(serving_metadata["bundle_sha256"]),
            "source_game_ids": source_game_ids,
            "source_game_checksums": {
                game_id: str(source_game_checksums[game_id]) for game_id in source_game_ids
            },
            "stint_count": int(len(stints)),
            "lineup_count": int(len(lineups)),
            "player_team_exposure_count": int(len(player_team_exposure)),
            "artifact_sha256": artifact_hashes,
        }
        bundle_sha = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest = {
            "bundle_sha256": bundle_sha,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "identity": identity,
        }
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        relative = Path(
            f"season={identity['season']}/through_date={identity['evidence_through_date']}"
        ) / f"bundle_sha256={bundle_sha}"
        target = root_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            verify_current_season_lineup_bundle(target)
            shutil.rmtree(staging)
        else:
            staging.replace(target)
        pointer = {
            "schema_version": 1,
            "season": identity["season"],
            "evidence_through_date": identity["evidence_through_date"],
            "serving_bundle_sha256": identity["serving_bundle_sha256"],
            "bundle_sha256": bundle_sha,
            "bundle_path": str(relative),
            "promoted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        root_path.mkdir(parents=True, exist_ok=True)
        pointer_tmp = root_path / ".latest.json.tmp"
        pointer_tmp.write_text(
            json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pointer_tmp.replace(root_path / "latest.json")
        return target
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_current_season_lineup_bundle(path: Path | str) -> dict[str, Any]:
    """Validate files, hashes, schema, and content-addressed directory name."""

    bundle = Path(path).resolve()
    manifest = _read_json(bundle / MANIFEST_FILENAME)
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Current-season lineup manifest has no identity")
    expected_bundle_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if str(manifest.get("bundle_sha256")) != expected_bundle_sha:
        raise ValueError("Current-season lineup manifest checksum is invalid")
    if bundle.name != f"bundle_sha256={expected_bundle_sha}":
        raise ValueError("Current-season lineup directory is not content-addressed")
    artifact_hashes = identity.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("Current-season lineup artifact hashes are missing")
    for filename in (LINEUPS_FILENAME, STINTS_FILENAME, EXPOSURE_FILENAME):
        artifact_path = bundle / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Current-season lineup artifact is missing: {filename}")
        if _file_sha256(artifact_path) != str(artifact_hashes.get(filename)):
            raise ValueError(f"Current-season lineup artifact checksum failed: {filename}")
    lineups = pd.read_parquet(bundle / LINEUPS_FILENAME)
    stints = pd.read_parquet(bundle / STINTS_FILENAME)
    exposure = pd.read_parquet(bundle / EXPOSURE_FILENAME)
    _validate_frames(lineups, stints, exposure)
    if int(identity.get("stint_count", -1)) != len(stints):
        raise ValueError("Current-season lineup stint count does not match manifest")
    if int(identity.get("lineup_count", -1)) != len(lineups):
        raise ValueError("Current-season lineup count does not match manifest")
    source_ids = sorted(stints["game_id"].astype(str).unique().tolist())
    if source_ids != sorted(map(str, identity.get("source_game_ids", []))):
        raise ValueError("Current-season lineup source-game coverage is inconsistent")
    return manifest


def build_player_team_exposure(stints: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reconciled on-court exposure by current-season player and team."""

    rows: list[dict[str, Any]] = []
    for row in stints.itertuples(index=False):
        for side in ("home", "away"):
            team_id = int(getattr(row, f"{side}_team_id"))
            team = str(getattr(row, f"{side}_team_tricode"))
            for player_id in getattr(row, f"{side}_player_ids"):
                rows.append(
                    {
                        "season": str(getattr(row, "season", "2026-27")),
                        "game_id": str(row.game_id),
                        "team_id": team_id,
                        "team": team,
                        "player_id": int(player_id),
                        "possessions": float(row.possessions),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=["season", "team_id", "team", "player_id", "possessions", "games"]
        )
    return (
        frame.groupby(["season", "team_id", "team", "player_id"], as_index=False)
        .agg(possessions=("possessions", "sum"), games=("game_id", "nunique"))
        .sort_values(
            ["team", "possessions", "player_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _validate_frames(
    lineups: pd.DataFrame,
    stints: pd.DataFrame,
    player_team_exposure: pd.DataFrame,
) -> None:
    lineup_required = {
        "team_id",
        "team",
        "player_ids",
        "possessions",
        "games",
        "gestalt_score",
        "actual_net_rating",
    }
    stint_required = {
        "game_id",
        "home_team_id",
        "away_team_id",
        "home_team_tricode",
        "away_team_tricode",
        "home_player_ids",
        "away_player_ids",
        "possessions",
    }
    exposure_required = {"season", "team_id", "team", "player_id", "possessions", "games"}
    for label, frame, required in (
        ("lineups", lineups, lineup_required),
        ("stints", stints, stint_required),
        ("player-team exposure", player_team_exposure, exposure_required),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Current-season {label} are missing: {', '.join(missing)}")
    if lineups["player_ids"].map(lambda ids: len(list(ids)) != 5).any():
        raise ValueError("Current-season observed lineups must contain exactly five players")
    for frame, columns in (
        (lineups, ["possessions", "games", "gestalt_score", "actual_net_rating"]),
        (stints, ["possessions"]),
        (player_team_exposure, ["possessions", "games"]),
    ):
        values = frame.loc[:, columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Current-season lineup artifacts contain non-finite values")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
