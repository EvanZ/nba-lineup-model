"""Persist coverage of the historical RAPM modeling inputs against the catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("data/audit/historical_modeling_coverage")
DEFAULT_WARNING_THRESHOLD_PCT = 95.0
DEFAULT_CRITICAL_THRESHOLD_PCT = 90.0


@dataclass(frozen=True)
class ModelingCoverageAudit:
    """Materialized coverage tables and provenance for historical model inputs."""

    season_coverage: pd.DataFrame
    team_coverage: pd.DataFrame
    paths: dict[str, Path]


def build_modeling_coverage_audit(
    *,
    catalog_path: Path | str = Path("data/catalog/games.parquet"),
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    warning_threshold_pct: float = DEFAULT_WARNING_THRESHOLD_PCT,
    critical_threshold_pct: float = DEFAULT_CRITICAL_THRESHOLD_PCT,
) -> ModelingCoverageAudit:
    """Audit regular-season RAPM stint coverage by season and team.

    The audit deliberately measures the actual ``rapm_stints`` inputs rather
    than raw-source availability. That makes its ``modeled_games`` counts the
    correct coverage denominator for historical ratings and the player panel.
    """

    if not 0.0 <= critical_threshold_pct <= warning_threshold_pct <= 100.0:
        raise ValueError("Coverage thresholds must satisfy 0 <= critical <= warning <= 100")

    catalog_path = Path(catalog_path)
    analytical_dir = Path(analytical_dir)
    curated_dir = Path(curated_dir)
    output_dir = Path(output_dir)
    catalog = pd.read_parquet(catalog_path)
    required_catalog = {
        "season",
        "season_type",
        "game_id",
        "home_team_id",
        "home_team_tricode",
        "away_team_id",
        "away_team_tricode",
    }
    missing_catalog = required_catalog - set(catalog.columns)
    if missing_catalog:
        raise ValueError(f"Game catalog missing columns: {sorted(missing_catalog)}")
    catalog = catalog.loc[catalog["season_type"].eq("regular")].copy()
    if catalog.empty:
        raise ValueError("Game catalog has no regular-season rows")

    season_rows: list[dict[str, object]] = []
    team_frames: list[pd.DataFrame] = []
    source_manifests: list[dict[str, object]] = []
    for season in sorted(catalog["season"].astype(str).unique(), key=lambda value: int(value[:4])):
        expected_games = catalog.loc[catalog["season"].eq(season)].copy()
        stint_dir = analytical_dir / "rapm_stints" / season / "regular"
        stint_manifest = stint_dir / "_manifest.json"
        part_paths = sorted(stint_dir.glob("part-*.parquet"))
        if not stint_manifest.is_file() or not part_paths:
            raise FileNotFoundError(f"Missing regular RAPM stints for {season}: {stint_dir}")
        stints = pd.concat(
            [
                pd.read_parquet(
                    path,
                    columns=[
                        "game_id",
                        "home_team_id",
                        "home_team_tricode",
                        "away_team_id",
                        "away_team_tricode",
                    ],
                )
                for path in part_paths
            ],
            ignore_index=True,
        )
        modeled_games = _modeled_games(stints)
        expected_game_ids = set(expected_games["game_id"].astype(str))
        unexpected_game_ids = set(modeled_games["game_id"]) - expected_game_ids
        if unexpected_game_ids:
            example = ", ".join(sorted(unexpected_game_ids)[:5])
            raise ValueError(
                f"RAPM stints for {season} contain games outside the catalog: {example}"
            )

        expected_team_games = _catalog_team_games(expected_games)
        modeled_team_games = _modeled_team_games(modeled_games)
        team_coverage = expected_team_games.merge(
            modeled_team_games,
            on=["team_id", "team"],
            how="left",
            validate="one_to_one",
        ).fillna({"modeled_games": 0})
        team_coverage["modeled_games"] = team_coverage["modeled_games"].astype("int64")
        team_coverage["missing_games"] = (
            team_coverage["catalog_games"] - team_coverage["modeled_games"]
        )
        team_coverage["coverage_pct"] = (
            100.0 * team_coverage["modeled_games"] / team_coverage["catalog_games"]
        )
        team_coverage["coverage_status"] = team_coverage["coverage_pct"].map(
            lambda value: _coverage_status(
                float(value),
                warning_threshold_pct=warning_threshold_pct,
                critical_threshold_pct=critical_threshold_pct,
            )
        )
        team_coverage.insert(0, "season", season)
        team_frames.append(team_coverage)

        catalog_game_count = int(expected_games["game_id"].nunique())
        modeled_game_count = int(modeled_games["game_id"].nunique())
        player_manifest = curated_dir / "players" / season / "regular" / "_manifest.json"
        season_rows.append(
            {
                "season": season,
                "catalog_games": catalog_game_count,
                "modeled_games": modeled_game_count,
                "missing_games": catalog_game_count - modeled_game_count,
                "coverage_pct": 100.0 * modeled_game_count / catalog_game_count,
                "coverage_status": _coverage_status(
                    100.0 * modeled_game_count / catalog_game_count,
                    warning_threshold_pct=warning_threshold_pct,
                    critical_threshold_pct=critical_threshold_pct,
                ),
                "minimum_team_coverage_pct": float(team_coverage["coverage_pct"].min()),
                "median_team_coverage_pct": float(team_coverage["coverage_pct"].median()),
                "rapm_stints_manifest_sha256": _sha256_file(stint_manifest),
                "players_manifest_sha256": (
                    _sha256_file(player_manifest) if player_manifest.is_file() else None
                ),
            }
        )
        source_manifests.append(
            {
                "season": season,
                "rapm_stints_manifest": str(stint_manifest),
                "rapm_stints_manifest_sha256": _sha256_file(stint_manifest),
                "players_manifest": str(player_manifest),
                "players_manifest_sha256": (
                    _sha256_file(player_manifest) if player_manifest.is_file() else None
                ),
            }
        )

    season_coverage = pd.DataFrame(season_rows).sort_values("season", kind="stable")
    team_coverage = pd.concat(team_frames, ignore_index=True).sort_values(
        ["season", "coverage_pct", "team"], kind="stable"
    )
    paths = {
        "season_coverage": output_dir / "season_coverage.parquet",
        "team_coverage": output_dir / "team_coverage.parquet",
        "manifest": output_dir / "manifest.json",
    }
    _write_parquet(season_coverage, paths["season_coverage"])
    _write_parquet(team_coverage, paths["team_coverage"])
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "regular-season historical RAPM stint inputs",
        "definitions": {
            "catalog_games": "Final regular-season games listed in data/catalog/games.parquet.",
            "modeled_games": "Distinct catalog games represented in data/analytical/rapm_stints.",
            "modeled_gp": (
                "A player's positive-minute box-score games within the modeled subset; "
                "not official NBA GP."
            ),
        },
        "thresholds": {
            "warning_coverage_pct": warning_threshold_pct,
            "critical_coverage_pct": critical_threshold_pct,
        },
        "inputs": {
            "catalog_path": str(catalog_path),
            "catalog_sha256": _sha256_file(catalog_path),
            "analytical_dir": str(analytical_dir),
            "curated_dir": str(curated_dir),
            "season_manifests": source_manifests,
        },
        "outputs": {
            "season_coverage": {
                "path": str(paths["season_coverage"]),
                "sha256": _sha256_file(paths["season_coverage"]),
            },
            "team_coverage": {
                "path": str(paths["team_coverage"]),
                "sha256": _sha256_file(paths["team_coverage"]),
            },
        },
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return ModelingCoverageAudit(season_coverage, team_coverage, paths)


def _modeled_games(stints: pd.DataFrame) -> pd.DataFrame:
    """Return one row per game from a regular RAPM stint partition."""

    required = {
        "game_id",
        "home_team_id",
        "home_team_tricode",
        "away_team_id",
        "away_team_tricode",
    }
    missing = required - set(stints.columns)
    if missing:
        raise ValueError(f"RAPM stints missing columns: {sorted(missing)}")
    games = stints.loc[:, sorted(required)].drop_duplicates("game_id")
    if games["game_id"].duplicated().any():
        raise ValueError("RAPM stints have conflicting team identities within a game")
    games["game_id"] = games["game_id"].astype(str)
    return games


def _catalog_team_games(catalog_games: pd.DataFrame) -> pd.DataFrame:
    """Expand catalog games to their expected team-game observations."""

    home = catalog_games.loc[:, ["game_id", "home_team_id", "home_team_tricode"]].rename(
        columns={"home_team_id": "team_id", "home_team_tricode": "team"}
    )
    away = catalog_games.loc[:, ["game_id", "away_team_id", "away_team_tricode"]].rename(
        columns={"away_team_id": "team_id", "away_team_tricode": "team"}
    )
    return (
        pd.concat([home, away], ignore_index=True)
        .assign(game_id=lambda frame: frame["game_id"].astype(str))
        .groupby(["team_id", "team"], as_index=False, sort=True)
        .agg(catalog_games=("game_id", "nunique"))
    )


def _modeled_team_games(modeled_games: pd.DataFrame) -> pd.DataFrame:
    """Expand modeled games to their observed team-game observations."""

    home = modeled_games.loc[:, ["game_id", "home_team_id", "home_team_tricode"]].rename(
        columns={"home_team_id": "team_id", "home_team_tricode": "team"}
    )
    away = modeled_games.loc[:, ["game_id", "away_team_id", "away_team_tricode"]].rename(
        columns={"away_team_id": "team_id", "away_team_tricode": "team"}
    )
    return (
        pd.concat([home, away], ignore_index=True)
        .groupby(["team_id", "team"], as_index=False, sort=True)
        .agg(modeled_games=("game_id", "nunique"))
    )


def _coverage_status(
    coverage_pct: float,
    *,
    warning_threshold_pct: float,
    critical_threshold_pct: float,
) -> str:
    if coverage_pct < critical_threshold_pct:
        return "critical"
    if coverage_pct < warning_threshold_pct:
        return "warning"
    return "pass"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit historical regular-season RAPM modeling coverage."
    )
    parser.add_argument("--catalog-path", default="data/catalog/games.parquet")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--warning-threshold-pct",
        type=float,
        default=DEFAULT_WARNING_THRESHOLD_PCT,
    )
    parser.add_argument(
        "--critical-threshold-pct",
        type=float,
        default=DEFAULT_CRITICAL_THRESHOLD_PCT,
    )
    args = parser.parse_args()
    result = build_modeling_coverage_audit(
        catalog_path=args.catalog_path,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        output_dir=args.output_dir,
        warning_threshold_pct=args.warning_threshold_pct,
        critical_threshold_pct=args.critical_threshold_pct,
    )
    critical_seasons = int(result.season_coverage["coverage_status"].eq("critical").sum())
    warning_seasons = int(result.season_coverage["coverage_status"].eq("warning").sum())
    print(
        f"Audited {len(result.season_coverage)} seasons: {critical_seasons} critical, "
        f"{warning_seasons} warning"
    )
    for name, path in result.paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
