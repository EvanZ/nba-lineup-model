"""Era-standardized retrospective player-season RAPM comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.tracking import track_completed_run

DEFAULT_TARGET_SEASON = "2025-26"
DEFAULT_MINIMUM_MINUTES = 2_000.0
DEFAULT_REFERENCE_MINUTES = 2_000.0
REGULATION_TEAM_PLAYER_MINUTES_PER_GAME = 5.0 * 48.0


def build_era_comparison(
    *,
    forward_lagged_run_dir: Path | str,
    forward_calibration_run_dir: Path | str,
    analytical_dir: Path | str = Path("data/analytical"),
    player_catalog_path: Path | str = Path("data/catalog/players.parquet"),
    artifacts_dir: Path | str = Path("artifacts/reports"),
    target_season: str = DEFAULT_TARGET_SEASON,
    minimum_minutes: float = DEFAULT_MINIMUM_MINUTES,
    reference_minutes: float = DEFAULT_REFERENCE_MINUTES,
) -> Path:
    """Create an era-standardized player-season comparison report.

    Each season's full-season RAPM is centered and scaled against that same
    season's reconstructed player exposure. The calibrated win conversion is
    a common rate, not an assertion that player effects are independent of role
    or teammates.
    """

    if minimum_minutes <= 0 or reference_minutes <= 0:
        raise ValueError("Minute thresholds must be positive")
    rapm_root = Path(forward_lagged_run_dir)
    calibration_root = Path(forward_calibration_run_dir)
    rapm_manifest = _read_manifest(rapm_root)
    calibration_manifest = _read_manifest(calibration_root)
    _validate_sources(rapm_manifest, calibration_manifest, target_season)
    calibration_parameters = pd.read_parquet(calibration_root / "target_model_parameters.parquet")
    if len(calibration_parameters) != 1 or "prior_rapm_slope" not in calibration_parameters:
        raise ValueError("Forward calibration must have exactly one target slope")
    win_pct_per_standardized_team_unit = float(calibration_parameters["prior_rapm_slope"].iloc[0])

    player_coefficients = _load_player_coefficients(rapm_root, target_season)
    exposure = _load_player_exposure(
        tuple(sorted(player_coefficients["season"].unique(), key=_season_start_year)),
        analytical_dir=analytical_dir,
    )
    catalog = _load_player_catalog(player_catalog_path)
    player_seasons = _standardize_player_seasons(player_coefficients, exposure, catalog)
    player_seasons["wins_above_average_actual_minutes"] = (
        win_pct_per_standardized_team_unit
        * player_seasons["era_standardized_rapm"]
        * player_seasons["minutes"]
        / REGULATION_TEAM_PLAYER_MINUTES_PER_GAME
    )
    player_seasons["wins_above_average_per_reference_minutes"] = (
        win_pct_per_standardized_team_unit
        * player_seasons["era_standardized_rapm"]
        * reference_minutes
        / REGULATION_TEAM_PLAYER_MINUTES_PER_GAME
    )
    player_seasons["qualified"] = player_seasons["minutes"].ge(minimum_minutes)
    qualified = player_seasons.loc[player_seasons["qualified"]].copy()
    qualified = qualified.sort_values(
        ["wins_above_average_per_reference_minutes", "minutes", "player_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    qualified["qualified_rank"] = np.arange(1, len(qualified) + 1)
    top_25 = qualified.head(25).copy()

    run_id = f"era-comparison-{target_season}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = Path(artifacts_dir) / "era_comparison" / target_season
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / run_id
    temporary_dir = root / f".{run_id}.tmp"
    temporary_dir.mkdir()
    try:
        player_seasons.to_parquet(temporary_dir / "player_season_comparisons.parquet", index=False)
        qualified.to_parquet(temporary_dir / "qualified_peak_seasons.parquet", index=False)
        top_25.to_parquet(temporary_dir / "top_25_qualified_peak_seasons.parquet", index=False)
        metadata = {
            "run_id": run_id,
            "season": target_season,
            "report": "era_standardized_player_season_rapm_comparison",
            "source_forward_lagged_rapm_run_id": rapm_manifest["run_id"],
            "source_forward_lagged_rapm_manifest_sha256": _sha256_file(rapm_root / "manifest.json"),
            "source_forward_calibration_run_id": calibration_manifest["run_id"],
            "source_forward_calibration_manifest_sha256": _sha256_file(
                calibration_root / "manifest.json"
            ),
            "seasons": sorted(player_seasons["season"].unique(), key=_season_start_year),
            "minimum_minutes": minimum_minutes,
            "reference_minutes": reference_minutes,
            "calibration_win_pct_per_standardized_team_unit": (win_pct_per_standardized_team_unit),
            "regulation_team_player_minutes_per_game": REGULATION_TEAM_PLAYER_MINUTES_PER_GAME,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (temporary_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        artifacts = [
            {
                "filename": path.name,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary_dir.iterdir())
            if path.is_file()
        ]
        (temporary_dir / "manifest.json").write_text(
            json.dumps({"schema_version": 1, **metadata, "artifacts": artifacts}, indent=2) + "\n"
        )
        temporary_dir.replace(output_dir)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    track_completed_run(output_dir)
    return output_dir


def _load_player_coefficients(root: Path, target_season: str) -> pd.DataFrame:
    historical = pd.read_parquet(root / "historical_player_coefficients.parquet")
    required_historical = {"season", "player_id", "rapm"}
    missing = required_historical - set(historical)
    if missing:
        raise ValueError(f"Historical coefficients missing columns: {sorted(missing)}")
    target = pd.read_parquet(root / "full_season_player_coefficients.parquet")
    required_target = {"player_id", "rapm"}
    missing_target = required_target - set(target)
    if missing_target:
        raise ValueError(
            f"Target full-season coefficients missing columns: {sorted(missing_target)}"
        )
    target = target.assign(season=target_season)
    output = pd.concat(
        [historical.loc[:, ["season", "player_id", "rapm"]], target],
        ignore_index=True,
    )
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype("int64")
    if output.duplicated(["season", "player_id"]).any():
        raise ValueError("Player-season RAPM coefficients must be unique")
    return output


def _load_player_exposure(seasons: tuple[str, ...], *, analytical_dir: Path | str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        frames.append(_player_season_exposure(stints))
    exposure = pd.concat(frames, ignore_index=True)
    if exposure.duplicated(["season", "player_id"]).any():
        raise ValueError("Player-season exposure must be unique")
    return exposure


def _player_season_exposure(stints: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "home_team_id",
        "away_team_id",
        "home_team_tricode",
        "away_team_tricode",
        "home_player_ids",
        "away_player_ids",
        "duration_seconds",
    }
    missing = required - set(stints)
    if missing:
        raise ValueError(f"RAPM stints missing columns for player exposure: {sorted(missing)}")
    if stints["duration_seconds"].lt(0).any():
        raise ValueError("RAPM stint duration must not be negative")
    positive = stints.loc[stints["duration_seconds"].gt(0)].copy()
    if positive.empty:
        raise ValueError("RAPM stints contain no positive-duration player exposure")
    frames: list[pd.DataFrame] = []
    for side in ("home", "away"):
        frame = positive.loc[
            :,
            [
                "season",
                f"{side}_team_id",
                f"{side}_team_tricode",
                f"{side}_player_ids",
                "duration_seconds",
            ],
        ].rename(
            columns={
                f"{side}_team_id": "team_id",
                f"{side}_team_tricode": "team_tricode",
                f"{side}_player_ids": "player_id",
            }
        )
        frames.append(frame.explode("player_id", ignore_index=True))
    player_team = pd.concat(frames, ignore_index=True)
    if player_team["player_id"].isna().any():
        raise ValueError("RAPM stint contains a null lineup player")
    player_team["player_id"] = pd.to_numeric(player_team["player_id"], errors="raise").astype(
        "int64"
    )
    player_team["team_id"] = pd.to_numeric(player_team["team_id"], errors="raise").astype("int64")
    player_team = player_team.groupby(
        ["season", "player_id", "team_id", "team_tricode"], as_index=False
    ).agg(seconds=("duration_seconds", "sum"))
    primary_team = player_team.sort_values(
        ["season", "player_id", "seconds", "team_id"],
        ascending=[True, True, False, True],
        kind="stable",
    ).drop_duplicates(["season", "player_id"], keep="first")
    totals = player_team.groupby(["season", "player_id"], as_index=False).agg(
        seconds=("seconds", "sum"), team_count=("team_id", "nunique")
    )
    output = totals.merge(
        primary_team.loc[:, ["season", "player_id", "team_id", "team_tricode"]],
        on=["season", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    output["minutes"] = output["seconds"] / 60.0
    return output


def _standardize_player_seasons(
    coefficients: pd.DataFrame, exposure: pd.DataFrame, catalog: pd.DataFrame
) -> pd.DataFrame:
    output = coefficients.merge(
        exposure,
        on=["season", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(output) != len(coefficients):
        raise ValueError("Every player-season RAPM coefficient requires lineup exposure")
    weighted = (
        output.assign(
            weighted_rapm=lambda frame: frame["rapm"] * frame["seconds"],
            weighted_rapm_squared=lambda frame: frame["rapm"] ** 2 * frame["seconds"],
        )
        .groupby("season", as_index=False)
        .agg(
            reference_seconds=("seconds", "sum"),
            weighted_rapm=("weighted_rapm", "sum"),
            weighted_rapm_squared=("weighted_rapm_squared", "sum"),
        )
    )
    weighted["season_rapm_mean"] = weighted["weighted_rapm"] / weighted["reference_seconds"]
    weighted["season_rapm_scale"] = np.sqrt(
        weighted["weighted_rapm_squared"] / weighted["reference_seconds"]
        - weighted["season_rapm_mean"] ** 2
    )
    output = output.merge(
        weighted.loc[:, ["season", "season_rapm_mean", "season_rapm_scale"]],
        on="season",
        how="inner",
        validate="many_to_one",
    )
    if output["season_rapm_scale"].le(1e-12).any():
        raise ValueError("Season RAPM scale must be positive")
    output["era_standardized_rapm"] = (output["rapm"] - output["season_rapm_mean"]) / output[
        "season_rapm_scale"
    ]
    output["season_player_percentile"] = (
        output.groupby("season")["era_standardized_rapm"].rank(pct=True) * 100.0
    )
    output = output.merge(catalog, on="player_id", how="left", validate="many_to_one")
    output["player_name"] = output["display_name"].fillna(output["player_id"].astype(str))
    return (
        output.loc[
            :,
            [
                "season",
                "player_id",
                "player_name",
                "team_id",
                "team_tricode",
                "team_count",
                "seconds",
                "minutes",
                "rapm",
                "season_rapm_mean",
                "season_rapm_scale",
                "era_standardized_rapm",
                "season_player_percentile",
            ],
        ]
        .sort_values(["season", "player_id"], kind="stable")
        .reset_index(drop=True)
    )


def _load_player_catalog(path: Path | str) -> pd.DataFrame:
    catalog = pd.read_parquet(path)
    required = {"player_id", "display_name"}
    missing = required - set(catalog)
    if missing:
        raise ValueError(f"Player catalog missing columns: {sorted(missing)}")
    output = catalog.loc[:, ["player_id", "display_name"]].copy()
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype("int64")
    if output["player_id"].duplicated().any():
        raise ValueError("Player catalog must have unique player IDs")
    return output


def _validate_sources(
    rapm_manifest: dict[str, object], calibration_manifest: dict[str, object], target_season: str
) -> None:
    if rapm_manifest.get("model") != "forward_lagged_prior_centered_ridge_rapm":
        raise ValueError("Era comparison requires a forward lagged-prior RAPM run")
    if calibration_manifest.get("model") != "forward_usage_conditional_prior_rapm_win_calibration":
        raise ValueError("Era comparison requires a forward RAPM win calibration run")
    if rapm_manifest.get("target_season") != target_season:
        raise ValueError("Forward RAPM run target season does not match era comparison target")
    if calibration_manifest.get("target_season") != target_season:
        raise ValueError("Forward calibration target season does not match era comparison target")
    if calibration_manifest.get("source_forward_lagged_rapm_run_id") != rapm_manifest.get("run_id"):
        raise ValueError("Forward calibration was not built from the supplied RAPM run")


def _read_manifest(root: Path) -> dict[str, object]:
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Manifest not found: {path}")
    return json.loads(path.read_text())


def _season_start_year(season: str) -> int:
    return int(str(season)[:4])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main() -> None:
    """Build the era-standardized player-season comparison report."""

    parser = argparse.ArgumentParser(description="Build an era-standardized RAPM comparison")
    parser.add_argument("--forward-lagged-run-dir", required=True)
    parser.add_argument("--forward-calibration-run-dir", required=True)
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--player-catalog-path", default="data/catalog/players.parquet")
    parser.add_argument("--artifacts-dir", default="artifacts/reports")
    parser.add_argument("--target-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--minimum-minutes", type=float, default=DEFAULT_MINIMUM_MINUTES)
    parser.add_argument("--reference-minutes", type=float, default=DEFAULT_REFERENCE_MINUTES)
    args = parser.parse_args()
    print(
        build_era_comparison(
            forward_lagged_run_dir=args.forward_lagged_run_dir,
            forward_calibration_run_dir=args.forward_calibration_run_dir,
            analytical_dir=args.analytical_dir,
            player_catalog_path=args.player_catalog_path,
            artifacts_dir=args.artifacts_dir,
            target_season=args.target_season,
            minimum_minutes=args.minimum_minutes,
            reference_minutes=args.reference_minutes,
        )
    )


if __name__ == "__main__":
    main()
