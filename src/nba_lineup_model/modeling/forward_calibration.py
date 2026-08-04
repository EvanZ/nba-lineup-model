"""Forward, usage-conditional calibration from lagged RAPM to team wins."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nba_lineup_model.modeling.prior_rapm import HISTORICAL_SEASONS
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.tracking import track_completed_run

DEFAULT_TARGET_SEASON = "2025-26"
DEFAULT_WARMUP_SEASONS = 4


@dataclass(frozen=True)
class WinCalibrationModel:
    """One weighted linear mapping from team RAPM to regular-season win percentage."""

    intercept: float
    prior_rapm_slope: float
    training_team_season_count: int
    training_seasons: tuple[str, ...]

    def predict(self, team_prior_rapm: pd.Series | np.ndarray) -> np.ndarray:
        """Predict a bounded regular-season win percentage."""

        values = np.asarray(team_prior_rapm, dtype=float)
        return np.clip(self.intercept + self.prior_rapm_slope * values, 0.0, 1.0)


def fit_win_calibration(training: pd.DataFrame) -> WinCalibrationModel:
    """Fit a game-count-weighted linear win calibration on completed seasons.

    The target is team regular-season win percentage. A team-season is weighted
    by its number of games so shortened seasons retain their appropriate scale.
    This intentionally has no player-age, draft, or roster-quality covariates.
    """

    _validate_team_seasons(training)
    weights = np.sqrt(training["games"].to_numpy(dtype=float))
    design = np.column_stack(
        [np.ones(len(training), dtype=float), training["team_prior_rapm"].to_numpy(dtype=float)]
    )
    target = training["win_pct"].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * weights[:, None], target * weights, rcond=None
    )
    if rank != 2:
        raise ValueError("Win calibration design is rank deficient")
    return WinCalibrationModel(
        intercept=float(coefficients[0]),
        prior_rapm_slope=float(coefficients[1]),
        training_team_season_count=len(training),
        training_seasons=tuple(sorted(training["season"].unique(), key=_season_start_year)),
    )


def build_team_season_inputs(
    prior_by_season: dict[str, pd.DataFrame],
    *,
    analytical_dir: Path | str = Path("data/analytical"),
) -> pd.DataFrame:
    """Aggregate frozen player priors and realized stint seconds to team-seasons."""

    frames: list[pd.DataFrame] = []
    for season in sorted(prior_by_season, key=_season_start_year):
        priors = prior_by_season[season]
        _validate_player_priors(priors, season)
        prior_source_season = priors["prior_source_season"].iloc[0]
        source_stints = read_rapm_stints(prior_source_season, analytical_dir=analytical_dir)
        prior_mean, prior_scale = _prior_reference(priors, source_stints)
        stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        exposures = _player_team_exposure(stints)
        outcomes = _team_outcomes(stints)
        merged = exposures.merge(
            priors.loc[:, ["player_id", "prior_rapm"]],
            on="player_id",
            how="left",
            validate="many_to_one",
        )
        merged["prior_available"] = merged["prior_rapm"].notna()
        merged["prior_rapm"] = merged["prior_rapm"].fillna(0.0)
        merged["standardized_prior_rapm"] = (merged["prior_rapm"] - prior_mean) / prior_scale
        team = (
            merged.assign(
                weighted_prior=lambda frame: frame["player_seconds"] * frame["prior_rapm"],
                weighted_standardized_prior=lambda frame: (
                    frame["player_seconds"] * frame["standardized_prior_rapm"]
                ),
                prior_seconds=lambda frame: np.where(
                    frame["prior_available"], frame["player_seconds"], 0.0
                ),
            )
            .groupby(["season", "team_id", "team_tricode"], as_index=False, sort=True)
            .agg(
                team_player_seconds=("player_seconds", "sum"),
                weighted_prior=("weighted_prior", "sum"),
                weighted_standardized_prior=("weighted_standardized_prior", "sum"),
                prior_seconds=("prior_seconds", "sum"),
                player_count=("player_id", "nunique"),
            )
        )
        team["prior_source_season"] = prior_source_season
        team["prior_reference_mean"] = prior_mean
        team["prior_reference_scale"] = prior_scale
        team["team_prior_rapm_raw"] = team["weighted_prior"] / team["team_player_seconds"]
        team["team_prior_rapm"] = team["weighted_standardized_prior"] / team["team_player_seconds"]
        team["prior_exposure_fraction"] = team["prior_seconds"] / team["team_player_seconds"]
        team = team.merge(
            outcomes,
            on=["season", "team_id", "team_tricode"],
            how="inner",
            validate="one_to_one",
        )
        if len(team) != len(outcomes):
            raise ValueError(f"{season} has team outcomes without lineup exposure")
        frames.append(team)
    result = pd.concat(frames, ignore_index=True)
    _validate_team_seasons(result)
    return result.sort_values(["season", "team_id"], kind="stable").reset_index(drop=True)


def build_prior_by_season(
    forward_lagged_run_dir: Path | str,
    *,
    target_season: str = DEFAULT_TARGET_SEASON,
) -> dict[str, pd.DataFrame]:
    """Recover the pre-season lagged RAPM tables from an immutable RAPM run."""

    root = Path(forward_lagged_run_dir)
    historical = pd.read_parquet(root / "historical_player_coefficients.parquet")
    required_historical = {"season", "player_id", "rapm"}
    missing = required_historical - set(historical)
    if missing:
        raise ValueError(f"Historical coefficients missing columns: {sorted(missing)}")
    available = set(historical["season"].astype(str))
    expected = set(HISTORICAL_SEASONS)
    if not expected.issubset(available):
        raise ValueError("Forward RAPM run does not contain every historical prior season")

    priors: dict[str, pd.DataFrame] = {}
    ordered = tuple(HISTORICAL_SEASONS)
    for previous, season in zip(ordered[:-1], ordered[1:], strict=True):
        priors[season] = (
            historical.loc[historical["season"].eq(previous), ["player_id", "rapm"]]
            .rename(columns={"rapm": "prior_rapm"})
            .assign(prior_source_season=previous)
            .copy()
        )
    target = pd.read_parquet(root / "target_player_priors.parquet")
    required_target = {"player_id", "prior_rapm_mean"}
    missing_target = required_target - set(target)
    if missing_target:
        raise ValueError(f"Target prior table missing columns: {sorted(missing_target)}")
    target_source = ordered[-1]
    expected_target = historical.loc[
        historical["season"].eq(target_source), ["player_id", "rapm"]
    ].rename(columns={"rapm": "prior_rapm"})
    available_target = target.loc[target["prior_available"], ["player_id", "prior_rapm_mean"]]
    checked = available_target.merge(
        expected_target,
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    if checked["prior_rapm"].isna().any() or not np.allclose(
        checked["prior_rapm_mean"], checked["prior_rapm"]
    ):
        raise ValueError(
            f"Target frozen priors do not match the {target_source} forward RAPM state"
        )
    priors[target_season] = expected_target.assign(prior_source_season=target_source)
    return priors


def run_forward_calibration(
    *,
    forward_lagged_run_dir: Path | str,
    analytical_dir: Path | str = Path("data/analytical"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    target_season: str = DEFAULT_TARGET_SEASON,
    warmup_seasons: int = DEFAULT_WARMUP_SEASONS,
) -> Path:
    """Create forward team-win calibration artifacts from frozen prior RAPM."""

    if warmup_seasons < 1:
        raise ValueError("Warmup seasons must be positive")
    source_run = Path(forward_lagged_run_dir)
    source_manifest = source_run / "manifest.json"
    if not source_manifest.is_file():
        raise ValueError(f"Forward RAPM manifest not found: {source_manifest}")
    source_metadata = json.loads(source_manifest.read_text())
    if source_metadata.get("model") != "forward_lagged_prior_centered_ridge_rapm":
        raise ValueError("Forward calibration requires a forward lagged-prior RAPM run")

    priors = build_prior_by_season(source_run, target_season=target_season)
    team_seasons = build_team_season_inputs(priors, analytical_dir=analytical_dir)
    ordered_seasons = tuple(sorted(team_seasons["season"].unique(), key=_season_start_year))
    if ordered_seasons[-1] != target_season:
        raise ValueError("Target season is missing from team-season calibration inputs")

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []
    for index, season in enumerate(ordered_seasons):
        if index < warmup_seasons:
            continue
        training_seasons = ordered_seasons[:index]
        training = team_seasons.loc[team_seasons["season"].isin(training_seasons)].copy()
        evaluation = team_seasons.loc[team_seasons["season"].eq(season)].copy()
        model = fit_win_calibration(training)
        predictions = _prediction_frame(evaluation, model, training_seasons)
        prediction_frames.append(predictions)
        metric_rows.append(_metric_row(predictions, model))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values("season", kind="stable").reset_index(drop=True)
    final_predictions = predictions.loc[predictions["season"].eq(target_season)].copy()
    final_metrics = metrics.loc[metrics["season"].eq(target_season)].copy()
    if len(final_predictions) == 0 or len(final_metrics) != 1:
        raise ValueError("Forward calibration did not produce the target-season evaluation")
    final_model = fit_win_calibration(team_seasons.loc[team_seasons["season"].ne(target_season)])

    run_id = (
        f"forward-calibration-{target_season}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    root = Path(artifacts_dir) / "forward_calibration" / target_season
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / run_id
    temporary_dir = root / f".{run_id}.tmp"
    temporary_dir.mkdir()
    try:
        team_seasons.to_parquet(temporary_dir / "team_season_inputs.parquet", index=False)
        predictions.to_parquet(temporary_dir / "forward_predictions.parquet", index=False)
        final_predictions.to_parquet(temporary_dir / "target_predictions.parquet", index=False)
        metrics.to_parquet(temporary_dir / "season_metrics.parquet", index=False)
        final_metrics.to_parquet(temporary_dir / "target_metrics.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "intercept": final_model.intercept,
                    "prior_rapm_slope": final_model.prior_rapm_slope,
                    "training_team_season_count": final_model.training_team_season_count,
                    "training_seasons": json.dumps(final_model.training_seasons),
                }
            ]
        ).to_parquet(temporary_dir / "target_model_parameters.parquet", index=False)
        metadata = {
            "run_id": run_id,
            "season": target_season,
            "model": "forward_usage_conditional_prior_rapm_win_calibration",
            "source_forward_lagged_rapm_run_id": source_metadata["run_id"],
            "source_forward_lagged_rapm_manifest_sha256": _sha256_file(source_manifest),
            "input_seasons": list(ordered_seasons),
            "target_season": target_season,
            "warmup_seasons": warmup_seasons,
            "calibration": {
                "form": "weighted_linear_probability",
                "target": "team_regular_season_win_percentage",
                "predictor": "realized_stint_seconds_weighted_frozen_prior_rapm",
                "weights": "team_regular_season_game_count",
                "cold_start_prior_rapm": 0.0,
                "prediction_bounds": [0.0, 1.0],
                "target_model_training_seasons": list(final_model.training_seasons),
            },
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


def _player_team_exposure(stints: pd.DataFrame) -> pd.DataFrame:
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
        raise ValueError(f"RAPM stints missing columns for team exposure: {sorted(missing)}")
    if stints["duration_seconds"].lt(0).any():
        raise ValueError("RAPM stint duration must not be negative for team exposure")
    stints = stints.loc[stints["duration_seconds"].gt(0)].copy()
    if stints.empty:
        raise ValueError("RAPM stints contain no positive-duration player exposure")
    frames: list[pd.DataFrame] = []
    for side in ("home", "away"):
        frame = (
            stints.loc[
                :,
                [
                    "season",
                    f"{side}_team_id",
                    f"{side}_team_tricode",
                    f"{side}_player_ids",
                    "duration_seconds",
                ],
            ]
            .rename(
                columns={
                    f"{side}_team_id": "team_id",
                    f"{side}_team_tricode": "team_tricode",
                    f"{side}_player_ids": "player_id",
                }
            )
            .explode("player_id", ignore_index=True)
        )
        if frame["player_id"].isna().any():
            raise ValueError("RAPM stint contains a null lineup player")
        frames.append(frame)
    exposure = pd.concat(frames, ignore_index=True)
    exposure["player_id"] = pd.to_numeric(exposure["player_id"], errors="raise").astype("int64")
    exposure["team_id"] = pd.to_numeric(exposure["team_id"], errors="raise").astype("int64")
    return (
        exposure.groupby(["season", "team_id", "team_tricode", "player_id"], as_index=False)
        .agg(player_seconds=("duration_seconds", "sum"))
        .sort_values(["season", "team_id", "player_id"], kind="stable")
        .reset_index(drop=True)
    )


def _team_outcomes(stints: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "game_id",
        "home_team_id",
        "away_team_id",
        "home_team_tricode",
        "away_team_tricode",
        "home_margin",
    }
    missing = required - set(stints)
    if missing:
        raise ValueError(f"RAPM stints missing columns for team outcomes: {sorted(missing)}")
    games = (
        stints.groupby(
            [
                "season",
                "game_id",
                "home_team_id",
                "away_team_id",
                "home_team_tricode",
                "away_team_tricode",
            ],
            as_index=False,
        )
        .agg(home_margin=("home_margin", "sum"))
        .reset_index(drop=True)
    )
    if games["home_margin"].eq(0).any():
        raise ValueError("Regular-season game has a tied final margin")
    home = games.loc[:, ["season", "home_team_id", "home_team_tricode", "home_margin"]].rename(
        columns={
            "home_team_id": "team_id",
            "home_team_tricode": "team_tricode",
            "home_margin": "margin",
        }
    )
    away = games.loc[:, ["season", "away_team_id", "away_team_tricode", "home_margin"]].rename(
        columns={
            "away_team_id": "team_id",
            "away_team_tricode": "team_tricode",
            "home_margin": "margin",
        }
    )
    away["margin"] *= -1.0
    records = pd.concat([home, away], ignore_index=True)
    records["win"] = records["margin"].gt(0).astype(int)
    output = (
        records.groupby(["season", "team_id", "team_tricode"], as_index=False)
        .agg(games=("win", "size"), wins=("win", "sum"), total_margin=("margin", "sum"))
        .reset_index(drop=True)
    )
    output["losses"] = output["games"] - output["wins"]
    output["win_pct"] = output["wins"] / output["games"]
    return output


def _prior_reference(priors: pd.DataFrame, source_stints: pd.DataFrame) -> tuple[float, float]:
    """Return the identified location and scale of one completed RAPM state."""

    source_exposure = _player_team_exposure(source_stints)
    weights = source_exposure.groupby("player_id", as_index=False).agg(
        source_player_seconds=("player_seconds", "sum")
    )
    reference = priors.loc[:, ["player_id", "prior_rapm"]].merge(
        weights,
        on="player_id",
        how="inner",
        validate="one_to_one",
    )
    if reference.empty:
        raise ValueError("Prior reference has no player exposure in its source season")
    player_weights = reference["source_player_seconds"].to_numpy(dtype=float)
    ratings = reference["prior_rapm"].to_numpy(dtype=float)
    mean = float(np.average(ratings, weights=player_weights))
    scale = float(np.sqrt(np.average((ratings - mean) ** 2, weights=player_weights)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Prior reference RAPM scale must be positive")
    return mean, scale


def _prediction_frame(
    evaluation: pd.DataFrame, model: WinCalibrationModel, training_seasons: tuple[str, ...]
) -> pd.DataFrame:
    result = evaluation.copy()
    result["predicted_win_pct"] = model.predict(result["team_prior_rapm"])
    result["predicted_wins"] = result["games"] * result["predicted_win_pct"]
    result["prediction_error_win_pct"] = result["predicted_win_pct"] - result["win_pct"]
    result["prediction_error_wins"] = result["predicted_wins"] - result["wins"]
    result["calibration_intercept"] = model.intercept
    result["calibration_prior_rapm_slope"] = model.prior_rapm_slope
    result["calibration_training_seasons"] = json.dumps(training_seasons)
    return result


def _metric_row(
    predictions: pd.DataFrame, model: WinCalibrationModel
) -> dict[str, float | int | str]:
    errors = predictions["prediction_error_win_pct"].to_numpy(dtype=float)
    win_errors = predictions["prediction_error_wins"].to_numpy(dtype=float)
    actual = predictions["win_pct"].to_numpy(dtype=float)
    predicted = predictions["predicted_win_pct"].to_numpy(dtype=float)
    mean_rmse = float(np.sqrt(np.mean((actual - 0.5) ** 2)))
    model_rmse = float(np.sqrt(np.mean(errors**2)))
    correlation = spearmanr(actual, predicted).statistic
    return {
        "season": str(predictions["season"].iloc[0]),
        "team_count": len(predictions),
        "games": int(predictions["games"].sum()),
        "training_season_count": len(model.training_seasons),
        "training_team_season_count": model.training_team_season_count,
        "intercept": model.intercept,
        "prior_rapm_slope": model.prior_rapm_slope,
        "win_pct_rmse": model_rmse,
        "win_pct_mae": float(np.mean(np.abs(errors))),
        "win_rmse": float(np.sqrt(np.mean(win_errors**2))),
        "win_mae": float(np.mean(np.abs(win_errors))),
        "win_pct_skill_vs_500": float(1.0 - model_rmse**2 / mean_rmse**2),
        "spearman_rank_correlation": float(correlation),
    }


def _validate_player_priors(priors: pd.DataFrame, season: str) -> None:
    required = {"player_id", "prior_rapm", "prior_source_season"}
    missing = required - set(priors)
    if missing:
        raise ValueError(f"{season} player priors missing columns: {sorted(missing)}")
    if priors["player_id"].duplicated().any():
        raise ValueError(f"{season} player priors contain duplicate player IDs")
    if priors["prior_source_season"].nunique() != 1:
        raise ValueError(f"{season} player priors must have one source season")
    if not np.isfinite(priors["prior_rapm"].to_numpy(dtype=float)).all():
        raise ValueError(f"{season} player priors contain non-finite values")


def _validate_team_seasons(team_seasons: pd.DataFrame) -> None:
    required = {
        "season",
        "team_id",
        "team_tricode",
        "team_prior_rapm",
        "games",
        "wins",
        "losses",
        "win_pct",
    }
    missing = required - set(team_seasons)
    if missing:
        raise ValueError(f"Team-season inputs missing columns: {sorted(missing)}")
    if team_seasons.duplicated(["season", "team_id"]).any():
        raise ValueError("Team-season inputs must be unique by season and team")
    numeric = ["team_prior_rapm", "games", "wins", "losses", "win_pct"]
    if not np.isfinite(team_seasons[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Team-season inputs contain non-finite values")
    if team_seasons["games"].le(0).any() or team_seasons["wins"].lt(0).any():
        raise ValueError("Team-season games and wins must be non-negative")
    if not np.allclose(team_seasons["wins"] + team_seasons["losses"], team_seasons["games"]):
        raise ValueError("Team-season records do not reconcile")


def _season_start_year(season: str) -> int:
    return int(str(season)[:4])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main() -> None:
    """Run the forward RAPM-to-wins calibration exemplar."""

    parser = argparse.ArgumentParser(description="Calibrate forward RAPM to team wins")
    parser.add_argument("--forward-lagged-run-dir", required=True)
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--target-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--warmup-seasons", type=int, default=DEFAULT_WARMUP_SEASONS)
    args = parser.parse_args()
    print(
        run_forward_calibration(
            forward_lagged_run_dir=args.forward_lagged_run_dir,
            analytical_dir=args.analytical_dir,
            artifacts_dir=args.artifacts_dir,
            target_season=args.target_season,
            warmup_seasons=args.warmup_seasons,
        )
    )


if __name__ == "__main__":
    main()
