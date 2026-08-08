"""Frozen lineup-composition residual on top of forward exposure-gated RAPM."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nba_lineup_model.evaluation.metrics import mean_squared_error
from nba_lineup_model.modeling.contextual_features import (
    contextual_feature_columns,
    lineup_context_features,
)
from nba_lineup_model.modeling.contextual_profiles import build_contextual_player_profiles
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.frozen_prior_evaluation import (
    _game_prediction_frame,
    _historical_team_seasons,
    _read_playoff_possessions,
    _read_regular_possessions,
    _recover_home_intercept,
    _team_net_rating_metrics,
    _team_win_evaluation,
    fit_pythagorean_win_model,
    score_frozen_possessions,
    score_possession_cohort,
)
from nba_lineup_model.modeling.neural_data import read_neural_possessions
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.season.schema import validate_season

DEFAULT_TARGET_SEASON = "2025-26"
DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_ALPHA_GRID = (1.0, 10.0, 100.0, 1_000.0, 10_000.0)
DEFAULT_TRAINING_START_SEASON = "2019-20"
MODEL_NAME = "frozen_contextual_prior_spline_ridge"
RUN_PREFIX = "contextual-prior"


@dataclass(frozen=True)
class ContextualPriorRun:
    """Immutable fitted contextual residual artifact."""

    run_dir: Path
    run_id: str
    selected_alpha: float


def train_contextual_prior(
    *,
    target_season: str = DEFAULT_TARGET_SEASON,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    training_start_season: str = DEFAULT_TRAINING_START_SEASON,
) -> ContextualPriorRun:
    """Fit an earlier-season composition residual and score a frozen target year."""

    target = validate_season(target_season)
    if target != DEFAULT_TARGET_SEASON:
        raise ValueError("The first contextual exemplar is defined for the 2025-26 frozen holdout")
    _validate_alpha_grid(alpha_grid)
    panel = pd.read_parquet(player_season_panel_path)
    artifact_root = Path(artifacts_dir)
    forward_root = _latest_run(artifact_root / "forward_exposure_gated_rapm" / target)
    priors, coefficients = _forward_state(forward_root)
    training_start = validate_season(training_start_season)
    seasons = _training_seasons(priors, target, training_start)
    source_season = _previous_season(target)
    print(f"Preparing exact replacement exposure through {source_season}", flush=True)
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(source_season)],
        through_season=source_season,
        analytical_dir=analytical_dir,
    )
    frames: dict[str, pd.DataFrame] = {}
    profiles_by_season: dict[str, pd.DataFrame] = {}
    for season in (*seasons, target):
        print(f"Building frozen contextual features for {season}", flush=True)
        frames[season] = _season_training_frame(
            season,
            panel=panel,
            priors=priors,
            coefficients=coefficients,
            analytical_dir=str(analytical_dir),
            exposure_cohort=exposure_cohort,
        )
        profiles = frames[season].attrs.pop("profiles")
        if not isinstance(profiles, pd.DataFrame):
            raise ValueError("Contextual feature construction did not retain player profiles")
        profiles_by_season[season] = profiles
    print("Selecting contextual Ridge regularization", flush=True)
    selected_alpha, cross_validation = _select_alpha(frames, seasons, alpha_grid)
    training_frame = pd.concat([frames[season] for season in seasons], ignore_index=True)
    print(f"Fitting contextual Ridge alpha={selected_alpha:g}", flush=True)
    model = _fit_model(training_frame, selected_alpha)
    target_stints = frames[target].copy()
    target_stints["contextual_correction_net_rating"] = model.predict(
        target_stints.loc[:, contextual_feature_columns()]
    )
    target_stints["prediction_home_net_rating"] = (
        target_stints["baseline_home_net_rating"]
        + target_stints["contextual_correction_net_rating"]
    )
    profiles = profiles_by_season[target]
    print(f"Scoring frozen {target} regular season and playoffs", flush=True)
    evaluation = _evaluate_target(
        target,
        model=model,
        profiles=profiles,
        priors=priors,
        coefficients=coefficients,
        analytical_dir=Path(analytical_dir),
        curated_dir=Path(curated_dir),
    )
    return _write_run(
        target=target,
        training_seasons=seasons,
        forward_root=forward_root,
        selected_alpha=selected_alpha,
        cross_validation=cross_validation,
        model=model,
        target_profiles=profiles,
        target_stints=target_stints,
        evaluation=evaluation,
        artifacts_dir=artifact_root,
    )


def _training_seasons(
    priors: pd.DataFrame, target: str, training_start: str
) -> tuple[str, ...]:
    seasons = tuple(
        sorted(
            {
                str(season)
                for season in priors["season"].unique()
                if training_start <= str(season) < target
            },
            key=lambda value: int(value[:4]),
        )
    )
    if len(seasons) < 5:
        raise ValueError("Contextual prior requires at least five earlier target seasons")
    return seasons


def _forward_state(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    priors = pd.read_parquet(root / "season_player_priors.parquet")
    coefficients = pd.read_parquet(root / "historical_player_coefficients.parquet")
    required_priors = {"season", "player_id", "lagged_rapm_prior"}
    required_coefficients = {"season", "player_id", "rapm"}
    if required_priors - set(priors) or required_coefficients - set(coefficients):
        raise ValueError("Forward exposure-gated artifact lacks the required player state")
    if priors.duplicated(["season", "player_id"]).any():
        raise ValueError("Forward player priors must be unique by season and player")
    return priors.rename(columns={"lagged_rapm_prior": "prior_rapm"}), coefficients


def _season_training_frame(
    season: str,
    *,
    panel: pd.DataFrame,
    priors: pd.DataFrame,
    coefficients: pd.DataFrame,
    analytical_dir: str,
    exposure_cohort: pd.DataFrame,
) -> pd.DataFrame:
    stints = read_rapm_stints(season, analytical_dir=analytical_dir)
    participants = set().union(*stints["home_player_ids"], *stints["away_player_ids"])
    profiles = build_contextual_player_profiles(
        panel,
        target_season=season,
        target_player_ids=participants,
        analytical_dir=analytical_dir,
        exposure_cohort=exposure_cohort,
    )
    feature_frame = lineup_context_features(
        stints["home_player_ids"].tolist(),
        stints["away_player_ids"].tolist(),
        profiles,
    )
    player_priors = priors.loc[priors["season"].eq(season), ["player_id", "prior_rapm"]]
    source_coefficients = coefficients.loc[
        coefficients["season"].eq(_previous_season(season)), ["player_id", "rapm"]
    ]
    if source_coefficients.empty:
        raise ValueError(f"Contextual prior has no source coefficient state for {season}")
    home_intercept = _recover_home_intercept(
        read_rapm_stints(_previous_season(season), analytical_dir=analytical_dir),
        source_coefficients,
    )
    prior_map = dict(
        zip(player_priors["player_id"].astype(int), player_priors["prior_rapm"], strict=True)
    )
    effects, unknown = _lineup_effects(stints, prior_map)
    output = stints.loc[
        :,
        [
            "season",
            "game_id",
            "home_team_id",
            "away_team_id",
            "possessions",
            "home_margin",
            "target_home_net_rating",
        ],
    ].copy()
    output = pd.concat([output.reset_index(drop=True), feature_frame], axis=1)
    output["baseline_home_net_rating"] = effects + home_intercept
    output["target_residual_net_rating"] = (
        output["target_home_net_rating"] - output["baseline_home_net_rating"]
    )
    output["unknown_player_exposures"] = unknown
    output.attrs["profiles"] = profiles
    return output


def _lineup_effects(
    stints: pd.DataFrame, prior_map: dict[int, float]
) -> tuple[np.ndarray, np.ndarray]:
    effects = np.empty(len(stints), dtype=float)
    unknown = np.empty(len(stints), dtype=int)
    for index, (home, away) in enumerate(
        zip(stints["home_player_ids"], stints["away_player_ids"], strict=True)
    ):
        home_ids = [int(player_id) for player_id in home]
        away_ids = [int(player_id) for player_id in away]
        effects[index] = sum(prior_map.get(player_id, 0.0) for player_id in home_ids) - sum(
            prior_map.get(player_id, 0.0) for player_id in away_ids
        )
        unknown[index] = sum(player_id not in prior_map for player_id in (*home_ids, *away_ids))
    return effects, unknown


def _select_alpha(
    frames: dict[str, pd.DataFrame],
    seasons: tuple[str, ...],
    alpha_grid: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    validation_seasons = seasons[-3:]
    rows: list[dict[str, float | int | str]] = []
    for validation in validation_seasons:
        train = pd.concat(
            [frame for season, frame in frames.items() if season < validation], ignore_index=True
        )
        held_out = frames[validation]
        for alpha in alpha_grid:
            model = _fit_model(train, alpha)
            prediction = model.predict(held_out.loc[:, contextual_feature_columns()])
            rows.append(
                {
                    "validation_season": validation,
                    "alpha": alpha,
                    "training_stint_count": len(train),
                    "validation_stint_count": len(held_out),
                    "weighted_mse": mean_squared_error(
                        held_out["target_residual_net_rating"],
                        prediction,
                        sample_weight=held_out["possessions"],
                    ),
                    "weighted_rmse": float(
                        np.sqrt(
                            mean_squared_error(
                                held_out["target_residual_net_rating"],
                                prediction,
                                sample_weight=held_out["possessions"],
                            )
                        )
                    ),
                }
            )
    results = pd.DataFrame(rows)
    summary = results.groupby("alpha", as_index=False).agg(
        mean_weighted_mse=("weighted_mse", "mean")
    )
    selected = summary.sort_values(["mean_weighted_mse", "alpha"], kind="stable").iloc[0]
    return float(selected["alpha"]), results.sort_values(
        ["alpha", "validation_season"]
    ).reset_index(drop=True)


def _fit_model(frame: pd.DataFrame, alpha: float) -> Pipeline:
    model = Pipeline(
        [
            ("spline", SplineTransformer(n_knots=4, degree=2, extrapolation="linear")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(
        frame.loc[:, contextual_feature_columns()],
        frame["target_residual_net_rating"],
        ridge__sample_weight=frame["possessions"],
    )
    return model


def _evaluate_target(
    target: str,
    *,
    model: Pipeline,
    profiles: pd.DataFrame,
    priors: pd.DataFrame,
    coefficients: pd.DataFrame,
    analytical_dir: Path,
    curated_dir: Path,
    evaluation_model: str = MODEL_NAME,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    source = _previous_season(target)
    prior_frame = priors.loc[priors["season"].eq(target), ["player_id", "prior_rapm"]].rename(
        columns={"prior_rapm": "prior_rapm_mean"}
    )
    source_coefficients = coefficients.loc[coefficients["season"].eq(source), ["player_id", "rapm"]]
    source_stints = read_rapm_stints(source, analytical_dir=analytical_dir)
    source_home_intercept = _recover_home_intercept(source_stints, source_coefficients)
    source_possessions, _ = _read_regular_possessions(
        source,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    source_mean = float(source_possessions["target_offense_margin"].mean())
    regular = read_neural_possessions(target, analytical_dir=analytical_dir)
    playoffs, _ = _read_playoff_possessions(target, curated_dir)
    regular_predictions = _score_possessions(
        regular,
        cohort="regular_season",
        profiles=profiles,
        model=model,
        priors=prior_frame,
        source_mean=source_mean,
        source_home_intercept=source_home_intercept,
    )
    playoff_predictions = _score_possessions(
        playoffs,
        cohort="playoffs",
        profiles=profiles,
        model=model,
        priors=prior_frame,
        source_mean=source_mean,
        source_home_intercept=source_home_intercept,
    )
    possession_predictions = pd.concat(
        [regular_predictions, playoff_predictions], ignore_index=True
    )
    cohort_metrics = pd.concat(
        [
            score_possession_cohort(
                regular_predictions,
                source_mean=source_mean,
                model=evaluation_model,
            ),
            score_possession_cohort(
                playoff_predictions,
                source_mean=source_mean,
                model=evaluation_model,
            ),
        ],
        ignore_index=True,
    )
    target_stints = read_rapm_stints(target, analytical_dir=analytical_dir)
    regular_games, team_predictions = _contextual_stint_predictions(
        target_stints,
        profiles=profiles,
        model=model,
        priors=prior_frame,
        source_home_intercept=source_home_intercept,
    )
    team_metrics = _team_net_rating_metrics(team_predictions, model=evaluation_model)
    calibration = _historical_team_seasons(analytical_dir=analytical_dir, through_season=source)
    pythagorean = fit_pythagorean_win_model(calibration)
    team_wins, team_win_metrics = _team_win_evaluation(
        regular_games,
        team_predictions,
        pythagorean,
        model=evaluation_model,
    )
    source_state: dict[str, object] = {
        "target_season": target,
        "source_season": source,
        "player_prior_method": "forward exposure-gated RAPM plus contextual spline residual",
        "profile_information_boundary": "all profiles and exposure gates end with source season",
        "source_offense_margin_mean": source_mean,
        "source_home_intercept_net_rating": source_home_intercept,
        "target_season_refit": False,
        "target_regular_outcomes_used_for_fit": False,
        "target_playoff_outcomes_used_for_fit": False,
        "oracle_information": "realized target-season lineups and exposure only",
    }
    return {
        "source_state": source_state,
        "cohort_metrics": cohort_metrics,
        "possession_predictions": possession_predictions,
        "game_predictions": _game_prediction_frame(possession_predictions),
        "regular_game_predictions": regular_games,
        "team_net_rating_predictions": team_predictions,
        "team_net_rating_metrics": team_metrics,
        "team_win_predictions": team_wins,
        "team_win_metrics": team_win_metrics,
    }


def _score_possessions(
    possessions: pd.DataFrame,
    *,
    cohort: str,
    profiles: pd.DataFrame,
    model: Pipeline,
    priors: pd.DataFrame,
    source_mean: float,
    source_home_intercept: float,
) -> pd.DataFrame:
    output = score_frozen_possessions(
        possessions,
        priors,
        source_mean=source_mean,
        source_home_intercept=source_home_intercept,
        cohort=cohort,
    )
    home_lineups = [
        tuple(int(player_id) for player_id in (offense if home_offense else defense))
        for offense, defense, home_offense in zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            possessions["home_offense"],
            strict=True,
        )
    ]
    away_lineups = [
        tuple(int(player_id) for player_id in (defense if home_offense else offense))
        for offense, defense, home_offense in zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            possessions["home_offense"],
            strict=True,
        )
    ]
    pairs = pd.DataFrame({"home": home_lineups, "away": away_lineups}).drop_duplicates()
    pair_prediction = model.predict(
        lineup_context_features(pairs["home"].tolist(), pairs["away"].tolist(), profiles)
    )
    correction_map = dict(
        zip(zip(pairs["home"], pairs["away"], strict=True), pair_prediction, strict=True)
    )
    correction = np.array(
        [
            correction_map[(home, away)]
            for home, away in zip(home_lineups, away_lineups, strict=True)
        ]
    )
    sign = output["home_offense_sign"].to_numpy(dtype=float)
    output["contextual_correction_home_net_rating"] = correction
    output["prediction_home_margin"] += correction / 200.0
    output["prediction_offense_margin"] += sign * correction / 200.0
    output["residual_offense_margin"] = (
        output["target_offense_margin"] - output["prediction_offense_margin"]
    )
    return output


def _contextual_stint_predictions(
    stints: pd.DataFrame,
    *,
    profiles: pd.DataFrame,
    model: Pipeline,
    priors: pd.DataFrame,
    source_home_intercept: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior_map = dict(zip(priors["player_id"].astype(int), priors["prior_rapm_mean"], strict=True))
    effects, _ = _lineup_effects(stints, prior_map)
    correction = model.predict(
        lineup_context_features(
            stints["home_player_ids"].tolist(),
            stints["away_player_ids"].tolist(),
            profiles,
        )
    )
    base = stints.loc[
        :,
        [
            "game_id",
            "home_team_id",
            "away_team_id",
            "home_team_tricode",
            "away_team_tricode",
            "possessions",
            "home_margin",
        ],
    ].copy()
    base["predicted_home_margin"] = (
        (effects + source_home_intercept + correction) * base["possessions"] / 100.0
    )
    games = base.groupby(
        ["game_id", "home_team_id", "away_team_id", "home_team_tricode", "away_team_tricode"],
        as_index=False,
        sort=False,
    ).agg(
        actual_home_margin=("home_margin", "sum"),
        predicted_home_margin=("predicted_home_margin", "sum"),
    )
    games["predicted_tie"] = games["predicted_home_margin"].eq(0.0)
    games["actual_home_win"] = games["actual_home_margin"].gt(0)
    games["predicted_home_win"] = games["predicted_home_margin"].gt(0)
    games["margin_error"] = games["predicted_home_margin"] - games["actual_home_margin"]
    home = base.rename(
        columns={
            "home_team_id": "team_id",
            "home_team_tricode": "team_tricode",
            "home_margin": "actual_margin",
            "predicted_home_margin": "predicted_margin",
        }
    ).loc[:, ["team_id", "team_tricode", "possessions", "actual_margin", "predicted_margin"]]
    away = base.rename(
        columns={
            "away_team_id": "team_id",
            "away_team_tricode": "team_tricode",
            "home_margin": "actual_margin",
            "predicted_home_margin": "predicted_margin",
        }
    ).loc[:, ["team_id", "team_tricode", "possessions", "actual_margin", "predicted_margin"]]
    away[["actual_margin", "predicted_margin"]] *= -1.0
    teams = (
        pd.concat([home, away], ignore_index=True)
        .groupby(["team_id", "team_tricode"], as_index=False, sort=True)
        .agg(
            possessions=("possessions", "sum"),
            actual_total_margin=("actual_margin", "sum"),
            predicted_total_margin=("predicted_margin", "sum"),
        )
    )
    teams["actual_net_rating"] = 100.0 * teams["actual_total_margin"] / teams["possessions"]
    teams["predicted_net_rating"] = 100.0 * teams["predicted_total_margin"] / teams["possessions"]
    teams["net_rating_error"] = teams["predicted_net_rating"] - teams["actual_net_rating"]
    return games.sort_values("game_id", kind="stable").reset_index(drop=True), teams


def _write_run(
    *,
    target: str,
    training_seasons: tuple[str, ...],
    forward_root: Path,
    selected_alpha: float,
    cross_validation: pd.DataFrame,
    model: Pipeline,
    target_profiles: pd.DataFrame,
    target_stints: pd.DataFrame,
    evaluation: dict[str, pd.DataFrame | dict[str, object]],
    artifacts_dir: Path,
) -> ContextualPriorRun:
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}-{target}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "contextual_prior" / target
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        tables: dict[str, pd.DataFrame] = {
            "cross_validation.parquet": cross_validation,
            "target_player_profiles.parquet": target_profiles,
            "target_stint_predictions.parquet": target_stints.drop(
                columns=list(contextual_feature_columns())
            ),
            "cohort_metrics.parquet": evaluation["cohort_metrics"],  # type: ignore[dict-item]
            "possession_predictions.parquet": evaluation["possession_predictions"],  # type: ignore[dict-item]
            "game_predictions.parquet": evaluation["game_predictions"],  # type: ignore[dict-item]
            "regular_game_predictions.parquet": evaluation["regular_game_predictions"],  # type: ignore[dict-item]
            "team_net_rating_predictions.parquet": evaluation["team_net_rating_predictions"],  # type: ignore[dict-item]
            "team_net_rating_metrics.parquet": evaluation["team_net_rating_metrics"],  # type: ignore[dict-item]
            "team_win_predictions.parquet": evaluation["team_win_predictions"],  # type: ignore[dict-item]
            "team_win_metrics.parquet": evaluation["team_win_metrics"],  # type: ignore[dict-item]
        }
        for filename, frame in tables.items():
            frame.to_parquet(temporary / filename, index=False)
        joblib.dump(model, temporary / "model.joblib")
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "target_season": target,
            "training_seasons": list(training_seasons),
            "training_start_season": training_seasons[0],
            "source_forward_run": str(forward_root),
            "selected_alpha": selected_alpha,
            "feature_columns": list(contextual_feature_columns()),
            "profile_contract": "prior-season rates; exposure-gated rookie/replacement blends",
            "residual_target": "home net rating minus frozen forward RAPM prediction",
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint(
                (
                    Path(__file__),
                    Path(__file__).with_name("contextual_profiles.py"),
                    Path(__file__).with_name("contextual_features.py"),
                )
            ),
            "source_state": evaluation["source_state"],
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {"filename": path.name, "byte_count": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        latest = root / "latest.json.tmp"
        latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest.replace(root / "latest.json")
        return ContextualPriorRun(output, run_id, selected_alpha)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validate_alpha_grid(values: tuple[float, ...]) -> None:
    if not values or any(value <= 0 or not np.isfinite(value) for value in values):
        raise ValueError("Contextual Ridge alpha grid must contain only positive finite values")


def _previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen contextual lineup prior")
    parser.add_argument("--target-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--curated-dir", default=str(DEFAULT_CURATED_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--training-start-season", default=DEFAULT_TRAINING_START_SEASON)
    args = parser.parse_args()
    run = train_contextual_prior(
        target_season=args.target_season,
        player_season_panel_path=args.player_season_panel_path,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        training_start_season=args.training_start_season,
    )
    print(f"Contextual prior: run={run.run_dir} alpha={run.selected_alpha:g}")


if __name__ == "__main__":
    main()
