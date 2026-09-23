"""Descriptive, factor-specific player RAPM for Dean Oliver's Four Factors.

This module intentionally fits completed seasons only.  Its outputs are an
explainability layer measured in factor-rate percentage points, not a
preseason player-rating candidate or a replacement for NAIL-RAPM.
"""

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
from scipy import sparse
from scipy.sparse.linalg import lsqr

from nba_lineup_model.modeling.four_factors import FOUR_FACTORS
from nba_lineup_model.modeling.prior_rapm import HISTORICAL_SEASONS
from nba_lineup_model.modeling.schedule_controls import (
    BACK_TO_BACK_COLUMN,
    attach_back_to_back_feature,
    build_back_to_back_game_features,
)
from nba_lineup_model.modeling.schema import ChronologicalSplitConfig
from nba_lineup_model.modeling.stints import modeling_code_fingerprint
from nba_lineup_model.modeling.train import (
    DEFAULT_LAMBDA_GRID,
    GameSplitPlan,
    chronological_game_splits,
)
from nba_lineup_model.models.baselines import (
    entity_vocabulary,
    signed_entity_matrix,
    vocabulary_mapping,
)
from nba_lineup_model.season.schema import validate_season


DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_GAME_CATALOG_PATH = Path("data/catalog/games.parquet")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models/analysis/four_factor_rapm")
DEFAULT_HISTORY_ARTIFACTS_DIR = Path("artifacts/models/analysis/four_factor_rapm_history")
DESCRIPTIVE_SEASONS = (*HISTORICAL_SEASONS, "2025-26")


@dataclass(frozen=True)
class FactorSpec:
    """Target and opportunity contract for one Four Factor."""

    name: str
    label: str
    home_rate_column: str
    away_rate_column: str
    home_opportunity_column: str
    away_opportunity_column: str
    direction: float


FACTOR_SPECS = {
    "efg_pct": FactorSpec(
        name="efg_pct",
        label="Effective field-goal percentage",
        home_rate_column="home_efg_pct",
        away_rate_column="away_efg_pct",
        home_opportunity_column="home_efg_denominator",
        away_opportunity_column="away_efg_denominator",
        direction=1.0,
    ),
    "tov_pct": FactorSpec(
        name="tov_pct",
        label="Turnover avoidance",
        home_rate_column="home_tov_pct",
        away_rate_column="away_tov_pct",
        home_opportunity_column="home_tov_denominator",
        away_opportunity_column="away_tov_denominator",
        direction=-1.0,
    ),
    "orb_pct": FactorSpec(
        name="orb_pct",
        label="Offensive rebound percentage",
        home_rate_column="home_orb_pct",
        away_rate_column="away_orb_pct",
        home_opportunity_column="home_orb_denominator",
        away_opportunity_column="away_orb_denominator",
        direction=1.0,
    ),
    "ftr": FactorSpec(
        name="ftr",
        label="Free-throw rate (FTA/FGA)",
        home_rate_column="home_ftr",
        away_rate_column="away_ftr",
        home_opportunity_column="home_ftr_denominator",
        away_opportunity_column="away_ftr_denominator",
        direction=1.0,
    ),
}


@dataclass(frozen=True)
class FactorDesign:
    """Sparse player design and valid rows for one factor target."""

    factor: FactorSpec
    player_ids: tuple[int, ...]
    features: sparse.csr_matrix
    target: np.ndarray
    weights: np.ndarray
    back_to_back: np.ndarray
    game_ids: np.ndarray
    player_opportunities: np.ndarray


@dataclass(frozen=True)
class FactorFit:
    """One completed-season factor fit and its schedule controls."""

    factor: FactorSpec
    selected_lambda: float
    player_ids: tuple[int, ...]
    player_effects: np.ndarray
    player_opportunities: np.ndarray
    home_court_effect: float
    back_to_back_effect: float
    row_count: int
    effective_opportunity: float
    cv_results: pd.DataFrame


def read_four_factor_stints(
    season: str,
    *,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
) -> pd.DataFrame:
    """Read the audited regular-season Four Factors stint mart."""

    season = validate_season(season)
    path = Path(analytical_dir) / "four_factor_stints" / season / "regular" / "part-00000.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Four Factors mart is missing: {path}")
    frame = pd.read_parquet(path)
    required = {
        "game_id",
        "game_date",
        "game_time_utc",
        "home_player_ids",
        "away_player_ids",
        "is_four_factor_eligible",
        *(
            column
            for spec in FACTOR_SPECS.values()
            for column in (
                spec.home_rate_column,
                spec.away_rate_column,
                spec.home_opportunity_column,
                spec.away_opportunity_column,
            )
        ),
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Four Factors mart lacks columns: {sorted(missing)}")
    return frame.loc[frame["is_four_factor_eligible"].astype(bool)].copy()


def build_factor_design(
    stints: pd.DataFrame,
    factor: str,
) -> FactorDesign:
    """Build the signed player design using a factor-specific harmonic weight.

    A rate difference has approximate sampling precision proportional to the
    harmonic opportunity, ``n_home * n_away / (n_home + n_away)``.  This avoids
    treating a one-attempt stint as equally informative as a long one while
    preserving each factor's native denominator.
    """

    if factor not in FACTOR_SPECS:
        raise ValueError(f"Unknown Four Factor: {factor!r}")
    spec = FACTOR_SPECS[factor]
    required = {
        "game_id",
        "home_player_ids",
        "away_player_ids",
        BACK_TO_BACK_COLUMN,
        spec.home_rate_column,
        spec.away_rate_column,
        spec.home_opportunity_column,
        spec.away_opportunity_column,
    }
    missing = required - set(stints)
    if missing:
        raise ValueError(f"Four Factor fit rows lack columns: {sorted(missing)}")

    home_rates = stints[spec.home_rate_column].to_numpy(dtype=float)
    away_rates = stints[spec.away_rate_column].to_numpy(dtype=float)
    home_opportunities = stints[spec.home_opportunity_column].to_numpy(dtype=float)
    away_opportunities = stints[spec.away_opportunity_column].to_numpy(dtype=float)
    harmonic_weights = np.divide(
        home_opportunities * away_opportunities,
        home_opportunities + away_opportunities,
        out=np.zeros_like(home_opportunities, dtype=float),
        where=(home_opportunities + away_opportunities) > 0,
    )
    valid = (
        np.isfinite(home_rates)
        & np.isfinite(away_rates)
        & np.isfinite(home_opportunities)
        & np.isfinite(away_opportunities)
        & np.isfinite(harmonic_weights)
        & (home_opportunities > 0)
        & (away_opportunities > 0)
        & (harmonic_weights > 0)
    )
    selected = stints.loc[valid].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"Four Factor {factor} has no finite positive-opportunity stints")

    player_ids = entity_vocabulary(
        selected,
        "home_player_ids",
        "away_player_ids",
        multiple=True,
    )
    mapping = vocabulary_mapping(player_ids)
    features = signed_entity_matrix(
        selected,
        "home_player_ids",
        "away_player_ids",
        mapping,
        multiple=True,
    )
    selected_home_opportunities = home_opportunities[valid]
    selected_away_opportunities = away_opportunities[valid]
    player_opportunities = np.zeros(len(player_ids), dtype=float)
    for row, (home_lineup, away_lineup) in enumerate(
        zip(selected["home_player_ids"], selected["away_player_ids"], strict=True)
    ):
        for player_id in home_lineup:
            player_opportunities[mapping[int(player_id)]] += selected_home_opportunities[row]
        for player_id in away_lineup:
            player_opportunities[mapping[int(player_id)]] += selected_away_opportunities[row]

    target = spec.direction * (home_rates[valid] - away_rates[valid])
    return FactorDesign(
        factor=spec,
        player_ids=player_ids,
        features=features,
        target=target,
        weights=harmonic_weights[valid],
        back_to_back=selected[BACK_TO_BACK_COLUMN].to_numpy(dtype=float),
        game_ids=selected["game_id"].astype(str).to_numpy(),
        player_opportunities=player_opportunities,
    )


def fit_four_factor_season(
    season: str,
    *,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> tuple[list[FactorFit], pd.DataFrame]:
    """Tune and refit all four descriptive player-factor models for a season."""

    if not lambda_grid or any(value < 0 for value in lambda_grid):
        raise ValueError("Four Factor lambda grid must contain non-negative values")
    if len(set(lambda_grid)) != len(lambda_grid):
        raise ValueError("Four Factor lambda grid values must be unique")
    stints = read_four_factor_stints(season, analytical_dir=analytical_dir)
    schedule_features = build_back_to_back_game_features(pd.read_parquet(game_catalog_path))
    stints = attach_back_to_back_feature(stints, schedule_features)
    split_plan = chronological_game_splits(stints, ChronologicalSplitConfig())
    fits: list[FactorFit] = []
    coefficient_frames: list[pd.DataFrame] = []
    for factor in FOUR_FACTORS:
        design = build_factor_design(stints, factor)
        cv_results = _cross_validate_factor(design, split_plan, lambda_grid)
        selected_lambda = _select_lambda(cv_results)
        player_effects, home_court, back_to_back = _fit_with_schedule_controls(
            design.features,
            design.target,
            design.weights,
            design.back_to_back,
            selected_lambda,
        )
        fit = FactorFit(
            factor=design.factor,
            selected_lambda=selected_lambda,
            player_ids=design.player_ids,
            player_effects=player_effects,
            player_opportunities=design.player_opportunities,
            home_court_effect=home_court,
            back_to_back_effect=back_to_back,
            row_count=len(design.target),
            effective_opportunity=float(design.weights.sum()),
            cv_results=cv_results,
        )
        fits.append(fit)
        coefficient_frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "factor": factor,
                    "factor_label": design.factor.label,
                    "player_id": design.player_ids,
                    "effect_percentage_points": 100.0 * player_effects,
                    "factor_opportunities": design.player_opportunities,
                    "selected_lambda": selected_lambda,
                }
            )
        )
    return fits, pd.concat(coefficient_frames, ignore_index=True)


def build_four_factor_rapm_artifact(
    season: str,
    *,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> Path:
    """Write an immutable completed-season Four Factors RAPM artifact."""

    season = validate_season(season)
    fits, coefficients = fit_four_factor_season(
        season,
        analytical_dir=analytical_dir,
        game_catalog_path=game_catalog_path,
        lambda_grid=lambda_grid,
    )
    cv_results = pd.concat(
        [
            fit.cv_results.assign(factor=fit.factor.name)
            for fit in fits
        ],
        ignore_index=True,
    )
    controls = pd.DataFrame(
        [
            {
                "season": season,
                "factor": fit.factor.name,
                "factor_label": fit.factor.label,
                "home_court_effect_percentage_points": 100.0 * fit.home_court_effect,
                "back_to_back_effect_percentage_points": 100.0 * fit.back_to_back_effect,
                "selected_lambda": fit.selected_lambda,
                "row_count": fit.row_count,
                "effective_opportunity": fit.effective_opportunity,
            }
            for fit in fits
        ]
    )
    code_version = modeling_code_fingerprint((Path(__file__), Path(__file__).with_name("four_factors.py")))
    run_id = f"four-factor-rapm-{season}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{code_version.split(':', 1)[1][:8]}"
    root = Path(artifacts_dir) / season
    target = root / run_id
    temporary = root / f".{run_id}.tmp-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        coefficients.to_parquet(temporary / "player_factor_effects.parquet", index=False)
        controls.to_parquet(temporary / "schedule_controls.parquet", index=False)
        cv_results.to_parquet(temporary / "lambda_cv.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "season": season,
            "created_at": datetime.now(UTC).isoformat(),
            "code_version": code_version,
            "purpose": "completed_season_descriptive_explainability",
            "target": {
                "definition": "signed home-minus-away factor rate",
                "units": "rate fraction during fitting; percentage points in player output",
                "turnover_convention": "turnover avoidance equals negative TOV%; larger player effects are favorable",
            },
            "weights": {
                "definition": "harmonic factor opportunity: n_home * n_away / (n_home + n_away)",
                "efg_pct_and_ftr": "field-goal attempts",
                "tov_pct": "FGA + 0.44 * FTA + TOV",
                "orb_pct": "ORB + opponent DRB",
            },
            "schedule_controls": {
                "home_court": "unpenalized fitted intercept in the home-minus-away target",
                "back_to_back": "unpenalized fitted coefficient on home B2B minus away B2B",
            },
            "regularization": {
                "player_penalty": "sample-size-normalized ridge selected separately for each factor using expanding chronological folds",
                "lambda_grid": list(lambda_grid),
                "selected": {fit.factor.name: fit.selected_lambda for fit in fits},
            },
            "quality": "only exact score-reconciled Four Factors games included",
            "factor_definitions": {
                "efg_pct": "(FGM + 0.5 * 3PM) / FGA",
                "tov_pct": "TOV / (FGA + 0.44 * FTA + TOV)",
                "orb_pct": "ORB / (ORB + opponent DRB)",
                "ftr": "FTA / FGA",
            },
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def build_four_factor_rapm_history_artifact(
    *,
    seasons: tuple[str, ...] = DESCRIPTIVE_SEASONS,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
    artifacts_dir: Path | str = DEFAULT_HISTORY_ARTIFACTS_DIR,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> Path:
    """Fit every audited season under one stable descriptive contract."""

    normalized_seasons = tuple(validate_season(season) for season in seasons)
    if not normalized_seasons:
        raise ValueError("Four Factor history requires at least one season")
    if len(set(normalized_seasons)) != len(normalized_seasons):
        raise ValueError("Four Factor history seasons must be unique")
    coefficient_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    cv_frames: list[pd.DataFrame] = []
    for season in normalized_seasons:
        print(f"Four Factors RAPM: {season}", flush=True)
        fits, coefficients = fit_four_factor_season(
            season,
            analytical_dir=analytical_dir,
            game_catalog_path=game_catalog_path,
            lambda_grid=lambda_grid,
        )
        coefficient_frames.append(coefficients)
        cv_frames.extend(
            fit.cv_results.assign(season=season, factor=fit.factor.name)
            for fit in fits
        )
        control_frames.append(
            pd.DataFrame(
                [
                    {
                        "season": season,
                        "factor": fit.factor.name,
                        "factor_label": fit.factor.label,
                        "home_court_effect_percentage_points": 100.0 * fit.home_court_effect,
                        "back_to_back_effect_percentage_points": 100.0 * fit.back_to_back_effect,
                        "selected_lambda": fit.selected_lambda,
                        "row_count": fit.row_count,
                        "effective_opportunity": fit.effective_opportunity,
                    }
                    for fit in fits
                ]
            )
        )
    code_version = modeling_code_fingerprint((Path(__file__), Path(__file__).with_name("four_factors.py")))
    run_id = (
        f"four-factor-rapm-history-{normalized_seasons[0]}-to-{normalized_seasons[-1]}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{code_version.split(':', 1)[1][:8]}"
    )
    root = Path(artifacts_dir)
    target = root / run_id
    temporary = root / f".{run_id}.tmp-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        pd.concat(coefficient_frames, ignore_index=True).to_parquet(
            temporary / "historical_player_factor_effects.parquet", index=False
        )
        pd.concat(control_frames, ignore_index=True).to_parquet(
            temporary / "historical_schedule_controls.parquet", index=False
        )
        pd.concat(cv_frames, ignore_index=True).to_parquet(
            temporary / "historical_lambda_cv.parquet", index=False
        )
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "seasons": list(normalized_seasons),
            "season_count": len(normalized_seasons),
            "code_version": code_version,
            "contract": "identical to four_factor_rapm season artifact; each season is fit independently after factor-specific chronological lambda selection",
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def _cross_validate_factor(
    design: FactorDesign,
    split_plan: GameSplitPlan,
    lambda_grid: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fold in split_plan.folds:
        train = np.isin(design.game_ids, fold.train_game_ids)
        validation = np.isin(design.game_ids, fold.validation_game_ids)
        if not train.any() or not validation.any():
            raise ValueError(f"Four Factor {design.factor.name} has an empty chronological fold")
        for regularization in lambda_grid:
            player_effects, home_court, back_to_back = _fit_with_schedule_controls(
                design.features[train],
                design.target[train],
                design.weights[train],
                design.back_to_back[train],
                regularization,
            )
            predictions = (
                np.asarray(design.features[validation] @ player_effects).reshape(-1)
                + home_court
                + back_to_back * design.back_to_back[validation]
            )
            errors = design.target[validation] - predictions
            weights = design.weights[validation]
            rows.append(
                {
                    "fold": fold.fold,
                    "regularization": regularization,
                    "squared_error_sum": float(np.dot(weights, errors**2)),
                    "validation_effective_opportunity": float(weights.sum()),
                    "validation_row_count": int(validation.sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["regularization", "fold"], kind="stable")


def _select_lambda(cv_results: pd.DataFrame) -> float:
    summary = (
        cv_results.groupby("regularization", as_index=False)
        .agg(
            squared_error_sum=("squared_error_sum", "sum"),
            validation_effective_opportunity=("validation_effective_opportunity", "sum"),
        )
        .sort_values("regularization", kind="stable")
    )
    summary["weighted_mse"] = (
        summary["squared_error_sum"] / summary["validation_effective_opportunity"]
    )
    return float(summary.sort_values(["weighted_mse", "regularization"], kind="stable").iloc[0]["regularization"])


def _fit_with_schedule_controls(
    player_features: sparse.csr_matrix,
    target: np.ndarray,
    weights: np.ndarray,
    back_to_back: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, float, float]:
    """Fit player ridge jointly with unpenalized home and B2B controls."""

    if regularization < 0:
        raise ValueError("Four Factor regularization must be non-negative")
    values = np.asarray(target, dtype=float)
    raw_weights = np.asarray(weights, dtype=float)
    b2b = np.asarray(back_to_back, dtype=float)
    if (
        player_features.shape[0] != len(values)
        or len(values) != len(raw_weights)
        or len(values) != len(b2b)
    ):
        raise ValueError("Four Factor features, target, weights, and controls must align")
    if not np.isfinite(values).all() or not np.isfinite(raw_weights).all() or (raw_weights <= 0).any():
        raise ValueError("Four Factor target and weights must be finite with positive weights")
    if not np.isfinite(b2b).all():
        raise ValueError("Four Factor B2B control must be finite")

    normalized_weights = raw_weights / raw_weights.mean()
    control_features = sparse.csr_matrix(
        np.column_stack([np.ones(len(values), dtype=float), b2b])
    )
    design = sparse.hstack([player_features, control_features], format="csr")
    weighted_design = design.multiply(np.sqrt(normalized_weights)[:, None])
    weighted_target = values * np.sqrt(normalized_weights)
    alpha = regularization * len(values)
    if alpha > 0:
        penalty = sparse.hstack(
            [
                sparse.eye(player_features.shape[1], format="csr") * np.sqrt(alpha),
                sparse.csr_matrix((player_features.shape[1], 2)),
            ],
            format="csr",
        )
        design = sparse.vstack([weighted_design, penalty], format="csr")
        fit_target = np.concatenate([weighted_target, np.zeros(player_features.shape[1])])
    else:
        design = weighted_design
        fit_target = weighted_target
    result = lsqr(design, fit_target, atol=1e-10, btol=1e-10, iter_lim=20_000)
    coefficients = np.asarray(result[0], dtype=float)
    player_count = player_features.shape[1]
    return coefficients[:player_count], float(coefficients[player_count]), float(coefficients[player_count + 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit descriptive Four Factors player RAPM")
    parser.add_argument("season")
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--game-catalog-path", default=str(DEFAULT_GAME_CATALOG_PATH))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDA_GRID))
    parser.add_argument("--history", action="store_true", help="fit every historical season")
    args = parser.parse_args()
    lambdas = tuple(float(value) for value in args.lambdas.split(","))
    if args.history:
        print(
            build_four_factor_rapm_history_artifact(
                analytical_dir=args.analytical_dir,
                game_catalog_path=args.game_catalog_path,
                artifacts_dir=args.artifacts_dir,
                lambda_grid=lambdas,
            )
        )
    else:
        print(
            build_four_factor_rapm_artifact(
                args.season,
                analytical_dir=args.analytical_dir,
                game_catalog_path=args.game_catalog_path,
                artifacts_dir=args.artifacts_dir,
                lambda_grid=lambdas,
            )
        )


if __name__ == "__main__":
    main()
