"""Prior-centered offense/defense player effects for the Four Factors.

The module is deliberately factor-native: each side of a stint receives its
own rate target and its own denominator weight.  It does not reuse a total
NAIL rating or any NAIL context feature.
"""

from __future__ import annotations

import argparse
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

from nba_lineup_model.modeling.four_factor_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_GAME_CATALOG_PATH,
    FACTOR_SPECS,
    FOUR_FACTORS,
    FactorSpec,
    read_four_factor_stints,
)
from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.modeling.schema import ChronologicalSplitConfig
from nba_lineup_model.modeling.stints import modeling_code_fingerprint
from nba_lineup_model.modeling.train import (
    DEFAULT_LAMBDA_GRID,
    GameSplitPlan,
    chronological_game_splits,
)
from nba_lineup_model.season.schema import validate_season


SIDE_CONTROL_COLUMNS = (
    "intercept",
    "home_court",
    "own_back_to_back",
    "opponent_back_to_back",
)
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models/analysis/four_factor_offense_defense")


@dataclass(frozen=True)
class OffenseDefenseFactorDesign:
    """Two side-rate observations per eligible lineup stint for one factor."""

    factor: FactorSpec
    player_ids: tuple[int, ...]
    player_features: sparse.csr_matrix
    controls: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    game_ids: np.ndarray
    offense_opportunities: np.ndarray
    defense_opportunities: np.ndarray


@dataclass(frozen=True)
class OffenseDefenseFactorFit:
    """A completed-season prior-centered O/D fit for one Four Factor."""

    factor: FactorSpec
    player_ids: tuple[int, ...]
    offense_effects: np.ndarray
    defense_effects: np.ndarray
    offense_priors: np.ndarray
    defense_priors: np.ndarray
    offense_opportunities: np.ndarray
    defense_opportunities: np.ndarray
    selected_offense_lambda: float
    selected_defense_lambda: float
    controls: dict[str, float]
    cv_results: pd.DataFrame


def build_offense_defense_factor_design(
    stints: pd.DataFrame,
    factor: str,
) -> OffenseDefenseFactorDesign:
    """Create side-rate O/D rows with factor-specific opportunity weights.

    For a positively oriented factor rate ``q`` (eFG%, ORB%, FTR), the home
    row is ``q_home = intercept + home_court + sum(o_home) - sum(d_away)``.
    Turnover avoidance uses ``q = -TOV%`` so positive offensive effects mean
    ball security and positive defensive effects mean creating turnovers.
    """

    if factor not in FACTOR_SPECS:
        raise ValueError(f"Unknown Four Factor: {factor!r}")
    spec = FACTOR_SPECS[factor]
    required = {
        "game_id",
        "home_player_ids",
        "away_player_ids",
        "home_back_to_back",
        "away_back_to_back",
        spec.home_rate_column,
        spec.away_rate_column,
        spec.home_opportunity_column,
        spec.away_opportunity_column,
    }
    missing = required - set(stints)
    if missing:
        raise ValueError(f"O/D Four Factor rows lack columns: {sorted(missing)}")

    home_rate = stints[spec.home_rate_column].to_numpy(dtype=float)
    away_rate = stints[spec.away_rate_column].to_numpy(dtype=float)
    home_opportunity = stints[spec.home_opportunity_column].to_numpy(dtype=float)
    away_opportunity = stints[spec.away_opportunity_column].to_numpy(dtype=float)
    valid = (
        np.isfinite(home_rate)
        & np.isfinite(away_rate)
        & np.isfinite(home_opportunity)
        & np.isfinite(away_opportunity)
        & (home_opportunity > 0)
        & (away_opportunity > 0)
    )
    selected = stints.loc[valid].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"Four Factor {factor} has no valid O/D side rows")
    home_rate = home_rate[valid]
    away_rate = away_rate[valid]
    home_opportunity = home_opportunity[valid]
    away_opportunity = away_opportunity[valid]
    player_ids = tuple(
        sorted(
            {
                int(player_id)
                for column in ("home_player_ids", "away_player_ids")
                for lineup in selected[column]
                for player_id in lineup
            }
        )
    )
    player_columns = {player_id: index for index, player_id in enumerate(player_ids)}
    player_count = len(player_ids)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    targets: list[float] = []
    weights: list[float] = []
    controls: list[tuple[float, float, float, float]] = []
    game_ids: list[str] = []
    offense_opportunities = np.zeros(player_count, dtype=float)
    defense_opportunities = np.zeros(player_count, dtype=float)
    home_b2b = selected["home_back_to_back"].to_numpy(dtype=float)
    away_b2b = selected["away_back_to_back"].to_numpy(dtype=float)

    for stint_index, (home_lineup, away_lineup) in enumerate(
        zip(selected["home_player_ids"], selected["away_player_ids"], strict=True)
    ):
        for is_home, own_lineup, opponent_lineup, own_rate, own_opportunity, own_b2b, opponent_b2b in (
            (
                True,
                home_lineup,
                away_lineup,
                home_rate[stint_index],
                home_opportunity[stint_index],
                home_b2b[stint_index],
                away_b2b[stint_index],
            ),
            (
                False,
                away_lineup,
                home_lineup,
                away_rate[stint_index],
                away_opportunity[stint_index],
                away_b2b[stint_index],
                home_b2b[stint_index],
            ),
        ):
            row = len(targets)
            for player_id in own_lineup:
                column = player_columns[int(player_id)]
                rows.append(row)
                columns.append(column)
                values.append(1.0)
                offense_opportunities[column] += own_opportunity
            for player_id in opponent_lineup:
                column = player_columns[int(player_id)]
                rows.append(row)
                columns.append(player_count + column)
                values.append(-1.0)
                defense_opportunities[column] += own_opportunity
            targets.append(spec.direction * own_rate)
            weights.append(own_opportunity)
            controls.append(
                (1.0, 1.0 if is_home else -1.0, own_b2b, opponent_b2b)
            )
            game_ids.append(str(selected.iloc[stint_index]["game_id"]))

    return OffenseDefenseFactorDesign(
        factor=spec,
        player_ids=player_ids,
        player_features=sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(len(targets), 2 * player_count),
            dtype=float,
        ).tocsr(),
        controls=np.asarray(controls, dtype=float),
        target=np.asarray(targets, dtype=float),
        weights=np.asarray(weights, dtype=float),
        game_ids=np.asarray(game_ids, dtype=str),
        offense_opportunities=offense_opportunities,
        defense_opportunities=defense_opportunities,
    )


def fit_prior_centered_offense_defense_factor(
    design: OffenseDefenseFactorDesign,
    *,
    offense_prior: np.ndarray | None = None,
    defense_prior: np.ndarray | None = None,
    regularization: float | tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit one factor with player ridge centered on frozen O/D prior states."""

    player_count = len(design.player_ids)
    offense = _validated_prior(offense_prior, player_count)
    defense = _validated_prior(defense_prior, player_count)
    prior = np.concatenate([offense, defense])
    coefficients, control_coefficients = _fit_prior_centered_with_controls(
        design.player_features,
        design.controls,
        design.target,
        design.weights,
        prior,
        regularization,
    )
    return (
        coefficients[:player_count],
        coefficients[player_count:],
        dict(zip(SIDE_CONTROL_COLUMNS, control_coefficients, strict=True)),
    )


def cross_validate_offense_defense_factor(
    design: OffenseDefenseFactorDesign,
    split_plan: GameSplitPlan,
    *,
    offense_prior: np.ndarray | None = None,
    defense_prior: np.ndarray | None = None,
    lambda_grid: tuple[float, ...],
    regularization_pairs: tuple[tuple[float, float], ...] | None = None,
) -> pd.DataFrame:
    """Select ridge lambda on frozen player priors and chronological folds."""

    if not lambda_grid or any(value < 0 for value in lambda_grid):
        raise ValueError("O/D Four Factor lambda grid must be non-negative")
    pairs = regularization_pairs or tuple((value, value) for value in lambda_grid)
    if not pairs or any(
        len(pair) != 2 or pair[0] < 0 or pair[1] < 0 for pair in pairs
    ):
        raise ValueError("O/D Four Factor penalty pairs must be non-negative O/D pairs")
    player_count = len(design.player_ids)
    prior = np.concatenate(
        [_validated_prior(offense_prior, player_count), _validated_prior(defense_prior, player_count)]
    )
    rows: list[dict[str, float | int]] = []
    for fold in split_plan.folds:
        train = np.isin(design.game_ids, fold.train_game_ids)
        validation = np.isin(design.game_ids, fold.validation_game_ids)
        if not train.any() or not validation.any():
            raise ValueError(f"Four Factor {design.factor.name} has an empty O/D fold")
        for offense_lambda, defense_lambda in pairs:
            coefficients, control_coefficients = _fit_prior_centered_with_controls(
                design.player_features[train],
                design.controls[train],
                design.target[train],
                design.weights[train],
                prior,
                (offense_lambda, defense_lambda),
            )
            prediction = (
                np.asarray(design.player_features[validation] @ coefficients).reshape(-1)
                + design.controls[validation] @ control_coefficients
            )
            errors = design.target[validation] - prediction
            weights = design.weights[validation]
            rows.append(
                {
                    "fold": fold.fold,
                    "offense_regularization": offense_lambda,
                    "defense_regularization": defense_lambda,
                    "squared_error_sum": float(np.dot(weights, errors**2)),
                    "validation_opportunities": float(weights.sum()),
                    "validation_row_count": int(validation.sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["offense_regularization", "defense_regularization", "fold"],
        kind="stable",
    )


def select_offense_defense_penalties(cv_results: pd.DataFrame) -> tuple[float, float]:
    """Choose the lowest aggregate opportunity-weighted chronological MSE."""

    summary = (
        cv_results.groupby(
            ["offense_regularization", "defense_regularization"], as_index=False
        )
        .agg(
            squared_error_sum=("squared_error_sum", "sum"),
            validation_opportunities=("validation_opportunities", "sum"),
        )
        .sort_values(["offense_regularization", "defense_regularization"], kind="stable")
    )
    summary["weighted_mse"] = (
        summary["squared_error_sum"] / summary["validation_opportunities"]
    )
    winner = summary.sort_values(
        ["weighted_mse", "offense_regularization", "defense_regularization"],
        kind="stable",
    ).iloc[0]
    return float(winner["offense_regularization"]), float(winner["defense_regularization"])


def prepare_offense_defense_factor_stints(
    season: str,
    *,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
) -> pd.DataFrame:
    """Read audited stints and attach both game-side B2B flags."""

    stints = read_four_factor_stints(season, analytical_dir=analytical_dir)
    schedule = build_back_to_back_game_features(pd.read_parquet(game_catalog_path))
    schedule = schedule.loc[:, ["game_id", "home_back_to_back", "away_back_to_back"]].copy()
    output = stints.merge(schedule, on="game_id", how="left", validate="many_to_one")
    if output[["home_back_to_back", "away_back_to_back"]].isna().any().any():
        raise ValueError("O/D Four Factors is missing a B2B schedule flag")
    return output


def fit_completed_offense_defense_factor_season(
    season: str,
    *,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    prior_by_factor: dict[str, pd.DataFrame] | None = None,
    regularization_pairs: tuple[tuple[float, float], ...] | None = None,
) -> tuple[list[OffenseDefenseFactorFit], pd.DataFrame]:
    """Tune and refit all four completed-season O/D factor components.

    ``prior_by_factor`` supplies pre-season O/D states with columns
    ``player_id``, ``offense_prior``, and ``defense_prior``.  Omitting it is a
    zero-prior structural baseline only, not the intended forward model.
    """

    season = validate_season(season)
    stints = prepare_offense_defense_factor_stints(
        season,
        analytical_dir=analytical_dir,
        game_catalog_path=game_catalog_path,
    )
    split_plan = chronological_game_splits(stints, config=ChronologicalSplitConfig())
    fits: list[OffenseDefenseFactorFit] = []
    frames: list[pd.DataFrame] = []
    for factor in FOUR_FACTORS:
        design = build_offense_defense_factor_design(stints, factor)
        source_prior = (prior_by_factor or {}).get(factor)
        offense_prior, defense_prior, prior_available = _prior_vectors(design.player_ids, source_prior)
        cv = cross_validate_offense_defense_factor(
            design,
            split_plan,
            offense_prior=offense_prior,
            defense_prior=defense_prior,
            lambda_grid=lambda_grid,
            regularization_pairs=regularization_pairs,
        )
        selected_offense_lambda, selected_defense_lambda = select_offense_defense_penalties(cv)
        offense, defense, controls = fit_prior_centered_offense_defense_factor(
            design,
            offense_prior=offense_prior,
            defense_prior=defense_prior,
            regularization=(selected_offense_lambda, selected_defense_lambda),
        )
        fits.append(
            OffenseDefenseFactorFit(
                factor=design.factor,
                player_ids=design.player_ids,
                offense_effects=offense,
                defense_effects=defense,
                offense_priors=offense_prior,
                defense_priors=defense_prior,
                offense_opportunities=design.offense_opportunities,
                defense_opportunities=design.defense_opportunities,
                selected_offense_lambda=selected_offense_lambda,
                selected_defense_lambda=selected_defense_lambda,
                controls=controls,
                cv_results=cv,
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "factor": factor,
                    "factor_label": design.factor.label,
                    "player_id": design.player_ids,
                    "offense_effect_percentage_points": 100.0 * offense,
                    "defense_effect_percentage_points": 100.0 * defense,
                    "offense_prior_percentage_points": 100.0 * offense_prior,
                    "defense_prior_percentage_points": 100.0 * defense_prior,
                    "prior_available": prior_available,
                    "offense_opportunities": design.offense_opportunities,
                    "defense_opportunities": design.defense_opportunities,
                    "selected_offense_lambda": selected_offense_lambda,
                    "selected_defense_lambda": selected_defense_lambda,
                }
            )
        )
    return fits, pd.concat(frames, ignore_index=True)


def build_completed_offense_defense_factor_artifact(
    season: str,
    *,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    prior_by_factor: dict[str, pd.DataFrame] | None = None,
    regularization_pairs: tuple[tuple[float, float], ...] | None = None,
) -> Path:
    """Persist the structural completed-season O/D baseline for inspection."""

    season = validate_season(season)
    fits, effects = fit_completed_offense_defense_factor_season(
        season,
        analytical_dir=analytical_dir,
        game_catalog_path=game_catalog_path,
        lambda_grid=lambda_grid,
        prior_by_factor=prior_by_factor,
        regularization_pairs=regularization_pairs,
    )
    controls = pd.DataFrame(
        [
            {
                "season": season,
                "factor": fit.factor.name,
                "factor_label": fit.factor.label,
                "selected_offense_lambda": fit.selected_offense_lambda,
                "selected_defense_lambda": fit.selected_defense_lambda,
                **{f"{name}_percentage_points": 100.0 * value for name, value in fit.controls.items()},
            }
            for fit in fits
        ]
    )
    cv = pd.concat(
        [fit.cv_results.assign(season=season, factor=fit.factor.name) for fit in fits],
        ignore_index=True,
    )
    code_version = modeling_code_fingerprint((Path(__file__),))
    run_id = f"four-factor-od-{season}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{code_version.split(':', 1)[1][:8]}"
    root = Path(artifacts_dir) / season
    target = root / run_id
    temporary = root / f".{run_id}.tmp-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        effects.to_parquet(temporary / "player_factor_side_effects.parquet", index=False)
        controls.to_parquet(temporary / "factor_side_schedule_controls.parquet", index=False)
        cv.to_parquet(temporary / "factor_side_lambda_cv.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "season": season,
            "created_at": datetime.now(UTC).isoformat(),
            "code_version": code_version,
            "purpose": "structural_completed_season_od_factor_baseline",
            "prior_contract": (
                "zero-centered structural baseline"
                if prior_by_factor is None
                else "caller-supplied strictly-forward factor-side priors"
            ),
            "target_contract": {
                "home_row": "own offense positive, opponent defense negative",
                "away_row": "own offense positive, opponent defense negative",
                "turnover_target": "negative TOV%; offense is ball security and defense is forced turnovers",
            },
            "weight_contract": "side-specific native factor opportunity",
            "controls": list(SIDE_CONTROL_COLUMNS),
            "lambda_grid": list(lambda_grid),
            "regularization_pairs": (
                [list(pair) for pair in regularization_pairs]
                if regularization_pairs is not None
                else "diagonal lambda_grid pairs"
            ),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def _fit_prior_centered_with_controls(
    player_features: sparse.csr_matrix,
    controls: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    prior: np.ndarray,
    regularization: float | tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the weighted objective with unpenalized schedule controls.

    The player penalty is centered at ``prior``.  Augmenting the weighted
    sparse system with identity rows applies ridge only to player adjustments,
    leaving intercept, HCA, and B2B controls unpenalized.
    """

    offense_regularization, defense_regularization = _regularization_pair(regularization)
    values = np.asarray(target, dtype=float)
    raw_weights = np.asarray(weights, dtype=float)
    prior = np.asarray(prior, dtype=float)
    control_values = np.asarray(controls, dtype=float)
    player_count = player_features.shape[1]
    if (
        player_features.shape[0] != len(values)
        or len(values) != len(raw_weights)
        or control_values.shape != (len(values), len(SIDE_CONTROL_COLUMNS))
        or prior.shape != (player_count,)
    ):
        raise ValueError("O/D Four Factor fit inputs do not align")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(raw_weights).all()
        or (raw_weights <= 0).any()
        or not np.isfinite(control_values).all()
        or not np.isfinite(prior).all()
    ):
        raise ValueError("O/D Four Factor fit inputs must be finite with positive weights")

    normalized_weights = raw_weights / raw_weights.mean()
    root_weight = np.sqrt(normalized_weights)
    full_design = sparse.hstack([player_features, sparse.csr_matrix(control_values)], format="csr")
    weighted_design = full_design.multiply(root_weight[:, None])
    residual_target = values - np.asarray(player_features @ prior).reshape(-1)
    weighted_target = residual_target * root_weight
    penalties = np.concatenate(
        [
            np.repeat(np.sqrt(offense_regularization * len(values)), player_count // 2),
            np.repeat(np.sqrt(defense_regularization * len(values)), player_count // 2),
        ]
    )
    if np.any(penalties > 0):
        penalty = sparse.hstack(
            [
                sparse.diags(penalties, format="csr"),
                sparse.csr_matrix((player_count, len(SIDE_CONTROL_COLUMNS))),
            ],
            format="csr",
        )
        solved_design = sparse.vstack([weighted_design, penalty], format="csr")
        solved_target = np.concatenate([weighted_target, np.zeros(player_count)])
    else:
        solved_design = weighted_design
        solved_target = weighted_target
    solved = lsqr(solved_design, solved_target, atol=1e-10, btol=1e-10, iter_lim=20_000)[0]
    return prior + solved[:player_count], solved[player_count:]


def _regularization_pair(regularization: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(regularization, tuple):
        if len(regularization) != 2:
            raise ValueError("O/D Four Factor regularization must contain O/D values")
        offense, defense = (float(value) for value in regularization)
    else:
        offense = defense = float(regularization)
    if offense < 0 or defense < 0:
        raise ValueError("O/D Four Factor regularization must be non-negative")
    return offense, defense


def _validated_prior(prior: np.ndarray | None, player_count: int) -> np.ndarray:
    if prior is None:
        return np.zeros(player_count, dtype=float)
    values = np.asarray(prior, dtype=float)
    if values.shape != (player_count,) or not np.isfinite(values).all():
        raise ValueError("O/D Four Factor prior must be finite and align with player columns")
    return values


def _prior_vectors(
    player_ids: tuple[int, ...],
    priors: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if priors is None:
        zeros = np.zeros(len(player_ids), dtype=float)
        return zeros, zeros.copy(), np.zeros(len(player_ids), dtype=bool)
    required = {"player_id", "offense_prior", "defense_prior"}
    missing = required - set(priors)
    if missing:
        raise ValueError(f"O/D Four Factor priors lack columns: {sorted(missing)}")
    if priors["player_id"].duplicated().any():
        raise ValueError("O/D Four Factor priors must be unique by player")
    indexed = priors.set_index("player_id")
    offense = np.asarray(
        [
            float(indexed.at[player_id, "offense_prior"])
            if player_id in indexed.index
            else 0.0
            for player_id in player_ids
        ]
    )
    defense = np.asarray(
        [
            float(indexed.at[player_id, "defense_prior"])
            if player_id in indexed.index
            else 0.0
            for player_id in player_ids
        ]
    )
    if not np.isfinite(offense).all() or not np.isfinite(defense).all():
        raise ValueError("O/D Four Factor priors must be finite")
    return offense, defense, np.asarray([player_id in indexed.index for player_id in player_ids])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit completed-season Four Factors O/D RAPM")
    parser.add_argument("season")
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--game-catalog-path", default=str(DEFAULT_GAME_CATALOG_PATH))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    args = parser.parse_args()
    print(
        build_completed_offense_defense_factor_artifact(
            args.season,
            analytical_dir=args.analytical_dir,
            game_catalog_path=args.game_catalog_path,
            artifacts_dir=args.artifacts_dir,
        )
    )


if __name__ == "__main__":
    main()
