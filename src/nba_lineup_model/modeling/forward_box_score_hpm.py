"""Recursive HPM with a lagged box-score residual prior for returning players."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.box_score_prior import (
    build_box_score_prior_features,
)
from nba_lineup_model.modeling.box_score_rapm import (
    DEFAULT_REGULARIZATION_GRID,
    fit_box_score_pipeline,
)
from nba_lineup_model.modeling.forward_aging_player_prior import (
    build_centered_value_conditioned_aging_exposure_gated_priors,
    center_player_priors,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONTEXT_ALPHA,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    DEFAULT_TARGET_SEASON,
)
from nba_lineup_model.modeling.forward_hierarchical_pspline_contextual_rapm import (
    DEFAULT_CONTEXT_CURVATURE_ALPHA,
    DEFAULT_CONTEXT_TEMPORAL_ALPHA,
)
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    ForwardPortableMatchupContextualRapmRun,
    train_forward_portable_matchup_contextual_rapm,
)
from nba_lineup_model.modeling.matchup_contextual import (
    bounded_model_metadata,
    fit_bounded_hierarchical_matchup_contextual_model,
)
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN, ForwardLaggedRapmSeason

MODEL_NAME = "forward_box_score_residual_hpm"
RUN_PREFIX = "forward-box-score-residual-hpm"

# The aging/cold-start HPM prior already carries prior RAPM, biography, draft,
# and physical information. This layer is intentionally limited to lagged
# possession-native box-score evidence and does not duplicate those predictors.
BOX_SCORE_RESIDUAL_FEATURE_COLUMNS = (
    "prior_log_on_court_possessions",
    "prior_fga_per_100_on_court_possessions",
    "prior_three_pa_per_100_on_court_possessions",
    "prior_fta_per_100_on_court_possessions",
    "prior_assists_per_100_on_court_possessions",
    "prior_turnovers_per_100_on_court_possessions",
    "prior_offensive_rebounds_per_100_on_court_possessions",
    "prior_defensive_rebounds_per_100_on_court_possessions",
    "prior_steals_per_100_on_court_possessions",
    "prior_blocks_per_100_on_court_possessions",
    "prior_personal_fouls_per_100_on_court_possessions",
    "prior_stabilized_effective_field_goal_percentage",
    "prior_stabilized_three_point_percentage",
    "prior_stabilized_free_throw_percentage",
)

# These products represent a deliberately narrow set of player archetypes.  They
# are created from the same lagged possession-native profile as the additive
# features and are standardized within the annual ridge pipeline.
BOX_SCORE_INTERACTION_FEATURE_COLUMNS = (
    "prior_usage_assists_interaction",
    "prior_usage_turnovers_interaction",
    "prior_three_point_volume_efficiency_interaction",
    "prior_field_goal_free_throw_interaction",
    "prior_offensive_defensive_rebounds_interaction",
    "prior_steals_blocks_interaction",
)
BOX_SCORE_INTERACTION_RESIDUAL_FEATURE_COLUMNS = (
    *BOX_SCORE_RESIDUAL_FEATURE_COLUMNS,
    *BOX_SCORE_INTERACTION_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class BoxScoreResidualSelection:
    """Chronological regularization choice for one next-season residual prior."""

    selected_regularization: float
    validation_weighted_rmse: float
    training_target_seasons: tuple[str, ...]
    validation_fold_count: int
    training_player_count: int
    summary: pd.DataFrame


def _season_key(season: str) -> int:
    return int(str(season)[:4])


def _weighted_rmse(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.average(np.square(actual - predicted), weights=weights)))


def select_box_score_residual_regularization(
    training: pd.DataFrame,
    *,
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID,
    feature_columns: tuple[str, ...] = BOX_SCORE_RESIDUAL_FEATURE_COLUMNS,
) -> BoxScoreResidualSelection:
    """Select ridge penalty by expanding completed-HPM-season folds only."""

    if not regularization_grid or any(value < 0 for value in regularization_grid):
        raise ValueError("Box-score regularization grid must contain non-negative values")
    seasons = tuple(sorted(training["target_season"].unique(), key=_season_key))
    if len(seasons) < 2:
        raise ValueError("Box-score residual prior requires two completed target seasons")
    rows: list[dict[str, float | int]] = []
    for regularization in regularization_grid:
        squared_error_sum = 0.0
        weight_sum = 0.0
        for fold, validation_season in enumerate(seasons[1:]):
            train = training.loc[
                training["target_season"].map(_season_key) < _season_key(validation_season)
            ]
            validation = training.loc[training["target_season"].eq(validation_season)]
            model = fit_box_score_pipeline(
                train,
                regularization=regularization,
                feature_columns=feature_columns,
            )
            predicted = model.predict(validation.loc[:, feature_columns])
            actual = validation["target_rapm"].to_numpy(dtype=float)
            weights = validation["target_rapm_possessions"].to_numpy(dtype=float)
            squared_error_sum += float(np.dot(weights, np.square(actual - predicted)))
            weight_sum += float(weights.sum())
            rows.append(
                {
                    "regularization": regularization,
                    "fold": fold,
                    "train_target_season_count": train["target_season"].nunique(),
                    "validation_target_season": validation_season,
                    "validation_player_count": len(validation),
                    "validation_weight_sum": float(weights.sum()),
                    "weighted_squared_error_sum": float(
                        np.dot(weights, np.square(actual - predicted))
                    ),
                }
            )
        rows.append(
            {
                "regularization": regularization,
                "fold": -1,
                "train_target_season_count": len(seasons) - 1,
                "validation_target_season": "summary",
                "validation_player_count": 0,
                "validation_weight_sum": weight_sum,
                "weighted_squared_error_sum": squared_error_sum,
            }
        )
    detail = pd.DataFrame(rows)
    summary = detail.loc[detail["fold"].eq(-1)].copy()
    summary["weighted_validation_rmse"] = np.sqrt(
        summary["weighted_squared_error_sum"] / summary["validation_weight_sum"]
    )
    summary = summary.sort_values(
        ["weighted_validation_rmse", "regularization"], kind="stable"
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1, dtype=int)
    summary["selected"] = summary["rank"].eq(1)
    selected = summary.iloc[0]
    return BoxScoreResidualSelection(
        selected_regularization=float(selected["regularization"]),
        validation_weighted_rmse=float(selected["weighted_validation_rmse"]),
        training_target_seasons=seasons,
        validation_fold_count=len(seasons) - 1,
        training_player_count=len(training),
        summary=summary,
    )


@dataclass
class ForwardBoxScoreResidualPriorBuilder:
    """Stateful forward builder that learns only from completed HPM transitions."""

    features: pd.DataFrame
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID
    feature_columns: tuple[str, ...] = BOX_SCORE_RESIDUAL_FEATURE_COLUMNS
    prior_method_suffix: str = "lagged_box_score_residual"
    base_priors_by_season: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __call__(
        self,
        *,
        season: str,
        panel: pd.DataFrame,
        completed_results: list[ForwardLaggedRapmSeason],
        exposure_history: list[pd.DataFrame],
        replacement_tokens: list[dict[str, object]],
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        base_priors, metadata = build_centered_value_conditioned_aging_exposure_gated_priors(
            season=season,
            panel=panel,
            completed_results=completed_results,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
        )
        self.base_priors_by_season[season] = base_priors.copy()
        if len(completed_results) < 3:
            return base_priors, {
                **metadata,
                "box_score_residual_enabled": False,
                "box_score_residual_reason": "fewer_than_three_completed_hpm_seasons",
            }

        training = _completed_residual_rows(
            self.features,
            results=completed_results,
            exposures=exposure_history,
            base_priors_by_season=self.base_priors_by_season,
            feature_columns=self.feature_columns,
        )
        if training["target_season"].nunique() < 2:
            return base_priors, {
                **metadata,
                "box_score_residual_enabled": False,
                "box_score_residual_reason": "insufficient_completed_box_score_transitions",
            }
        selection = select_box_score_residual_regularization(
            training,
            regularization_grid=self.regularization_grid,
            feature_columns=self.feature_columns,
        )
        fitted = fit_box_score_pipeline(
            training,
            regularization=selection.selected_regularization,
            feature_columns=self.feature_columns,
        )
        target = _target_box_score_rows(
            self.features,
            season=season,
            priors=base_priors,
            feature_columns=self.feature_columns,
        )
        adjustment = fitted.predict(target.loc[:, self.feature_columns])
        adjusted = base_priors.merge(
            target.loc[:, ["player_id"]].assign(box_score_residual_adjustment=adjustment),
            on="player_id",
            how="left",
            validate="one_to_one",
        )
        adjusted["box_score_residual_adjustment"] = adjusted[
            "box_score_residual_adjustment"
        ].fillna(0.0)
        adjusted[PRIOR_MEAN_COLUMN] = (
            adjusted[PRIOR_MEAN_COLUMN] + adjusted["box_score_residual_adjustment"]
        )
        adjusted = adjusted.drop(columns="box_score_residual_adjustment")
        centered, centering = center_player_priors(
            adjusted,
            previous_exposure=exposure_history[-1] if exposure_history else None,
        )
        return centered, {
            **metadata,
            **centering,
            "player_prior_method": (
                "centered_value_conditioned_aging_exposure_gated_prior_plus_"
                f"{self.prior_method_suffix}"
            ),
            "box_score_residual_enabled": True,
            "box_score_residual_selected_regularization": selection.selected_regularization,
            "box_score_residual_validation_weighted_rmse": selection.validation_weighted_rmse,
            "box_score_residual_training_target_seasons": list(selection.training_target_seasons),
            "box_score_residual_validation_fold_count": selection.validation_fold_count,
            "box_score_residual_training_player_count": selection.training_player_count,
            "box_score_residual_target_player_count": len(target),
            "box_score_residual_feature_columns": list(self.feature_columns),
            "box_score_residual_selection": selection.summary.to_dict(orient="records"),
            "_box_score_residual_model": fitted,
        }


def _completed_residual_rows(
    features: pd.DataFrame,
    *,
    results: Sequence[ForwardLaggedRapmSeason],
    exposures: Sequence[pd.DataFrame],
    base_priors_by_season: dict[str, pd.DataFrame],
    feature_columns: tuple[str, ...] = BOX_SCORE_RESIDUAL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    if len(results) != len(exposures):
        raise ValueError("Completed HPM results and exposure history must align")
    ratings = pd.concat(
        [
            result.player_estimates.loc[:, ["season", "player_id", "rapm"]].rename(
                columns={"season": "target_season", "rapm": "hpm_target_rapm"}
            )
            for result in results
        ],
        ignore_index=True,
    )
    weights = pd.concat(
        [
            exposure.loc[:, ["player_id", "on_court_possessions"]]
            .assign(target_season=result.season)
            .rename(columns={"on_court_possessions": "target_rapm_possessions"})
            for result, exposure in zip(results, exposures, strict=True)
        ],
        ignore_index=True,
    )
    priors = pd.concat(
        [
            value.loc[:, ["player_id", PRIOR_MEAN_COLUMN]]
            .assign(target_season=season)
            .rename(columns={PRIOR_MEAN_COLUMN: "base_hpm_prior"})
            for season, value in base_priors_by_season.items()
            if season in set(ratings["target_season"])
        ],
        ignore_index=True,
    )
    source_features = features.drop(
        columns=["target_rapm", "target_rapm_possessions"], errors="ignore"
    )
    rows = (
        source_features.merge(
            ratings,
            on=["target_season", "player_id"],
            how="inner",
            validate="one_to_one",
        )
        .merge(weights, on=["target_season", "player_id"], how="inner", validate="one_to_one")
        .merge(priors, on=["target_season", "player_id"], how="inner", validate="one_to_one")
    )
    rows["target_rapm"] = rows["hpm_target_rapm"] - rows["base_hpm_prior"]
    eligible = _eligible_box_score_rows(rows, feature_columns=feature_columns)
    return (
        eligible.sort_values(["target_season", "player_id"], kind="stable")
        .reset_index(drop=True)
    )


def _target_box_score_rows(
    features: pd.DataFrame,
    *,
    season: str,
    priors: pd.DataFrame,
    feature_columns: tuple[str, ...] = BOX_SCORE_RESIDUAL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    target = features.loc[features["target_season"].eq(season)].copy()
    target = target.merge(priors.loc[:, ["player_id"]], on="player_id", how="inner")
    target["target_rapm"] = 0.0
    target["target_rapm_possessions"] = 1.0
    target = _eligible_box_score_rows(target, feature_columns=feature_columns)
    return target.sort_values("player_id", kind="stable").reset_index(drop=True)


def _eligible_box_score_rows(
    rows: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = BOX_SCORE_RESIDUAL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Retain only returners with the narrow residual model's lagged profile."""

    required = {
        "target_season",
        "player_id",
        "target_rapm",
        "target_rapm_possessions",
        "has_prior_season",
        "prior_rapm_available",
        "prior_boxscore_features_available",
        *feature_columns,
    }
    missing = required - set(rows)
    if missing:
        raise ValueError(f"Box-score residual rows missing columns: {sorted(missing)}")
    output = rows.loc[
        rows["has_prior_season"].astype(bool)
        & rows["prior_rapm_available"].astype(bool)
        & rows["prior_boxscore_features_available"].astype(bool)
    ].copy()
    if output.empty:
        raise ValueError("Box-score residual prior requires at least one eligible returner")
    if output.duplicated(["target_season", "player_id"]).any():
        raise ValueError("Box-score residual rows must be unique by player-season")
    output["target_rapm"] = pd.to_numeric(output["target_rapm"], errors="raise")
    output["target_rapm_possessions"] = pd.to_numeric(
        output["target_rapm_possessions"], errors="raise"
    )
    if not output["target_rapm_possessions"].gt(0).all():
        raise ValueError("Box-score residual target possessions must be positive")
    return output


def add_box_score_interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    """Append six interpretable lagged box-score interactions to a feature panel."""

    output = features.copy()
    fga = output["prior_fga_per_100_on_court_possessions"]
    three_pa = output["prior_three_pa_per_100_on_court_possessions"]
    fta = output["prior_fta_per_100_on_court_possessions"]
    assists = output["prior_assists_per_100_on_court_possessions"]
    turnovers = output["prior_turnovers_per_100_on_court_possessions"]
    offensive_rebounds = output["prior_offensive_rebounds_per_100_on_court_possessions"]
    defensive_rebounds = output["prior_defensive_rebounds_per_100_on_court_possessions"]
    steals = output["prior_steals_per_100_on_court_possessions"]
    blocks = output["prior_blocks_per_100_on_court_possessions"]
    effective_field_goal_percentage = output[
        "prior_stabilized_effective_field_goal_percentage"
    ]
    usage_events = fga + 0.44 * fta + turnovers

    output["prior_usage_assists_interaction"] = usage_events * assists
    output["prior_usage_turnovers_interaction"] = usage_events * turnovers
    output["prior_three_point_volume_efficiency_interaction"] = (
        three_pa * effective_field_goal_percentage
    )
    output["prior_field_goal_free_throw_interaction"] = fga * fta
    output["prior_offensive_defensive_rebounds_interaction"] = (
        offensive_rebounds * defensive_rebounds
    )
    output["prior_steals_blocks_interaction"] = steals * blocks
    return output


def train_forward_box_score_residual_hpm(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    context_curvature_alpha: float = DEFAULT_CONTEXT_CURVATURE_ALPHA,
    context_temporal_alpha: float = DEFAULT_CONTEXT_TEMPORAL_ALPHA,
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Train HPM with a strictly lagged, incrementally learned box-score prior."""

    panel = pd.read_parquet(player_season_panel_path)
    features, _, _ = build_box_score_prior_features(panel)
    builder = ForwardBoxScoreResidualPriorBuilder(
        features=features,
        regularization_grid=regularization_grid,
    )
    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=context_curvature_alpha,
        context_temporal_alpha=context_temporal_alpha,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_prior_builder=builder,
        player_prior_description=(
            "prior-season centered value-conditioned aging/exposure-gated HPM prior plus "
            "a strictly lagged possession-native box-score residual for returning players, "
            "with bounded portable-matchup context"
        ),
        context_fit=fit_bounded_hierarchical_matchup_contextual_model,
        context_metadata=bounded_model_metadata,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def train_forward_box_score_interaction_hpm(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    context_curvature_alpha: float = DEFAULT_CONTEXT_CURVATURE_ALPHA,
    context_temporal_alpha: float = DEFAULT_CONTEXT_TEMPORAL_ALPHA,
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Train the residual HPM with six predeclared lagged box-score products."""

    panel = pd.read_parquet(player_season_panel_path)
    features, _, _ = build_box_score_prior_features(panel)
    builder = ForwardBoxScoreResidualPriorBuilder(
        features=add_box_score_interaction_features(features),
        regularization_grid=regularization_grid,
        feature_columns=BOX_SCORE_INTERACTION_RESIDUAL_FEATURE_COLUMNS,
        prior_method_suffix="lagged_box_score_interaction_residual",
    )
    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=context_curvature_alpha,
        context_temporal_alpha=context_temporal_alpha,
        model_name="forward_box_score_interaction_hpm",
        run_prefix="forward-box-score-interaction-hpm",
        player_prior_builder=builder,
        player_prior_description=(
            "prior-season centered value-conditioned aging/exposure-gated HPM prior plus "
            "a strictly lagged possession-native box-score residual with six predeclared "
            "player-profile interactions for returning players, with bounded portable-matchup "
            "context"
        ),
        context_fit=fit_bounded_hierarchical_matchup_contextual_model,
        context_metadata=bounded_model_metadata,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )
def main() -> None:
    parser = argparse.ArgumentParser(description="Train forward box-score residual HPM")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    args = parser.parse_args()
    run = train_forward_box_score_residual_hpm(through_season=args.through_season)
    print(f"Forward box-score residual HPM: run={run.run_dir}")


def interaction_main() -> None:
    parser = argparse.ArgumentParser(description="Train forward box-score interaction HPM")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    args = parser.parse_args()
    run = train_forward_box_score_interaction_hpm(through_season=args.through_season)
    print(f"Forward box-score interaction HPM: run={run.run_dir}")


if __name__ == "__main__":
    main()
