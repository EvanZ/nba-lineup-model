"""Strict pre-season evaluation of a frozen lagged-RAPM player prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.stats import pearsonr, spearmanr

from nba_lineup_model.evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    possession_game_margin_rmse,
    rmse,
    skill_score,
)
from nba_lineup_model.modeling.aging import validate_aging_model_run
from nba_lineup_model.modeling.aging_prior_rapm import aging_prior_frame
from nba_lineup_model.modeling.draft_prior import validate_draft_prior_study
from nba_lineup_model.modeling.exposure_gated_cold_start import (
    validate_exposure_gated_cold_start_prior,
)
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN
from nba_lineup_model.modeling.schema import CODE_VERSION_PATTERN, ArtifactRecord
from nba_lineup_model.modeling.single_lineup_possessions import (
    single_lineup_possessions_frame,
)
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.season.compact import (
    read_curated_partition_manifest,
    validate_curated_partition,
)
from nba_lineup_model.season.layout import CuratedDatasetLayout, CuratedPartition
from nba_lineup_model.season.schema import SEASON_PATTERN, SHA256_PATTERN, validate_season

DEFAULT_SEASON = "2025-26"
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
_FORBIDDEN_EXTERNAL_PRIOR_COLUMNS = {
    "target_rapm",
    "target_rapm_possessions",
    "target_rapm_seconds",
    "target_rapm_exposure_eligible",
}


class FrozenPriorEvaluationManifest(BaseModel):
    """Integrity and information-boundary contract for one frozen evaluation."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    created_at: datetime
    season: str = Field(pattern=SEASON_PATTERN)
    model: Literal[
        "frozen_regular_only_lagged_rapm",
        "frozen_one_year_rapm_no_priors",
        "frozen_three_year_rapm_no_priors",
        "frozen_aging_prior",
        "frozen_combined_box_score_prior",
        "frozen_draft_cold_start_prior",
        "frozen_exposure_gated_cold_start_prior",
    ]
    source_season: str = Field(pattern=SEASON_PATTERN)
    evaluation_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    prior_run_id: str = Field(min_length=1)
    prior_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_rapm_stints_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_possessions_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    regular_possessions_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    regular_rapm_stints_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    playoff_segments_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    player_prior_count: int = Field(ge=1)
    regular_game_count: int = Field(ge=1)
    playoff_game_count: int = Field(ge=1)
    team_count: int = Field(ge=2)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=10)

    @field_validator("season", "source_season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Frozen evaluation timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_contract(self) -> FrozenPriorEvaluationManifest:
        if int(self.source_season[:4]) + 1 != int(self.season[:4]):
            raise ValueError("Frozen prior source must immediately precede evaluation season")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Frozen evaluation artifact names must be unique")
        return self


@dataclass(frozen=True)
class FrozenEvaluation:
    """All output tables for one strict frozen-prior evaluation."""

    source_state: dict[str, Any]
    frozen_player_priors: pd.DataFrame
    cohort_metrics: pd.DataFrame
    possession_predictions: pd.DataFrame
    game_predictions: pd.DataFrame
    regular_game_predictions: pd.DataFrame
    team_net_rating_predictions: pd.DataFrame
    team_net_rating_metrics: pd.DataFrame
    team_win_predictions: pd.DataFrame
    team_win_metrics: pd.DataFrame
    pythagorean_calibration_team_seasons: pd.DataFrame


@dataclass(frozen=True)
class PythagoreanWinModel:
    """Forward historical linear mapping from net rating to win percentage."""

    intercept: float
    net_rating_slope: float
    training_seasons: tuple[str, ...]
    training_team_season_count: int
    historical_win_total_rmse: float

    def predict_win_pct(self, net_rating: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(net_rating, dtype=float)
        return np.clip(self.intercept + self.net_rating_slope * values, 0.0, 1.0)


def frozen_prior_code_fingerprint() -> str:
    """Hash code that defines the strict frozen evaluation."""

    digest = hashlib.sha256()
    paths = (
        Path(__file__),
        Path(__file__).with_name("forward_calibration.py"),
        Path(__file__).parents[1] / "evaluation" / "metrics.py",
    )
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def run_frozen_lagged_evaluation(
    *,
    season: str = DEFAULT_SEASON,
    prior_run_dir: Path | str,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    player_priors_override: pd.DataFrame | None = None,
    source_state_overrides: dict[str, Any] | None = None,
    evaluation_model: str = "frozen_lagged_rapm",
) -> FrozenEvaluation:
    """Evaluate a prior fixed before the target season on every target outcome."""

    target_season = validate_season(season)
    source_season = _previous_season(target_season)
    prior_root = Path(prior_run_dir)
    prior_metadata = _validate_forward_prior_run(prior_root, target_season)
    historical = pd.read_parquet(prior_root / "historical_player_coefficients.parquet")
    source_coefficients = historical.loc[
        historical["season"].eq(source_season), ["player_id", "rapm"]
    ].copy()
    if source_coefficients.empty:
        raise ValueError(f"Forward prior run has no completed {source_season} RAPM state")
    if player_priors_override is None:
        published_priors = pd.read_parquet(prior_root / "target_player_priors.parquet")
        priors = _validate_frozen_priors(
            published_priors,
            source_coefficients,
            target_season=target_season,
            source_season=source_season,
        )
    else:
        priors = _validate_external_frozen_priors(
            player_priors_override,
            target_season=target_season,
            source_season=source_season,
        )

    source_stints = read_rapm_stints(source_season, analytical_dir=analytical_dir)
    source_possessions, source_possessions_manifest = _read_regular_possessions(
        source_season,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    source_mean = float(source_possessions["target_offense_margin"].mean())
    source_home_intercept = _recover_home_intercept(source_stints, source_coefficients)
    source_state = {
        "target_season": target_season,
        "source_season": source_season,
        "prior_run_id": str(prior_metadata["run_id"]),
        "player_prior_method": "completed prior-season regular-only forward RAPM",
        "cold_start_prior": 0.0,
        "source_offense_margin_mean": source_mean,
        "source_home_intercept_net_rating": source_home_intercept,
        "source_possessions_manifest_sha256": _sha256_file(source_possessions_manifest),
        "possession_translation": (
            "source_mean + offense_minus_defense_prior/200 "
            "+ home_offense_sign*source_home_intercept/200"
        ),
        "target_season_refit": False,
        "target_regular_outcomes_used_for_fit": False,
        "target_playoff_outcomes_used_for_fit": False,
        "oracle_information": "realized target-season lineups and exposure only",
    }
    if source_state_overrides:
        source_state.update(source_state_overrides)

    regular_possessions, target_regular_manifest = _read_regular_possessions(
        target_season,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    source_state["target_regular_possessions_manifest_sha256"] = _sha256_file(
        target_regular_manifest
    )
    playoff_possessions, _ = _read_playoff_possessions(target_season, curated_dir)
    regular_predictions = score_frozen_possessions(
        regular_possessions,
        priors,
        source_mean=source_mean,
        source_home_intercept=source_home_intercept,
        cohort="regular_season",
    )
    playoff_predictions = score_frozen_possessions(
        playoff_possessions,
        priors,
        source_mean=source_mean,
        source_home_intercept=source_home_intercept,
        cohort="playoffs",
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
    game_predictions = _game_prediction_frame(possession_predictions)

    target_stints = read_rapm_stints(target_season, analytical_dir=analytical_dir)
    regular_game_predictions, team_net_rating_predictions = _regular_stint_predictions(
        target_stints,
        priors,
        source_home_intercept=source_home_intercept,
    )
    team_net_rating_metrics = _team_net_rating_metrics(
        team_net_rating_predictions,
        model=evaluation_model,
    )
    calibration_team_seasons = _historical_team_seasons(
        analytical_dir=analytical_dir,
        through_season=source_season,
    )
    pythagorean_model = fit_pythagorean_win_model(calibration_team_seasons)
    source_state["pythagorean_win_model"] = {
        "form": "weighted_linear_win_percentage_from_net_rating",
        "intercept": pythagorean_model.intercept,
        "net_rating_slope": pythagorean_model.net_rating_slope,
        "wins_per_net_rating_point_82_games": 82.0 * pythagorean_model.net_rating_slope,
        "training_first_season": pythagorean_model.training_seasons[0],
        "training_last_season": pythagorean_model.training_seasons[-1],
        "training_season_count": len(pythagorean_model.training_seasons),
        "training_team_season_count": pythagorean_model.training_team_season_count,
        "historical_win_total_rmse": pythagorean_model.historical_win_total_rmse,
    }
    team_win_predictions, team_win_metrics = _team_win_evaluation(
        regular_game_predictions,
        team_net_rating_predictions,
        pythagorean_model,
        model=evaluation_model,
    )
    if set(team_net_rating_predictions["team_id"]) != set(team_win_predictions["team_id"]):
        raise ValueError("Team net-rating and win evaluations cover different teams")
    return FrozenEvaluation(
        source_state=source_state,
        frozen_player_priors=priors,
        cohort_metrics=cohort_metrics,
        possession_predictions=possession_predictions,
        game_predictions=game_predictions,
        regular_game_predictions=regular_game_predictions,
        team_net_rating_predictions=team_net_rating_predictions,
        team_net_rating_metrics=team_net_rating_metrics,
        team_win_predictions=team_win_predictions,
        team_win_metrics=team_win_metrics,
        pythagorean_calibration_team_seasons=calibration_team_seasons,
    )


def score_frozen_possessions(
    possessions: pd.DataFrame,
    priors: pd.DataFrame,
    *,
    source_mean: float,
    source_home_intercept: float,
    cohort: str,
) -> pd.DataFrame:
    """Score target possessions directly from a pre-season player vector."""

    required = {
        "game_id",
        "possession_id",
        "home_offense_sign",
        "offense_player_ids",
        "defense_player_ids",
        "target_offense_margin",
    }
    missing = required - set(possessions)
    if missing:
        raise ValueError(f"Frozen evaluation possessions missing columns: {sorted(missing)}")
    if priors["player_id"].duplicated().any():
        raise ValueError("Frozen player priors contain duplicate players")
    prior_map = dict(
        zip(
            priors["player_id"].astype(int),
            priors["prior_rapm_mean"].astype(float),
            strict=True,
        )
    )
    effects = np.empty(len(possessions), dtype=float)
    unknown_counts = np.empty(len(possessions), dtype=int)
    for row, (offense, defense) in enumerate(
        zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            strict=True,
        )
    ):
        offense_ids = [int(player_id) for player_id in offense]
        defense_ids = [int(player_id) for player_id in defense]
        unknown_counts[row] = sum(player_id not in prior_map for player_id in offense_ids) + sum(
            player_id not in prior_map for player_id in defense_ids
        )
        effects[row] = sum(prior_map.get(player_id, 0.0) for player_id in offense_ids) - sum(
            prior_map.get(player_id, 0.0) for player_id in defense_ids
        )
    signs = possessions["home_offense_sign"].to_numpy(dtype=float)
    prediction = source_mean + effects / 200.0 + signs * source_home_intercept / 200.0
    output_columns = [
        "game_id",
        "possession_id",
        "home_offense_sign",
        "target_offense_margin",
    ]
    for optional in (
        "season",
        "season_type",
        "game_date",
        "game_time_utc",
        "possession_index",
        "offense_team_id",
        "defense_team_id",
        "offense_team_tricode",
        "defense_team_tricode",
    ):
        if optional in possessions:
            output_columns.append(optional)
    output = possessions.loc[:, output_columns].copy()
    output.insert(0, "cohort", cohort)
    output["prediction_offense_margin"] = prediction
    output["target_home_margin"] = output["target_offense_margin"] * signs
    output["prediction_home_margin"] = prediction * signs
    output["residual_offense_margin"] = output["target_offense_margin"] - prediction
    output["unknown_player_exposures"] = unknown_counts
    return output


def score_possession_cohort(
    predictions: pd.DataFrame,
    *,
    source_mean: float,
    model: str = "frozen_lagged_rapm",
) -> pd.DataFrame:
    """Compute possession and eligible-possession game metrics for one cohort."""

    if predictions["cohort"].nunique() != 1:
        raise ValueError("Possession metric input must contain one cohort")
    actual = predictions["target_offense_margin"].to_numpy(dtype=float)
    predicted = predictions["prediction_offense_margin"].to_numpy(dtype=float)
    signs = predictions["home_offense_sign"].to_numpy(dtype=float)
    mean_prediction = np.full(len(predictions), source_mean, dtype=float)
    model_mse = mean_squared_error(actual, predicted)
    mean_mse = mean_squared_error(actual, mean_prediction)
    model_game_rmse = possession_game_margin_rmse(
        predictions["game_id"], actual, predicted, signs
    )
    mean_game_rmse = possession_game_margin_rmse(
        predictions["game_id"], actual, mean_prediction, signs
    )
    return pd.DataFrame(
        [
            {
                "model": model,
                "cohort": str(predictions["cohort"].iloc[0]),
                "information_cutoff": "end_of_prior_regular_season",
                "game_count": int(predictions["game_id"].nunique()),
                "possession_count": len(predictions),
                "possession_mse": model_mse,
                "possession_rmse": rmse(actual, predicted),
                "possession_mae": mean_absolute_error(actual, predicted),
                "eligible_possession_game_margin_rmse": model_game_rmse,
                "possession_skill_vs_frozen_mean": skill_score(model_mse, mean_mse),
                "game_margin_skill_vs_frozen_mean": skill_score(
                    model_game_rmse**2, mean_game_rmse**2
                ),
                "frozen_mean_reference_possession_rmse": float(np.sqrt(mean_mse)),
                "frozen_mean_reference_game_margin_rmse": mean_game_rmse,
                "unknown_player_exposures": int(
                    predictions["unknown_player_exposures"].sum()
                ),
            }
        ]
    )


def train_frozen_lagged_prior_evaluation(
    *,
    season: str = DEFAULT_SEASON,
    prior_run_id: str | None = None,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> tuple[FrozenPriorEvaluationManifest, Path]:
    """Evaluate and atomically publish one immutable frozen-prior run."""

    target_season = validate_season(season)
    artifact_root = Path(artifacts_dir)
    prior_root = _resolve_run(artifact_root / "prior_rapm" / target_season, prior_run_id)
    evaluation = run_frozen_lagged_evaluation(
        season=target_season,
        prior_run_dir=prior_root,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    return _write_run(
        evaluation,
        prior_root=prior_root,
        analytical_dir=Path(analytical_dir),
        curated_dir=Path(curated_dir),
        artifacts_dir=artifact_root,
    )


def train_frozen_aging_prior_evaluation(
    *,
    season: str = DEFAULT_SEASON,
    aging_run_id: str | None = None,
    reference_prior_run_id: str | None = None,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> tuple[FrozenPriorEvaluationManifest, Path]:
    """Evaluate the age/draft/physical player prior without a target-season refit."""

    target_season = validate_season(season)
    artifact_root = Path(artifacts_dir)
    reference_root = _resolve_run(
        artifact_root / "prior_rapm" / target_season,
        reference_prior_run_id,
    )
    reference_metadata = _validate_forward_prior_run(reference_root, target_season)
    aging_root = _resolve_run(artifact_root / "aging" / target_season, aging_run_id)
    aging_manifest = validate_aging_model_run(aging_root)
    aging_priors = (
        aging_prior_frame(
            pd.read_parquet(aging_root / "player_priors.parquet"),
            target_season=target_season,
        )
        .rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm_mean"})
        .copy()
    )
    aging_priors["prior_available"] = True
    evaluation = run_frozen_lagged_evaluation(
        season=target_season,
        prior_run_dir=reference_root,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        player_priors_override=aging_priors,
        source_state_overrides={
            "prior_run_id": aging_manifest.run_id,
            "player_prior_method": (
                "forward aging ridge using age, experience, prior RAPM exposure, "
                "draft profile, and physical profile"
            ),
            "cold_start_prior": "forward aging profile estimate",
            "aging_run_id": aging_manifest.run_id,
            "aging_manifest_sha256": _sha256_file(aging_root / "manifest.json"),
            "aging_training_last_target_season": aging_manifest.training_target_seasons[-1],
            "reference_lagged_prior_run_id": str(reference_metadata["run_id"]),
            "reference_lagged_prior_manifest_sha256": _sha256_file(
                reference_root / "manifest.json"
            ),
        },
        evaluation_model="frozen_aging_prior",
    )
    return _write_run(
        evaluation,
        prior_root=aging_root,
        analytical_dir=Path(analytical_dir),
        curated_dir=Path(curated_dir),
        artifacts_dir=artifact_root,
        run_prefix="frozen-aging-prior",
        manifest_model="frozen_aging_prior",
    )


def train_frozen_combined_box_score_prior_evaluation(
    *,
    season: str = DEFAULT_SEASON,
    combined_run_id: str | None = None,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> tuple[FrozenPriorEvaluationManifest, Path]:
    """Score the selected box-score/cold-start prior without a target refit."""

    target_season = validate_season(season)
    artifact_root = Path(artifacts_dir)
    combined_root = _resolve_run(
        artifact_root / "combined_box_score_prior_rapm" / target_season,
        combined_run_id,
    )
    combined_metadata = json.loads((combined_root / "metadata.json").read_text())
    if combined_metadata.get("model") != (
        "combined_box_score_and_cold_start_prior_centered_ridge_rapm"
    ):
        raise ValueError("Frozen box-score evaluation requires a combined-prior run")
    if combined_metadata.get("season") != target_season:
        raise ValueError("Combined box-score prior season does not match evaluation")
    lagged_run_id = combined_metadata.get("lagged_run_id")
    if not isinstance(lagged_run_id, str):
        raise ValueError("Combined box-score prior does not record its lagged source")
    reference_root = _resolve_run(
        artifact_root / "prior_rapm" / target_season,
        lagged_run_id,
    )
    _validate_forward_prior_run(reference_root, target_season)
    combined_priors = pd.read_parquet(combined_root / "combined_player_priors.parquet")
    required = {"player_id", PRIOR_MEAN_COLUMN, "prior_branch"}
    missing = required - set(combined_priors)
    if missing:
        raise ValueError(f"Combined box-score priors missing columns: {sorted(missing)}")
    frozen_priors = combined_priors.loc[:, ["player_id", PRIOR_MEAN_COLUMN]].rename(
        columns={PRIOR_MEAN_COLUMN: "prior_rapm_mean"}
    )
    lagged_priors = pd.read_parquet(reference_root / "target_player_priors.parquet")
    compared = frozen_priors.merge(
        lagged_priors.loc[:, ["player_id", "prior_rapm_mean"]],
        on="player_id",
        how="outer",
        suffixes=("_combined", "_lagged"),
        validate="one_to_one",
    )
    equal_to_lagged = bool(
        len(compared) == len(frozen_priors)
        and compared[["prior_rapm_mean_combined", "prior_rapm_mean_lagged"]]
        .notna()
        .all()
        .all()
        and np.allclose(
            compared["prior_rapm_mean_combined"],
            compared["prior_rapm_mean_lagged"],
        )
    )
    evaluation = run_frozen_lagged_evaluation(
        season=target_season,
        prior_run_dir=reference_root,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        player_priors_override=frozen_priors,
        source_state_overrides={
            "prior_run_id": combined_metadata["run_id"],
            "player_prior_method": "selected combined box-score and cold-start prior",
            "combined_box_score_prior_run_id": combined_metadata["run_id"],
            "combined_box_score_prior_manifest_sha256": _sha256_file(
                combined_root / "manifest.json"
            ),
            "reference_lagged_prior_run_id": lagged_run_id,
            "reference_lagged_prior_manifest_sha256": _sha256_file(
                reference_root / "manifest.json"
            ),
            "selected_box_score_weight": combined_metadata["selected_box_score_weight"],
            "prior_vector_equals_lagged": equal_to_lagged,
            "prior_branch_counts": combined_metadata["prior_branch_counts"],
        },
        evaluation_model="frozen_combined_box_score_prior",
    )
    return _write_run(
        evaluation,
        prior_root=combined_root,
        analytical_dir=Path(analytical_dir),
        curated_dir=Path(curated_dir),
        artifacts_dir=artifact_root,
        run_prefix="frozen-combined-box-score-prior",
        manifest_model="frozen_combined_box_score_prior",
    )


def draft_cold_start_prior_frame(
    lagged_priors: pd.DataFrame,
    draft_rankings: pd.DataFrame,
) -> pd.DataFrame:
    """Replace only first-NBA-season zero priors with frozen draft-prior values."""

    lagged_required = {"player_id", "prior_rapm_mean", "prior_available"}
    ranking_required = {"player_id", "draft_prior"}
    missing_lagged = lagged_required - set(lagged_priors)
    missing_rankings = ranking_required - set(draft_rankings)
    if missing_lagged or missing_rankings:
        raise ValueError(
            "Draft cold-start inputs missing columns: "
            f"lagged={sorted(missing_lagged)}, rankings={sorted(missing_rankings)}"
        )
    if (
        lagged_priors["player_id"].duplicated().any()
        or draft_rankings["player_id"].duplicated().any()
    ):
        raise ValueError("Draft cold-start inputs contain duplicate players")
    forbidden = _FORBIDDEN_EXTERNAL_PRIOR_COLUMNS & set(draft_rankings)
    if forbidden:
        raise ValueError(f"Draft rankings contain target outcomes: {sorted(forbidden)}")
    draft = draft_rankings.loc[:, ["player_id", "draft_prior"]].copy()
    draft["player_id"] = pd.to_numeric(draft["player_id"], errors="raise").astype(int)
    draft["draft_prior"] = pd.to_numeric(draft["draft_prior"], errors="raise")
    if not np.isfinite(draft["draft_prior"].to_numpy(dtype=float)).all():
        raise ValueError("Draft rankings contain non-finite prior values")
    output = lagged_priors.loc[:, ["player_id", "prior_rapm_mean", "prior_available"]].copy()
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    output = output.merge(draft, on="player_id", how="left", validate="one_to_one")
    matched = output["draft_prior"].notna()
    returning = output["prior_available"].astype(bool)
    if (matched & returning).any():
        raise ValueError("Draft rankings overlap players with a lagged RAPM prior")
    if len(draft) != int(matched.sum()):
        raise ValueError("Draft rankings include players absent from the frozen prior universe")
    output["prior_branch"] = np.select(
        [returning, matched],
        ["lagged_rapm", "draft_cold_start"],
        default="zero_cold_start",
    )
    output.loc[matched, "prior_rapm_mean"] = output.loc[matched, "draft_prior"]
    return output.drop(columns="draft_prior").sort_values("player_id", kind="stable").reset_index(
        drop=True
    )


def train_frozen_draft_cold_start_prior_evaluation(
    *,
    season: str = DEFAULT_SEASON,
    draft_run_id: str | None = None,
    reference_prior_run_id: str | None = None,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> tuple[FrozenPriorEvaluationManifest, Path]:
    """Evaluate a draft-informed replacement for the frozen zero cold-start prior."""

    target_season = validate_season(season)
    artifact_root = Path(artifacts_dir)
    reference_root = _resolve_run(
        artifact_root / "prior_rapm" / target_season,
        reference_prior_run_id,
    )
    reference_metadata = _validate_forward_prior_run(reference_root, target_season)
    draft_root = _resolve_run(artifact_root / "draft_prior" / target_season, draft_run_id)
    draft_metadata = validate_draft_prior_study(draft_root)
    if draft_metadata.get("target_season") != target_season:
        raise ValueError("Draft-prior target season does not match frozen evaluation")
    if draft_metadata.get("training_last_season") != _previous_season(target_season):
        raise ValueError("Draft-prior study does not end with the source season")
    if draft_metadata.get("target_outcomes_used_for_fit") is not False:
        raise ValueError("Draft-prior study uses target-season outcomes")
    lagged_priors = pd.read_parquet(reference_root / "target_player_priors.parquet")
    draft_rankings = pd.read_parquet(draft_root / "rookie_rankings.parquet")
    draft_priors = draft_cold_start_prior_frame(lagged_priors, draft_rankings)
    branch_counts = draft_priors.groupby("prior_branch").size().to_dict()
    evaluation = run_frozen_lagged_evaluation(
        season=target_season,
        prior_run_dir=reference_root,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        player_priors_override=draft_priors,
        source_state_overrides={
            "prior_run_id": draft_metadata["run_id"],
            "player_prior_method": (
                "2024-25-or-earlier draft-profile ridge for first-NBA-season players; "
                "otherwise completed 2024-25 regular-only lagged RAPM or zero cold start"
            ),
            "cold_start_prior": "draft profile for first-NBA-season players; zero otherwise",
            "draft_prior_run_id": draft_metadata["run_id"],
            "draft_prior_manifest_sha256": _sha256_file(draft_root / "manifest.json"),
            "draft_prior_training_last_season": draft_metadata["training_last_season"],
            "draft_prior_selected_regularization": draft_metadata["selected_regularization"],
            "reference_lagged_prior_run_id": str(reference_metadata["run_id"]),
            "reference_lagged_prior_manifest_sha256": _sha256_file(
                reference_root / "manifest.json"
            ),
            "prior_branch_counts": branch_counts,
        },
        evaluation_model="frozen_draft_cold_start_prior",
    )
    return _write_run(
        evaluation,
        prior_root=draft_root,
        analytical_dir=Path(analytical_dir),
        curated_dir=Path(curated_dir),
        artifacts_dir=artifact_root,
        run_prefix="frozen-draft-cold-start-prior",
        manifest_model="frozen_draft_cold_start_prior",
    )


def exposure_gated_cold_start_prior_frame(
    lagged_priors: pd.DataFrame,
    revised_rankings: pd.DataFrame,
) -> pd.DataFrame:
    """Replace first-year zero priors with the continuous exposure-gated blend."""

    lagged_required = {"player_id", "prior_rapm_mean", "prior_available"}
    ranking_required = {"player_id", "blended_cold_start_prior"}
    missing_lagged = lagged_required - set(lagged_priors)
    missing_rankings = ranking_required - set(revised_rankings)
    if missing_lagged or missing_rankings:
        raise ValueError(
            "Exposure-gated cold-start inputs missing columns: "
            f"lagged={sorted(missing_lagged)}, rankings={sorted(missing_rankings)}"
        )
    if (
        lagged_priors["player_id"].duplicated().any()
        or revised_rankings["player_id"].duplicated().any()
    ):
        raise ValueError("Exposure-gated cold-start inputs contain duplicate players")
    forbidden = _FORBIDDEN_EXTERNAL_PRIOR_COLUMNS & set(revised_rankings)
    if forbidden:
        raise ValueError(f"Exposure-gated rankings contain target outcomes: {sorted(forbidden)}")
    gated = revised_rankings.loc[:, ["player_id", "blended_cold_start_prior"]].copy()
    gated["player_id"] = pd.to_numeric(gated["player_id"], errors="raise").astype(int)
    gated["blended_cold_start_prior"] = pd.to_numeric(
        gated["blended_cold_start_prior"], errors="raise"
    )
    if not np.isfinite(gated["blended_cold_start_prior"].to_numpy(dtype=float)).all():
        raise ValueError("Exposure-gated rankings contain non-finite prior values")
    output = lagged_priors.loc[:, ["player_id", "prior_rapm_mean", "prior_available"]].copy()
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    output = output.merge(gated, on="player_id", how="left", validate="one_to_one")
    matched = output["blended_cold_start_prior"].notna()
    returning = output["prior_available"].astype(bool)
    if (matched & returning).any():
        raise ValueError("Exposure-gated rankings overlap players with a lagged RAPM prior")
    if len(gated) != int(matched.sum()):
        raise ValueError(
            "Exposure-gated rankings include players absent from frozen prior universe"
        )
    output["prior_branch"] = np.select(
        [returning, matched],
        ["lagged_rapm", "exposure_gated_cold_start"],
        default="zero_cold_start",
    )
    output.loc[matched, "prior_rapm_mean"] = output.loc[matched, "blended_cold_start_prior"]
    return output.drop(columns="blended_cold_start_prior").sort_values(
        "player_id", kind="stable"
    ).reset_index(drop=True)


def train_frozen_exposure_gated_cold_start_prior_evaluation(
    *,
    season: str = DEFAULT_SEASON,
    exposure_gated_run_id: str | None = None,
    reference_prior_run_id: str | None = None,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> tuple[FrozenPriorEvaluationManifest, Path]:
    """Evaluate the continuous replacement/draft cold-start blend as a frozen prior."""

    target_season = validate_season(season)
    artifact_root = Path(artifacts_dir)
    reference_root = _resolve_run(
        artifact_root / "prior_rapm" / target_season,
        reference_prior_run_id,
    )
    reference_metadata = _validate_forward_prior_run(reference_root, target_season)
    gated_root = _resolve_run(
        artifact_root / "exposure_gated_cold_start" / target_season,
        exposure_gated_run_id,
    )
    gated_metadata = validate_exposure_gated_cold_start_prior(gated_root)
    if gated_metadata.get("target_season") != target_season:
        raise ValueError("Exposure-gated cold-start target season does not match evaluation")
    if gated_metadata.get("source_season") != _previous_season(target_season):
        raise ValueError("Exposure-gated cold-start prior does not end with the source season")
    if gated_metadata.get("target_outcomes_used_for_fit") is not False:
        raise ValueError("Exposure-gated cold-start prior uses target-season outcomes")
    lagged_priors = pd.read_parquet(reference_root / "target_player_priors.parquet")
    revised_rankings = pd.read_parquet(gated_root / "revised_rookie_rankings.parquet")
    frozen_priors = exposure_gated_cold_start_prior_frame(lagged_priors, revised_rankings)
    branch_counts = frozen_priors.groupby("prior_branch").size().to_dict()
    evaluation = run_frozen_lagged_evaluation(
        season=target_season,
        prior_run_dir=reference_root,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        player_priors_override=frozen_priors,
        source_state_overrides={
            "prior_run_id": gated_metadata["run_id"],
            "player_prior_method": (
                "2024-25-or-earlier draft rate blended continuously with a 2024-25-or-earlier "
                "pooled replacement token for first-NBA-season players; otherwise lagged RAPM"
            ),
            "cold_start_prior": "exposure-gated draft/replacement blend for first-year players",
            "exposure_gated_cold_start_run_id": gated_metadata["run_id"],
            "exposure_gated_cold_start_manifest_sha256": _sha256_file(
                gated_root / "manifest.json"
            ),
            "replacement_rapm": gated_metadata["replacement_rapm"],
            "reference_lagged_prior_run_id": str(reference_metadata["run_id"]),
            "reference_lagged_prior_manifest_sha256": _sha256_file(
                reference_root / "manifest.json"
            ),
            "prior_branch_counts": branch_counts,
        },
        evaluation_model="frozen_exposure_gated_cold_start_prior",
    )
    return _write_run(
        evaluation,
        prior_root=gated_root,
        analytical_dir=Path(analytical_dir),
        curated_dir=Path(curated_dir),
        artifacts_dir=artifact_root,
        run_prefix="frozen-exposure-gated-cold-start-prior",
        manifest_model="frozen_exposure_gated_cold_start_prior",
    )


def validate_frozen_prior_evaluation_run(
    run_dir: Path | str,
) -> FrozenPriorEvaluationManifest:
    """Validate artifact hashes, rows, cohort separation, and team coverage."""

    root = Path(run_dir)
    manifest = FrozenPriorEvaluationManifest.model_validate_json(
        (root / "manifest.json").read_text()
    )
    expected = {artifact.filename for artifact in manifest.artifacts}
    actual = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    if actual != expected:
        raise ValueError("Frozen evaluation files do not match manifest")
    required = {
        "cohort_metrics.parquet",
        "possession_predictions.parquet",
        "game_predictions.parquet",
        "regular_game_predictions.parquet",
        "team_net_rating_predictions.parquet",
        "team_net_rating_metrics.parquet",
        "team_win_predictions.parquet",
        "team_win_metrics.parquet",
        "pythagorean_calibration_team_seasons.parquet",
        "frozen_player_priors.parquet",
        "source_state.json",
        "metadata.json",
    }
    if not required <= expected:
        raise ValueError("Frozen evaluation is missing required artifacts")
    records = {artifact.filename: artifact for artifact in manifest.artifacts}
    for filename, record in records.items():
        path = root / filename
        if path.stat().st_size != record.byte_count or _sha256_file(path) != record.sha256:
            raise ValueError(f"Frozen evaluation artifact integrity changed: {filename}")
        if record.row_count is not None and len(pd.read_parquet(path)) != record.row_count:
            raise ValueError(f"Frozen evaluation artifact row count changed: {filename}")
    metrics = pd.read_parquet(root / "cohort_metrics.parquet")
    if set(metrics["cohort"]) != {"regular_season", "playoffs"}:
        raise ValueError("Frozen evaluation must report regular season and playoffs separately")
    teams = pd.read_parquet(root / "team_net_rating_predictions.parquet")
    wins = pd.read_parquet(root / "team_win_predictions.parquet")
    if len(teams) != manifest.team_count or len(wins) != manifest.team_count:
        raise ValueError("Frozen evaluation team counts changed")
    source = json.loads((root / "source_state.json").read_text())
    if source.get("target_season_refit") is not False:
        raise ValueError("Frozen evaluation source state permits a target-season refit")
    return manifest


def _validate_forward_prior_run(root: Path, season: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Forward prior manifest not found: {manifest_path}")
    metadata = json.loads(manifest_path.read_text())
    if metadata.get("model") != "forward_lagged_prior_centered_ridge_rapm":
        raise ValueError("Frozen baseline requires a forward lagged-prior RAPM run")
    if metadata.get("target_season") != season:
        raise ValueError("Forward prior target season does not match evaluation")
    if metadata.get("historical_training_variant") != "regular_only":
        raise ValueError("Frozen baseline requires the regular-only historical prior")
    return metadata


def _validate_frozen_priors(
    published: pd.DataFrame,
    source_coefficients: pd.DataFrame,
    *,
    target_season: str,
    source_season: str,
) -> pd.DataFrame:
    required = {"player_id", "prior_rapm_mean", "prior_available"}
    missing = required - set(published)
    if missing:
        raise ValueError(f"Published target priors missing columns: {sorted(missing)}")
    if published["player_id"].duplicated().any():
        raise ValueError("Published target priors contain duplicate players")
    source = source_coefficients.rename(columns={"rapm": "source_rapm"})
    checked = published.merge(source, on="player_id", how="left", validate="one_to_one")
    available = checked["prior_available"].astype(bool)
    if checked.loc[available, "source_rapm"].isna().any() or not np.allclose(
        checked.loc[available, "prior_rapm_mean"],
        checked.loc[available, "source_rapm"],
    ):
        raise ValueError("Published priors differ from the completed source-season state")
    if not np.allclose(checked.loc[~available, "prior_rapm_mean"], 0.0):
        raise ValueError("Cold-start lagged priors must equal zero")
    output = published.copy()
    output["target_season"] = target_season
    output["source_season"] = source_season
    output["prior_method"] = "regular_only_lagged_rapm"
    return output.sort_values("player_id", kind="stable").reset_index(drop=True)


def _validate_external_frozen_priors(
    priors: pd.DataFrame,
    *,
    target_season: str,
    source_season: str,
) -> pd.DataFrame:
    """Validate a label-free player-prior table supplied by another frozen model."""

    required = {"player_id", "prior_rapm_mean"}
    missing = required - set(priors)
    if missing:
        raise ValueError(f"External frozen prior table missing columns: {sorted(missing)}")
    forbidden = _FORBIDDEN_EXTERNAL_PRIOR_COLUMNS & set(priors)
    if forbidden:
        raise ValueError(
            f"External frozen prior table contains target outcomes: {sorted(forbidden)}"
        )
    if priors["player_id"].duplicated().any():
        raise ValueError("External frozen prior table contains duplicate players")
    values = priors["prior_rapm_mean"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("External frozen prior means must be finite")
    output = priors.copy()
    if "prior_available" not in output:
        output["prior_available"] = True
    output["target_season"] = target_season
    output["source_season"] = source_season
    return output.sort_values("player_id", kind="stable").reset_index(drop=True)


def _recover_home_intercept(stints: pd.DataFrame, coefficients: pd.DataFrame) -> float:
    values = dict(zip(coefficients["player_id"].astype(int), coefficients["rapm"], strict=True))
    lineup_effect = np.array(
        [
            sum(values[int(player)] for player in home)
            - sum(values[int(player)] for player in away)
            for home, away in zip(
                stints["home_player_ids"], stints["away_player_ids"], strict=True
            )
        ],
        dtype=float,
    )
    residual = stints["target_home_net_rating"].to_numpy(dtype=float) - lineup_effect
    return float(np.average(residual, weights=stints["possessions"].to_numpy(dtype=float)))


def _read_playoff_possessions(
    season: str,
    curated_dir: Path | str,
) -> tuple[pd.DataFrame, Path]:
    partition = CuratedPartition(
        table="possession_segments", season=season, season_type="playoffs"
    )
    manifest = read_curated_partition_manifest(partition, curated_dir)
    validate_curated_partition(manifest, curated_dir)
    partition_dir = CuratedDatasetLayout(Path(curated_dir)).partition_dir(partition)
    return single_lineup_possessions_frame(pd.read_parquet(partition_dir)), partition_dir


def _read_regular_possessions(
    season: str,
    *,
    analytical_dir: Path | str,
    curated_dir: Path | str,
) -> tuple[pd.DataFrame, Path]:
    partition = CuratedPartition(
        table="possession_segments", season=season, season_type="regular"
    )
    manifest = read_curated_partition_manifest(partition, curated_dir)
    validate_curated_partition(manifest, curated_dir)
    partition_dir = CuratedDatasetLayout(Path(curated_dir)).partition_dir(partition)
    return (
        single_lineup_possessions_frame(pd.read_parquet(partition_dir)),
        partition_dir / "_manifest.json",
    )


def _game_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    games = (
        predictions.groupby(["cohort", "game_id"], as_index=False, sort=False)
        .agg(
            eligible_possession_count=("possession_id", "size"),
            actual_home_margin=("target_home_margin", "sum"),
            predicted_home_margin=("prediction_home_margin", "sum"),
            unknown_player_exposures=("unknown_player_exposures", "sum"),
        )
        .reset_index(drop=True)
    )
    games["margin_error"] = games["predicted_home_margin"] - games["actual_home_margin"]
    return games


def _regular_stint_predictions(
    stints: pd.DataFrame,
    priors: pd.DataFrame,
    *,
    source_home_intercept: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior_map = dict(zip(priors["player_id"].astype(int), priors["prior_rapm_mean"], strict=True))
    home_effect = np.array(
        [
            sum(prior_map.get(int(player), 0.0) for player in home)
            - sum(prior_map.get(int(player), 0.0) for player in away)
            + source_home_intercept
            for home, away in zip(
                stints["home_player_ids"], stints["away_player_ids"], strict=True
            )
        ],
        dtype=float,
    )
    base = stints.loc[
        :,
        [
            "home_team_id",
            "away_team_id",
            "home_team_tricode",
            "away_team_tricode",
            "possessions",
            "home_margin",
        ],
    ].copy()
    base["predicted_home_margin"] = home_effect * base["possessions"] / 100.0
    game_columns = [
        "game_id",
        "home_team_id",
        "away_team_id",
        "home_team_tricode",
        "away_team_tricode",
    ]
    game_source = stints.loc[:, game_columns].copy()
    game_source["actual_home_margin"] = base["home_margin"].to_numpy(dtype=float)
    game_source["predicted_home_margin"] = base["predicted_home_margin"].to_numpy(dtype=float)
    games = game_source.groupby(game_columns, as_index=False, sort=False).agg(
        actual_home_margin=("actual_home_margin", "sum"),
        predicted_home_margin=("predicted_home_margin", "sum"),
    )
    if games["actual_home_margin"].eq(0).any():
        raise ValueError("Regular-season evaluation contains a tied actual final margin")
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
    away["actual_margin"] *= -1.0
    away["predicted_margin"] *= -1.0
    teams = pd.concat([home, away], ignore_index=True).groupby(
        ["team_id", "team_tricode"], as_index=False
    ).agg(
        possessions=("possessions", "sum"),
        actual_total_margin=("actual_margin", "sum"),
        predicted_total_margin=("predicted_margin", "sum"),
    )
    teams["actual_net_rating"] = 100.0 * teams["actual_total_margin"] / teams["possessions"]
    teams["predicted_net_rating"] = (
        100.0 * teams["predicted_total_margin"] / teams["possessions"]
    )
    teams["net_rating_error"] = teams["predicted_net_rating"] - teams["actual_net_rating"]
    return (
        games.sort_values("game_id", kind="stable").reset_index(drop=True),
        teams.sort_values("team_id", kind="stable").reset_index(drop=True),
    )


def _team_net_rating_metrics(
    predictions: pd.DataFrame,
    *,
    model: str = "frozen_lagged_rapm",
) -> pd.DataFrame:
    actual = predictions["actual_net_rating"].to_numpy(dtype=float)
    predicted = predictions["predicted_net_rating"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "model": model,
                "cohort": "regular_season_teams",
                "team_count": len(predictions),
                "net_rating_rmse": rmse(actual, predicted),
                "net_rating_mae": mean_absolute_error(actual, predicted),
                "pearson_correlation": float(pearsonr(actual, predicted).statistic),
                "spearman_rank_correlation": float(spearmanr(actual, predicted).statistic),
            }
        ]
    )


def _team_win_evaluation(
    games: pd.DataFrame,
    team_net_ratings: pd.DataFrame,
    pythagorean_model: PythagoreanWinModel,
    *,
    model: str = "frozen_lagged_rapm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    home = games.loc[
        :, ["home_team_id", "home_team_tricode", "actual_home_win", "predicted_home_win"]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_team_tricode": "team_tricode",
            "actual_home_win": "actual_win",
            "predicted_home_win": "predicted_win",
        }
    )
    away = games.loc[
        :, ["away_team_id", "away_team_tricode", "actual_home_win", "predicted_home_win"]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_team_tricode": "team_tricode",
            "actual_home_win": "actual_win",
            "predicted_home_win": "predicted_win",
        }
    )
    away["actual_win"] = ~away["actual_win"].astype(bool)
    away["predicted_win"] = ~away["predicted_win"].astype(bool)
    target = pd.concat([home, away], ignore_index=True).groupby(
        ["team_id", "team_tricode"], as_index=False
    ).agg(
        games=("actual_win", "size"),
        wins=("actual_win", "sum"),
        predicted_game_winner_count=("predicted_win", "sum"),
    )
    target = target.merge(
        team_net_ratings.loc[
            :, ["team_id", "actual_net_rating", "predicted_net_rating"]
        ],
        on="team_id",
        how="inner",
        validate="one_to_one",
    )
    target["losses"] = target["games"] - target["wins"]
    target["win_pct"] = target["wins"] / target["games"]
    target["pythagorean_win_pct"] = pythagorean_model.predict_win_pct(
        target["predicted_net_rating"]
    )
    target["pythagorean_wins"] = target["games"] * target["pythagorean_win_pct"]
    target["pythagorean_losses"] = target["games"] - target["pythagorean_wins"]
    target["pythagorean_win_error"] = target["pythagorean_wins"] - target["wins"]
    target["pythagorean_win_pct_error"] = target["pythagorean_win_pct"] - target["win_pct"]
    target["game_winner_count_error"] = target["predicted_game_winner_count"] - target["wins"]
    target["actual_rank"] = target["wins"].rank(method="min", ascending=False).astype(int)
    target["pythagorean_rank"] = (
        target["pythagorean_wins"].rank(method="min", ascending=False).astype(int)
    )
    metrics = pd.DataFrame(
        [
            {
                "model": model,
                "cohort": "regular_season_teams",
                "team_count": len(target),
                "pythagorean_win_total_rmse": rmse(
                    target["wins"], target["pythagorean_wins"]
                ),
                "pythagorean_win_total_mae": mean_absolute_error(
                    target["wins"].to_numpy(dtype=float),
                    target["pythagorean_wins"].to_numpy(dtype=float),
                ),
                "pythagorean_win_pct_rmse": rmse(
                    target["win_pct"], target["pythagorean_win_pct"]
                ),
                "pythagorean_win_pct_mae": mean_absolute_error(
                    target["win_pct"].to_numpy(dtype=float),
                    target["pythagorean_win_pct"].to_numpy(dtype=float),
                ),
                "pythagorean_spearman_rank_correlation": float(
                    spearmanr(target["wins"], target["pythagorean_wins"]).statistic
                ),
                "raw_game_winner_count_rmse": rmse(
                    target["wins"], target["predicted_game_winner_count"]
                ),
                "raw_game_winner_count_mae": mean_absolute_error(
                    target["wins"].to_numpy(dtype=float),
                    target["predicted_game_winner_count"].to_numpy(dtype=float),
                ),
                "actual_league_win_total": float(target["wins"].sum()),
                "pythagorean_league_win_total": float(target["pythagorean_wins"].sum()),
                "raw_game_winner_league_win_total": float(
                    target["predicted_game_winner_count"].sum()
                ),
                "prediction_rule": (
                    "pythagorean wins from forward historical net-rating mapping"
                ),
                "win_model_intercept": pythagorean_model.intercept,
                "win_model_net_rating_slope": pythagorean_model.net_rating_slope,
                "win_model_wins_per_net_rating_point_82_games": (
                    82.0 * pythagorean_model.net_rating_slope
                ),
                "win_model_training_last_season": pythagorean_model.training_seasons[-1],
                "win_model_training_team_season_count": (
                    pythagorean_model.training_team_season_count
                ),
                "predicted_tie_game_count": int(games["predicted_tie"].sum()),
            }
        ]
    )
    return target.sort_values("team_id", kind="stable").reset_index(drop=True), metrics


def fit_pythagorean_win_model(team_seasons: pd.DataFrame) -> PythagoreanWinModel:
    """Fit the forward-safe historical NetRtg-to-win relationship."""

    required = {"season", "games", "wins", "win_pct", "net_rating"}
    missing = required - set(team_seasons)
    if missing:
        raise ValueError(f"Pythagorean calibration rows missing columns: {sorted(missing)}")
    if team_seasons.empty:
        raise ValueError("Pythagorean calibration requires historical team-seasons")
    weights = np.sqrt(team_seasons["games"].to_numpy(dtype=float))
    design = np.column_stack(
        [
            np.ones(len(team_seasons), dtype=float),
            team_seasons["net_rating"].to_numpy(dtype=float),
        ]
    )
    target = team_seasons["win_pct"].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * weights[:, None], target * weights, rcond=None
    )
    if rank != 2:
        raise ValueError("Pythagorean win model design is rank deficient")
    predicted_win_pct = np.clip(design @ coefficients, 0.0, 1.0)
    predicted_wins = team_seasons["games"].to_numpy(dtype=float) * predicted_win_pct
    seasons = tuple(
        sorted(team_seasons["season"].astype(str).unique(), key=lambda value: int(value[:4]))
    )
    return PythagoreanWinModel(
        intercept=float(coefficients[0]),
        net_rating_slope=float(coefficients[1]),
        training_seasons=seasons,
        training_team_season_count=len(team_seasons),
        historical_win_total_rmse=rmse(
            team_seasons["wins"].to_numpy(dtype=float), predicted_wins
        ),
    )


def _historical_team_seasons(
    *,
    analytical_dir: Path | str,
    through_season: str,
) -> pd.DataFrame:
    root = Path(analytical_dir) / "rapm_stints"
    paths = sorted(
        root.glob("*/regular/part-00000.parquet"),
        key=lambda path: int(path.parents[1].name[:4]),
    )
    frames = []
    for path in paths:
        season = path.parents[1].name
        if int(season[:4]) > int(through_season[:4]):
            continue
        frames.append(_actual_team_season_frame(pd.read_parquet(path), season=season))
    if not frames:
        raise ValueError("No historical team-seasons are available for win calibration")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["season", "team_id"], kind="stable"
    ).reset_index(drop=True)


def _actual_team_season_frame(stints: pd.DataFrame, *, season: str) -> pd.DataFrame:
    game_keys = [
        "game_id",
        "home_team_id",
        "away_team_id",
        "home_team_tricode",
        "away_team_tricode",
    ]
    games = stints.groupby(game_keys, as_index=False, sort=False).agg(
        home_margin=("home_margin", "sum")
    )
    home_games = games.loc[
        :, ["home_team_id", "home_team_tricode", "home_margin"]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_team_tricode": "team_tricode",
            "home_margin": "margin",
        }
    )
    away_games = games.loc[
        :, ["away_team_id", "away_team_tricode", "home_margin"]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_team_tricode": "team_tricode",
            "home_margin": "margin",
        }
    )
    away_games["margin"] *= -1.0
    records = pd.concat([home_games, away_games], ignore_index=True)
    records["win"] = records["margin"].gt(0).astype(int)
    win_rows = records.groupby(["team_id", "team_tricode"], as_index=False).agg(
        games=("win", "size"), wins=("win", "sum")
    )

    base = stints.loc[
        :,
        [
            "home_team_id",
            "away_team_id",
            "home_team_tricode",
            "away_team_tricode",
            "possessions",
            "home_margin",
        ],
    ]
    home = base.loc[
        :, ["home_team_id", "home_team_tricode", "possessions", "home_margin"]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_team_tricode": "team_tricode",
            "home_margin": "margin",
        }
    )
    away = base.loc[
        :, ["away_team_id", "away_team_tricode", "possessions", "home_margin"]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_team_tricode": "team_tricode",
            "home_margin": "margin",
        }
    )
    away["margin"] *= -1.0
    net_rows = pd.concat([home, away], ignore_index=True).groupby(
        ["team_id", "team_tricode"], as_index=False
    ).agg(possessions=("possessions", "sum"), total_margin=("margin", "sum"))
    output = win_rows.merge(
        net_rows,
        on=["team_id", "team_tricode"],
        validate="one_to_one",
    )
    output["season"] = season
    output["win_pct"] = output["wins"] / output["games"]
    output["net_rating"] = 100.0 * output["total_margin"] / output["possessions"]
    return output


def _write_run(
    evaluation: FrozenEvaluation,
    *,
    prior_root: Path,
    analytical_dir: Path,
    curated_dir: Path,
    artifacts_dir: Path,
    run_prefix: str = "frozen-lagged-prior",
    manifest_model: Literal[
        "frozen_regular_only_lagged_rapm",
        "frozen_one_year_rapm_no_priors",
        "frozen_three_year_rapm_no_priors",
        "frozen_aging_prior",
        "frozen_combined_box_score_prior",
        "frozen_draft_cold_start_prior",
        "frozen_exposure_gated_cold_start_prior",
    ] = "frozen_regular_only_lagged_rapm",
) -> tuple[FrozenPriorEvaluationManifest, Path]:
    now = datetime.now(UTC)
    season = str(evaluation.source_state["target_season"])
    source_season = str(evaluation.source_state["source_season"])
    run_id = f"{run_prefix}-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    season_dir = artifacts_dir / "frozen_prior_evaluation" / season
    run_dir = season_dir / run_id
    temporary = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        parquet_outputs = {
            "cohort_metrics.parquet": evaluation.cohort_metrics,
            "possession_predictions.parquet": evaluation.possession_predictions,
            "game_predictions.parquet": evaluation.game_predictions,
            "regular_game_predictions.parquet": evaluation.regular_game_predictions,
            "team_net_rating_predictions.parquet": evaluation.team_net_rating_predictions,
            "team_net_rating_metrics.parquet": evaluation.team_net_rating_metrics,
            "team_win_predictions.parquet": evaluation.team_win_predictions,
            "team_win_metrics.parquet": evaluation.team_win_metrics,
            "pythagorean_calibration_team_seasons.parquet": (
                evaluation.pythagorean_calibration_team_seasons
            ),
            "frozen_player_priors.parquet": evaluation.frozen_player_priors,
        }
        for filename, frame in parquet_outputs.items():
            frame.to_parquet(temporary / filename, index=False)
        (temporary / "source_state.json").write_text(
            json.dumps(evaluation.source_state, indent=2, sort_keys=True) + "\n"
        )
        metadata = {
            "run_id": run_id,
            "season": season,
            "model": manifest_model,
            "source_season": source_season,
            "prior_run_id": evaluation.source_state["prior_run_id"],
            "information_boundary": "all fitted information ends with source regular season",
            "oracle_information": "target-season realized lineup exposure only",
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        artifacts = tuple(
            ArtifactRecord(
                filename=path.name,
                row_count=(
                    len(parquet_outputs[path.name]) if path.name in parquet_outputs else None
                ),
                byte_count=path.stat().st_size,
                sha256=_sha256_file(path),
            )
            for path in sorted(temporary.iterdir())
            if path.is_file()
        )
        playoff_partition = CuratedPartition(
            table="possession_segments", season=season, season_type="playoffs"
        )
        playoff_dir = CuratedDatasetLayout(curated_dir).partition_dir(playoff_partition)
        manifest = FrozenPriorEvaluationManifest(
            run_id=run_id,
            created_at=now,
            season=season,
            model=manifest_model,
            source_season=source_season,
            evaluation_code_version=frozen_prior_code_fingerprint(),
            prior_run_id=str(evaluation.source_state["prior_run_id"]),
            prior_manifest_sha256=_sha256_file(prior_root / "manifest.json"),
            source_rapm_stints_manifest_sha256=_sha256_file(
                analytical_dir / "rapm_stints" / source_season / "regular" / "_manifest.json"
            ),
            source_possessions_manifest_sha256=str(
                evaluation.source_state["source_possessions_manifest_sha256"]
            ),
            regular_possessions_manifest_sha256=str(
                evaluation.source_state["target_regular_possessions_manifest_sha256"]
            ),
            regular_rapm_stints_manifest_sha256=_sha256_file(
                analytical_dir / "rapm_stints" / season / "regular" / "_manifest.json"
            ),
            playoff_segments_manifest_sha256=_sha256_file(playoff_dir / "_manifest.json"),
            player_prior_count=len(evaluation.frozen_player_priors),
            regular_game_count=int(
                evaluation.cohort_metrics.loc[
                    evaluation.cohort_metrics["cohort"].eq("regular_season"), "game_count"
                ].item()
            ),
            playoff_game_count=int(
                evaluation.cohort_metrics.loc[
                    evaluation.cohort_metrics["cohort"].eq("playoffs"), "game_count"
                ].item()
            ),
            team_count=len(evaluation.team_net_rating_predictions),
            artifacts=artifacts,
        )
        (temporary / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        temporary.replace(run_dir)
        validate_frozen_prior_evaluation_run(run_dir)
        latest = season_dir / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return manifest, run_dir
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _resolve_run(season_dir: Path, run_id: str | None) -> Path:
    if run_id is None:
        latest = season_dir / "latest.json"
        if not latest.is_file():
            raise ValueError(f"No latest run pointer found: {latest}")
        run_id = str(json.loads(latest.read_text())["run_id"])
    run_dir = season_dir / run_id
    if not run_dir.is_dir():
        raise ValueError(f"Model run not found: {run_dir}")
    return run_dir


def _previous_season(season: str) -> str:
    year = int(season[:4]) - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-season frozen regular-only lagged RAPM prior."
    )
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--prior-run-id")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    return parser


def build_aging_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-season frozen age/draft/physical player prior."
    )
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--aging-run-id")
    parser.add_argument("--reference-prior-run-id")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    return parser


def build_combined_box_score_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-season frozen combined box-score prior."
    )
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--combined-run-id")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    return parser


def build_draft_cold_start_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-season frozen draft-informed cold-start prior."
    )
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--draft-run-id")
    parser.add_argument("--reference-prior-run-id")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    return parser


def build_exposure_gated_cold_start_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen exposure-gated cold-start prior."
    )
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--exposure-gated-run-id")
    parser.add_argument("--reference-prior-run-id")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, run_dir = train_frozen_lagged_prior_evaluation(
        season=args.season,
        prior_run_id=args.prior_run_id,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(
        f"Frozen lagged-RAPM evaluation: season={manifest.season}, "
        f"regular_games={manifest.regular_game_count}, "
        f"playoff_games={manifest.playoff_game_count}, "
        f"teams={manifest.team_count}; run={run_dir}{tracking_text}"
    )


def aging_main() -> None:
    args = build_aging_parser().parse_args()
    manifest, run_dir = train_frozen_aging_prior_evaluation(
        season=args.season,
        aging_run_id=args.aging_run_id,
        reference_prior_run_id=args.reference_prior_run_id,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(
        f"Frozen aging-prior evaluation: season={manifest.season}, "
        f"regular_games={manifest.regular_game_count}, "
        f"playoff_games={manifest.playoff_game_count}, "
        f"teams={manifest.team_count}; run={run_dir}{tracking_text}"
    )


def combined_box_score_main() -> None:
    args = build_combined_box_score_parser().parse_args()
    manifest, run_dir = train_frozen_combined_box_score_prior_evaluation(
        season=args.season,
        combined_run_id=args.combined_run_id,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(
        f"Frozen combined box-score-prior evaluation: season={manifest.season}, "
        f"regular_games={manifest.regular_game_count}, "
        f"playoff_games={manifest.playoff_game_count}, "
        f"teams={manifest.team_count}; run={run_dir}{tracking_text}"
    )


def draft_cold_start_main() -> None:
    args = build_draft_cold_start_parser().parse_args()
    manifest, run_dir = train_frozen_draft_cold_start_prior_evaluation(
        season=args.season,
        draft_run_id=args.draft_run_id,
        reference_prior_run_id=args.reference_prior_run_id,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(
        f"Frozen draft-cold-start evaluation: season={manifest.season}, "
        f"regular_games={manifest.regular_game_count}, "
        f"playoff_games={manifest.playoff_game_count}, "
        f"teams={manifest.team_count}; run={run_dir}{tracking_text}"
    )


def exposure_gated_cold_start_main() -> None:
    args = build_exposure_gated_cold_start_parser().parse_args()
    manifest, run_dir = train_frozen_exposure_gated_cold_start_prior_evaluation(
        season=args.season,
        exposure_gated_run_id=args.exposure_gated_run_id,
        reference_prior_run_id=args.reference_prior_run_id,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(
        f"Frozen exposure-gated cold-start evaluation: season={manifest.season}, "
        f"regular_games={manifest.regular_game_count}, "
        f"playoff_games={manifest.playoff_game_count}, "
        f"teams={manifest.team_count}; run={run_dir}{tracking_text}"
    )


if __name__ == "__main__":
    main()
