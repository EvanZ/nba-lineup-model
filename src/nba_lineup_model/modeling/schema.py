from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nba_lineup_model.season.schema import (
    SEASON_PATTERN,
    SHA256_PATTERN,
    validate_season,
)

CODE_VERSION_PATTERN = r"^sha256:[0-9a-f]{64}$"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Datetime values must include a timezone")
    return value.astimezone(UTC)


class RapmStintManifest(BaseModel):
    """Integrity and source evidence for one canonical RAPM stint dataset."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: Literal["regular"] = "regular"
    created_at: datetime
    builder_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    lineup_stints_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    possession_segments_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    lineup_stints_input_fingerprint: str = Field(pattern=CODE_VERSION_PATTERN)
    possession_segments_input_fingerprint: str = Field(pattern=CODE_VERSION_PATTERN)
    source_game_count: int = Field(ge=1)
    source_stint_count: int = Field(ge=1)
    source_possession_count: int = Field(ge=1)
    multi_segment_possession_count: int = Field(ge=0)
    included_stint_count: int = Field(ge=1)
    excluded_zero_exposure_stint_count: int = Field(ge=0)
    player_count: int = Field(ge=10)
    home_offensive_possessions: float = Field(gt=0)
    away_offensive_possessions: float = Field(gt=0)
    part_filename: Literal["part-00000.parquet"] = "part-00000.parquet"
    row_count: int = Field(ge=1)
    byte_count: int = Field(gt=0)
    part_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_counts(self) -> RapmStintManifest:
        if (
            self.included_stint_count + self.excluded_zero_exposure_stint_count
            != self.source_stint_count
        ):
            raise ValueError("Included and excluded stints must conserve source stints")
        if self.row_count != self.included_stint_count:
            raise ValueError("RAPM part rows must match included stints")
        if self.multi_segment_possession_count > self.source_possession_count:
            raise ValueError("Multi-segment possessions exceed source possessions")
        return self


class ChronologicalSplitConfig(BaseModel):
    """Expanding-window validation and final-test configuration."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    cv_folds: int = Field(default=3, ge=1, le=10)
    validation_fraction: float = Field(default=0.1, gt=0, lt=0.5)
    test_fraction: float = Field(default=0.15, gt=0, lt=0.5)

    @model_validator(mode="after")
    def validate_capacity(self) -> ChronologicalSplitConfig:
        held_out = self.cv_folds * self.validation_fraction + self.test_fraction
        if held_out >= 0.8:
            raise ValueError("Chronological splits must leave at least 20% for training")
        return self


class ChronologicalFold(BaseModel):
    """Game-level boundaries for one expanding validation fold."""

    model_config = ConfigDict(strict=True, extra="forbid")

    fold: int = Field(ge=0)
    train_game_count: int = Field(ge=1)
    validation_game_count: int = Field(ge=1)
    train_first_game_id: str
    train_last_game_id: str
    validation_first_game_id: str
    validation_last_game_id: str


class NeuralPossessionManifest(BaseModel):
    """Integrity and source evidence for the neural possession dataset."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: Literal["regular"] = "regular"
    created_at: datetime
    builder_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    possession_segments_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    possession_segments_input_fingerprint: str = Field(pattern=CODE_VERSION_PATTERN)
    source_game_count: int = Field(ge=1)
    source_segment_count: int = Field(ge=1)
    source_possession_count: int = Field(ge=1)
    included_possession_count: int = Field(ge=1)
    excluded_multi_segment_possession_count: int = Field(ge=0)
    player_count: int = Field(ge=10)
    home_offense_possession_count: int = Field(ge=1)
    away_offense_possession_count: int = Field(ge=1)
    part_filename: Literal["part-00000.parquet"] = "part-00000.parquet"
    row_count: int = Field(ge=1)
    byte_count: int = Field(gt=0)
    part_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_counts(self) -> NeuralPossessionManifest:
        if (
            self.included_possession_count
            + self.excluded_multi_segment_possession_count
            != self.source_possession_count
        ):
            raise ValueError("Included and excluded neural possessions must conserve sources")
        if self.row_count != self.included_possession_count:
            raise ValueError("Neural possession part rows must match included possessions")
        if (
            self.home_offense_possession_count + self.away_offense_possession_count
            != self.included_possession_count
        ):
            raise ValueError("Home and away offense counts must conserve neural possessions")
        return self


class ArtifactRecord(BaseModel):
    """Integrity metadata for one model-run artifact."""

    model_config = ConfigDict(strict=True, extra="forbid")

    filename: str = Field(min_length=1)
    row_count: int | None = Field(default=None, ge=0)
    byte_count: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class BaselineRunManifest(BaseModel):
    """Reproducibility contract for one null/team/RAPM experiment."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    created_at: datetime
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: Literal["regular"] = "regular"
    modeling_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_part_sha256: str = Field(pattern=SHA256_PATTERN)
    stint_count: int = Field(ge=1)
    game_count: int = Field(ge=1)
    player_count: int = Field(ge=10)
    team_count: int = Field(ge=2)
    split_config: ChronologicalSplitConfig
    folds: tuple[ChronologicalFold, ...] = Field(min_length=1)
    final_train_game_count: int = Field(ge=1)
    final_test_game_count: int = Field(ge=1)
    lambda_grid: tuple[float, ...] = Field(min_length=2)
    selected_team_lambda: float = Field(ge=0)
    selected_rapm_lambda: float = Field(ge=0)
    minimum_ranking_possessions: float = Field(ge=0)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=1)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @field_validator("lambda_grid")
    @classmethod
    def validate_lambdas(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(value < 0 for value in values):
            raise ValueError("Lambda values must be non-negative")
        if len(set(values)) != len(values):
            raise ValueError("Lambda values must be unique")
        return values

    @model_validator(mode="after")
    def validate_folds(self) -> BaselineRunManifest:
        if len(self.folds) != self.split_config.cv_folds:
            raise ValueError("Fold records must match the split configuration")
        if self.final_train_game_count + self.final_test_game_count != self.game_count:
            raise ValueError("Final train and test games must conserve all games")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Model artifact filenames must be unique")
        return self


class BayesianRapmRunManifest(BaseModel):
    """Reproducibility contract for one exact Bayesian RAPM run."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    created_at: datetime
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: Literal["regular"] = "regular"
    bayesian_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    source_model_run_id: str = Field(min_length=1)
    source_model_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_part_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_rapm_lambda: float = Field(gt=0)
    posterior_draws: int = Field(ge=1)
    posterior_seed: int = Field(ge=0)
    credible_interval_probability: float = Field(gt=0, lt=1)
    minimum_ranking_possessions: float = Field(ge=0)
    stint_count: int = Field(ge=1)
    game_count: int = Field(ge=1)
    player_count: int = Field(ge=10)
    final_train_game_count: int = Field(ge=1)
    final_test_game_count: int = Field(ge=1)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=1)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_run(self) -> BayesianRapmRunManifest:
        if self.final_train_game_count + self.final_test_game_count != self.game_count:
            raise ValueError("Final train and test games must conserve all games")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Bayesian RAPM artifact filenames must be unique")
        return self


class NeuralRapmRunManifest(BaseModel):
    """Reproducibility contract for one additive neural RAPM run."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1, 2, 3] = 1
    run_id: str = Field(min_length=1)
    created_at: datetime
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: Literal["regular"] = "regular"
    architecture: Literal["additive", "deep_sets"]
    neural_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_part_sha256: str = Field(pattern=SHA256_PATTERN)
    possession_count: int = Field(ge=1)
    game_count: int = Field(ge=1)
    player_count: int = Field(ge=10)
    split_config: ChronologicalSplitConfig
    folds: tuple[ChronologicalFold, ...] = Field(min_length=1)
    selection_train_game_count: int = Field(ge=1)
    selection_validation_game_count: int = Field(ge=1)
    final_train_game_count: int = Field(ge=1)
    final_test_game_count: int = Field(ge=1)
    random_seed: int = Field(ge=0)
    batch_size: int = Field(ge=1)
    max_epochs: int = Field(ge=1)
    early_stopping_patience: int = Field(ge=0)
    selected_epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    learning_rate_candidates: tuple[float, ...] = ()
    weight_decay_candidates: tuple[float, ...] = ()
    hyperparameter_selection_metric: Literal[
        "validation_possession_weighted_mse"
    ] = "validation_possession_weighted_mse"
    player_embedding_dim: int | None = Field(default=None, ge=1)
    role_embedding_dim: int | None = Field(default=None, ge=1)
    player_hidden_dim: int | None = Field(default=None, ge=1)
    pooled_dim: int | None = Field(default=None, ge=1)
    lineup_hidden_dims: tuple[int, int] | None = None
    parameter_count: int | None = Field(default=None, ge=1)
    refit_seeds: tuple[int, ...] = ()
    leaderboard_seed: int | None = Field(default=None, ge=0)
    requested_accelerator: Literal["cpu", "mps", "auto"]
    resolved_accelerator: str = Field(min_length=1)
    target: Literal["offense_points_minus_defense_points"]
    minimum_ranking_possessions: float = Field(ge=0)
    torch_version: str = Field(min_length=1)
    lightning_version: str = Field(min_length=1)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=1)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_run(self) -> NeuralRapmRunManifest:
        if len(self.folds) != self.split_config.cv_folds:
            raise ValueError("Fold records must match the split configuration")
        if self.final_train_game_count + self.final_test_game_count != self.game_count:
            raise ValueError("Final train and test games must conserve all games")
        if self.selected_epochs > self.max_epochs:
            raise ValueError("Selected epochs cannot exceed the training limit")
        if self.schema_version >= 2:
            if not self.learning_rate_candidates or not self.weight_decay_candidates:
                raise ValueError("Grid-search manifests require candidate values")
            if any(value <= 0 for value in self.learning_rate_candidates):
                raise ValueError("Learning-rate candidates must be positive")
            if any(value < 0 for value in self.weight_decay_candidates):
                raise ValueError("Weight-decay candidates must be nonnegative")
            if len(set(self.learning_rate_candidates)) != len(
                self.learning_rate_candidates
            ):
                raise ValueError("Learning-rate candidates must be unique")
            if len(set(self.weight_decay_candidates)) != len(
                self.weight_decay_candidates
            ):
                raise ValueError("Weight-decay candidates must be unique")
            if self.learning_rate not in self.learning_rate_candidates:
                raise ValueError("Selected learning rate is absent from its grid")
            if self.weight_decay not in self.weight_decay_candidates:
                raise ValueError("Selected weight decay is absent from its grid")
        if self.schema_version >= 3:
            if self.architecture != "deep_sets":
                raise ValueError("Schema version 3 is reserved for Deep Sets")
            dimensions = (
                self.player_embedding_dim,
                self.role_embedding_dim,
                self.player_hidden_dim,
                self.pooled_dim,
                self.parameter_count,
            )
            if any(value is None for value in dimensions):
                raise ValueError("Deep Sets manifests require architecture dimensions")
            if self.lineup_hidden_dims is None or any(
                value < 1 for value in self.lineup_hidden_dims
            ):
                raise ValueError("Deep Sets manifests require lineup hidden dimensions")
            if len(self.refit_seeds) < 3 or len(set(self.refit_seeds)) != len(
                self.refit_seeds
            ):
                raise ValueError("Deep Sets requires at least three unique refit seeds")
            if self.leaderboard_seed not in self.refit_seeds:
                raise ValueError("Leaderboard seed must be one of the refit seeds")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Neural RAPM artifact filenames must be unique")
        return self


class CatBoostRunManifest(BaseModel):
    """Reproducibility contract for one categorical CatBoost lineup run."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    created_at: datetime
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: Literal["regular"] = "regular"
    architecture: Literal["categorical_player_state"] = "categorical_player_state"
    catboost_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_part_sha256: str = Field(pattern=SHA256_PATTERN)
    possession_count: int = Field(ge=1)
    game_count: int = Field(ge=1)
    player_count: int = Field(ge=10)
    feature_count: int = Field(ge=11)
    split_config: ChronologicalSplitConfig
    folds: tuple[ChronologicalFold, ...] = Field(min_length=1)
    selection_train_game_count: int = Field(ge=1)
    selection_validation_game_count: int = Field(ge=1)
    final_train_game_count: int = Field(ge=1)
    final_test_game_count: int = Field(ge=1)
    max_iterations: int = Field(ge=1)
    best_iteration: int = Field(ge=0)
    selected_tree_count: int = Field(ge=1)
    one_hot_max_size: Literal[3] = 3
    has_time: Literal[True] = True
    use_best_model: Literal[True] = True
    random_seed: int = Field(ge=0)
    resolved_learning_rate: float = Field(gt=0)
    target: Literal["offense_points_minus_defense_points"]
    catboost_version: str = Field(min_length=1)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=1)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_run(self) -> CatBoostRunManifest:
        if len(self.folds) != self.split_config.cv_folds:
            raise ValueError("Fold records must match the split configuration")
        if self.final_train_game_count + self.final_test_game_count != self.game_count:
            raise ValueError("Final train and test games must conserve all games")
        if self.feature_count != self.player_count + 1:
            raise ValueError("CatBoost features must be one per player plus home offense")
        if self.best_iteration + 1 != self.selected_tree_count:
            raise ValueError("Best iteration and selected tree count do not match")
        if self.selected_tree_count > self.max_iterations:
            raise ValueError("Selected trees cannot exceed the iteration ceiling")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("CatBoost artifact filenames must be unique")
        return self


class RapmBasePredictionManifest(BaseModel):
    """Integrity contract for stage-specific possession RAPM predictions."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: Literal["regular"] = "regular"
    created_at: datetime
    builder_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    source_rapm_run_id: str = Field(min_length=1)
    source_rapm_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    rapm_stints_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    rapm_stints_part_sha256: str = Field(pattern=SHA256_PATTERN)
    neural_possessions_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    neural_possessions_part_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_rapm_lambda: float = Field(ge=0)
    possession_count: int = Field(ge=1)
    game_count: int = Field(ge=1)
    player_count: int = Field(ge=10)
    split_config: ChronologicalSplitConfig
    folds: tuple[ChronologicalFold, ...] = Field(min_length=1)
    stage_count: int = Field(ge=3)
    prediction_row_count: int = Field(ge=1)
    in_sample_prediction_count: int = Field(ge=1)
    out_of_sample_prediction_count: int = Field(ge=1)
    player_coefficient_row_count: int = Field(ge=1)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=3)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_dataset(self) -> RapmBasePredictionManifest:
        if len(self.folds) != self.split_config.cv_folds:
            raise ValueError("Base-prediction folds must match the split configuration")
        if self.stage_count != self.split_config.cv_folds + 2:
            raise ValueError("Base-prediction stages must be CV folds plus final and all-season")
        if (
            self.in_sample_prediction_count
            + self.out_of_sample_prediction_count
            != self.prediction_row_count
        ):
            raise ValueError("Base-prediction sample roles must conserve rows")
        if self.player_coefficient_row_count != self.stage_count * self.player_count:
            raise ValueError("Every base-prediction stage must retain all player coefficients")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Base-prediction artifact filenames must be unique")
        return self


class RapmTransformerRunManifest(BaseModel):
    """Reproducibility contract for one frozen-RAPM Transformer residual run."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    created_at: datetime
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: Literal["regular"] = "regular"
    architecture: Literal["rapm_transformer"] = "rapm_transformer"
    transformer_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    source_rapm_run_id: str = Field(min_length=1)
    source_rapm_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_rapm_lambda: float = Field(ge=0)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_part_sha256: str = Field(pattern=SHA256_PATTERN)
    base_predictions_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    base_predictions_part_sha256: str = Field(pattern=SHA256_PATTERN)
    possession_count: int = Field(ge=1)
    game_count: int = Field(ge=1)
    player_count: int = Field(ge=10)
    split_config: ChronologicalSplitConfig
    folds: tuple[ChronologicalFold, ...] = Field(min_length=1)
    selection_train_game_count: int = Field(ge=1)
    selection_validation_game_count: int = Field(ge=1)
    final_train_game_count: int = Field(ge=1)
    final_test_game_count: int = Field(ge=1)
    random_seed: int = Field(ge=0)
    batch_size: int = Field(ge=1)
    max_epochs: int = Field(ge=1)
    early_stopping_patience: int = Field(ge=0)
    selected_epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    learning_rate_candidates: tuple[float, ...] = Field(min_length=1)
    weight_decay_candidates: tuple[float, ...] = Field(min_length=1)
    hyperparameter_selection_metric: Literal[
        "validation_possession_weighted_mse"
    ]
    d_model: int = Field(ge=1)
    attention_heads: int = Field(ge=1)
    transformer_layers: int = Field(ge=1)
    feedforward_dim: int = Field(ge=1)
    dropout: float = Field(ge=0, lt=1)
    parameter_count: int = Field(ge=1)
    refit_seeds: tuple[int, ...] = Field(min_length=3)
    leaderboard_seed: int = Field(ge=0)
    requested_accelerator: Literal["cpu", "mps", "auto"]
    resolved_accelerator: str = Field(min_length=1)
    target: Literal["offense_points_minus_defense_points"]
    torch_version: str = Field(min_length=1)
    lightning_version: str = Field(min_length=1)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=1)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_run(self) -> RapmTransformerRunManifest:
        if len(self.folds) != self.split_config.cv_folds:
            raise ValueError("Transformer folds must match the split configuration")
        if self.final_train_game_count + self.final_test_game_count != self.game_count:
            raise ValueError("Transformer final train and test games must conserve games")
        if self.selected_epochs > self.max_epochs:
            raise ValueError("Transformer selected epochs cannot exceed the training limit")
        if self.d_model % self.attention_heads != 0:
            raise ValueError("Transformer width must be divisible by attention heads")
        if len(set(self.learning_rate_candidates)) != len(
            self.learning_rate_candidates
        ):
            raise ValueError("Transformer learning-rate candidates must be unique")
        if len(set(self.weight_decay_candidates)) != len(
            self.weight_decay_candidates
        ):
            raise ValueError("Transformer weight-decay candidates must be unique")
        if self.learning_rate not in self.learning_rate_candidates:
            raise ValueError("Selected Transformer learning rate is absent from its grid")
        if self.weight_decay not in self.weight_decay_candidates:
            raise ValueError("Selected Transformer weight decay is absent from its grid")
        if len(set(self.refit_seeds)) != len(self.refit_seeds):
            raise ValueError("Transformer refit seeds must be unique")
        if self.leaderboard_seed not in self.refit_seeds:
            raise ValueError("Transformer leaderboard seed must be a refit seed")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Transformer artifact filenames must be unique")
        return self


class ModelEvaluationManifest(BaseModel):
    """Reproducibility contract for one cross-model evaluation report."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1, 2, 3] = 1
    run_id: str = Field(min_length=1)
    created_at: datetime
    season: str = Field(pattern=SEASON_PATTERN)
    evaluation_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    ridge_run_id: str = Field(min_length=1)
    ridge_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    bayesian_run_id: str = Field(min_length=1)
    bayesian_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    neural_run_id: str = Field(min_length=1)
    neural_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    neural_learning_rate: float | None = Field(default=None, gt=0)
    neural_weight_decay: float | None = Field(default=None, ge=0)
    neural_selected_epochs: int | None = Field(default=None, ge=1)
    deep_sets_run_id: str | None = None
    deep_sets_manifest_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    deep_sets_learning_rate: float | None = Field(default=None, gt=0)
    deep_sets_weight_decay: float | None = Field(default=None, ge=0)
    deep_sets_selected_epochs: int | None = Field(default=None, ge=1)
    deep_sets_leaderboard_seed: int | None = Field(default=None, ge=0)
    catboost_run_id: str | None = None
    catboost_manifest_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    catboost_max_iterations: int | None = Field(default=None, ge=1)
    catboost_best_iteration: int | None = Field(default=None, ge=0)
    catboost_selected_tree_count: int | None = Field(default=None, ge=1)
    catboost_resolved_learning_rate: float | None = Field(default=None, gt=0)
    rapm_transformer_run_id: str | None = None
    rapm_transformer_manifest_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    rapm_transformer_source_rapm_run_id: str | None = None
    rapm_transformer_learning_rate: float | None = Field(default=None, gt=0)
    rapm_transformer_weight_decay: float | None = Field(default=None, ge=0)
    rapm_transformer_selected_epochs: int | None = Field(default=None, ge=1)
    rapm_transformer_leaderboard_seed: int | None = Field(default=None, ge=0)
    regular_segments_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    regular_lineup_stints_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    playoff_segments_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    models: tuple[
        Literal[
            "ridge_rapm",
            "bayesian_rapm",
            "additive_neural",
            "deep_sets",
            "catboost",
            "rapm_transformer",
        ],
        ...,
    ] = Field(min_length=3)
    regular_holdout_game_count: int = Field(ge=1)
    regular_holdout_possession_count: int = Field(ge=1)
    playoff_game_count: int = Field(ge=1)
    playoff_possession_count: int = Field(ge=1)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=1)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_run(self) -> ModelEvaluationManifest:
        if len(set(self.models)) != len(self.models):
            raise ValueError("Evaluation model names must be unique")
        if "catboost" in self.models:
            catboost_values = (
                self.catboost_run_id,
                self.catboost_manifest_sha256,
                self.catboost_max_iterations,
                self.catboost_best_iteration,
                self.catboost_selected_tree_count,
                self.catboost_resolved_learning_rate,
            )
            if self.schema_version < 2 or any(
                value is None for value in catboost_values
            ):
                raise ValueError(
                    "CatBoost evaluation manifests require schema version 2 "
                    "and complete source parameters"
                )
            if (
                self.catboost_best_iteration is not None
                and self.catboost_selected_tree_count is not None
                and self.catboost_best_iteration + 1
                != self.catboost_selected_tree_count
            ):
                raise ValueError(
                    "CatBoost best iteration and selected tree count do not match"
                )
        if "rapm_transformer" in self.models:
            transformer_values = (
                self.rapm_transformer_run_id,
                self.rapm_transformer_manifest_sha256,
                self.rapm_transformer_source_rapm_run_id,
                self.rapm_transformer_learning_rate,
                self.rapm_transformer_weight_decay,
                self.rapm_transformer_selected_epochs,
                self.rapm_transformer_leaderboard_seed,
            )
            if self.schema_version < 3 or any(
                value is None for value in transformer_values
            ):
                raise ValueError(
                    "RAPM Transformer evaluation manifests require schema "
                    "version 3 and complete source parameters"
                )
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Evaluation artifact filenames must be unique")
        return self


class RapmDiagnosticsManifest(BaseModel):
    """Reproducibility contract for one RAPM stability diagnostic run."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    created_at: datetime
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: Literal["regular"] = "regular"
    diagnostics_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    source_model_run_id: str = Field(min_length=1)
    source_model_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_part_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_rapm_lambda: float = Field(gt=0)
    sensitivity_lambdas: tuple[float, ...] = Field(min_length=3)
    bootstrap_samples: int = Field(ge=1)
    bootstrap_seed: int = Field(ge=0)
    minimum_ranking_possessions: float = Field(ge=0)
    influence_player_count: int = Field(ge=1)
    influence_stints_per_player: int = Field(ge=1)
    delete_games_per_player: int = Field(ge=1)
    allocation_policies: tuple[
        Literal[
            "equal_segments",
            "starting_lineup",
            "terminal_lineup",
            "boundary_split",
            "exclude_multi_lineup",
        ],
        ...,
    ] = Field(min_length=2)
    player_count: int = Field(ge=10)
    game_count: int = Field(ge=1)
    stint_count: int = Field(ge=1)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=1)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @field_validator("sensitivity_lambdas")
    @classmethod
    def validate_lambdas(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(value <= 0 for value in values):
            raise ValueError("Sensitivity lambdas must be positive")
        if len(set(values)) != len(values):
            raise ValueError("Sensitivity lambdas must be unique")
        return values

    @model_validator(mode="after")
    def validate_diagnostics(self) -> RapmDiagnosticsManifest:
        if self.selected_rapm_lambda not in self.sensitivity_lambdas:
            raise ValueError("Sensitivity lambdas must include the selected lambda")
        if len(set(self.allocation_policies)) != len(self.allocation_policies):
            raise ValueError("Allocation policies must be unique")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Diagnostic artifact filenames must be unique")
        return self
