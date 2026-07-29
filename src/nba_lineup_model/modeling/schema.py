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
