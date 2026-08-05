"""Possession-native, leakage-safe player box-score prior features."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nba_lineup_model.modeling.player_history import (
    player_history_code_fingerprint,
    validate_player_season_panel,
)
from nba_lineup_model.modeling.schema import CODE_VERSION_PATTERN, ArtifactRecord
from nba_lineup_model.season.schema import SHA256_PATTERN, validate_season

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_OUTPUT_DIR = Path("data/analytical/box_score_prior_panel")
LOW_EXPOSURE_POSSESSIONS = 500.0
ESTABLISHED_POSSESSIONS = 1_500.0
SHOOTING_PSEUDO_ATTEMPTS = {
    "effective_field_goal_percentage": 300.0,
    "three_point_percentage": 150.0,
    "free_throw_percentage": 100.0,
}
RATE_COLUMNS = {
    "field_goals_attempted": "fga",
    "three_pointers_attempted": "three_pa",
    "free_throws_attempted": "fta",
    "assists": "assists",
    "turnovers": "turnovers",
    "rebounds_offensive": "offensive_rebounds",
    "rebounds_defensive": "defensive_rebounds",
    "steals": "steals",
    "blocks": "blocks",
    "fouls_personal": "personal_fouls",
}
STATIC_FEATURE_COLUMNS = (
    "preseason_age",
    "preseason_nba_experience_years",
    "preseason_is_rookie",
    "preseason_years_since_draft",
    "draft_year",
    "draft_round",
    "draft_number",
    "is_undrafted",
    "height_inches",
    "weight_pounds",
    "listed_position",
)


class BoxScorePriorPanelManifest(BaseModel):
    """Integrity and leakage contract for box-score prior features."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    created_at: datetime
    builder_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    source_player_season_panel_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_player_season_panel_path: str
    seasons: tuple[str, ...] = Field(min_length=2)
    target_seasons: tuple[str, ...] = Field(min_length=1)
    row_count: int = Field(ge=1)
    model_feature_columns: tuple[str, ...] = Field(min_length=1)
    target_label_columns: tuple[str, ...] = Field(min_length=1)
    low_exposure_possessions: float = Field(gt=0)
    established_possessions: float = Field(gt=0)
    shooting_pseudo_attempts: dict[str, float]
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=3)

    @field_validator("seasons", "target_seasons")
    @classmethod
    def validate_season_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_season(value) for value in values)

    @model_validator(mode="after")
    def validate_contract(self) -> BoxScorePriorPanelManifest:
        if self.seasons != tuple(sorted(self.seasons, key=_season_start_year)):
            raise ValueError("Box-score prior seasons must be in chronological order")
        if self.target_seasons != self.seasons[1:]:
            raise ValueError("Target seasons must exclude only the first source season")
        if self.established_possessions <= self.low_exposure_possessions:
            raise ValueError("Established exposure threshold must exceed low-exposure threshold")
        if any(value <= 0 for value in self.shooting_pseudo_attempts.values()):
            raise ValueError("Shooting pseudo-attempts must be positive")
        if any(column.startswith("target_") for column in self.model_feature_columns):
            raise ValueError("Model feature columns must not include target labels")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Box-score prior artifact names must be unique")
        return self


def build_box_score_prior_panel(
    *,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[BoxScorePriorPanelManifest, Path]:
    """Build and atomically publish leakage-safe player box-score prior features."""

    source_path = Path(player_season_panel_path)
    source_manifest_path = source_path.parent / "_manifest.json"
    validate_player_season_panel(source_path.parent)
    player_seasons = pd.read_parquet(source_path)
    features, summaries, shooting_references = build_box_score_prior_features(player_seasons)
    seasons = tuple(sorted(player_seasons["season"].astype(str).unique(), key=_season_start_year))
    target = Path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid4().hex}"
    backup = target.parent / f".{target.name}.bak-{uuid4().hex}"
    temporary.mkdir()
    try:
        outputs = {
            "player_prior_features.parquet": features,
            "season_cohort_summary.parquet": summaries,
            "league_shooting_references.parquet": shooting_references,
        }
        for filename, frame in outputs.items():
            frame.to_parquet(temporary / filename, index=False)
        artifacts = tuple(
            ArtifactRecord(
                filename=filename,
                row_count=len(frame),
                byte_count=(temporary / filename).stat().st_size,
                sha256=_sha256_file(temporary / filename),
            )
            for filename, frame in outputs.items()
        )
        manifest = BoxScorePriorPanelManifest(
            created_at=datetime.now(UTC),
            builder_code_version=player_history_code_fingerprint((Path(__file__),)),
            source_player_season_panel_manifest_sha256=_sha256_file(source_manifest_path),
            source_player_season_panel_path=str(source_path),
            seasons=seasons,
            target_seasons=seasons[1:],
            row_count=len(features),
            model_feature_columns=model_feature_columns(),
            target_label_columns=("target_rapm", "target_rapm_possessions"),
            low_exposure_possessions=LOW_EXPOSURE_POSSESSIONS,
            established_possessions=ESTABLISHED_POSSESSIONS,
            shooting_pseudo_attempts=SHOOTING_PSEUDO_ATTEMPTS,
            artifacts=artifacts,
        )
        (temporary / "_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        validate_box_score_prior_panel(temporary)
        if target.exists():
            target.replace(backup)
        try:
            temporary.replace(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        validate_box_score_prior_panel(target)
        return manifest, target
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def build_box_score_prior_features(
    player_seasons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive target rows, cohort summaries, and source-season shooting references."""

    _validate_player_seasons(player_seasons)
    seasons = tuple(sorted(player_seasons["season"].astype(str).unique(), key=_season_start_year))
    source = player_seasons.copy()
    source["target_season"] = source["season"].map(_next_season)
    source = source.loc[source["target_season"].isin(seasons)].copy()
    targets = player_seasons.loc[player_seasons["season"].isin(seasons[1:])].copy()
    targets = targets.rename(columns={"season": "target_season"})
    source_dynamic = _source_dynamic_features(source)
    shooting_references = _league_shooting_references(source)
    source_dynamic = source_dynamic.merge(
        shooting_references,
        on="target_season",
        how="left",
        validate="many_to_one",
    )
    source_dynamic = _stabilize_shooting(source_dynamic)
    output = _target_frame(targets).merge(
        source_dynamic,
        on=["target_season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    output["has_prior_season"] = output["prior_source_season"].notna()
    output["prior_rapm_available"] = output["prior_rapm"].notna()
    output["prior_boxscore_features_available"] = output[
        "prior_boxscore_features_available"
    ].eq(True)
    output["prior_exposure_cohort"] = _exposure_cohort(output)
    output["prior_log_on_court_possessions"] = np.log1p(output["prior_on_court_possessions"])
    _validate_output(output, seasons)
    summaries = _cohort_summary(output)
    return output, summaries, shooting_references


def model_feature_columns() -> tuple[str, ...]:
    """Return the explicit, label-free first-pass box-score prior feature set."""

    rate_features = tuple(
        f"prior_{name}_per_100_on_court_possessions" for name in RATE_COLUMNS.values()
    )
    return (
        *STATIC_FEATURE_COLUMNS,
        "has_prior_season",
        "prior_rapm_available",
        "prior_boxscore_features_available",
        "prior_log_on_court_possessions",
        "prior_rapm",
        *rate_features,
        "prior_stabilized_effective_field_goal_percentage",
        "prior_stabilized_three_point_percentage",
        "prior_stabilized_free_throw_percentage",
    )


def validate_box_score_prior_panel(path: Path | str) -> BoxScorePriorPanelManifest:
    """Validate published artifacts, hashes, and label/feature separation."""

    root = Path(path)
    manifest = BoxScorePriorPanelManifest.model_validate_json((root / "_manifest.json").read_text())
    expected = {artifact.filename for artifact in manifest.artifacts}
    actual = {
        item.name for item in root.iterdir() if item.is_file() and item.name != "_manifest.json"
    }
    if actual != expected:
        raise ValueError("Box-score prior panel files do not match manifest")
    artifacts = {artifact.filename: artifact for artifact in manifest.artifacts}
    for filename, artifact in artifacts.items():
        output = root / filename
        if output.stat().st_size != artifact.byte_count or _sha256_file(output) != artifact.sha256:
            raise ValueError(f"Box-score prior artifact integrity changed: {filename}")
    features = pd.read_parquet(root / "player_prior_features.parquet")
    if len(features) != manifest.row_count:
        raise ValueError("Box-score prior row count changed")
    _validate_output(features, manifest.seasons)
    if tuple(features["target_season"].drop_duplicates()) != manifest.target_seasons:
        raise ValueError("Box-score prior target season order changed")
    return manifest


def _source_dynamic_features(source: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target_season",
        "season",
        "player_id",
        "rapm",
        "rapm_possessions",
        "boxscore_features_available",
        *RATE_COLUMNS,
        "field_goals_made",
        "three_pointers_made",
        "free_throws_made",
    ]
    output = source.loc[:, columns].rename(
        columns={
            "season": "prior_source_season",
            "rapm": "prior_rapm",
            "rapm_possessions": "prior_on_court_possessions",
            "boxscore_features_available": "prior_boxscore_features_available",
            **{column: f"prior_{column}" for column in RATE_COLUMNS},
            "field_goals_made": "prior_field_goals_made",
            "three_pointers_made": "prior_three_pointers_made",
            "free_throws_made": "prior_free_throws_made",
        }
    )
    denominator = output["prior_on_court_possessions"]
    for source_name, rate_name in RATE_COLUMNS.items():
        output[f"prior_{rate_name}_per_100_on_court_possessions"] = (
            _safe_rate(output[f"prior_{source_name}"], denominator) * 100.0
        )
    if output.duplicated(["target_season", "player_id"]).any():
        raise ValueError("Prior source rows must be unique by target season and player")
    return output


def _target_frame(targets: pd.DataFrame) -> pd.DataFrame:
    output = targets.loc[
        :,
        [
            "target_season",
            "player_id",
            "player_name",
            "rapm",
            "rapm_possessions",
            "age",
            "nba_experience_years",
            "is_rookie",
            "years_since_draft",
            "draft_year",
            "draft_round",
            "draft_number",
            "is_undrafted",
            "height_inches",
            "weight_pounds",
            "listed_position",
        ],
    ].rename(
        columns={
            "rapm": "target_rapm",
            "rapm_possessions": "target_rapm_possessions",
            "age": "preseason_age",
            "nba_experience_years": "preseason_nba_experience_years",
            "is_rookie": "preseason_is_rookie",
            "years_since_draft": "preseason_years_since_draft",
        }
    )
    if output.duplicated(["target_season", "player_id"]).any():
        raise ValueError("Target rows must be unique by season and player")
    return output


def _league_shooting_references(source: pd.DataFrame) -> pd.DataFrame:
    usable = source.loc[source["boxscore_features_available"].fillna(False)].copy()
    if usable.empty:
        raise ValueError("Box-score prior panel requires usable source box-score rows")
    grouped = usable.groupby("target_season", as_index=False).agg(
        field_goals_attempted=("field_goals_attempted", "sum"),
        field_goals_made=("field_goals_made", "sum"),
        three_pointers_attempted=("three_pointers_attempted", "sum"),
        three_pointers_made=("three_pointers_made", "sum"),
        free_throws_attempted=("free_throws_attempted", "sum"),
        free_throws_made=("free_throws_made", "sum"),
    )
    grouped["league_effective_field_goal_percentage"] = _safe_rate(
        grouped["field_goals_made"] + 0.5 * grouped["three_pointers_made"],
        grouped["field_goals_attempted"],
    )
    grouped["league_three_point_percentage"] = _safe_rate(
        grouped["three_pointers_made"], grouped["three_pointers_attempted"]
    )
    grouped["league_free_throw_percentage"] = _safe_rate(
        grouped["free_throws_made"], grouped["free_throws_attempted"]
    )
    if grouped.filter(like="percentage").isna().any().any():
        raise ValueError("League shooting references require positive attempt denominators")
    return grouped


def _stabilize_shooting(source: pd.DataFrame) -> pd.DataFrame:
    output = source.copy()
    has_box = output["prior_boxscore_features_available"].fillna(False)
    specifications = (
        (
            "effective_field_goal_percentage",
            output["prior_field_goals_made"] + 0.5 * output["prior_three_pointers_made"],
            output["prior_field_goals_attempted"],
        ),
        (
            "three_point_percentage",
            output["prior_three_pointers_made"],
            output["prior_three_pointers_attempted"],
        ),
        (
            "free_throw_percentage",
            output["prior_free_throws_made"],
            output["prior_free_throws_attempted"],
        ),
    )
    for metric, made, attempts in specifications:
        pseudo_attempts = SHOOTING_PSEUDO_ATTEMPTS[metric]
        league = output[f"league_{metric}"]
        values = (made + pseudo_attempts * league) / (attempts + pseudo_attempts)
        output[f"prior_stabilized_{metric}"] = values.where(has_box)
    return output


def _exposure_cohort(frame: pd.DataFrame) -> pd.Series:
    exposure = frame["prior_on_court_possessions"]
    return pd.Series(
        np.select(
            [
                ~frame["has_prior_season"],
                exposure.lt(LOW_EXPOSURE_POSSESSIONS),
                exposure.lt(ESTABLISHED_POSSESSIONS),
            ],
            ["no_prior", "low_exposure", "developing"],
            default="established",
        ),
        index=frame.index,
        dtype="string",
    )


def _cohort_summary(features: pd.DataFrame) -> pd.DataFrame:
    summary = features.groupby(["target_season", "prior_exposure_cohort"], as_index=False).agg(
        player_count=("player_id", "size"),
        prior_rapm_available_count=("prior_rapm_available", "sum"),
        prior_boxscore_available_count=("prior_boxscore_features_available", "sum"),
        median_prior_on_court_possessions=("prior_on_court_possessions", "median"),
        median_target_rapm_possessions=("target_rapm_possessions", "median"),
    )
    return summary.sort_values(["target_season", "prior_exposure_cohort"], kind="stable")


def _validate_player_seasons(player_seasons: pd.DataFrame) -> None:
    required = {
        "season",
        "player_id",
        "player_name",
        "rapm",
        "rapm_possessions",
        "boxscore_features_available",
        *RATE_COLUMNS,
        "field_goals_made",
        "three_pointers_made",
        "free_throws_made",
        "age",
        "nba_experience_years",
        "is_rookie",
        "years_since_draft",
        "draft_year",
        "draft_round",
        "draft_number",
        "is_undrafted",
        "height_inches",
        "weight_pounds",
        "listed_position",
    }
    missing = required - set(player_seasons)
    if missing:
        raise ValueError(f"Player-season panel missing columns: {sorted(missing)}")
    if player_seasons.duplicated(["season", "player_id"]).any():
        raise ValueError("Player-season rows must be unique")
    if player_seasons["rapm_possessions"].lt(0).any():
        raise ValueError("Player on-court possessions must not be negative")


def _validate_output(features: pd.DataFrame, seasons: tuple[str, ...]) -> None:
    required = {
        "target_season",
        "player_id",
        "target_rapm",
        "target_rapm_possessions",
        "prior_source_season",
        "prior_on_court_possessions",
        "prior_exposure_cohort",
        *model_feature_columns(),
    }
    missing = required - set(features)
    if missing:
        raise ValueError(f"Box-score prior features missing columns: {sorted(missing)}")
    if features.duplicated(["target_season", "player_id"]).any():
        raise ValueError("Box-score prior rows must be unique")
    if not set(features["target_season"].astype(str)).issubset(set(seasons[1:])):
        raise ValueError("Box-score prior features contain unsupported target seasons")
    if features["target_rapm"].isna().any() or features["target_rapm_possessions"].isna().any():
        raise ValueError("Every target player requires a RAPM label and exposure")
    if features["prior_on_court_possessions"].dropna().lt(0).any():
        raise ValueError("Prior on-court possessions must not be negative")
    if features["prior_source_season"].notna().any():
        expected = features.loc[features["prior_source_season"].notna(), "target_season"].map(
            _previous_season
        )
        actual = features.loc[features["prior_source_season"].notna(), "prior_source_season"]
        if (
            not actual.astype(str)
            .reset_index(drop=True)
            .equals(expected.astype(str).reset_index(drop=True))
        ):
            raise ValueError("Prior source season must immediately precede the target season")
    if any(column.startswith("target_") for column in model_feature_columns()):
        raise ValueError("Model feature list contains a target label")


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = np.divide(
        numerator.to_numpy(dtype=float),
        denominator.to_numpy(dtype=float),
        out=np.full(len(numerator), np.nan, dtype=float),
        where=denominator.to_numpy(dtype=float) > 0,
    )
    return pd.Series(values, index=numerator.index, dtype=float)


def _previous_season(season: str) -> str:
    year = _season_start_year(season) - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _next_season(season: str) -> str:
    year = _season_start_year(season) + 1
    return f"{year}-{str(year + 1)[-2:]}"


def _season_start_year(season: str) -> int:
    return int(str(season)[:4])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Build the possession-native box-score prior panel."""

    parser = argparse.ArgumentParser(description="Build a possession-native box-score prior panel")
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest, path = build_box_score_prior_panel(
        player_season_panel_path=args.player_season_panel_path,
        output_dir=args.output_dir,
    )
    print(
        "Box-score prior panel: "
        f"targets={len(manifest.target_seasons)}, rows={manifest.row_count}; "
        f"path={path}"
    )
