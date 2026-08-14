"""Forward-safe player profile tokens for set-based neural lineup models.

The mart has one row per player available to a target season. It deliberately
stores raw, provenance-rich inputs rather than normalized tensors: a neural
training run must fit scalers using its own historical training window.
"""

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

from nba_lineup_model.modeling.contextual_profiles import (
    PROFILE_COLUMNS,
    _draft_profile_group,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.frozen_prior_evaluation import _read_regular_possessions
from nba_lineup_model.modeling.player_history import (
    player_history_code_fingerprint,
    validate_player_season_panel,
)
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.schema import ArtifactRecord
from nba_lineup_model.modeling.shot_taxonomy import validate_shot_taxonomy
from nba_lineup_model.season.schema import SHA256_PATTERN, validate_season

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_SHOT_TAXONOMY_DIR = Path("data/analytical/shot_taxonomy")
DEFAULT_PRIOR_RUN_ROOT = Path(
    "artifacts/models/forward_centered_value_conditioned_aging_no_context_rapm/2025-26"
)
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_OUTPUT_DIR = Path("data/analytical/profile_tokens")

SHOT_TOKEN_COLUMNS = (
    "rim_attempts_per_100",
    "rim_fg_pct_shrunk",
    "non_rim_two_attempts_per_100",
    "non_rim_two_fg_pct_shrunk",
    "three_attempts_per_100",
    "three_fg_pct_shrunk",
)
SHOT_HIERARCHICAL_COLUMNS = (
    "rim_fg_pct_hierarchical",
    "non_rim_two_fg_pct_hierarchical",
    "three_fg_pct_hierarchical",
)
SHOT_CAREER_ATTEMPT_COLUMNS = (
    "rim_career_attempts",
    "non_rim_two_career_attempts",
    "three_career_attempts",
)
FOUL_TOKEN_COLUMNS = (
    "free_throw_attempts_per_100",
    "free_throw_rate",
    "free_throw_pct_hierarchical",
    "free_throw_career_attempts",
)
SHOT_FAMILIES = ("rim", "non_rim_two", "three")
PROFILE_RATE_PSEUDO_POSSESSIONS = 300.0
RATE_PSEUDO_ATTEMPTS = 75.0
CAREER_LEAGUE_PSEUDO_ATTEMPTS = 75.0
BIO_TOKEN_COLUMNS = (
    "age",
    "nba_experience_years",
    "is_rookie",
    "height_inches",
    "weight_pounds",
    "draft_number",
    "has_draft_number",
    "is_undrafted",
)
PROVENANCE_COLUMNS = (
    "has_prior_player_season",
    "prior_on_court_possessions",
    "profile_imputed",
    "profile_replacement_weight",
    "shot_profile_imputed",
    "shot_profile_replacement_weight",
    "foul_profile_imputed",
    "foul_profile_replacement_weight",
    "age_imputed",
    "height_imputed",
    "weight_imputed",
)
TOKEN_FEATURE_COLUMNS = (
    "prior_rapm",
    *PROVENANCE_COLUMNS,
    *BIO_TOKEN_COLUMNS,
    *PROFILE_COLUMNS,
    *SHOT_TOKEN_COLUMNS,
    *SHOT_HIERARCHICAL_COLUMNS,
    *SHOT_CAREER_ATTEMPT_COLUMNS,
    *FOUL_TOKEN_COLUMNS,
)


class ProfileTokenManifest(BaseModel):
    """Integrity and temporal contract for a profile-token mart."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[2] = 2
    created_at: datetime
    builder_code_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_player_season_panel_path: str
    source_player_season_panel_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_shot_taxonomy_path: str
    source_shot_taxonomy_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_prior_run_path: str
    source_prior_run_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    target_seasons: tuple[str, ...] = Field(min_length=1)
    source_seasons: dict[str, str]
    row_count: int = Field(ge=1)
    token_feature_columns: tuple[str, ...] = Field(min_length=1)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=2)

    @field_validator("target_seasons")
    @classmethod
    def validate_target_seasons(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_season(value) for value in values)

    @model_validator(mode="after")
    def validate_contract(self) -> ProfileTokenManifest:
        if self.target_seasons != tuple(sorted(self.target_seasons, key=_season_start_year)):
            raise ValueError("Profile-token target seasons must be chronological")
        if tuple(self.source_seasons) != self.target_seasons:
            raise ValueError("Profile-token source seasons must match target season order")
        for target, source in self.source_seasons.items():
            if source != _previous_season(target):
                raise ValueError(
                    "Every profile-token source season must immediately precede target"
                )
        if self.token_feature_columns != TOKEN_FEATURE_COLUMNS:
            raise ValueError("Profile-token feature contract changed")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Profile-token artifact filenames must be unique")
        return self


def build_profile_token_mart(
    *,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    shot_taxonomy_dir: Path | str = DEFAULT_SHOT_TAXONOMY_DIR,
    prior_run_root: Path | str = DEFAULT_PRIOR_RUN_ROOT,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[ProfileTokenManifest, Path]:
    """Materialize one forward-safe token row for every prior-state player."""

    panel_path = Path(player_season_panel_path)
    panel_root = panel_path.parent
    validate_player_season_panel(panel_root)
    shot_root = Path(shot_taxonomy_dir)
    validate_shot_taxonomy(shot_root)
    prior_root = _resolve_prior_run(Path(prior_run_root))
    panel = pd.read_parquet(panel_path)
    shots = pd.read_parquet(shot_root / "player_season_shot_profiles.parquet")
    priors = pd.read_parquet(prior_root / "season_player_priors.parquet")
    _validate_sources(panel, shots, priors)

    panel_seasons = set(panel["season"].astype(str))
    target_seasons = tuple(
        season
        for season in sorted(priors["season"].astype(str).unique(), key=_season_start_year)
        if season in panel_seasons and _previous_season(season) in panel_seasons
    )
    if not target_seasons:
        raise ValueError("Profile-token mart found no target seasons with a prior player panel")
    latest_target = target_seasons[-1]
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(latest_target)],
        through_season=latest_target,
        analytical_dir=analytical_dir,
    )

    token_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    for target in target_seasons:
        print(f"target={target} building profile tokens", flush=True)
        target_priors = priors.loc[priors["season"].eq(target), ["player_id", "prior_rapm"]].copy()
        if target_priors["player_id"].duplicated().any():
            raise ValueError(f"Prior state has duplicate players in {target}")
        lineup_player_ids = _regular_lineup_player_ids(
            target,
            analytical_dir=Path(analytical_dir),
            curated_dir=Path(curated_dir),
        )
        target_ids = sorted(
            set(target_priors["player_id"].astype(int)).union(lineup_player_ids)
        )
        target_priors = (
            target_priors.set_index("player_id")
            .reindex(target_ids, fill_value=0.0)
            .rename_axis("player_id")
            .reset_index()
        )
        profiles = build_contextual_player_profiles(
            panel,
            target_season=target,
            target_player_ids=target_ids,
            analytical_dir=str(analytical_dir),
            curated_dir=str(curated_dir),
            exposure_cohort=exposure_cohort,
        )
        token_frame = _build_target_tokens(
            target=target,
            target_priors=target_priors,
            panel=panel,
            profiles=profiles,
            shots=shots,
        )
        token_frames.append(token_frame)
        coverage_rows.append(_coverage_row(token_frame))
        print(f"target={target} token_rows={len(token_frame):,}", flush=True)

    tokens = pd.concat(token_frames, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    _validate_tokens(tokens, target_seasons)
    return _write_mart(
        tokens=tokens,
        coverage=coverage,
        target_seasons=target_seasons,
        panel_path=panel_path,
        shot_root=shot_root,
        prior_root=prior_root,
        output_dir=Path(output_dir),
    )


def _regular_lineup_player_ids(
    season: str,
    *,
    analytical_dir: Path,
    curated_dir: Path,
) -> set[int]:
    """Return every player in the eligible regular-season possession corpus."""

    possessions, _ = _read_regular_possessions(
        season,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    return {
        int(player_id)
        for lineup in [
            *possessions["offense_player_ids"],
            *possessions["defense_player_ids"],
        ]
        for player_id in lineup
    }


def validate_profile_token_mart(path: Path | str) -> ProfileTokenManifest:
    """Validate profile-token artifacts, hashes, and temporal source boundaries."""

    root = Path(path)
    manifest = ProfileTokenManifest.model_validate_json((root / "_manifest.json").read_text())
    expected = {artifact.filename for artifact in manifest.artifacts}
    actual = {
        item.name for item in root.iterdir() if item.is_file() and item.name != "_manifest.json"
    }
    if actual != expected:
        raise ValueError("Profile-token files do not match manifest")
    for artifact in manifest.artifacts:
        artifact_path = root / artifact.filename
        if artifact_path.stat().st_size != artifact.byte_count:
            raise ValueError(f"Profile-token byte count changed: {artifact.filename}")
        if _sha256_file(artifact_path) != artifact.sha256:
            raise ValueError(f"Profile-token hash changed: {artifact.filename}")
        if (
            artifact.row_count is not None
            and len(pd.read_parquet(artifact_path)) != artifact.row_count
        ):
            raise ValueError(f"Profile-token row count changed: {artifact.filename}")
    tokens = pd.read_parquet(root / "player_profile_tokens.parquet")
    _validate_tokens(tokens, manifest.target_seasons)
    return manifest


def _build_target_tokens(
    *,
    target: str,
    target_priors: pd.DataFrame,
    panel: pd.DataFrame,
    profiles: pd.DataFrame,
    shots: pd.DataFrame,
) -> pd.DataFrame:
    source = _previous_season(target)
    target_bios = panel.loc[
        panel["season"].eq(target),
        [
            "player_id",
            "player_name",
            "age",
            "nba_experience_years",
            "is_rookie",
            "height_inches",
            "weight_pounds",
            "draft_number",
            "is_undrafted",
        ],
    ].copy()
    target_bios = target_bios.rename(columns={"draft_number": "draft_number_raw"})
    if target_bios["player_id"].duplicated().any():
        raise ValueError(f"Player panel bios have duplicates in {target}")
    source_panel = panel.loc[panel["season"].eq(source), ["player_id", "rapm_possessions"]].copy()
    source_panel = source_panel.rename(columns={"rapm_possessions": "prior_on_court_possessions"})
    if source_panel["player_id"].duplicated().any():
        raise ValueError(f"Player panel has duplicate prior rows in {source}")

    output = target_priors.merge(target_bios, on="player_id", how="left", validate="one_to_one")
    output = output.merge(source_panel, on="player_id", how="left", validate="one_to_one")
    output = output.merge(
        profiles.loc[
            :,
            [
                "player_id",
                "player_name",
                "is_rookie",
                "profile_source",
                "profile_imputed",
                "profile_replacement_weight",
                *PROFILE_COLUMNS,
            ],
        ].rename(
            columns={
                "player_name": "profile_player_name",
                "is_rookie": "profile_is_rookie",
            }
        ),
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    output["player_name"] = output["player_name"].fillna(output.pop("profile_player_name"))
    profile_is_rookie = output.pop("profile_is_rookie").astype("boolean")
    output["is_rookie"] = output["is_rookie"].astype("boolean").fillna(profile_is_rookie)
    if output["player_name"].isna().any():
        raise ValueError(f"Contextual profiles are missing players in {target}")
    output["has_prior_player_season"] = output["prior_on_court_possessions"].notna().astype("int64")
    output["prior_on_court_possessions"] = output["prior_on_court_possessions"].fillna(0.0)

    _fill_bio_tokens(output, source_panel=panel.loc[panel["season"].astype(str).lt(target)])
    output["draft_number"] = output.pop("draft_number_raw").fillna(0.0).astype(float)
    output["has_draft_number"] = output["draft_number"].gt(0).astype("int64")
    output["is_rookie"] = output["is_rookie"].fillna(False).astype("int64")
    output["is_undrafted"] = output["is_undrafted"].astype("boolean").fillna(False).astype("int64")
    output = _attach_shot_profiles(output, target=target, panel=panel, shots=shots)
    output = _attach_foul_profiles(output, target=target, panel=panel)
    output.insert(0, "target_season", target)
    output.insert(1, "source_season", source)
    return (
        output.loc[
            :,
            [
                "target_season",
                "source_season",
                "player_id",
                "player_name",
                "profile_source",
                "shot_profile_source",
                "foul_profile_source",
                *TOKEN_FEATURE_COLUMNS,
            ],
        ]
        .sort_values("player_id", kind="stable")
        .reset_index(drop=True)
    )


def _fill_bio_tokens(output: pd.DataFrame, *, source_panel: pd.DataFrame) -> None:
    for column in ("age", "height_inches", "weight_pounds"):
        values = pd.to_numeric(output[column], errors="coerce")
        reference = pd.to_numeric(source_panel[column], errors="coerce").dropna()
        if reference.empty:
            raise ValueError(f"No historical values available to impute {column}")
        output[f"{column.split('_')[0]}_imputed"] = values.isna().astype("int64")
        output[column] = values.fillna(float(reference.median())).astype(float)
    output["nba_experience_years"] = (
        pd.to_numeric(output["nba_experience_years"], errors="coerce").fillna(0.0).astype(float)
    )


def _attach_shot_profiles(
    output: pd.DataFrame,
    *,
    target: str,
    panel: pd.DataFrame,
    shots: pd.DataFrame,
) -> pd.DataFrame:
    source = _previous_season(target)
    source_raw_columns = tuple(
        column for family in SHOT_FAMILIES for column in (f"{family}_attempts", f"{family}_makes")
    )
    source_shots = shots.loc[
        shots["season"].eq(source),
        ["player_id", *SHOT_TOKEN_COLUMNS, *source_raw_columns],
    ].copy()
    source_shots = source_shots.rename(
        columns={column: f"source_{column}" for column in source_raw_columns}
    )
    if source_shots["player_id"].duplicated().any():
        raise ValueError(f"Shot taxonomy has duplicate player rows in {source}")
    result = output.merge(source_shots, on="player_id", how="left", validate="one_to_one")
    has_prior_shot_profile = result.loc[:, list(SHOT_TOKEN_COLUMNS)].notna().all(axis=1)
    result["shot_profile_imputed"] = (~has_prior_shot_profile).astype("int64")
    result["shot_profile_replacement_weight"] = 0.0
    result["shot_profile_source"] = np.where(has_prior_shot_profile, "prior_season", "cold_start")
    result = _attach_hierarchical_shooting(
        result,
        target=target,
        shots=shots,
        has_prior_shot_profile=has_prior_shot_profile,
    )

    cold = result.loc[~has_prior_shot_profile].copy()
    if cold.empty:
        return _drop_source_shot_counts(result)
    historical = _historical_shot_profiles(target=target, panel=panel, shots=shots)
    replacement = historical.loc[~historical["rapm_exposure_eligible"].fillna(False).astype(bool)]
    if replacement.empty:
        replacement = historical
    cold_columns = (*SHOT_TOKEN_COLUMNS, *SHOT_HIERARCHICAL_COLUMNS)
    replacement_profile = _weighted_profile(replacement, cold_columns)
    rookie_profiles = _rookie_profiles(historical, cold_columns)
    groups = rookie_profiles.set_index("draft_profile_group")
    fallback = groups.loc["all_rookies"]
    cold_groups = _draft_profile_group(cold)
    weights = cold["profile_replacement_weight"].to_numpy(dtype=float)
    for column in cold_columns:
        draft = (
            cold_groups.map(groups[column]).fillna(float(fallback[column])).to_numpy(dtype=float)
        )
        cold[column] = (1.0 - weights) * draft + weights * replacement_profile[column]
    cold["shot_profile_replacement_weight"] = weights
    cold["shot_profile_source"] = np.where(
        cold["is_rookie"].astype(bool),
        "exposure_gated_rookie",
        "replacement_profile",
    )
    result.loc[~has_prior_shot_profile, cold.columns] = cold
    return _drop_source_shot_counts(result)


def _attach_hierarchical_shooting(
    output: pd.DataFrame,
    *,
    target: str,
    shots: pd.DataFrame,
    has_prior_shot_profile: pd.Series,
) -> pd.DataFrame:
    """Add player-career/league and source-season shooting posteriors.

    The career prior ends before the source season. Source-season makes and
    attempts update that prior once, so target-season shooting is never read.
    """

    source = _previous_season(target)
    history = shots.loc[shots["season"].astype(str).lt(source)].copy()
    aggregate_columns = {
        f"{family}_career_attempts": (f"{family}_attempts", "sum") for family in SHOT_FAMILIES
    } | {f"{family}_career_makes": (f"{family}_makes", "sum") for family in SHOT_FAMILIES}
    career = history.groupby("player_id", as_index=False).agg(**aggregate_columns)
    result = output.merge(career, on="player_id", how="left", validate="one_to_one")
    source_rows = shots.loc[shots["season"].eq(source)]
    for family in SHOT_FAMILIES:
        attempts = float(source_rows[f"{family}_attempts"].sum())
        makes = float(source_rows[f"{family}_makes"].sum())
        if attempts <= 0:
            raise ValueError(f"Shot taxonomy has no {family} attempts in {source}")
        league_pct = makes / attempts
        career_attempts = result[f"{family}_career_attempts"].fillna(0.0).to_numpy(dtype=float)
        career_makes = result.pop(f"{family}_career_makes").fillna(0.0).to_numpy(dtype=float)
        career_prior = (career_makes + CAREER_LEAGUE_PSEUDO_ATTEMPTS * league_pct) / (
            career_attempts + CAREER_LEAGUE_PSEUDO_ATTEMPTS
        )
        source_attempts = result[f"source_{family}_attempts"].fillna(0.0).to_numpy(dtype=float)
        source_makes = result[f"source_{family}_makes"].fillna(0.0).to_numpy(dtype=float)
        result[f"{family}_fg_pct_hierarchical"] = (
            source_makes + RATE_PSEUDO_ATTEMPTS * career_prior
        ) / (source_attempts + RATE_PSEUDO_ATTEMPTS)
        result[f"{family}_career_attempts"] = career_attempts
    return result


def _historical_shot_profiles(
    *,
    target: str,
    panel: pd.DataFrame,
    shots: pd.DataFrame,
) -> pd.DataFrame:
    """Create forward historical rows for cold-start shot-profile blending."""

    history = shots.loc[shots["season"].astype(str).lt(target)].copy()
    bios = panel.loc[
        panel["season"].astype(str).lt(target),
        [
            "season",
            "player_id",
            "is_rookie",
            "draft_number",
            "is_undrafted",
            "rapm_exposure_eligible",
        ],
    ].copy()
    history = history.merge(bios, on=["season", "player_id"], how="inner", validate="one_to_one")
    for family in SHOT_FAMILIES:
        history[f"{family}_fg_pct_hierarchical"] = history[f"{family}_fg_pct_shrunk"]
    return history


def _drop_source_shot_counts(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        f"source_{family}_{kind}" for family in SHOT_FAMILIES for kind in ("attempts", "makes")
    ]
    return frame.drop(columns=columns)


def _weighted_profile(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    weight_column: str = "on_court_possessions",
) -> dict[str, float]:
    weights = pd.to_numeric(frame[weight_column], errors="raise").to_numpy(dtype=float)
    if not np.any(weights > 0):
        raise ValueError("Cold-start profile requires positive possession weights")
    return {
        column: float(np.average(frame[column].to_numpy(dtype=float), weights=weights))
        for column in columns
    }


def _rookie_profiles(
    historical: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    weight_column: str = "on_court_possessions",
) -> pd.DataFrame:
    rookies = historical.loc[historical["is_rookie"].fillna(False).astype(bool)].copy()
    if rookies.empty:
        raise ValueError("Cold-start profile requires historical rookie seasons")
    rookies["draft_profile_group"] = _draft_profile_group(rookies)
    rows: list[dict[str, object]] = []
    for group, frame in rookies.groupby("draft_profile_group", sort=True):
        rows.append(
            {
                "draft_profile_group": group,
                **_weighted_profile(frame, columns, weight_column=weight_column),
            }
        )
    rows.append(
        {
            "draft_profile_group": "all_rookies",
            **_weighted_profile(rookies, columns, weight_column=weight_column),
        }
    )
    return pd.DataFrame(rows)


def _attach_foul_profiles(
    output: pd.DataFrame,
    *,
    target: str,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Add forward-safe foul-drawing volume, rate, and hierarchical FT%."""

    source = _previous_season(target)
    required = (
        "player_id",
        "rapm_possessions",
        "field_goals_attempted",
        "free_throws_attempted",
        "free_throws_made",
    )
    source_rows = panel.loc[panel["season"].eq(source), list(required)].copy()
    if source_rows["player_id"].duplicated().any():
        raise ValueError(f"Player panel has duplicate foul rows in {source}")
    source_rows = source_rows.rename(
        columns={
            "rapm_possessions": "source_foul_possessions",
            "field_goals_attempted": "source_foul_fga",
            "free_throws_attempted": "source_foul_fta",
            "free_throws_made": "source_foul_ftm",
        }
    )
    result = output.merge(source_rows, on="player_id", how="left", validate="one_to_one")
    available = (
        result[["source_foul_possessions", "source_foul_fga", "source_foul_fta", "source_foul_ftm"]]
        .notna()
        .all(axis=1)
    )
    result["foul_profile_imputed"] = (~available).astype("int64")
    result["foul_profile_replacement_weight"] = 0.0
    result["foul_profile_source"] = np.where(available, "prior_season", "cold_start")

    source_total_possessions = float(source_rows["source_foul_possessions"].sum())
    source_total_fga = float(source_rows["source_foul_fga"].sum())
    source_total_fta = float(source_rows["source_foul_fta"].sum())
    source_total_ftm = float(source_rows["source_foul_ftm"].sum())
    if min(source_total_possessions, source_total_fga, source_total_fta) <= 0:
        raise ValueError(f"Invalid source foul totals in {source}")
    league_fta_per_100 = 100.0 * source_total_fta / source_total_possessions
    league_ftr = source_total_fta / source_total_fga
    league_ft_pct = source_total_ftm / source_total_fta

    history = panel.loc[panel["season"].astype(str).lt(source)].copy()
    career = history.groupby("player_id", as_index=False).agg(
        free_throw_career_attempts=("free_throws_attempted", "sum"),
        free_throw_career_makes=("free_throws_made", "sum"),
    )
    result = result.merge(career, on="player_id", how="left", validate="one_to_one")
    career_attempts = result["free_throw_career_attempts"].fillna(0.0).to_numpy(dtype=float)
    career_makes = result.pop("free_throw_career_makes").fillna(0.0).to_numpy(dtype=float)
    career_prior = (career_makes + CAREER_LEAGUE_PSEUDO_ATTEMPTS * league_ft_pct) / (
        career_attempts + CAREER_LEAGUE_PSEUDO_ATTEMPTS
    )
    possessions = result["source_foul_possessions"].fillna(0.0).to_numpy(dtype=float)
    fga = result["source_foul_fga"].fillna(0.0).to_numpy(dtype=float)
    fta = result["source_foul_fta"].fillna(0.0).to_numpy(dtype=float)
    ftm = result["source_foul_ftm"].fillna(0.0).to_numpy(dtype=float)
    result["free_throw_attempts_per_100"] = (
        100.0
        * (fta + PROFILE_RATE_PSEUDO_POSSESSIONS * league_fta_per_100 / 100.0)
        / (possessions + PROFILE_RATE_PSEUDO_POSSESSIONS)
    )
    result["free_throw_rate"] = (fta + RATE_PSEUDO_ATTEMPTS * league_ftr) / (
        fga + RATE_PSEUDO_ATTEMPTS
    )
    result["free_throw_pct_hierarchical"] = (ftm + RATE_PSEUDO_ATTEMPTS * career_prior) / (
        fta + RATE_PSEUDO_ATTEMPTS
    )
    result["free_throw_career_attempts"] = career_attempts

    cold = result.loc[~available].copy()
    if not cold.empty:
        historical = _historical_foul_profiles(target=target, panel=panel)
        replacement = historical.loc[
            ~historical["rapm_exposure_eligible"].fillna(False).astype(bool)
        ]
        if replacement.empty:
            replacement = historical
        replacement_profile = _weighted_profile(
            replacement,
            FOUL_TOKEN_COLUMNS[:-1],
            weight_column="rapm_possessions",
        )
        rookie_profiles = _rookie_profiles(
            historical,
            FOUL_TOKEN_COLUMNS[:-1],
            weight_column="rapm_possessions",
        )
        groups = rookie_profiles.set_index("draft_profile_group")
        fallback = groups.loc["all_rookies"]
        cold_groups = _draft_profile_group(cold)
        weights = cold["profile_replacement_weight"].to_numpy(dtype=float)
        for column in FOUL_TOKEN_COLUMNS[:-1]:
            draft = (
                cold_groups.map(groups[column])
                .fillna(float(fallback[column]))
                .to_numpy(dtype=float)
            )
            cold[column] = (1.0 - weights) * draft + weights * replacement_profile[column]
        cold["foul_profile_replacement_weight"] = weights
        cold["foul_profile_source"] = np.where(
            cold["is_rookie"].astype(bool),
            "exposure_gated_rookie",
            "replacement_profile",
        )
        result.loc[~available, cold.columns] = cold
    return result.drop(
        columns=["source_foul_possessions", "source_foul_fga", "source_foul_fta", "source_foul_ftm"]
    )


def _historical_foul_profiles(*, target: str, panel: pd.DataFrame) -> pd.DataFrame:
    """Create possession-native historical foul profiles for cold starts."""

    history = panel.loc[panel["season"].astype(str).lt(target)].copy()
    required = {
        "season",
        "rapm_possessions",
        "field_goals_attempted",
        "free_throws_attempted",
        "free_throws_made",
        "rapm_exposure_eligible",
    }
    missing = required - set(history)
    if missing:
        raise ValueError(f"Player panel is missing foul-profile columns: {sorted(missing)}")
    league = history.groupby("season", as_index=False).agg(
        league_possessions=("rapm_possessions", "sum"),
        league_fga=("field_goals_attempted", "sum"),
        league_fta=("free_throws_attempted", "sum"),
        league_ftm=("free_throws_made", "sum"),
    )
    history = history.merge(league, on="season", how="left", validate="many_to_one")
    history["free_throw_attempts_per_100"] = (
        100.0
        * (
            history["free_throws_attempted"]
            + PROFILE_RATE_PSEUDO_POSSESSIONS
            * history["league_fta"]
            / history["league_possessions"]
        )
        / (history["rapm_possessions"] + PROFILE_RATE_PSEUDO_POSSESSIONS)
    )
    history["free_throw_rate"] = (
        history["free_throws_attempted"]
        + RATE_PSEUDO_ATTEMPTS * history["league_fta"] / history["league_fga"]
    ) / (history["field_goals_attempted"] + RATE_PSEUDO_ATTEMPTS)
    history["free_throw_pct_hierarchical"] = (
        history["free_throws_made"]
        + RATE_PSEUDO_ATTEMPTS * history["league_ftm"] / history["league_fta"]
    ) / (history["free_throws_attempted"] + RATE_PSEUDO_ATTEMPTS)
    return history


def _coverage_row(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "target_season": str(frame["target_season"].iloc[0]),
        "source_season": str(frame["source_season"].iloc[0]),
        "player_count": int(len(frame)),
        "prior_player_season_available_count": int(frame["has_prior_player_season"].sum()),
        "context_profile_imputed_count": int(frame["profile_imputed"].sum()),
        "shot_profile_imputed_count": int(frame["shot_profile_imputed"].sum()),
        "foul_profile_imputed_count": int(frame["foul_profile_imputed"].sum()),
        "age_imputed_count": int(frame["age_imputed"].sum()),
        "height_imputed_count": int(frame["height_imputed"].sum()),
        "weight_imputed_count": int(frame["weight_imputed"].sum()),
    }


def _validate_sources(panel: pd.DataFrame, shots: pd.DataFrame, priors: pd.DataFrame) -> None:
    required_panel = {
        "season",
        "player_id",
        "player_name",
        "rapm_possessions",
        "rapm_exposure_eligible",
        "field_goals_attempted",
        "free_throws_attempted",
        "free_throws_made",
        "age",
        "nba_experience_years",
        "is_rookie",
        "height_inches",
        "weight_pounds",
        "draft_number",
        "is_undrafted",
    }
    required_shots = {"season", "player_id", "on_court_possessions", *SHOT_TOKEN_COLUMNS}
    required_priors = {"season", "player_id", "prior_rapm"}
    for name, frame, required in (
        ("player panel", panel, required_panel),
        ("shot taxonomy", shots, required_shots),
        ("prior state", priors, required_priors),
    ):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"Profile-token {name} is missing columns: {sorted(missing)}")
    if (
        panel.duplicated(["season", "player_id"]).any()
        or shots.duplicated(["season", "player_id"]).any()
    ):
        raise ValueError("Profile-token panel and shot rows must be unique by season-player")


def _validate_tokens(tokens: pd.DataFrame, target_seasons: tuple[str, ...]) -> None:
    required = {
        "target_season",
        "source_season",
        "player_id",
        "player_name",
        *TOKEN_FEATURE_COLUMNS,
    }
    missing = required - set(tokens)
    if missing:
        raise ValueError(f"Profile-token rows missing columns: {sorted(missing)}")
    if tokens.duplicated(["target_season", "player_id"]).any():
        raise ValueError("Profile-token rows must be unique by target season-player")
    actual_seasons = tuple(
        sorted(tokens["target_season"].astype(str).unique(), key=_season_start_year)
    )
    if actual_seasons != target_seasons:
        raise ValueError("Profile-token target-season coverage changed")
    if (
        not tokens["source_season"]
        .astype(str)
        .eq(tokens["target_season"].map(_previous_season))
        .all()
    ):
        raise ValueError("Profile-token rows use a non-prior source season")
    if tokens.loc[:, list(TOKEN_FEATURE_COLUMNS)].isna().any(axis=None):
        raise ValueError("Profile-token feature columns must not contain missing values")
    if not np.isfinite(tokens.loc[:, list(TOKEN_FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Profile-token feature columns must be finite")
    binary_columns = (
        "has_prior_player_season",
        "profile_imputed",
        "shot_profile_imputed",
        "foul_profile_imputed",
        "age_imputed",
        "height_imputed",
        "weight_imputed",
        "has_draft_number",
        "is_rookie",
        "is_undrafted",
    )
    for column in binary_columns:
        if not tokens[column].isin([0, 1]).all():
            raise ValueError(f"Profile-token binary column changed: {column}")
    if tokens["prior_on_court_possessions"].lt(0).any():
        raise ValueError("Prior possession counts must be non-negative")
    for column in (
        "profile_replacement_weight",
        "shot_profile_replacement_weight",
        "foul_profile_replacement_weight",
    ):
        if tokens[column].lt(0).any() or tokens[column].gt(1).any():
            raise ValueError(f"Profile-token weights must lie in [0, 1]: {column}")


def _write_mart(
    *,
    tokens: pd.DataFrame,
    coverage: pd.DataFrame,
    target_seasons: tuple[str, ...],
    panel_path: Path,
    shot_root: Path,
    prior_root: Path,
    output_dir: Path,
) -> tuple[ProfileTokenManifest, Path]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid4().hex}"
    backup = output_dir.parent / f".{output_dir.name}.bak-{uuid4().hex}"
    temporary.mkdir()
    try:
        outputs = {
            "player_profile_tokens.parquet": tokens,
            "season_profile_token_coverage.parquet": coverage,
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
        manifest = ProfileTokenManifest(
            created_at=datetime.now(UTC),
            builder_code_version=player_history_code_fingerprint((Path(__file__),)),
            source_player_season_panel_path=str(panel_path),
            source_player_season_panel_manifest_sha256=_sha256_file(
                panel_path.parent / "_manifest.json"
            ),
            source_shot_taxonomy_path=str(shot_root),
            source_shot_taxonomy_manifest_sha256=_sha256_file(shot_root / "_manifest.json"),
            source_prior_run_path=str(prior_root),
            source_prior_run_manifest_sha256=_sha256_file(prior_root / "manifest.json"),
            target_seasons=target_seasons,
            source_seasons={season: _previous_season(season) for season in target_seasons},
            row_count=len(tokens),
            token_feature_columns=TOKEN_FEATURE_COLUMNS,
            artifacts=artifacts,
        )
        (temporary / "_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        validate_profile_token_mart(temporary)
        if output_dir.exists():
            output_dir.replace(backup)
        try:
            temporary.replace(output_dir)
        except Exception:
            if backup.exists() and not output_dir.exists():
                backup.replace(output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        validate_profile_token_mart(output_dir)
        return manifest, output_dir
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not output_dir.exists():
            backup.replace(output_dir)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def _resolve_prior_run(root: Path) -> Path:
    if (root / "season_player_priors.parquet").is_file():
        return root
    if not (root / "latest.json").is_file():
        raise FileNotFoundError(f"Profile-token prior root has no latest run: {root}")
    run = _latest_run(root)
    if (
        not (run / "season_player_priors.parquet").is_file()
        or not (run / "manifest.json").is_file()
    ):
        raise FileNotFoundError("Profile-token prior run is incomplete")
    return run


def _previous_season(season: str) -> str:
    start = int(validate_season(season)[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _season_start_year(season: str) -> int:
    return int(validate_season(season)[:4])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Build the validated forward profile-token mart."""

    parser = argparse.ArgumentParser(description="Build forward-safe player profile tokens")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prior-run-root", default=str(DEFAULT_PRIOR_RUN_ROOT))
    args = parser.parse_args()
    manifest, output = build_profile_token_mart(
        output_dir=args.output_dir,
        prior_run_root=args.prior_run_root,
    )
    print(
        f"Profile tokens: seasons={len(manifest.target_seasons)} "
        f"rows={manifest.row_count:,} output={output}"
    )


if __name__ == "__main__":
    main()
