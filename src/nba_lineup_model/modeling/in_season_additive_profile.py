"""Leakage-safe in-season updates for NAIL's additive player profiles.

The production context coefficients and reference field remain frozen.  This
module updates only the eight player-level profile inputs, then exposes the
resulting additive-rating movement for a separately residualized RAPM update.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import side_context_feature_columns
from nba_lineup_model.modeling.contextual_profiles import MEDVEDOVSKY_2020_PROFILE_PADDING
from nba_lineup_model.season.schema import validate_season
from nba_lineup_model.web_api.inference import _linear_x3_additive_feature_map

PROFILE_SCHEMA_VERSION = 1
PROFILE_STATE_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64

FEATURE_TO_PROFILE_COLUMN = {
    "three_pa_per_100": "three_pa_per_100",
    "three_pm_per_100": "three_pm_per_100",
    "assists_per_100": "assists_per_100",
    "turnovers_per_100": "turnovers_per_100",
    "usage_pct": "usage_pct",
    "steals_per_100": "steals_per_100",
    "blocks_per_100": "blocks_per_100",
    "offensive_rebound_claim_total": "offensive_rebound_pct",
}
PROFILE_FEATURES = tuple(FEATURE_TO_PROFILE_COLUMN)
PROFILE_COLUMNS = tuple(FEATURE_TO_PROFILE_COLUMN.values())

# Three-point makes preserve production's volume-times-accuracy construction.
PRIOR_COMPONENT_EXPOSURES = {
    "three_pa_per_100": float(
        MEDVEDOVSKY_2020_PROFILE_PADDING.rate_pseudo_possessions["three_pa"]
    ),
    "three_point_percentage": float(
        MEDVEDOVSKY_2020_PROFILE_PADDING.three_point_percentage_attempts or 0.0
    ),
    "assists_per_100": float(
        MEDVEDOVSKY_2020_PROFILE_PADDING.rate_pseudo_possessions["assists"]
    ),
    "turnovers_per_100": float(
        MEDVEDOVSKY_2020_PROFILE_PADDING.rate_pseudo_possessions["turnovers"]
    ),
    "usage_pct": float(
        MEDVEDOVSKY_2020_PROFILE_PADDING.usage_percentage_pseudo_possessions
    ),
    "steals_per_100": float(
        MEDVEDOVSKY_2020_PROFILE_PADDING.rate_pseudo_possessions["steals"]
    ),
    "blocks_per_100": float(
        MEDVEDOVSKY_2020_PROFILE_PADDING.rate_pseudo_possessions["blocks"]
    ),
    "offensive_rebound_claim_total": float(
        (MEDVEDOVSKY_2020_PROFILE_PADDING.rebound_percentage_pseudo_possessions or {})[
            "offensive_rebound_pct"
        ]
    ),
}
PRIOR_COMPONENTS = tuple(PRIOR_COMPONENT_EXPOSURES)

EVIDENCE_COLUMNS = (
    "player_id",
    "on_court_possessions",
    "minutes",
    "three_pointers_attempted",
    "three_pointers_made",
    "assists",
    "turnovers",
    "steals",
    "blocks",
    "usage_events",
    "usage_opportunities",
    "offensive_rebound_pct_minutes",
)


@dataclass(frozen=True)
class AdditiveProfileUpdaterConfig:
    """Versioned shrinkage parameters for one profile-state candidate."""

    updater_version: str = "in-season-additive-profile-v0.1.0"
    prior_multipliers: Mapping[str, float] = field(
        default_factory=lambda: {component: 1.0 for component in PRIOR_COMPONENTS}
    )

    def validate(self) -> None:
        if not self.updater_version.strip():
            raise ValueError("Profile updater version must be non-empty")
        missing = set(PRIOR_COMPONENTS) - set(self.prior_multipliers)
        extra = set(self.prior_multipliers) - set(PRIOR_COMPONENTS)
        if missing or extra:
            raise ValueError(
                f"Profile prior multipliers mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        values = np.asarray(list(self.prior_multipliers.values()), dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Profile prior multipliers must be finite and positive")

    def prior_exposure(self, component: str) -> float:
        self.validate()
        return PRIOR_COMPONENT_EXPOSURES[component] * float(
            self.prior_multipliers[component]
        )


@dataclass(frozen=True)
class FrozenAdditiveProfileContract:
    """Exact preseason profiles and frozen raw-unit production coefficients."""

    season: str
    core_model_version: str
    source_run_id: str
    profiles: pd.DataFrame
    coefficients: Mapping[str, float]
    reference_values: Mapping[str, float]
    published_adjustments: Mapping[int, float]
    lineage_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        validate_season(self.season)
        if not self.core_model_version.strip() or not self.source_run_id.strip():
            raise ValueError("Frozen profile model identifiers must be non-empty")
        required = {"player_id", *PROFILE_COLUMNS}
        missing = required - set(self.profiles)
        if missing:
            raise ValueError(f"Frozen profiles missing columns: {sorted(missing)}")
        profiles = self.profiles.loc[:, ["player_id", *PROFILE_COLUMNS]].copy()
        profiles["player_id"] = pd.to_numeric(
            profiles["player_id"], errors="raise"
        ).astype("int64")
        profiles = profiles.sort_values("player_id", kind="stable").reset_index(drop=True)
        if profiles.empty or profiles["player_id"].duplicated().any():
            raise ValueError("Frozen profiles must contain unique players")
        values = profiles.loc[:, PROFILE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Frozen profile values must be finite")
        _validate_feature_mapping(self.coefficients, "coefficients")
        _validate_feature_mapping(self.reference_values, "reference values")
        coefficient_values = np.asarray(
            [self.coefficients[feature] for feature in PROFILE_FEATURES], dtype=float
        )
        reference_values = np.asarray(
            [self.reference_values[feature] for feature in PROFILE_FEATURES], dtype=float
        )
        if not np.isfinite(coefficient_values).all() or not np.isfinite(
            reference_values
        ).all():
            raise ValueError("Profile coefficients and references must be finite")
        published = {int(key): float(value) for key, value in self.published_adjustments.items()}
        missing_adjustments = set(profiles["player_id"]) - set(published)
        if missing_adjustments:
            raise ValueError(
                f"Published additive adjustment missing players: {sorted(missing_adjustments)}"
            )
        if not np.isfinite(np.asarray(list(published.values()), dtype=float)).all():
            raise ValueError("Published additive adjustments must be finite")

        semantic = {
            "season": self.season,
            "core_model_version": self.core_model_version,
            "source_run_id": self.source_run_id,
            "profiles_sha256": _frame_sha256(profiles),
            "coefficients": {key: float(self.coefficients[key]) for key in PROFILE_FEATURES},
            "reference_values": {
                key: float(self.reference_values[key]) for key in PROFILE_FEATURES
            },
            "published_adjustments": sorted(published.items()),
        }
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "published_adjustments", published)
        object.__setattr__(
            self,
            "lineage_sha256",
            hashlib.sha256(_canonical_json(semantic)).hexdigest(),
        )

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(self.profiles["player_id"].astype(int))


@dataclass(frozen=True)
class ProfileGameContribution:
    """Immutable per-game box-score/profile sufficient statistics."""

    season: str
    game_id: str
    game_time_utc: datetime
    finalized_at: datetime
    source_sha256: str
    source_uri: str
    evidence: pd.DataFrame
    schema_version: int = PROFILE_SCHEMA_VERSION
    contribution_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        validate_season(self.season)
        if not self.game_id.strip() or not self.source_uri.strip():
            raise ValueError("Profile contribution IDs and source URI must be non-empty")
        game_time = _as_utc(self.game_time_utc, "game_time_utc")
        finalized = _as_utc(self.finalized_at, "finalized_at")
        if finalized < game_time:
            raise ValueError("Profile contribution cannot finalize before game time")
        _validate_sha256(self.source_sha256, "source_sha256")
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError("Unsupported profile contribution schema")
        missing = set(EVIDENCE_COLUMNS) - set(self.evidence)
        if missing:
            raise ValueError(f"Profile evidence missing columns: {sorted(missing)}")
        evidence = self.evidence.loc[:, EVIDENCE_COLUMNS].copy()
        evidence["player_id"] = pd.to_numeric(
            evidence["player_id"], errors="raise"
        ).astype("int64")
        evidence = evidence.sort_values("player_id", kind="stable").reset_index(drop=True)
        if evidence.empty or evidence["player_id"].duplicated().any():
            raise ValueError("Profile game evidence must contain unique players")
        numeric = evidence.drop(columns="player_id").to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ValueError("Profile game evidence must be finite and non-negative")
        if (evidence["three_pointers_made"] > evidence["three_pointers_attempted"]).any():
            raise ValueError("Three-point makes cannot exceed attempts")
        if (evidence["on_court_possessions"] <= 0).any() or (evidence["minutes"] <= 0).any():
            raise ValueError("Profile evidence requires positive minutes and possessions")
        if (evidence["usage_opportunities"] <= 0).any():
            raise ValueError("Profile usage opportunities must be positive")

        semantic = {
            "schema_version": self.schema_version,
            "season": self.season,
            "game_id": self.game_id,
            "game_time_utc": game_time.isoformat(),
            "finalized_at": finalized.isoformat(),
            "source_sha256": self.source_sha256,
            "source_uri": self.source_uri,
            "evidence_sha256": _frame_sha256(evidence),
        }
        object.__setattr__(self, "game_time_utc", game_time)
        object.__setattr__(self, "finalized_at", finalized)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(
            self,
            "contribution_sha256",
            hashlib.sha256(_canonical_json(semantic)).hexdigest(),
        )


@dataclass
class ProfileContributionLedger:
    """Select the current immutable source revision for every game."""

    _by_game_id: dict[str, ProfileGameContribution] = field(default_factory=dict)

    def upsert(self, contribution: ProfileGameContribution) -> str:
        prior = self._by_game_id.get(contribution.game_id)
        if prior is not None and prior.contribution_sha256 == contribution.contribution_sha256:
            return "unchanged"
        self._by_game_id[contribution.game_id] = contribution
        return "revised" if prior is not None else "added"

    def values(self) -> tuple[ProfileGameContribution, ...]:
        return tuple(
            sorted(
                self._by_game_id.values(),
                key=lambda row: (row.finalized_at, row.game_time_utc, row.game_id),
            )
        )


def load_published_profile_contract(
    rankings_path: Path | str,
    profiles_path: Path | str,
    *,
    core_model_version: str = "NAIL-RAPM v1.2.1.3",
) -> FrozenAdditiveProfileContract:
    """Recover the exact served profile contract from published web artifacts."""

    rankings_file = Path(rankings_path)
    profiles_file = Path(profiles_path)
    rankings = pd.read_parquet(rankings_file)
    profiles = pd.read_parquet(profiles_file)
    required = {
        "season",
        "player_id",
        "additive_profile_adjustment",
        "additive_profile_breakdown_json",
    }
    missing = required - set(rankings)
    if missing:
        raise ValueError(f"Published rankings missing columns: {sorted(missing)}")
    seasons = tuple(rankings["season"].astype(str).unique())
    if len(seasons) != 1:
        raise ValueError("Published profile contract must contain one season")
    season = validate_season(seasons[0])
    profile_season_column = "target_season" if "target_season" in profiles else "season"
    if not profiles[profile_season_column].astype(str).eq(season).all():
        raise ValueError("Published rankings and profiles cover different seasons")

    breakdown_rows: list[dict[str, float | int | str]] = []
    for row in rankings.itertuples(index=False):
        serialized = row.additive_profile_breakdown_json
        if pd.isna(serialized) or not str(serialized).strip():
            # Draft/cold-start ratings fold their preseason profile into the
            # separately published prior and intentionally have no additive
            # display adjustment.  They can still move from that profile once
            # current-season evidence arrives.
            continue
        payload = json.loads(str(serialized))
        by_feature = {str(item["feature"]): item for item in payload}
        if set(by_feature) != set(PROFILE_FEATURES):
            raise ValueError("Published additive breakdown does not match production bundle")
        breakdown_total = sum(float(item["contribution"]) for item in payload)
        # A handful of gap-returner rows carry a small frozen display-center
        # bridge in the published total.  Preserve that total as the baseline;
        # the feature slopes still determine the exact in-season delta.
        if not np.isclose(
            breakdown_total,
            float(row.additive_profile_adjustment),
            atol=1e-2,
        ):
            raise ValueError("Published additive breakdown does not reconstruct its total")
        for feature in PROFILE_FEATURES:
            item = by_feature[feature]
            breakdown_rows.append(
                {
                    "player_id": int(row.player_id),
                    "feature": feature,
                    "player_value": float(item["player_value"]),
                    "reference_value": float(item["reference_value"]),
                    "contribution": float(item["contribution"]),
                }
            )
    breakdown = pd.DataFrame(breakdown_rows)
    coefficients: dict[str, float] = {}
    references: dict[str, float] = {}
    for feature, rows in breakdown.groupby("feature", sort=False):
        difference = rows["player_value"] - rows["reference_value"]
        usable = difference.abs().gt(1e-10)
        estimates = rows.loc[usable, "contribution"] / difference.loc[usable]
        if estimates.empty:
            raise ValueError(f"Cannot recover frozen coefficient for {feature}")
        coefficient = float(estimates.median())
        if not np.allclose(estimates, coefficient, atol=1e-10, rtol=1e-10):
            raise ValueError(f"Published coefficient is inconsistent for {feature}")
        reference = float(rows["reference_value"].median())
        coefficients[str(feature)] = coefficient
        references[str(feature)] = reference

    selected = profiles.loc[:, ["player_id", *PROFILE_COLUMNS]].merge(
        rankings.loc[:, ["player_id"]], on="player_id", how="inner", validate="one_to_one"
    )
    published = dict(
        zip(
            rankings["player_id"].astype(int),
            rankings["additive_profile_adjustment"].fillna(0.0).astype(float),
            strict=True,
        )
    )
    metadata_path = rankings_file.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    run_id = str(metadata.get("run_id") or rankings_file.parent.name)
    contract = FrozenAdditiveProfileContract(
        season=season,
        core_model_version=core_model_version,
        source_run_id=run_id,
        profiles=selected,
        coefficients=coefficients,
        reference_values=references,
        published_adjustments=published,
    )
    # The published file contains two frozen display-reference cohorts for a
    # small set of gap returners.  The slope is common, and only the slope is
    # used for an in-season delta.  Baseline totals above remain the exact
    # published values rather than being reconstructed from a pooled center.
    return contract


def profile_contract_from_context_model(
    *,
    season: str,
    core_model_version: str,
    source_run_id: str,
    profiles: pd.DataFrame,
    context_model: object,
) -> FrozenAdditiveProfileContract:
    """Build the same raw-unit contract from a historical model state."""

    feature_map = _linear_x3_additive_feature_map(context_model.feature_set)
    if feature_map != FEATURE_TO_PROFILE_COLUMN:
        raise ValueError("Historical context model does not use the production profile bundle")
    scale = context_model.pipeline.named_steps["scale"]
    ridge = context_model.pipeline.named_steps["ridge"]
    raw = np.asarray(ridge.coef_, dtype=float) / np.asarray(scale.scale_, dtype=float)
    columns = side_context_feature_columns(context_model.feature_set)
    by_feature = dict(zip(columns, raw, strict=True))
    coefficients = {feature: float(by_feature[feature]) for feature in PROFILE_FEATURES}
    selected = profiles.loc[:, ["player_id", *PROFILE_COLUMNS]].copy()
    references = {
        feature: float(selected[profile_column].mean())
        for feature, profile_column in FEATURE_TO_PROFILE_COLUMN.items()
    }
    published = {
        int(row.player_id): float(
            sum(
                coefficients[feature]
                * (float(getattr(row, profile_column)) - references[feature])
                for feature, profile_column in FEATURE_TO_PROFILE_COLUMN.items()
            )
        )
        for row in selected.itertuples(index=False)
    }
    return FrozenAdditiveProfileContract(
        season=season,
        core_model_version=core_model_version,
        source_run_id=source_run_id,
        profiles=selected,
        coefficients=coefficients,
        reference_values=references,
        published_adjustments=published,
    )


def build_profile_game_contribution(
    players: pd.DataFrame,
    stints: pd.DataFrame,
    *,
    season: str,
    game_id: str,
    game_time_utc: datetime,
    finalized_at: datetime,
    source_sha256: str,
    source_uri: str,
) -> ProfileGameContribution:
    """Reduce one final box score and its lineups to profile sufficient statistics."""

    required_players = {
        "game_id",
        "team_side",
        "team_id",
        "personId",
        "statistics_minutes",
        "statistics_fieldGoalsAttempted",
        "statistics_freeThrowsAttempted",
        "statistics_threePointersAttempted",
        "statistics_threePointersMade",
        "statistics_assists",
        "statistics_turnovers",
        "statistics_steals",
        "statistics_blocks",
        "statistics_reboundsOffensive",
        "statistics_reboundsDefensive",
    }
    missing = required_players - set(players)
    if missing:
        raise ValueError(f"Player box score missing profile columns: {sorted(missing)}")
    required_stints = {"game_id", "home_player_ids", "away_player_ids", "possessions"}
    missing_stints = required_stints - set(stints)
    if missing_stints:
        raise ValueError(f"Lineup stints missing profile columns: {sorted(missing_stints)}")
    player_games = tuple(players["game_id"].astype(str).unique())
    stint_games = tuple(stints["game_id"].astype(str).unique())
    if player_games != (str(game_id),) or stint_games != (str(game_id),):
        raise ValueError("Profile contribution inputs must contain exactly the requested game")

    box = players.copy()
    box["minutes"] = _minutes_series(box["statistics_minutes"])
    box = box.loc[box["minutes"].gt(0)].copy()
    if box.empty:
        raise ValueError("Profile contribution requires positive-minute players")
    box["player_id"] = pd.to_numeric(box["personId"], errors="raise").astype("int64")
    if box["player_id"].duplicated().any():
        raise ValueError("Profile contribution has duplicate player box-score rows")

    count_columns = {
        "three_pointers_attempted": "statistics_threePointersAttempted",
        "three_pointers_made": "statistics_threePointersMade",
        "assists": "statistics_assists",
        "turnovers": "statistics_turnovers",
        "steals": "statistics_steals",
        "blocks": "statistics_blocks",
        "offensive_rebounds": "statistics_reboundsOffensive",
        "defensive_rebounds": "statistics_reboundsDefensive",
        "field_goals_attempted": "statistics_fieldGoalsAttempted",
        "free_throws_attempted": "statistics_freeThrowsAttempted",
    }
    for output, source in count_columns.items():
        box[output] = pd.to_numeric(box[source], errors="raise").astype(float)
    box["usage_events"] = (
        box["field_goals_attempted"]
        + 0.44 * box["free_throws_attempted"]
        + box["turnovers"]
    )

    team = box.groupby(["team_side", "team_id"], as_index=False).agg(
        team_minutes=("minutes", "sum"),
        team_usage_events=("usage_events", "sum"),
        team_offensive_rebounds=("offensive_rebounds", "sum"),
        team_defensive_rebounds=("defensive_rebounds", "sum"),
    )
    if len(team) != 2 or set(team["team_side"].astype(str)) != {"home", "away"}:
        raise ValueError("Profile contribution requires one home and one away team")
    other = team.loc[
        :, ["team_side", "team_offensive_rebounds", "team_defensive_rebounds"]
    ].rename(
        columns={
            "team_side": "opponent_side",
            "team_offensive_rebounds": "opponent_offensive_rebounds",
            "team_defensive_rebounds": "opponent_defensive_rebounds",
        }
    )
    box = box.merge(team, on=["team_side", "team_id"], validate="many_to_one")
    box["opponent_side"] = np.where(box["team_side"].eq("home"), "away", "home")
    box = box.merge(other, on="opponent_side", validate="many_to_one")
    box["usage_opportunities"] = (
        box["minutes"] / (box["team_minutes"] / 5.0)
    ) * box["team_usage_events"]
    offensive_opportunities = (
        box["team_offensive_rebounds"] + box["opponent_defensive_rebounds"]
    )
    if offensive_opportunities.le(0).any():
        raise ValueError("Profile contribution has no offensive rebound opportunity")
    minute_share_inverse = box["team_minutes"] / (5.0 * box["minutes"])
    box["offensive_rebound_pct"] = (
        100.0 * box["offensive_rebounds"] * minute_share_inverse / offensive_opportunities
    )
    box["offensive_rebound_pct_minutes"] = box["offensive_rebound_pct"] * box["minutes"]

    possessions: dict[int, float] = {}
    for row in stints.itertuples(index=False):
        weight = float(row.possessions)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Profile contribution stint possessions must be positive")
        for player_id in (*row.home_player_ids, *row.away_player_ids):
            key = int(player_id)
            possessions[key] = possessions.get(key, 0.0) + weight
    box["on_court_possessions"] = box["player_id"].map(possessions)
    # A small number of short appearances are omitted from the accepted RAPM
    # stint reconstruction.  Retain their box-score events in the team totals
    # above, but do not invent a possession exposure or player-state update for
    # a player absent from the model's lineup evidence.
    box = box.loc[box["on_court_possessions"].notna()].copy()
    if box.empty:
        raise ValueError("No positive-minute box-score player appears in accepted stints")

    evidence = box.loc[:, EVIDENCE_COLUMNS].copy()
    return ProfileGameContribution(
        season=season,
        game_id=str(game_id),
        game_time_utc=game_time_utc,
        finalized_at=finalized_at,
        source_sha256=source_sha256,
        source_uri=source_uri,
        evidence=evidence,
    )


def solve_additive_profile_state(
    contract: FrozenAdditiveProfileContract,
    contributions: Sequence[ProfileGameContribution],
    *,
    cutoff_time: datetime,
    config: AdditiveProfileUpdaterConfig | None = None,
) -> pd.DataFrame:
    """Return the dated player profile state strictly before ``cutoff_time``."""

    active_config = config or AdditiveProfileUpdaterConfig()
    active_config.validate()
    cutoff = _as_utc(cutoff_time, "cutoff_time")
    selected = _select_contributions(
        contributions, season=contract.season, cutoff_time=cutoff
    )
    evidence = _aggregate_evidence(selected, contract.player_ids)
    profile = contract.profiles.set_index("player_id")
    result = pd.DataFrame({"player_id": contract.player_ids})
    result["schema_version"] = PROFILE_STATE_SCHEMA_VERSION
    result["season"] = contract.season
    result["core_model_version"] = contract.core_model_version
    result["profile_updater_version"] = active_config.updater_version
    result["cutoff_time"] = cutoff
    result["profile_contract_sha256"] = contract.lineage_sha256
    result["included_game_ids_json"] = json.dumps(
        [row.game_id for row in selected], separators=(",", ":")
    )
    result["contribution_lineage_json"] = json.dumps(
        [
            {
                "game_id": row.game_id,
                "source_sha256": row.source_sha256,
                "contribution_sha256": row.contribution_sha256,
            }
            for row in selected
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    result = result.set_index("player_id")

    possessions = evidence["on_court_possessions"].to_numpy(dtype=float)
    three_pa, three_pa_sd, three_pa_weight = _posterior_count_rate(
        profile["three_pa_per_100"].to_numpy(dtype=float),
        evidence["three_pointers_attempted"].to_numpy(dtype=float),
        possessions,
        active_config.prior_exposure("three_pa_per_100"),
    )
    prior_three_pa = profile["three_pa_per_100"].to_numpy(dtype=float)
    prior_three_pm = profile["three_pm_per_100"].to_numpy(dtype=float)
    prior_three_pct = np.divide(
        prior_three_pm,
        prior_three_pa,
        out=np.zeros_like(prior_three_pm),
        where=prior_three_pa > 0,
    )
    three_pct, three_pct_sd, three_pct_weight = _posterior_percentage(
        prior_three_pct,
        evidence["three_pointers_made"].to_numpy(dtype=float),
        evidence["three_pointers_attempted"].to_numpy(dtype=float),
        active_config.prior_exposure("three_point_percentage"),
    )
    three_pm = three_pa * three_pct
    no_three_point_evidence = (
        (possessions == 0.0)
        & (evidence["three_pointers_attempted"].to_numpy(dtype=float) == 0.0)
    )
    three_pm = np.where(no_three_point_evidence, prior_three_pm, three_pm)
    three_pm_sd = np.sqrt(
        np.square(three_pct * three_pa_sd) + np.square(three_pa * three_pct_sd)
    )

    updated: dict[str, np.ndarray] = {
        "three_pa_per_100": three_pa,
        "three_pm_per_100": three_pm,
    }
    standard_error: dict[str, np.ndarray] = {
        "three_pa_per_100": three_pa_sd,
        "three_pm_per_100": three_pm_sd,
    }
    evidence_weight: dict[str, np.ndarray] = {
        "three_pa_per_100": three_pa_weight,
        "three_pm_per_100": 1.0 - (1.0 - three_pa_weight) * (1.0 - three_pct_weight),
    }
    count_feature_sources = {
        "assists_per_100": "assists",
        "turnovers_per_100": "turnovers",
        "steals_per_100": "steals",
        "blocks_per_100": "blocks",
    }
    for feature, source in count_feature_sources.items():
        posterior, sd, weight = _posterior_count_rate(
            profile[feature].to_numpy(dtype=float),
            evidence[source].to_numpy(dtype=float),
            possessions,
            active_config.prior_exposure(feature),
        )
        updated[feature] = posterior
        standard_error[feature] = sd
        evidence_weight[feature] = weight

    usage, usage_sd, usage_weight = _posterior_percent_rate(
        profile["usage_pct"].to_numpy(dtype=float),
        evidence["usage_events"].to_numpy(dtype=float),
        evidence["usage_opportunities"].to_numpy(dtype=float),
        active_config.prior_exposure("usage_pct"),
    )
    updated["usage_pct"] = usage
    standard_error["usage_pct"] = usage_sd
    evidence_weight["usage_pct"] = usage_weight

    minutes = evidence["minutes"].to_numpy(dtype=float)
    raw_orb = np.divide(
        evidence["offensive_rebound_pct_minutes"].to_numpy(dtype=float),
        minutes,
        out=profile["offensive_rebound_pct"].to_numpy(dtype=float).copy(),
        where=minutes > 0,
    )
    orb, orb_sd, orb_weight = _posterior_continuous_percentage(
        profile["offensive_rebound_pct"].to_numpy(dtype=float),
        raw_orb,
        possessions,
        active_config.prior_exposure("offensive_rebound_claim_total"),
    )
    updated["offensive_rebound_claim_total"] = orb
    standard_error["offensive_rebound_claim_total"] = orb_sd
    evidence_weight["offensive_rebound_claim_total"] = orb_weight

    contribution_variance = np.zeros(len(result), dtype=float)
    base_total = np.asarray(
        [contract.published_adjustments[player_id] for player_id in contract.player_ids],
        dtype=float,
    )
    delta_total = np.zeros(len(result), dtype=float)
    for feature, profile_column in FEATURE_TO_PROFILE_COLUMN.items():
        prior_values = profile[profile_column].to_numpy(dtype=float)
        values = updated[feature]
        coefficient = float(contract.coefficients[feature])
        delta = coefficient * (values - prior_values)
        result[f"preseason_{feature}"] = prior_values
        result[f"updated_{feature}"] = values
        result[f"profile_delta_{feature}"] = values - prior_values
        result[f"evidence_weight_{feature}"] = evidence_weight[feature]
        result[f"posterior_standard_error_{feature}"] = standard_error[feature]
        result[f"rating_delta_{feature}"] = delta
        delta_total += delta
        contribution_variance += np.square(coefficient * standard_error[feature])
    result["preseason_additive_profile"] = base_total
    result["additive_profile_delta"] = delta_total
    result["updated_additive_profile"] = base_total + delta_total
    result["additive_profile_standard_error"] = np.sqrt(contribution_variance)
    result["raw_on_court_possessions"] = possessions
    result["observed_games"] = evidence["observed_games"].to_numpy(dtype=int)
    result = result.reset_index()
    state_sha = _profile_state_sha256(result)
    result["profile_state_sha256"] = state_sha
    validate_additive_profile_state(result, contract=contract)
    return result


def validate_additive_profile_state(
    state: pd.DataFrame,
    *,
    contract: FrozenAdditiveProfileContract | None = None,
) -> None:
    """Validate the reconstruction and additive-delta arithmetic."""

    required = {
        "player_id",
        "season",
        "cutoff_time",
        "preseason_additive_profile",
        "additive_profile_delta",
        "updated_additive_profile",
        "profile_state_sha256",
    }
    for feature in PROFILE_FEATURES:
        required.update(
            {
                f"preseason_{feature}",
                f"updated_{feature}",
                f"profile_delta_{feature}",
                f"evidence_weight_{feature}",
                f"posterior_standard_error_{feature}",
                f"rating_delta_{feature}",
            }
        )
    missing = required - set(state)
    if missing:
        raise ValueError(f"Profile state missing columns: {sorted(missing)}")
    if state.empty or state["player_id"].duplicated().any():
        raise ValueError("Profile state must contain unique players")
    numeric = state.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Profile state numeric values must be finite")
    if not np.allclose(
        state["updated_additive_profile"],
        state["preseason_additive_profile"] + state["additive_profile_delta"],
        atol=1e-12,
    ):
        raise ValueError("Updated additive profile does not reconstruct")
    feature_delta = sum(state[f"rating_delta_{feature}"] for feature in PROFILE_FEATURES)
    if not np.allclose(feature_delta, state["additive_profile_delta"], atol=1e-12):
        raise ValueError("Feature rating deltas do not reconstruct additive movement")
    for feature in PROFILE_FEATURES:
        weight = state[f"evidence_weight_{feature}"].to_numpy(dtype=float)
        if (weight < 0).any() or (weight > 1).any():
            raise ValueError(f"Profile evidence weight is outside [0, 1]: {feature}")
    hashes = tuple(state["profile_state_sha256"].astype(str).unique())
    if len(hashes) != 1:
        raise ValueError("Profile state must contain one semantic checksum")
    _validate_sha256(hashes[0], "profile_state_sha256")
    if contract is not None:
        expected_players = contract.player_ids
        actual_players = tuple(state.sort_values("player_id")["player_id"].astype(int))
        if actual_players != expected_players:
            raise ValueError("Profile state player vocabulary changed")
        zero = state["raw_on_court_possessions"].eq(0)
        for feature in PROFILE_FEATURES:
            if not np.allclose(
                state.loc[zero, f"preseason_{feature}"],
                state.loc[zero, f"updated_{feature}"],
                atol=0.0,
                rtol=0.0,
            ):
                raise ValueError(f"Zero-evidence profile moved: {feature}")


def profile_delta_map(state: pd.DataFrame) -> dict[int, float]:
    validate_additive_profile_state(state)
    return dict(
        zip(
            state["player_id"].astype(int),
            state["additive_profile_delta"].astype(float),
            strict=True,
        )
    )


def publish_profile_game_contribution(
    contribution: ProfileGameContribution,
    root: Path | str,
) -> tuple[Path, bool]:
    """Publish one immutable profile contribution."""

    base = Path(root)
    target = (
        base
        / f"season={contribution.season}"
        / f"game_id={contribution.game_id}"
        / f"contribution_sha256={contribution.contribution_sha256}"
    )
    if target.exists():
        loaded = read_profile_game_contribution(target)
        if loaded.contribution_sha256 != contribution.contribution_sha256:
            raise ValueError("Existing profile contribution changed")
        return target, False
    staging = base / f".profile-contribution.tmp-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        evidence_path = staging / "evidence.parquet"
        contribution.evidence.to_parquet(evidence_path, index=False)
        manifest = {
            "schema_version": contribution.schema_version,
            "season": contribution.season,
            "game_id": contribution.game_id,
            "game_time_utc": contribution.game_time_utc.isoformat(),
            "finalized_at": contribution.finalized_at.isoformat(),
            "source_sha256": contribution.source_sha256,
            "source_uri": contribution.source_uri,
            "contribution_sha256": contribution.contribution_sha256,
            "evidence_sha256": _file_sha256(evidence_path),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, True


def read_profile_game_contribution(path: Path | str) -> ProfileGameContribution:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    evidence_path = root / "evidence.parquet"
    if _file_sha256(evidence_path) != manifest["evidence_sha256"]:
        raise ValueError("Profile contribution evidence checksum changed")
    contribution = ProfileGameContribution(
        schema_version=int(manifest["schema_version"]),
        season=str(manifest["season"]),
        game_id=str(manifest["game_id"]),
        game_time_utc=datetime.fromisoformat(str(manifest["game_time_utc"])),
        finalized_at=datetime.fromisoformat(str(manifest["finalized_at"])),
        source_sha256=str(manifest["source_sha256"]),
        source_uri=str(manifest["source_uri"]),
        evidence=pd.read_parquet(evidence_path),
    )
    if contribution.contribution_sha256 != manifest["contribution_sha256"]:
        raise ValueError("Profile contribution semantic checksum changed")
    return contribution


def publish_additive_profile_state(
    state: pd.DataFrame,
    root: Path | str,
) -> tuple[Path, bool]:
    """Atomically publish one content-addressed profile state."""

    validate_additive_profile_state(state)
    season_values = tuple(state["season"].astype(str).unique())
    cutoff_values = pd.to_datetime(state["cutoff_time"], utc=True).unique()
    hashes = tuple(state["profile_state_sha256"].astype(str).unique())
    if len(season_values) != 1 or len(cutoff_values) != 1 or len(hashes) != 1:
        raise ValueError("Profile state publication requires one season, cutoff, and checksum")
    cutoff = pd.Timestamp(cutoff_values[0]).to_pydatetime().astimezone(UTC)
    target = (
        Path(root)
        / f"season={season_values[0]}"
        / f"cutoff={cutoff.strftime('%Y%m%dT%H%M%SZ')}"
        / f"state_sha256={hashes[0]}"
    )
    if target.exists():
        return target, False
    staging = Path(root) / f".profile-state.tmp-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        state_path = staging / "profile_state.parquet"
        state.sort_values("player_id", kind="stable").to_parquet(state_path, index=False)
        manifest = {
            "schema_version": PROFILE_STATE_SCHEMA_VERSION,
            "season": season_values[0],
            "cutoff_time": cutoff.isoformat(),
            "profile_state_sha256": hashes[0],
            "file_sha256": _file_sha256(state_path),
            "player_count": len(state),
            "included_game_ids": json.loads(state["included_game_ids_json"].iloc[0]),
            "contribution_lineage": json.loads(
                state["contribution_lineage_json"].iloc[0]
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, True


def _select_contributions(
    contributions: Sequence[ProfileGameContribution],
    *,
    season: str,
    cutoff_time: datetime,
) -> tuple[ProfileGameContribution, ...]:
    selected: dict[str, ProfileGameContribution] = {}
    for contribution in contributions:
        if contribution.season != season:
            raise ValueError("Profile contribution season mismatch")
        if contribution.finalized_at >= cutoff_time:
            continue
        prior = selected.get(contribution.game_id)
        if prior is not None and prior.contribution_sha256 != contribution.contribution_sha256:
            raise ValueError(
                "Multiple profile revisions supplied for one game; use the current ledger"
            )
        selected[contribution.game_id] = contribution
    return tuple(
        sorted(
            selected.values(),
            key=lambda row: (row.finalized_at, row.game_time_utc, row.game_id),
        )
    )


def _aggregate_evidence(
    contributions: Sequence[ProfileGameContribution],
    player_ids: Sequence[int],
) -> pd.DataFrame:
    rows = [contribution.evidence for contribution in contributions]
    if rows:
        combined = pd.concat(rows, ignore_index=True)
        aggregate_columns = [column for column in EVIDENCE_COLUMNS if column != "player_id"]
        aggregate = combined.groupby("player_id", as_index=False)[aggregate_columns].sum()
        games = combined.groupby("player_id")["player_id"].size().rename("observed_games")
        aggregate = aggregate.merge(
            games, left_on="player_id", right_index=True, validate="one_to_one"
        )
    else:
        aggregate = pd.DataFrame(columns=[*EVIDENCE_COLUMNS, "observed_games"])
    output = pd.DataFrame({"player_id": tuple(int(value) for value in player_ids)}).merge(
        aggregate, on="player_id", how="left", validate="one_to_one"
    )
    for column in [*EVIDENCE_COLUMNS[1:], "observed_games"]:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    unknown = set(aggregate.get("player_id", pd.Series(dtype=int)).astype(int)) - set(player_ids)
    if unknown:
        raise ValueError(f"Profile evidence contains players outside contract: {sorted(unknown)}")
    return output.set_index("player_id").loc[list(player_ids)]


def _posterior_count_rate(
    prior_rate: np.ndarray,
    counts: np.ndarray,
    exposure: np.ndarray,
    prior_exposure: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = prior_exposure + exposure
    posterior = (prior_exposure * prior_rate + 100.0 * counts) / total
    posterior = np.where(exposure == 0.0, prior_rate, posterior)
    expected_count = np.maximum(posterior / 100.0 * total, 0.0)
    standard_error = 100.0 * np.sqrt(expected_count) / total
    return posterior, standard_error, exposure / total


def _posterior_percentage(
    prior_probability: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    prior_trials: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = prior_trials + trials
    posterior = (prior_trials * prior_probability + successes) / total
    posterior = np.clip(posterior, 0.0, 1.0)
    posterior = np.where(trials == 0.0, prior_probability, posterior)
    standard_error = np.sqrt(posterior * (1.0 - posterior) / total)
    return posterior, standard_error, trials / total


def _posterior_percent_rate(
    prior_percent: np.ndarray,
    events: np.ndarray,
    opportunities: np.ndarray,
    prior_opportunities: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability, standard_error, weight = _posterior_percentage(
        prior_percent / 100.0,
        events,
        opportunities,
        prior_opportunities,
    )
    posterior = 100.0 * probability
    posterior = np.where(opportunities == 0.0, prior_percent, posterior)
    return posterior, 100.0 * standard_error, weight


def _posterior_continuous_percentage(
    prior_percent: np.ndarray,
    observed_percent: np.ndarray,
    exposure: np.ndarray,
    prior_exposure: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = prior_exposure + exposure
    posterior = (prior_exposure * prior_percent + exposure * observed_percent) / total
    posterior = np.where(exposure == 0.0, prior_percent, posterior)
    probability = np.clip(posterior / 100.0, 0.0, 1.0)
    approximate_standard_error = 100.0 * np.sqrt(probability * (1.0 - probability) / total)
    return posterior, approximate_standard_error, exposure / total


def _base_additive_adjustments(contract: FrozenAdditiveProfileContract) -> np.ndarray:
    profile = contract.profiles
    total = np.zeros(len(profile), dtype=float)
    for feature, column in FEATURE_TO_PROFILE_COLUMN.items():
        total += float(contract.coefficients[feature]) * (
            profile[column].to_numpy(dtype=float) - float(contract.reference_values[feature])
        )
    return total


def _profile_state_sha256(state: pd.DataFrame) -> str:
    excluded = state.drop(columns=["profile_state_sha256"], errors="ignore").copy()
    return _frame_sha256(excluded.sort_values("player_id", kind="stable"))


def _frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.select_dtypes(include=["datetime", "datetimetz"]).columns:
        normalized[column] = pd.to_datetime(normalized[column], utc=True).astype(str)
    payload = normalized.to_json(
        orient="split", date_format="iso", date_unit="us", double_precision=15
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _minutes_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").astype(float)
    return pd.to_timedelta(values, errors="coerce").dt.total_seconds() / 60.0


def _validate_feature_mapping(values: Mapping[str, float], label: str) -> None:
    missing = set(PROFILE_FEATURES) - set(values)
    extra = set(values) - set(PROFILE_FEATURES)
    if missing or extra:
        raise ValueError(
            f"Profile {label} mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _as_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
