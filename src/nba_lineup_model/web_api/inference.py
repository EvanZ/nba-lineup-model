"""Compact in-memory inference for the NBA GESTALT Lineup Lab."""

from __future__ import annotations

import json
import random
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE,
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT,
    CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
    LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
    LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES,
    contextual_feature_columns,
    lineup_side_context_features,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.matchup_contextual import (
    BoundedMatchupContextualModel,
    MatchupContextualModel,
    isolated_feature_component,
)
from nba_lineup_model.modeling.stints import read_rapm_stints

if TYPE_CHECKING:
    from nba_lineup_model.modeling.contextual_profiles import ProfilePaddingContract

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_RESPONSE_CACHE_DIR = Path("artifacts/web/response_curve_cache")
DEFAULT_LINEUP_RANKINGS_CACHE_DIR = Path("artifacts/web/lineup_rankings")
DEFAULT_PLAYER_TEAM_SPLITS_CACHE_DIR = Path("artifacts/web/player_team_splits")
DEFAULT_EXPOSURE_COHORT_CACHE_DIR = Path("artifacts/web/exposure_cohorts")
DEFAULT_HISTORICAL_PROFILE_CACHE_DIR = Path("artifacts/web/historical_profiles")
DEFAULT_HISTORICAL_REALIZED_PROFILE_CACHE_DIR = Path(
    "artifacts/web/historical_realized_profiles"
)
DEFAULT_PRESEASON_RANKINGS_CACHE_DIR = Path("artifacts/web/preseason_rankings")
DEFAULT_TEAM_ROSTERS_DIR = Path("data/curated/team_rosters")
DEFAULT_FORWARD_DRAFT_COLD_START_DIR = Path(
    "artifacts/models/forward_draft_history_cold_start"
)
# Keep the artifact identifier distinct from the public release name.
MODEL_ARTIFACT = "forward_nail_rapm_v1212_back_to_back"
MODEL_NAME = "forward_nail_rapm_v1212_back_to_back"
MODEL_DISPLAY_NAME = "NAIL-RAPM v1.2.1.2"
DISPLAY_SEASON = "2025-26"
PRESEASON_PREVIEW_SEASON = "2026-27"
RESPONSE_CURVE_POINTS = 33
# The client renders the same 33-point polyline, so a denser server cache only
# adds warmup cost without increasing the published curve resolution.
WARM_RESPONSE_CURVE_POINTS = RESPONSE_CURVE_POINTS
LINEUP_REFERENCE_SAMPLE_SIZE = 512
WIN_PCT_INTERCEPT = 0.499583
WIN_PCT_PER_NET_RATING = 0.030250


def build_contextual_player_profiles(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Lazily load the profile builder, avoiding web startup plotting imports."""

    from nba_lineup_model.modeling.contextual_profiles import (
        build_contextual_player_profiles as implementation,
    )

    return implementation(*args, **kwargs)


def _published_profile_padding_contract(
    metadata: dict[str, Any],
) -> ProfilePaddingContract:
    """Validate and return the exact profile-padding contract for this release."""

    from nba_lineup_model.modeling.contextual_profiles import (
        MEDVEDOVSKY_2020_PROFILE_PADDING,
    )

    actual = metadata.get("profile_padding_contract")
    expected = MEDVEDOVSKY_2020_PROFILE_PADDING.metadata()
    # v1.2.1 predates the standard-USG% coordinate. Its immutable metadata
    # therefore omits this new padding field, whose default remains 300.
    if isinstance(actual, dict) and "usage_percentage_pseudo_possessions" not in actual:
        expected = {
            key: value
            for key, value in expected.items()
            if key != "usage_percentage_pseudo_possessions"
        }
    if not isinstance(actual, dict) or any(
        actual.get(key) != value for key, value in expected.items()
    ):
        raise LineupEvaluationError(
            f"The selected {MODEL_DISPLAY_NAME} artifact does not contain the "
            "published statistic-specific profile-padding coefficients"
        )
    return MEDVEDOVSKY_2020_PROFILE_PADDING


def prepare_player_exposure_cohort(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Lazily load the cohort builder when an uncached historical state is needed."""

    from nba_lineup_model.modeling.replacement_level import (
        prepare_player_exposure_cohort as implementation,
    )

    return implementation(*args, **kwargs)


def _previous_season(season: str) -> str:
    """Return the prior NBA season label without importing training modules."""

    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def projected_win_pct(net_rating: float) -> float:
    """Convert a neutral-court NetRtg estimate into the published win-rate scale."""

    return float(
        np.clip(WIN_PCT_INTERCEPT + WIN_PCT_PER_NET_RATING * net_rating, 0.0, 1.0)
    )


def _mean_reverted_schedule_controls(
    season_metadata: pd.DataFrame,
    schedule_metadata: pd.DataFrame,
) -> MeanRevertedScheduleControls:
    """Pool completed schedule states into stable Lab reference coefficients."""

    home_court = _weighted_completed_mean(
        season_metadata,
        value_column="context_home_intercept",
        weight_column="context_training_weight_sum",
        label="home-court",
    )
    back_to_back = _weighted_completed_mean(
        schedule_metadata,
        value_column="schedule_control_raw_weight",
        weight_column="schedule_training_stint_count",
        label="back-to-back",
    )
    source_seasons = int(
        min(
            season_metadata["season"].astype(str).nunique(),
            schedule_metadata["season"].astype(str).nunique(),
        )
    )
    if source_seasons < 1:
        raise LineupEvaluationError("The artifact has no completed schedule-control states")
    return MeanRevertedScheduleControls(
        home_court=home_court,
        back_to_back=back_to_back,
        source_season_count=source_seasons,
    )


def _weighted_completed_mean(
    frame: pd.DataFrame,
    *,
    value_column: str,
    weight_column: str,
    label: str,
) -> float:
    """Return a possession-weighted historical mean after rejecting stale metadata."""

    required = {value_column, weight_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LineupEvaluationError(f"The artifact lacks {label} control metadata: {missing}")
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(frame[weight_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise LineupEvaluationError(f"The artifact has no valid {label} control states")
    return float(np.average(values[valid], weights=weights[valid]))


def _apply_schedule_scenario(
    result: dict[str, Any],
    *,
    controls: MeanRevertedScheduleControls | None,
    court: str,
    unit_back_to_back: bool,
    opponent_back_to_back: bool,
) -> dict[str, Any]:
    """Overlay a transparent, date-free schedule scenario on a matchup edge."""

    if controls is None:
        raise LineupEvaluationError("The artifact does not contain Lab schedule controls")
    court_sign = {"neutral": 0.0, "unit_home": 1.0, "opponent_home": -1.0}[court]
    home_court_adjustment = court_sign * controls.home_court
    back_to_back_difference = int(unit_back_to_back) - int(opponent_back_to_back)
    back_to_back_adjustment = back_to_back_difference * controls.back_to_back
    schedule_adjustment = home_court_adjustment + back_to_back_adjustment
    base_net_rating = float(result["predicted_net_rating"])
    adjusted_net_rating = base_net_rating + schedule_adjustment
    updated = result.copy()
    updated.update(
        {
            "base_predicted_net_rating": base_net_rating,
            "court": court,
            "unit_back_to_back": unit_back_to_back,
            "opponent_back_to_back": opponent_back_to_back,
            "home_court_adjustment": home_court_adjustment,
            "back_to_back_adjustment": back_to_back_adjustment,
            "schedule_adjustment": schedule_adjustment,
            "home_court_reference": controls.home_court,
            "back_to_back_reference": controls.back_to_back,
            "schedule_control_source_season_count": controls.source_season_count,
            "predicted_net_rating": adjusted_net_rating,
            "predicted_win_pct": projected_win_pct(adjusted_net_rating),
        }
    )
    return updated

_FEATURE_LABELS = {
    "home_minus_away_three_pa_per_100": "Three-point attempt volume",
    "home_minus_away_three_pm_per_100": "Three-point makes",
    "home_minus_away_assists_per_100": "Assists",
    "home_minus_away_turnovers_per_100": "Turnovers",
    "home_minus_away_usage_per_100": "Usage events",
    "home_minus_away_usage_pct": "Usage percentage",
    "home_minus_away_offensive_rebounds_per_100": "Offensive rebounds",
    "home_minus_away_defensive_rebounds_per_100": "Defensive rebounds",
    "home_minus_away_steals_per_100": "Steals",
    "home_minus_away_blocks_per_100": "Blocks",
    "home_minus_away_bottom_two_three_pm": "Bottom-two three-point makes",
    "home_minus_away_credible_shooter_count": "Credible-shooter count",
    "home_minus_away_top_two_assists": "Top-two assists",
    "home_minus_away_usage_concentration": "Usage concentration",
    "home_minus_away_sqrt_offensive_rebounds": "Diminishing offensive rebounding",
    "home_minus_away_sqrt_defensive_rebounds": "Diminishing defensive rebounding",
    "home_minus_away_imputed_count": "Imputed-profile count",
    "home_minus_away_replacement_weight": "Replacement-profile weight",
    "home_minus_away_shooting_usage_interaction": "Shooting-by-usage",
    "home_minus_away_shooter_passing_interaction": "Shooter-by-passing",
    "home_minus_away_rebounding_usage_interaction": "Rebounding-by-usage",
}

# These NAIL-RAPM coordinates are exact sums of player-profile values. A linear
# Ridge coefficient on each coordinate can therefore be compiled into the
# individual player rating without changing any matchup prediction.
_LINEAR_X3_ADDITIVE_FEATURE_TO_PROFILE = {
    "three_pa_per_100": "three_pa_per_100",
    "three_pm_per_100": "three_pm_per_100",
    "assists_per_100": "assists_per_100",
    "turnovers_per_100": "turnovers_per_100",
    "usage_per_100": "usage_per_100",
    "steals_per_100": "steals_per_100",
    "blocks_per_100": "blocks_per_100",
    "imputed_count": "profile_imputed",
    "replacement_weight": "profile_replacement_weight",
    "offensive_rebound_claim_total": "offensive_rebound_pct",
}


def _linear_x3_additive_feature_map(feature_set: str) -> dict[str, str]:
    """Return the player-compilable coordinates for one x3 artifact contract."""

    if feature_set == CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE:
        return {
            feature: _LINEAR_X3_ADDITIVE_FEATURE_TO_PROFILE[feature]
            for feature in LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES
        }
    if feature_set == CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE:
        return {
            feature: (
                "usage_pct"
                if feature == "usage_pct"
                else _LINEAR_X3_ADDITIVE_FEATURE_TO_PROFILE[feature]
            )
            for feature in LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES
        }
    if feature_set == CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY:
        return {
            feature: profile_column
            for feature, profile_column in _LINEAR_X3_ADDITIVE_FEATURE_TO_PROFILE.items()
            if feature not in {"imputed_count", "replacement_weight"}
        }
    if feature_set == CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT:
        return _LINEAR_X3_ADDITIVE_FEATURE_TO_PROFILE
    raise LineupEvaluationError(f"Unsupported compiled-linear x3 contract: {feature_set}")


class LineupEvaluationError(ValueError):
    """Raised when an input cannot be evaluated against the loaded model state."""


@dataclass(frozen=True)
class SeasonLineupState:
    """One historical player pool scored in its completed-fit season."""

    season: str
    coefficients: pd.DataFrame
    profiles: pd.DataFrame
    players: pd.DataFrame


@dataclass(frozen=True)
class MeanRevertedScheduleControls:
    """Long-run schedule coefficients used for date-free Lab scenarios."""

    home_court: float
    back_to_back: float
    source_season_count: int


@dataclass(frozen=True)
class LineupEvaluator:
    """Serve one completed contextual RAPM state without reading raw possession data."""

    season: str
    run_id: str
    coefficients: pd.DataFrame
    profiles: pd.DataFrame
    players: pd.DataFrame
    context_model: MatchupContextualModel
    response_cache: dict[int, tuple[np.ndarray, np.ndarray]]
    context_alpha: float | None = None
    schedule_controls: MeanRevertedScheduleControls | None = None
    profile_padding_contract: ProfilePaddingContract | None = None
    use_last_observed_profile: bool = False
    lineup_rankings_root: Path | None = None
    player_rating_histories: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    player_league_leader_histories: dict[int, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    player_team_splits: dict[tuple[str, int], list[dict[str, Any]]] = field(
        default_factory=dict
    )
    player_latest_teams: dict[tuple[str, int], dict[str, Any]] = field(
        default_factory=dict
    )
    historical_rankings: pd.DataFrame = field(default_factory=pd.DataFrame)
    preseason_rankings: pd.DataFrame = field(default_factory=pd.DataFrame)
    observed_lineups: pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_coefficients: pd.DataFrame = field(default_factory=pd.DataFrame)
    seasonal_ratings: pd.DataFrame = field(default_factory=pd.DataFrame)
    season_context_models: dict[str, MatchupContextualModel] = field(default_factory=dict)
    player_season_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    exposure_cohort: pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_profiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_realized_profiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    season_states: dict[str, SeasonLineupState] = field(default_factory=dict)
    response_caches: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )

    @classmethod
    def from_latest_artifact(
        cls,
        *,
        artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
        season: str = DISPLAY_SEASON,
        panel_path: Path | str = "data/analytical/player_season_panel/player_seasons.parquet",
    ) -> LineupEvaluator:
        """Load the published completed-fit model state used by the local Lineup Lab."""

        root = Path(artifacts_dir) / MODEL_ARTIFACT / season
        latest_path = root / "latest.json"
        if not latest_path.is_file():
            raise FileNotFoundError(
                f"No published {MODEL_DISPLAY_NAME} artifact for {season}: {latest_path}"
            )
        run_id = str(json.loads(latest_path.read_text())["run_id"])
        run_dir = root / run_id
        metadata = json.loads((run_dir / "metadata.json").read_text())
        if metadata.get("model") != MODEL_NAME:
            raise LineupEvaluationError(
                f"The selected artifact is not the published {MODEL_DISPLAY_NAME} model"
            )
        if str(metadata.get("target_season")) != season:
            raise LineupEvaluationError("The selected artifact has an unexpected target season")

        profile_padding_contract = _published_profile_padding_contract(metadata)
        use_last_observed_profile = (
            metadata.get("profile_padding_contract", {}).get("gap_returner_profile_method")
            == "last_observed_padded_profile"
        )
        historical_coefficients = pd.read_parquet(
            run_dir / "historical_player_coefficients.parquet"
        )
        coefficients = historical_coefficients.loc[
            historical_coefficients["season"].eq(season), ["player_id", "rapm"]
        ].copy()
        if coefficients.empty or coefficients["player_id"].duplicated().any():
            raise LineupEvaluationError("The artifact has invalid completed player coefficients")
        prior_profiles = pd.read_parquet(run_dir / "target_player_profiles.parquet")
        if prior_profiles["player_id"].duplicated().any():
            raise LineupEvaluationError("The artifact has duplicate player profiles")
        raw_seasonal_ratings = pd.read_parquet(run_dir / "player_season_ratings.parquet")
        schedule_controls = _mean_reverted_schedule_controls(
            pd.read_parquet(run_dir / "season_model_metadata.parquet"),
            pd.read_parquet(run_dir / "season_schedule_control_metadata.parquet"),
        )
        models = joblib.load(run_dir / "season_context_models.joblib")
        context_model = models.get(season)
        if context_model is None:
            raise LineupEvaluationError(
                f"The artifact has no completed {MODEL_DISPLAY_NAME} function"
            )
        if not isinstance(context_model, MatchupContextualModel):
            raise LineupEvaluationError("The artifact has an incompatible contextual model")
        published_ratings_path = published_player_ratings_path(MODEL_ARTIFACT, run_id)
        seasonal_ratings = (
            pd.read_parquet(published_ratings_path)
            if published_ratings_path.is_file()
            else raw_seasonal_ratings
        )
        rookie_seasons = _rookie_seasons(seasonal_ratings)
        team_splits_cache_path = player_team_splits_path(MODEL_ARTIFACT, run_id)
        team_splits_frame = (
            pd.read_parquet(team_splits_cache_path)
            if team_splits_cache_path.is_file()
            else pd.DataFrame()
        )
        player_team_splits = (
            _player_team_splits_by_season(team_splits_frame)
            if not team_splits_frame.empty
            else {}
        )
        player_latest_teams = (
            _player_latest_teams_by_season(team_splits_frame)
            if not team_splits_frame.empty
            else {}
        )
        context_exposure_path = player_context_exposure_path(MODEL_ARTIFACT, run_id)
        context_exposure = (
            pd.read_parquet(context_exposure_path)
            if context_exposure_path.is_file()
            else pd.DataFrame(columns=["season", "player_id", "observed_context_exposure"])
        )
        historical_rankings = _historical_ranking_catalog(
            seasonal_ratings,
            panel_path=Path(panel_path),
            context_exposure=context_exposure,
            rookie_seasons=rookie_seasons,
            player_latest_teams=player_latest_teams,
        )
        preseason_path = preseason_rankings_path(
            MODEL_ARTIFACT, run_id, PRESEASON_PREVIEW_SEASON
        )
        preseason_rankings = (
            pd.read_parquet(preseason_path)
            if preseason_path.is_file()
            else _preseason_ranking_catalog(
                historical_rankings,
                roster_path=team_roster_path(PRESEASON_PREVIEW_SEASON),
                draft_rankings_path=forward_draft_cold_start_rankings_path(
                    PRESEASON_PREVIEW_SEASON
                ),
                completed_season=season,
                preview_season=PRESEASON_PREVIEW_SEASON,
            )
        )
        player_rating_histories = _player_rating_histories(
            seasonal_ratings,
            panel_path=Path(panel_path),
            context_exposure=context_exposure,
            player_team_splits=player_team_splits,
            player_latest_teams=player_latest_teams,
        )
        player_league_leader_histories = _player_league_leader_histories(
            seasonal_ratings,
            player_rating_histories,
            active_through_years=_player_active_through_years(Path(panel_path)),
        )
        cache_path = response_cache_path(MODEL_ARTIFACT, run_id)
        response_cache = joblib.load(cache_path) if cache_path.is_file() else {}
        player_season_panel = pd.read_parquet(panel_path)
        cohort_path = exposure_cohort_path(MODEL_ARTIFACT, run_id)
        exposure_cohort = (
            pd.read_parquet(cohort_path)
            if cohort_path.is_file()
            else prepare_player_exposure_cohort(
                player_season_panel.loc[
                    player_season_panel["season"].astype(str).le(season)
                ],
                through_season=season,
            )
        )
        profile_cache_path = historical_profiles_path(MODEL_ARTIFACT, run_id)
        historical_profiles = (
            pd.read_parquet(profile_cache_path)
            if profile_cache_path.is_file()
            else pd.DataFrame()
        )
        realized_profile_cache_path = historical_realized_profiles_path(MODEL_ARTIFACT, run_id)
        historical_realized_profiles = (
            pd.read_parquet(realized_profile_cache_path)
            if realized_profile_cache_path.is_file()
            else pd.DataFrame()
        )
        realized_profiles = historical_realized_profiles.loc[
            historical_realized_profiles.get("season", pd.Series(dtype=str))
            .astype(str)
            .eq(season)
        ].copy()
        if not realized_profiles.empty:
            realized_profiles = realized_profiles.drop(columns="season")
        expected_player_ids = set(coefficients["player_id"].astype(int))
        if (
            realized_profiles.empty
            or set(realized_profiles.get("player_id", pd.Series(dtype=int)).astype(int))
            != expected_player_ids
            or realized_profiles["player_id"].duplicated().any()
        ):
            realized_profiles = build_contextual_player_profiles(
                player_season_panel,
                target_season=season,
                target_player_ids=expected_player_ids,
                exposure_cohort=exposure_cohort,
                **(
                    {"padding_contract": profile_padding_contract}
                    if profile_padding_contract is not None
                    else {}
                ),
                profile_timing="realized",
            )
        display_coefficients = _compiled_linear_x3_coefficients(
            coefficients,
            prior_profiles,
            context_model,
        )
        players = _player_catalog(
            display_coefficients,
            realized_profiles,
            panel_path=Path(panel_path),
            season=season,
            player_latest_teams=player_latest_teams,
        )
        display_coefficients = _compiled_linear_x3_coefficients(
            coefficients,
            prior_profiles,
            context_model,
            center=_player_rating_center(display_coefficients, players),
        )
        players = _player_catalog(
            display_coefficients,
            realized_profiles,
            panel_path=Path(panel_path),
            season=season,
            player_latest_teams=player_latest_teams,
        )
        players = _attach_observed_context_exposure(
            players,
            context_exposure=context_exposure,
            season=season,
        )
        _replace_current_season_display_ratings(
            historical_rankings,
            player_rating_histories,
            season=season,
            players=players,
        )
        players["rating_history"] = players["player_id"].map(player_rating_histories).map(
            lambda history: history or []
        )
        players["league_leader_history"] = players["player_id"].map(
            player_league_leader_histories
        ).map(lambda history: history or [])
        players["rookie_season"] = players["player_id"].map(rookie_seasons)
        _assign_draft_class_year(players)
        players["age"] = players["player_id"].map(
            lambda player_id: (
                player_rating_histories.get(int(player_id), [{}])[-1].get("age")
            )
        )
        rankings_path = lineup_rankings_path(MODEL_ARTIFACT, run_id, season)
        observed_lineups = (
            pd.read_parquet(rankings_path) if rankings_path.is_file() else pd.DataFrame()
        )
        return cls(
            season=season,
            run_id=run_id,
            coefficients=coefficients,
            profiles=realized_profiles,
            players=players,
            context_model=context_model,
            response_cache=response_cache,
            context_alpha=float(metadata["context_alpha"]),
            schedule_controls=schedule_controls,
            profile_padding_contract=profile_padding_contract,
            use_last_observed_profile=use_last_observed_profile,
            lineup_rankings_root=DEFAULT_LINEUP_RANKINGS_CACHE_DIR / MODEL_ARTIFACT / run_id,
            player_rating_histories=player_rating_histories,
            player_league_leader_histories=player_league_leader_histories,
            player_team_splits=player_team_splits,
            player_latest_teams=player_latest_teams,
            historical_rankings=historical_rankings,
            preseason_rankings=preseason_rankings,
            observed_lineups=observed_lineups,
            historical_coefficients=historical_coefficients,
            seasonal_ratings=seasonal_ratings,
            season_context_models=models,
            player_season_panel=player_season_panel,
            exposure_cohort=exposure_cohort,
            historical_profiles=historical_profiles,
            historical_realized_profiles=historical_realized_profiles,
            season_states={
                season: SeasonLineupState(
                    season=season,
                    coefficients=coefficients,
                    profiles=realized_profiles,
                    players=players,
                )
            },
            response_caches={season: response_cache},
        )

    def available_lab_seasons(self) -> list[str]:
        """Return seasons that have completed player and contextual states."""

        return sorted(self.season_context_models, reverse=True) or [self.season]

    def search_players(
        self,
        query: str,
        *,
        season: str | None = None,
        team: str | None = None,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Return a short, deterministic player-name search result list."""

        normalized = _normalize_search_text(query)
        if not normalized:
            return []
        players = self._season_state(season or self.season).players
        names = players["player_name"].map(_normalize_search_text)
        matches = players.loc[names.str.contains(normalized, regex=False)]
        if team:
            matches = matches.loc[matches["team"].eq(team)]
        return _records(matches.head(max(1, min(limit, 25))))

    def teams(self, *, season: str | None = None) -> list[str]:
        """Return the current player-pool teams for one completed season."""

        players = self._season_state(season or self.season).players
        return sorted(players["team"].dropna().astype(str).unique().tolist())

    def players_by_id(
        self, player_ids: list[int], *, season: str | None = None
    ) -> list[dict[str, Any]]:
        """Return an ordered five-player unit from one completed season state."""

        selected_season = season or self.season
        if not player_ids:
            return []
        players = self._season_state(selected_season).players.set_index("player_id")
        missing = sorted(set(player_ids) - set(players.index.astype(int)))
        if missing:
            raise LineupEvaluationError(
                f"Players are unavailable for {selected_season}: {', '.join(map(str, missing))}"
            )
        return _records(players.loc[player_ids].reset_index())

    def player(self, player_id: int) -> dict[str, Any]:
        """Return a career profile for a player present in any completed fit."""

        preseason_row = (
            self.preseason_rankings.loc[
                self.preseason_rankings["player_id"].eq(player_id)
            ]
            if "player_id" in self.preseason_rankings
            else pd.DataFrame()
        )
        row = self.players.loc[self.players["player_id"].eq(player_id)]
        if not row.empty:
            player = _records(row)[0]
            if not preseason_row.empty:
                preview = _records(preseason_row)[0]
                for column in (
                    "team",
                    "position",
                    "draft_year",
                    "draft_round",
                    "draft_number",
                    "is_undrafted",
                    "draft_class_year",
                    "age",
                    "rapm",
                    "prior_rating",
                    "season_update",
                    "additive_profile_adjustment",
                    "additive_profile_breakdown",
                    "rookie_season",
                    "profile_source",
                ):
                    if column == "rookie_season" and preview[column] is None:
                        continue
                    player[column] = preview[column]
                player["rating_season"] = PRESEASON_PREVIEW_SEASON
            player.setdefault("draft_class_year", None)
            player.setdefault("rating_season", self.season)
            player["league_leader_history"] = self.player_league_leader_histories.get(
                player_id, player.get("league_leader_history", [])
            )
            return player

        history = self.player_rating_histories.get(player_id, [])
        historical_rows = (
            self.historical_rankings.loc[
                self.historical_rankings["player_id"].eq(player_id)
            ]
            if "player_id" in self.historical_rankings
            else pd.DataFrame()
        )
        if historical_rows.empty:
            if preseason_row.empty:
                raise LineupEvaluationError(
                    f"Player {player_id} is not available in published fits"
                )
            preview = _records(preseason_row)[0]
            return {
                **preview,
                "rating_season": PRESEASON_PREVIEW_SEASON,
                "profile_imputed": None,
                "profile_replacement_weight": None,
                "three_pa_per_100": None,
                "three_pm_per_100": None,
                "assists_per_100": None,
                "turnovers_per_100": None,
                "usage_per_100": None,
                "steals_per_100": None,
                "blocks_per_100": None,
                "offensive_rebound_pct": None,
                "rating_history": [],
                "league_leader_history": [],
            }
        latest = historical_rows.sort_values("season", ascending=False, kind="stable").iloc[0]
        latest_history = history[-1] if history else {}
        return {
            "player_id": int(latest["player_id"]),
            "player_name": str(latest["player_name"]),
            "team": str(latest["team"]),
            "position": str(latest["position"]),
            "draft_year": _optional_int(latest.get("draft_year")),
            "draft_round": _optional_int(latest.get("draft_round")),
            "draft_number": _optional_int(latest.get("draft_number")),
            "is_undrafted": _optional_bool(latest.get("is_undrafted")),
            "draft_class_year": _optional_int(latest.get("draft_class_year")),
            "age": latest_history.get("age"),
            "rapm": float(latest["rapm"]),
            "possessions": float(latest["possessions"]),
            "games": int(latest["games"]),
            "rating_season": str(latest["season"]),
            "profile_source": "career_history",
            "profile_imputed": None,
            "profile_replacement_weight": None,
            "three_pa_per_100": None,
            "three_pm_per_100": None,
            "assists_per_100": None,
            "turnovers_per_100": None,
            "usage_per_100": None,
            "steals_per_100": None,
            "blocks_per_100": None,
            "offensive_rebound_pct": None,
            "rookie_season": history[0]["season"] if history else None,
            "rating_history": history,
            "league_leader_history": self.player_league_leader_histories.get(player_id, []),
        }

    def available_ranking_seasons(self) -> list[str]:
        """Return completed-fit ranking seasons, newest first."""

        seasons = set()
        if not self.historical_rankings.empty:
            seasons.update(self.historical_rankings["season"].astype(str).unique())
        else:
            seasons.add(self.season)
        if not self.preseason_rankings.empty:
            seasons.update(self.preseason_rankings["season"].astype(str).unique())
        return sorted(seasons, reverse=True)

    def rankings(self, season: str | None = None) -> list[dict[str, Any]]:
        """Return one selected season's completed-fit coefficients in rank order."""

        selected_season = season or self.season
        if (
            not self.preseason_rankings.empty
            and self.preseason_rankings["season"].eq(selected_season).any()
        ):
            rows = self.preseason_rankings.loc[
                self.preseason_rankings["season"].eq(selected_season)
            ]
        elif self.historical_rankings.empty:
            if selected_season != self.season:
                raise LineupEvaluationError(
                    f"Completed-fit rankings are unavailable for {selected_season}"
                )
            rows = self.players
        else:
            rows = self.historical_rankings.loc[
                self.historical_rankings["season"].eq(selected_season)
            ]
            if rows.empty:
                raise LineupEvaluationError(
                    f"Completed-fit rankings are unavailable for {selected_season}"
                )
        rows = rows.copy()
        for column in (
            "prior_rating",
            "season_update",
            "additive_profile_adjustment",
            "observed_context_exposure",
            "draft_class_year",
        ):
            if column not in rows:
                rows[column] = np.nan
        ranked = rows.sort_values(
            ["rapm", "player_name", "player_id"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
        return _records(ranked)

    def lineups(
        self,
        *,
        season: str | None = None,
        minimum_possessions: float = 500.0,
        player_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Return observed target-season five-man units for lineup-context ranking."""

        if minimum_possessions < 0:
            raise LineupEvaluationError("Minimum possessions cannot be negative")
        selected_season = season or self.season
        if selected_season == self.season:
            source = self.observed_lineups
        elif self.lineup_rankings_root is not None:
            path = self.lineup_rankings_root / f"{selected_season}.parquet"
            source = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
        else:
            source = pd.DataFrame()
        if source.empty:
            raise LineupEvaluationError(
                f"Observed lineup rankings are not materialized for {selected_season}"
            )
        rows = source.loc[source["possessions"].ge(float(minimum_possessions))].copy()
        selected = player_ids or set()
        if selected:
            rows = rows.loc[
                rows["player_ids"].map(
                    lambda lineup: selected.issubset({int(player_id) for player_id in lineup})
                )
            ]
        ranked = rows.sort_values(
            ["context_edge", "possessions", "team", "lineup_label"],
            ascending=[False, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
        for column in ("player_ids", "player_names"):
            ranked[column] = ranked[column].map(
                lambda values: values.tolist()
                if isinstance(values, np.ndarray)
                else list(values)
            )
        return _records(ranked)

    def available_lineup_seasons(self) -> list[str]:
        """Return materialized completed-fit lineup-ranking seasons, newest first."""

        seasons = {self.season} if not self.observed_lineups.empty else set()
        if self.lineup_rankings_root is not None and self.lineup_rankings_root.is_dir():
            seasons.update(path.stem for path in self.lineup_rankings_root.glob("*.parquet"))
        return sorted(seasons, reverse=True)

    def default_opponent(
        self,
        *,
        season: str | None = None,
        team: str | None = None,
        excluded_player_ids: set[int] | None = None,
        count: int = 5,
    ) -> list[dict[str, Any]]:
        """Sample high-possession players, optionally from one team."""

        if count < 1 or count > 5:
            raise LineupEvaluationError("Random lineup count must be between one and five")
        excluded = excluded_player_ids or set()
        players = self._season_state(season or self.season).players
        eligible = players.loc[~players["player_id"].isin(excluded)].copy()
        if team:
            eligible = eligible.loc[eligible["team"].eq(team)]
            high_exposure_count = max(count, int(np.ceil(len(eligible) * 0.5)))
            eligible = eligible.nlargest(high_exposure_count, "possessions", keep="all")
        else:
            eligible = eligible.loc[
                eligible["possessions"].ge(eligible["possessions"].quantile(0.75))
            ]
        if len(eligible) < count:
            scope = f" for {team}" if team else ""
            raise LineupEvaluationError(
                f"The loaded player catalog has too few high-possession players{scope}"
            )
        sample = eligible.sample(
            n=count,
            replace=False,
            random_state=random.SystemRandom().randrange(2**32),
        )
        return _records(sample.sort_values("player_name", kind="stable"))

    def evaluate(
        self,
        unit_player_ids: list[int],
        opponent_player_ids: list[int],
        *,
        unit_season: str | None = None,
        opponent_season: str | None = None,
        environment: str = "unit",
        court: str = "neutral",
        unit_back_to_back: bool = False,
        opponent_back_to_back: bool = False,
        include_response_curves: bool = False,
        response_curve_feature_id: str | None = None,
        response_curve_kind: str | None = None,
    ) -> dict[str, Any]:
        """Score two season-scoped units in a selected historical environment."""

        selected_unit_season = unit_season or self.season
        selected_opponent_season = opponent_season or self.season
        if environment not in {"unit", "opponent", "neutral"}:
            raise LineupEvaluationError("Environment must be unit, neutral, or opponent")
        if court not in {"neutral", "unit_home", "opponent_home"}:
            raise LineupEvaluationError("Court must be neutral, unit_home, or opponent_home")
        unit_state = self._season_state(selected_unit_season)
        opponent_state = self._season_state(selected_opponent_season)
        if environment == "unit":
            result = self._evaluate_in_environment(
                unit_player_ids,
                opponent_player_ids,
                unit_state=unit_state,
                opponent_state=opponent_state,
                environment_season=selected_unit_season,
                include_response_curves=include_response_curves,
                response_curve_feature_id=response_curve_feature_id,
                response_curve_kind=response_curve_kind,
            )
            environment_seasons = [selected_unit_season]
        elif environment == "opponent":
            result = self._evaluate_in_environment(
                unit_player_ids,
                opponent_player_ids,
                unit_state=unit_state,
                opponent_state=opponent_state,
                environment_season=selected_opponent_season,
                include_response_curves=include_response_curves,
                response_curve_feature_id=response_curve_feature_id,
                response_curve_kind=response_curve_kind,
            )
            environment_seasons = [selected_opponent_season]
        else:
            unit_environment = self._evaluate_in_environment(
                unit_player_ids,
                opponent_player_ids,
                unit_state=unit_state,
                opponent_state=opponent_state,
                environment_season=selected_unit_season,
                include_response_curves=include_response_curves,
                response_curve_feature_id=response_curve_feature_id,
                response_curve_kind=response_curve_kind,
            )
            opponent_environment = self._evaluate_in_environment(
                unit_player_ids,
                opponent_player_ids,
                unit_state=unit_state,
                opponent_state=opponent_state,
                environment_season=selected_opponent_season,
                include_response_curves=include_response_curves,
                response_curve_feature_id=response_curve_feature_id,
                response_curve_kind=response_curve_kind,
            )
            result = _mean_evaluation(unit_environment, opponent_environment)
            environment_seasons = [selected_unit_season, selected_opponent_season]
        result.update(
            {
                "unit_season": selected_unit_season,
                "opponent_season": selected_opponent_season,
                "environment": environment,
                "environment_seasons": environment_seasons,
            }
        )
        return _apply_schedule_scenario(
            result,
            controls=self.schedule_controls,
            court=court,
            unit_back_to_back=unit_back_to_back,
            opponent_back_to_back=opponent_back_to_back,
        )

    def _evaluate_in_environment(
        self,
        unit_player_ids: list[int],
        opponent_player_ids: list[int],
        *,
        unit_state: SeasonLineupState,
        opponent_state: SeasonLineupState,
        environment_season: str,
        include_response_curves: bool,
        response_curve_feature_id: str | None,
        response_curve_kind: str | None,
    ) -> dict[str, Any]:
        """Apply one season's contextual surface to two season-scoped player pools."""

        _validate_lineup("unit", unit_player_ids)
        _validate_lineup("opponent", opponent_player_ids)
        overlap = set(unit_player_ids) & set(opponent_player_ids)
        if overlap:
            raise LineupEvaluationError("A player cannot appear on both sides of a matchup")
        unit_available = set(unit_state.players["player_id"].astype(int))
        opponent_available = set(opponent_state.players["player_id"].astype(int))
        missing_unit = sorted(set(unit_player_ids) - unit_available)
        missing_opponent = sorted(set(opponent_player_ids) - opponent_available)
        if missing_unit:
            raise LineupEvaluationError(
                f"Players are unavailable in your unit's {unit_state.season} pool: {missing_unit}"
            )
        if missing_opponent:
            raise LineupEvaluationError(
                "Players are unavailable in the opponent's "
                f"{opponent_state.season} pool: {missing_opponent}"
            )

        unit_coefficient_map = dict(
            zip(
                unit_state.coefficients["player_id"].astype(int),
                unit_state.coefficients["rapm"].astype(float),
                strict=True,
            )
        )
        opponent_coefficient_map = dict(
            zip(
                opponent_state.coefficients["player_id"].astype(int),
                opponent_state.coefficients["rapm"].astype(float),
                strict=True,
            )
        )
        context_model = self._context_model(environment_season)
        raw_unit_features = lineup_side_context_features(
            [unit_player_ids],
            unit_state.profiles,
            feature_set=context_model.feature_set,
        )
        raw_opponent_features = lineup_side_context_features(
            [opponent_player_ids],
            opponent_state.profiles,
            feature_set=context_model.feature_set,
        )
        unit_features, opponent_features, features = _model_feature_inputs(
            context_model,
            raw_unit_features,
            raw_opponent_features,
        )
        if _is_compiled_linear_x3(context_model):
            return self._evaluate_compiled_linear_x3(
                unit_player_ids,
                opponent_player_ids,
                unit_state=unit_state,
                opponent_state=opponent_state,
                context_model=context_model,
                environment_season=environment_season,
                unit_features=unit_features,
                opponent_features=opponent_features,
                features=features,
            )
        components = context_model.decompose_side_pairs(
            unit_features, opponent_features
        ).iloc[0]
        contextual_adjustment = float(components["total_context_net_rating"])
        portable_composition_margin = float(
            components["home_portable_context_net_rating"]
            - components["away_portable_context_net_rating"]
        )
        matchup_adjustment = float(components["matchup_context_net_rating"])
        unit_composition_rating = float(components["home_portable_context_net_rating"])
        opponent_composition_rating = float(components["away_portable_context_net_rating"])
        total_contributions = _antisymmetric_feature_contributions(context_model, features)[0]
        unit_composition_contributions = _side_feature_contributions(
            context_model, unit_features
        )[0]
        opponent_composition_contributions = _side_feature_contributions(
            context_model, opponent_features
        )[0]
        composition_contributions = (
            unit_composition_contributions - opponent_composition_contributions
        )
        matchup_contributions = total_contributions - composition_contributions
        additive_unit = float(sum(unit_coefficient_map[player_id] for player_id in unit_player_ids))
        additive_opponent = float(
            sum(opponent_coefficient_map[player_id] for player_id in opponent_player_ids)
        )
        additive_margin = additive_unit - additive_opponent
        feature_rows = _feature_rows(
            features, total_contributions, feature_set=context_model.feature_set
        )
        composition_feature_rows = _feature_rows(
            features, composition_contributions, feature_set=context_model.feature_set
        )
        matchup_feature_rows = _feature_rows(
            features, matchup_contributions, feature_set=context_model.feature_set
        )
        composition_feature_ids = {row["id"] for row in composition_feature_rows}
        matchup_feature_ids = {row["id"] for row in matchup_feature_rows}
        if response_curve_feature_id is not None:
            if response_curve_kind == "composition":
                composition_feature_ids = {response_curve_feature_id}
                matchup_feature_ids = set()
            elif response_curve_kind == "matchup":
                composition_feature_ids = set()
                matchup_feature_ids = {response_curve_feature_id}
            else:
                raise LineupEvaluationError(
                    "A response curve kind is required for a selected feature"
                )
        composition_response_curves = _composition_response_curves(
            context_model,
            self._response_cache(environment_season),
            unit_features,
            opponent_features,
            unit_composition_contributions,
            opponent_composition_contributions,
            feature_ids=composition_feature_ids,
        ) if include_response_curves else []
        matchup_response_curves = _matchup_response_curves(
            context_model,
            self._response_cache(environment_season),
            unit_features,
            opponent_features,
            opponent_composition_contributions,
            feature_ids=matchup_feature_ids,
        ) if include_response_curves else []
        predicted_net_rating = additive_margin + contextual_adjustment
        response = {
            "season": environment_season,
            "run_id": self.run_id,
            "retrospective": True,
            "unit": self._side(unit_player_ids, unit_coefficient_map, unit_state.players),
            "opponent": self._side(
                opponent_player_ids, opponent_coefficient_map, opponent_state.players
            ),
            "additive_margin": additive_margin,
            "contextual_adjustment": contextual_adjustment,
            "unit_composition_rating": unit_composition_rating,
            "opponent_composition_rating": opponent_composition_rating,
            "portable_composition_margin": portable_composition_margin,
            "matchup_adjustment": matchup_adjustment,
            "predicted_net_rating": predicted_net_rating,
            "predicted_win_pct": projected_win_pct(predicted_net_rating),
            "feature_contributions": feature_rows,
            "composition_feature_contributions": composition_feature_rows,
            "matchup_feature_contributions": matchup_feature_rows,
        }
        if include_response_curves:
            response["composition_response_curves"] = composition_response_curves
            response["matchup_response_curves"] = matchup_response_curves
        return response

    def _evaluate_compiled_linear_x3(
        self,
        unit_player_ids: list[int],
        opponent_player_ids: list[int],
        *,
        unit_state: SeasonLineupState,
        opponent_state: SeasonLineupState,
        context_model: MatchupContextualModel,
        environment_season: str,
        unit_features: pd.DataFrame,
        opponent_features: pd.DataFrame,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        """Score NAIL-RAPM with additive profile terms compiled into players."""

        # Player ratings are fixed at their selected source-season NAIL fit.
        # Evaluation Era is reserved for the non-additive lineup surface.
        unit_map = dict(
            zip(
                unit_state.players["player_id"].astype(int),
                unit_state.players["rapm"].astype(float),
                strict=True,
            )
        )
        opponent_map = dict(
            zip(
                opponent_state.players["player_id"].astype(int),
                opponent_state.players["rapm"].astype(float),
                strict=True,
            )
        )
        total_contributions = _antisymmetric_feature_contributions(context_model, features)[0]
        additive_ids = {
            f"home_minus_away_{column}"
            for column in _linear_x3_additive_feature_map(context_model.feature_set)
        }
        feature_ids = contextual_feature_columns(context_model.feature_set)
        additive_context = float(
            sum(
                contribution
                for feature_id, contribution in zip(
                    feature_ids, total_contributions, strict=True
                )
                if feature_id in additive_ids
            )
        )
        total_context = float(
            context_model.predict_side_pairs(unit_features, opponent_features)[0]
        )
        shape_context = total_context - additive_context
        unit_composition_rating, opponent_composition_rating = (
            _compiled_linear_x3_nonadditive_side_scores(
                context_model,
                unit_features,
                opponent_features,
            )
        )
        if not np.isclose(
            unit_composition_rating - opponent_composition_rating,
            shape_context,
            atol=1e-8,
        ):
            raise LineupEvaluationError(
                "NAIL-RAPM non-additive side scores failed reconstruction"
            )
        shape_rows = _feature_rows(
            features,
            total_contributions,
            feature_set=context_model.feature_set,
            include_ids=set(feature_ids) - additive_ids,
            details=_compiled_linear_feature_details(
                context_model,
                unit_features,
                opponent_features,
                unit_player_ids,
                opponent_player_ids,
                unit_state,
                opponent_state,
            ),
        )
        additive_unit = float(sum(unit_map[player_id] for player_id in unit_player_ids))
        additive_opponent = float(
            sum(opponent_map[player_id] for player_id in opponent_player_ids)
        )
        additive_margin = additive_unit - additive_opponent
        predicted_net_rating = additive_margin + shape_context
        return {
            "season": environment_season,
            "run_id": self.run_id,
            "retrospective": True,
            "model_form": "compiled_linear_x3",
            "unit": self._side(unit_player_ids, unit_map, unit_state.players),
            "opponent": self._side(
                opponent_player_ids, opponent_map, opponent_state.players
            ),
            "additive_margin": additive_margin,
            "contextual_adjustment": shape_context,
            "unit_composition_rating": unit_composition_rating,
            "opponent_composition_rating": opponent_composition_rating,
            "portable_composition_margin": shape_context,
            "matchup_adjustment": 0.0,
            "predicted_net_rating": predicted_net_rating,
            "predicted_win_pct": projected_win_pct(predicted_net_rating),
            "feature_contributions": shape_rows,
            "composition_feature_contributions": shape_rows,
            "matchup_feature_contributions": [],
            "composition_response_curves": [],
            "matchup_response_curves": [],
        }

    def _side(
        self,
        player_ids: list[int],
        coefficient_map: dict[int, float],
        players: pd.DataFrame,
    ) -> dict[str, Any]:
        rows = players.set_index("player_id").loc[player_ids].reset_index()
        players = _records(rows)
        for player in players:
            player["rapm"] = coefficient_map[int(player["player_id"])]
        return {"additive_rating": sum(player["rapm"] for player in players), "players": players}

    def _season_state(self, season: str) -> SeasonLineupState:
        """Build and cache a selected historical player pool on first use."""

        cached = self.season_states.get(season)
        if cached is not None:
            return cached
        if season == self.season:
            state = SeasonLineupState(season, self.coefficients, self.profiles, self.players)
            self.season_states[season] = state
            return state
        if season not in self.season_context_models:
            raise LineupEvaluationError(
                f"The completed NAIL-RAPM state is unavailable for {season}"
            )
        coefficients = self.historical_coefficients.loc[
            self.historical_coefficients["season"].eq(season), ["player_id", "rapm"]
        ].copy()
        if coefficients.empty:
            raise LineupEvaluationError(f"Completed player ratings are unavailable for {season}")
        # The public runtime ships this compact all-season cache, not raw stints.
        # Filtering it preserves the historical cold-start contract without an
        # on-demand reconstruction from the large analytical archive.
        exposure_cohort = self.exposure_cohort.loc[
            self.exposure_cohort["season"].astype(str).le(season)
        ].copy()
        if exposure_cohort.empty:
            exposure_cohort = prepare_player_exposure_cohort(
                self.player_season_panel.loc[
                    self.player_season_panel["season"].astype(str).le(season)
                ],
                through_season=season,
            )
        profiles = self.historical_realized_profiles.loc[
            self.historical_realized_profiles.get("season", pd.Series(dtype=str))
            .astype(str)
            .eq(season)
        ].copy()
        if not profiles.empty:
            profiles = profiles.drop(columns="season")
        expected_player_ids = set(coefficients["player_id"].astype(int))
        cached_player_ids = set(profiles.get("player_id", pd.Series(dtype=int)).astype(int))
        cache_is_complete = (
            not profiles.empty
            and cached_player_ids == expected_player_ids
            and not profiles["player_id"].duplicated().any()
        )
        if not cache_is_complete:
            profiles = build_contextual_player_profiles(
                self.player_season_panel,
                target_season=season,
                target_player_ids=expected_player_ids,
                exposure_cohort=exposure_cohort,
                **(
                    {"padding_contract": self.profile_padding_contract}
                    if self.profile_padding_contract is not None
                    else {}
                ),
                use_last_observed_profile=self.use_last_observed_profile,
                profile_timing="realized",
            )
        display_coefficients = _compiled_linear_x3_coefficients(
            coefficients,
            profiles,
            self._context_model(season),
        )
        players = _player_catalog(
            display_coefficients,
            profiles,
            panel_path=Path("data/analytical/player_season_panel/player_seasons.parquet"),
            season=season,
            player_latest_teams=self.player_latest_teams,
        )
        display_coefficients = _compiled_linear_x3_coefficients(
            coefficients,
            profiles,
            self._context_model(season),
            center=_player_rating_center(display_coefficients, players),
        )
        players = _player_catalog(
            display_coefficients,
            profiles,
            panel_path=Path("data/analytical/player_season_panel/player_seasons.parquet"),
            season=season,
            player_latest_teams=self.player_latest_teams,
        )
        season_ratings = self.seasonal_ratings.loc[
            self.seasonal_ratings["season"].eq(season), ["player_id", "age"]
        ]
        players["age"] = players["player_id"].map(
            dict(zip(season_ratings["player_id"], season_ratings["age"], strict=True))
        )
        players["rookie_season"] = players["player_id"].map(
            _rookie_seasons(self.seasonal_ratings)
        )
        _assign_draft_class_year(players)
        players["rating_history"] = players["player_id"].map(
            lambda player_id: [
                point
                for point in self.player_rating_histories.get(int(player_id), [])
                if str(point["season"]) <= season
            ]
        )
        rookie_seasons = _rookie_seasons(self.seasonal_ratings)
        players["rookie_season"] = players["player_id"].map(rookie_seasons)
        state = SeasonLineupState(season, coefficients, profiles, players)
        self.season_states[season] = state
        return state

    def _context_model(self, season: str) -> MatchupContextualModel:
        """Return a completed historical contextual model for one environment."""

        if season == self.season:
            return self.context_model
        try:
            return self.season_context_models[season]
        except KeyError as error:
            raise LineupEvaluationError(
                f"The contextual environment is unavailable for {season}"
            ) from error

    def _response_cache(self, season: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Warm the small per-environment response cache only once per API process."""

        cache = self.response_caches.get(season)
        if cache is None:
            path = response_cache_path(MODEL_ARTIFACT, self.run_id, season)
            cache = joblib.load(path) if path.is_file() else _warm_response_cache(
                self._context_model(season)
            )
            self.response_caches[season] = cache
        return cache


def _mean_evaluation(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    """Average two directional historical-environment evaluations for neutral mode."""

    result = first.copy()
    for key in (
        "additive_margin",
        "contextual_adjustment",
        "unit_composition_rating",
        "opponent_composition_rating",
        "portable_composition_margin",
        "matchup_adjustment",
        "predicted_net_rating",
    ):
        result[key] = (float(first[key]) + float(second[key])) / 2.0
    result["unit"] = _mean_side(first["unit"], second["unit"])
    result["opponent"] = _mean_side(first["opponent"], second["opponent"])
    for key in (
        "feature_contributions",
        "composition_feature_contributions",
        "matchup_feature_contributions",
    ):
        result[key] = _mean_feature_rows(first[key], second[key])
    for key in ("composition_response_curves", "matchup_response_curves"):
        result[key] = _mean_response_curves(first.get(key, []), second.get(key, []))
    return result


def _mean_side(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Average environment-specific compiled player values for neutral mode."""

    result = first.copy()
    result["additive_rating"] = (
        float(first["additive_rating"]) + float(second["additive_rating"])
    ) / 2.0
    second_players = {int(player["player_id"]): player for player in second["players"]}
    players = []
    for player in first["players"]:
        combined = player.copy()
        partner = second_players[int(player["player_id"])]
        combined["rapm"] = (float(player["rapm"]) + float(partner["rapm"])) / 2.0
        players.append(combined)
    result["players"] = players
    return result


def _mean_feature_rows(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    second_by_id = {str(row["id"]): row for row in second}
    rows = []
    for row in first:
        partner = second_by_id[str(row["id"])]
        combined = row.copy()
        combined["contribution"] = (
            float(row["contribution"]) + float(partner["contribution"])
        ) / 2.0
        rows.append(combined)
    return sorted(rows, key=lambda row: abs(float(row["contribution"])), reverse=True)


def _mean_response_curves(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    second_by_id = {str(curve["id"]): curve for curve in second}
    curves = []
    for curve in first:
        partner = second_by_id[str(curve["id"])]
        combined = curve.copy()
        for key in ("unit_contribution", "opponent_contribution"):
            combined[key] = (float(curve[key]) + float(partner[key])) / 2.0
        combined["points"] = [
            {
                "value": float(first_point["value"]),
                "contribution": (
                    float(first_point["contribution"]) + float(second_point["contribution"])
                )
                / 2.0,
            }
            for first_point, second_point in zip(curve["points"], partner["points"], strict=True)
        ]
        curves.append(combined)
    return curves


def _player_catalog(
    coefficients: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    panel_path: Path,
    season: str,
    player_latest_teams: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    panel_columns = [
        "player_id",
        "player_name",
        "primary_team_tricode",
        "listed_position",
        "draft_year",
        "draft_round",
        "draft_number",
        "is_undrafted",
        "rapm_possessions",
        "games",
        "points",
    ]
    available_panel_columns = [column for column in panel_columns if column in panel]
    season_panel = panel.loc[panel["season"].eq(season), available_panel_columns].copy()
    if season_panel["player_id"].duplicated().any():
        season_panel = (
            season_panel.sort_values("rapm_possessions", ascending=False)
            .drop_duplicates("player_id", keep="first")
        )
    profile_columns = [
        "player_id",
        "player_name",
        "profile_source",
        "profile_imputed",
        "profile_replacement_weight",
        "three_pa_per_100",
        "three_pm_per_100",
        "assists_per_100",
        "turnovers_per_100",
        "usage_per_100",
        "steals_per_100",
        "blocks_per_100",
        "offensive_rebound_pct",
    ]
    catalog = coefficients.merge(profiles.loc[:, profile_columns], on="player_id", how="inner")
    catalog = catalog.merge(
        season_panel.drop(columns=["player_name"], errors="ignore"), on="player_id", how="left"
    )
    catalog["team"] = catalog.get(
        "primary_team_tricode", pd.Series(index=catalog.index)
    ).fillna("-")
    latest_teams = player_latest_teams or {}
    if latest_teams:
        catalog["team"] = [
            latest_teams.get((season, int(player_id)), {}).get("team", team)
            for player_id, team in zip(catalog["player_id"], catalog["team"], strict=True)
        ]
    catalog["position"] = catalog.get(
        "listed_position", pd.Series(index=catalog.index)
    ).fillna("-")
    catalog["possessions"] = catalog.get(
        "rapm_possessions", pd.Series(index=catalog.index)
    ).fillna(0.0)
    catalog["games"] = catalog.get("games", pd.Series(index=catalog.index)).fillna(0).astype(int)
    _normalize_draft_metadata(catalog)
    catalog = catalog.sort_values(["player_name", "player_id"], kind="stable").reset_index(
        drop=True
    )
    return catalog.loc[
        :,
        [
            "player_id",
            "player_name",
            "team",
            "position",
            "draft_year",
            "draft_round",
            "draft_number",
            "is_undrafted",
            "rapm",
            "possessions",
            "games",
            "profile_source",
            "profile_imputed",
            "profile_replacement_weight",
            "three_pa_per_100",
            "three_pm_per_100",
            "assists_per_100",
            "turnovers_per_100",
            "usage_per_100",
            "steals_per_100",
            "blocks_per_100",
            "offensive_rebound_pct",
        ],
    ]


def _attach_observed_context_exposure(
    players: pd.DataFrame,
    *,
    context_exposure: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """Attach a current-season descriptive lineup-shape exposure when available."""

    if context_exposure.empty:
        output = players.copy()
        output["observed_context_exposure"] = np.nan
        return output
    exposure = context_exposure.loc[
        context_exposure["season"].eq(season),
        ["player_id", "observed_context_exposure"],
    ]
    return players.merge(exposure, on="player_id", how="left", validate="one_to_one")


def _replace_current_season_display_ratings(
    rankings: pd.DataFrame,
    histories: dict[int, list[dict[str, Any]]],
    *,
    season: str,
    players: pd.DataFrame,
) -> None:
    """Keep the public current-season catalog aligned with compiled player credit."""

    ratings = dict(
        zip(players["player_id"].astype(int), players["rapm"].astype(float), strict=True)
    )
    current = rankings["season"].eq(season)
    rankings.loc[current, "rapm"] = rankings.loc[current, "player_id"].map(ratings).fillna(
        rankings.loc[current, "rapm"]
    )
    for player_id, history in histories.items():
        for point in history:
            if str(point["season"]) == season and player_id in ratings:
                point["rating"] = ratings[player_id]


def build_published_player_ratings(
    ratings: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    models: dict[str, MatchupContextualModel],
    padding_contract: ProfilePaddingContract | None = None,
    use_last_observed_profile: bool = False,
) -> pd.DataFrame:
    """Compile additive NAIL credit into a consistent completed-fit rating history."""

    compiled_seasons: list[pd.DataFrame] = []
    for season, season_ratings in ratings.groupby("season", sort=True):
        season = str(season)
        context_model = models.get(season)
        if context_model is None:
            # The initial archive season initializes the recursive state. It has
            # observed RAPM, but no prior-season context model to compile.
            initial = season_ratings.copy()
            initial["additive_profile_adjustment"] = 0.0
            compiled_seasons.append(initial)
            continue
        player_ids = set(season_ratings["player_id"].astype(int))
        exposure_cohort = prepare_player_exposure_cohort(
            panel.loc[panel["season"].astype(str).le(season)],
            through_season=season,
        )
        profiles = build_contextual_player_profiles(
            panel,
            target_season=season,
            target_player_ids=player_ids,
            exposure_cohort=exposure_cohort,
            **(
                {"padding_contract": padding_contract}
                if padding_contract is not None
                else {}
            ),
            use_last_observed_profile=use_last_observed_profile,
        )
        base = season_ratings.loc[:, ["player_id", "rapm"]].copy()
        uncentered = _compiled_linear_x3_coefficients(base, profiles, context_model)
        season_panel = panel.loc[
            panel["season"].astype(str).eq(season), ["player_id", "rapm_possessions"]
        ].copy()
        if season_panel["player_id"].duplicated().any():
            season_panel = (
                season_panel.sort_values("rapm_possessions", ascending=False, kind="stable")
                .drop_duplicates("player_id", keep="first")
            )
        display_center = _player_rating_center(
            uncentered,
            season_panel.rename(columns={"rapm_possessions": "possessions"}),
        )
        compiled = _compiled_linear_x3_coefficients(
            base,
            profiles,
            context_model,
            center=display_center,
        )
        output = season_ratings.copy()
        compiled_by_player = compiled.set_index("player_id")["rapm"]
        output["additive_profile_adjustment"] = (
            output["player_id"].map(compiled_by_player).astype(float) - output["rapm"].astype(float)
        )
        output["rapm"] = output["player_id"].map(compiled_by_player).astype(float)
        compiled_seasons.append(_fold_imputed_profile_into_prior(output, profiles))
    if not compiled_seasons:
        raise LineupEvaluationError("No completed NAIL-RAPM seasonal ratings were compiled")
    return pd.concat(compiled_seasons, ignore_index=True)


def _fold_imputed_profile_into_prior(
    ratings: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Present imputed cold-start profiles as part of the player prior."""

    output = ratings.copy()
    imputed_ids = set(
        profiles.loc[
            profiles["profile_imputed"].astype(bool), "player_id"
        ].astype(int)
    )
    mask = output["player_id"].astype(int).isin(imputed_ids)
    output.loc[mask, "prior_rapm"] = (
        output.loc[mask, "prior_rapm"].astype(float)
        + output.loc[mask, "additive_profile_adjustment"].astype(float)
    )
    output.loc[mask, "additive_profile_adjustment"] = np.nan
    return output


def _historical_ranking_catalog(
    ratings: pd.DataFrame,
    *,
    panel_path: Path,
    context_exposure: pd.DataFrame | None = None,
    rookie_seasons: dict[int, str] | None = None,
    player_latest_teams: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Attach season-specific display metadata to completed player ratings."""

    required = {"season", "player_id", "player_name", "rapm"}
    missing = required - set(ratings.columns)
    if missing:
        raise LineupEvaluationError(
            "Player-season rating artifact is missing required columns: "
            + ", ".join(sorted(missing))
        )

    panel = pd.read_parquet(panel_path)
    panel_columns = [
        "season",
        "player_id",
        "primary_team_tricode",
        "listed_position",
        "draft_year",
        "draft_round",
        "draft_number",
        "is_undrafted",
        "rapm_possessions",
        "games",
    ]
    available_columns = [column for column in panel_columns if column in panel]
    metadata = panel.loc[:, available_columns].copy()
    if {"season", "player_id"} <= set(metadata.columns):
        sort_column = "rapm_possessions" if "rapm_possessions" in metadata else "player_id"
        metadata = metadata.sort_values(
            ["season", "player_id", sort_column],
            ascending=[True, True, False],
            kind="stable",
        ).drop_duplicates(["season", "player_id"], keep="first")
    else:
        metadata = pd.DataFrame(columns=["season", "player_id"])

    catalog = ratings.loc[:, ["season", "player_id", "player_name", "rapm"]].copy()
    component_columns = {
        "prior_rating": "prior_rapm",
        "season_update": "rapm_adjustment_from_prior",
        "additive_profile_adjustment": "additive_profile_adjustment",
    }
    for public_column, source_column in component_columns.items():
        catalog[public_column] = (
            ratings[source_column] if source_column in ratings else np.nan
        )
    catalog = catalog.merge(
        metadata,
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    if context_exposure is not None and not context_exposure.empty:
        catalog = catalog.merge(
            context_exposure.loc[:, ["season", "player_id", "observed_context_exposure"]],
            on=["season", "player_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        catalog["observed_context_exposure"] = np.nan
    catalog["team"] = catalog.get(
        "primary_team_tricode", pd.Series(index=catalog.index, dtype="object")
    ).fillna("-")
    latest_teams = player_latest_teams or {}
    if latest_teams:
        catalog["team"] = [
            latest_teams.get((str(season), int(player_id)), {}).get("team", team)
            for season, player_id, team in zip(
                catalog["season"], catalog["player_id"], catalog["team"], strict=True
            )
        ]
    catalog["position"] = catalog.get(
        "listed_position", pd.Series(index=catalog.index, dtype="object")
    ).fillna("-")
    catalog["possessions"] = catalog.get(
        "rapm_possessions", pd.Series(index=catalog.index, dtype="float64")
    ).fillna(0.0)
    catalog["games"] = catalog.get(
        "games", pd.Series(index=catalog.index, dtype="float64")
    ).fillna(0).astype(int)
    _normalize_draft_metadata(catalog)
    catalog["rookie_season"] = catalog["player_id"].map(rookie_seasons or {})
    _assign_draft_class_year(catalog)
    return catalog.loc[
        :,
        [
            "season",
            "player_id",
            "player_name",
            "team",
            "position",
            "draft_year",
            "draft_round",
            "draft_number",
            "is_undrafted",
            "draft_class_year",
            "rapm",
            "prior_rating",
            "season_update",
            "additive_profile_adjustment",
            "observed_context_exposure",
            "possessions",
            "games",
        ],
    ]


def _preseason_ranking_catalog(
    completed_rankings: pd.DataFrame,
    *,
    roster_path: Path,
    draft_rankings_path: Path,
    completed_season: str,
    preview_season: str,
) -> pd.DataFrame:
    """Combine a roster snapshot with published returner and cold-start values."""

    columns = [
        "season",
        "player_id",
        "player_name",
        "team",
        "position",
        "draft_year",
        "draft_round",
        "draft_number",
        "is_undrafted",
        "draft_class_year",
        "rapm",
        "prior_rating",
        "season_update",
        "additive_profile_adjustment",
        "observed_context_exposure",
        "possessions",
        "games",
        "age",
        "rookie_season",
        "profile_source",
    ]
    if not roster_path.is_file():
        return pd.DataFrame(columns=columns)

    roster = pd.read_parquet(roster_path).copy()
    required_roster_columns = {
        "player_id",
        "player_name",
        "team_abbreviation",
        "listed_position",
        "age",
        "experience",
    }
    missing = required_roster_columns - set(roster.columns)
    if missing:
        raise LineupEvaluationError(
            "Preseason roster is missing required columns: " + ", ".join(sorted(missing))
        )
    roster = roster.sort_values(["player_id", "team_abbreviation"], kind="stable").drop_duplicates(
        "player_id", keep="first"
    )

    completed = completed_rankings.loc[
        completed_rankings["season"].eq(completed_season)
    ].copy()
    completed = completed.sort_values("rapm", ascending=False, kind="stable").drop_duplicates(
        "player_id", keep="first"
    )
    returners = completed.set_index("player_id") if not completed.empty else pd.DataFrame()

    draft_rankings = pd.DataFrame()
    if draft_rankings_path.is_file():
        draft_rankings = pd.read_parquet(draft_rankings_path).copy()
        required_draft_columns = {"player_id", "draft_number", "cold_start_rapm_prior"}
        missing = required_draft_columns - set(draft_rankings.columns)
        if missing:
            raise LineupEvaluationError(
                "Forward draft cold-start ranking is missing required columns: "
                + ", ".join(sorted(missing))
            )
        draft_rankings["player_id"] = pd.to_numeric(
            draft_rankings["player_id"], errors="coerce"
        )
        draft_rankings = draft_rankings.loc[draft_rankings["player_id"].notna()].copy()
        draft_rankings["player_id"] = draft_rankings["player_id"].astype(int)
        draft_rankings = draft_rankings.drop_duplicates("player_id", keep="first")
        draft_rankings = draft_rankings.set_index("player_id")
    replacement_rating = (
        float(draft_rankings["replacement_rapm"].iloc[0])
        if not draft_rankings.empty and "replacement_rapm" in draft_rankings
        else np.nan
    )

    records: list[dict[str, Any]] = []
    for roster_row in roster.itertuples(index=False):
        player_id = int(roster_row.player_id)
        returning = returners.loc[player_id] if player_id in returners.index else None
        drafted = draft_rankings.loc[player_id] if player_id in draft_rankings.index else None
        experience = pd.to_numeric(pd.Series([roster_row.experience]), errors="coerce").iloc[0]
        is_new = pd.isna(experience) or float(experience) == 0.0
        is_undrafted = (
            bool(returning["is_undrafted"])
            if returning is not None and pd.notna(returning["is_undrafted"])
            else drafted is None
        )
        draft_year = (
            _optional_int(returning["draft_year"])
            if returning is not None
            else int(preview_season[:4]) if drafted is not None else None
        )
        draft_round = (
            _optional_int(returning["draft_round"])
            if returning is not None
            else _optional_int(drafted.get("draft_round")) if drafted is not None else None
        )
        draft_number = (
            _optional_int(returning["draft_number"])
            if returning is not None
            else _optional_int(drafted.get("draft_number")) if drafted is not None else None
        )
        rookie_season = (
            str(returning["rookie_season"])
            if returning is not None and pd.notna(returning.get("rookie_season"))
            else preview_season if is_new else None
        )
        if returning is not None:
            rating = float(returning["rapm"])
            source = "carried_forward_completed_fit"
        elif drafted is not None:
            rating = float(drafted["cold_start_rapm_prior"])
            source = "draft_cold_start_prior"
        else:
            rating = replacement_rating
            source = "replacement_cold_start_prior"
        draft_class_year = (
            draft_year
            if draft_year is not None
            else int(preview_season[:4])
            if is_new
            else None
        )
        records.append(
            {
                "season": preview_season,
                "player_id": player_id,
                "player_name": str(roster_row.player_name),
                "team": str(roster_row.team_abbreviation),
                "position": str(roster_row.listed_position),
                "draft_year": draft_year,
                "draft_round": draft_round,
                "draft_number": draft_number,
                "is_undrafted": is_undrafted,
                "draft_class_year": draft_class_year,
                "rapm": rating,
                "prior_rating": np.nan,
                "season_update": np.nan,
                "additive_profile_adjustment": np.nan,
                "observed_context_exposure": np.nan,
                "possessions": 0.0,
                "games": 0,
                "age": _optional_float(roster_row.age),
                "rookie_season": rookie_season,
                "profile_source": source,
            }
        )
    preview = pd.DataFrame(records, columns=columns)
    _normalize_draft_metadata(preview)
    _assign_draft_class_year(preview)
    return preview


def _player_rating_histories(
    ratings: pd.DataFrame,
    *,
    panel_path: Path,
    context_exposure: pd.DataFrame | None = None,
    player_team_splits: dict[tuple[str, int], list[dict[str, Any]]] | None = None,
    player_latest_teams: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Package completed-fit seasonal ratings into compact browser sparklines."""

    required = {"season", "player_id", "player_name", "rapm", "age"}
    missing = required - set(ratings.columns)
    if missing:
        raise LineupEvaluationError(
            "Player-season rating artifact is missing required columns: "
            + ", ".join(sorted(missing))
        )
    seasonal_bounds = (
        ratings.groupby("season", as_index=False)["rapm"]
        .agg(season_min_rating="min", season_max_rating="max")
    )
    seasonal_leaders = (
        ratings.loc[:, ["season", "player_id", "player_name", "rapm"]]
        .sort_values(
            ["season", "rapm", "player_name", "player_id"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        .drop_duplicates("season", keep="first")
        .loc[:, ["season", "player_id", "player_name"]]
        .rename(
            columns={
                "player_id": "season_max_player_id",
                "player_name": "season_max_player_name",
            }
        )
    )
    panel = pd.read_parquet(panel_path)
    team_columns = [
        "season",
        "player_id",
        "primary_team_id",
        "primary_team_tricode",
        "rapm_possessions",
        "games",
        "games_started",
    ]
    available_team_columns = [column for column in team_columns if column in panel]
    if {"season", "player_id", "primary_team_tricode"} <= set(available_team_columns):
        teams = panel.loc[:, available_team_columns].copy()
        for column in ["primary_team_id", "rapm_possessions", "games", "games_started"]:
            if column not in teams:
                teams[column] = pd.NA
        teams = teams.sort_values(
            "rapm_possessions" if "rapm_possessions" in teams else "primary_team_tricode",
            ascending=False,
            kind="stable",
        ).drop_duplicates(["season", "player_id"], keep="first")
        teams = teams.loc[
            :,
            [
                "season",
                "player_id",
                "primary_team_id",
                "primary_team_tricode",
                "rapm_possessions",
                "games",
                "games_started",
            ],
        ]
    else:
        teams = pd.DataFrame(
            columns=[
                "season",
                "player_id",
                "primary_team_id",
                "primary_team_tricode",
                "rapm_possessions",
                "games",
                "games_started",
            ]
        )
    enriched = ratings.sort_values(
        ["season", "rapm", "player_name", "player_id"],
        ascending=[True, False, True, True],
        kind="stable",
    ).copy()
    enriched["nail_rank"] = enriched.groupby("season", sort=False).cumcount() + 1
    for column in ("prior_rapm", "rapm_adjustment_from_prior", "additive_profile_adjustment"):
        if column not in enriched:
            enriched[column] = np.nan
    ordered = enriched.loc[
        :,
        [
            "season",
            "player_id",
                "rapm",
                "nail_rank",
                "prior_rapm",
                "rapm_adjustment_from_prior",
                "additive_profile_adjustment",
                "age",
        ],
    ].merge(
        seasonal_bounds.merge(seasonal_leaders, on="season", how="inner", validate="one_to_one"),
        on="season",
        how="left",
        validate="many_to_one",
    ).merge(
        teams,
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    if context_exposure is not None and not context_exposure.empty:
        ordered = ordered.merge(
            context_exposure.loc[:, ["season", "player_id", "observed_context_exposure"]],
            on=["season", "player_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        ordered["observed_context_exposure"] = np.nan
    ordered = ordered.sort_values(
        ["player_id", "season"], kind="stable"
    )
    team_splits = player_team_splits or {}
    latest_teams = player_latest_teams or {}
    return {
        int(player_id): [
            {
                "season": str(row.season),
                "rating": float(row.rapm),
                "nail_rank": int(row.nail_rank),
                "prior_rating": None if pd.isna(row.prior_rapm) else float(row.prior_rapm),
                "season_update": (
                    None
                    if pd.isna(row.rapm_adjustment_from_prior)
                    else float(row.rapm_adjustment_from_prior)
                ),
                "additive_profile_adjustment": (
                    None
                    if pd.isna(row.additive_profile_adjustment)
                    else float(row.additive_profile_adjustment)
                ),
                "observed_context_exposure": (
                    None
                    if pd.isna(row.observed_context_exposure)
                    else float(row.observed_context_exposure)
                ),
                "age": None if pd.isna(row.age) else float(row.age),
                "team_id": latest_teams.get(
                    (str(row.season), int(row.player_id)), {}
                ).get(
                    "team_id",
                    None if pd.isna(row.primary_team_id) else int(row.primary_team_id),
                ),
                "team": latest_teams.get(
                    (str(row.season), int(row.player_id)), {}
                ).get(
                    "team",
                    "-" if pd.isna(row.primary_team_tricode) else str(row.primary_team_tricode),
                ),
                "team_splits": team_splits.get(
                    (str(row.season), int(row.player_id)), []
                ),
                "possessions": (
                    0.0 if pd.isna(row.rapm_possessions) else float(row.rapm_possessions)
                ),
                "games": 0 if pd.isna(row.games) else int(row.games),
                "games_started": 0 if pd.isna(row.games_started) else int(row.games_started),
                "season_min_rating": float(row.season_min_rating),
                "season_max_rating": float(row.season_max_rating),
                "season_max_player_id": int(row.season_max_player_id),
                "season_max_player_name": str(row.season_max_player_name),
            }
            for row in rows.itertuples(index=False)
        ]
        for player_id, rows in ordered.groupby("player_id", sort=False)
    }


def _player_league_leader_histories(
    ratings: pd.DataFrame,
    player_histories: dict[int, list[dict[str, Any]]],
    *,
    active_through_years: dict[int, int] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Return the completed-fit league leader for every season in each player's career span."""

    required = {"season", "player_id", "player_name", "rapm"}
    missing = required - set(ratings.columns)
    if missing:
        raise LineupEvaluationError(
            "Player-season rating artifact is missing required columns: "
            + ", ".join(sorted(missing))
        )
    leaders = (
        ratings.loc[:, ["season", "player_id", "player_name", "rapm"]]
        .sort_values(
            ["season", "rapm", "player_name", "player_id"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        .drop_duplicates("season", keep="first")
        .sort_values("season", kind="stable")
    )
    records = [
        {
            "season": str(row.season),
            "rating": float(row.rapm),
            "player_id": int(row.player_id),
            "player_name": str(row.player_name),
        }
        for row in leaders.itertuples(index=False)
    ]
    latest_completed_year = max(int(record["season"][:4]) for record in records)
    active_through = active_through_years or {}
    histories: dict[int, list[dict[str, Any]]] = {}
    for player_id, history in player_histories.items():
        if not history:
            continue
        start = str(history[0]["season"])
        last_observed_year = int(str(history[-1]["season"])[:4])
        end_year = max(
            last_observed_year,
            min(latest_completed_year, active_through.get(player_id, last_observed_year)),
        )
        histories[player_id] = [
            record
            for record in records
            if start <= record["season"] and int(record["season"][:4]) <= end_year
        ]
    return histories


def _player_active_through_years(panel_path: Path) -> dict[int, int]:
    """Return the final season-start year implied by each player's catalog endpoint."""

    panel = pd.read_parquet(panel_path)
    required = {"player_id", "to_year"}
    missing = required - set(panel.columns)
    if missing:
        return {}
    values = panel.loc[:, ["player_id", "to_year"]].dropna()
    return {
        int(row.player_id): int(row.to_year) - 1
        for row in values.itertuples(index=False)
    }


def build_player_team_splits(
    ratings: pd.DataFrame,
    *,
    analytical_dir: Path | str = Path("data/analytical"),
) -> pd.DataFrame:
    """Aggregate player-team exposure and preserve primary and latest-team identity."""

    required = {"season", "player_id"}
    missing = required - set(ratings.columns)
    if missing:
        raise LineupEvaluationError(
            "Player-season rating artifact is missing required columns: "
            + ", ".join(sorted(missing))
        )
    frames: list[pd.DataFrame] = []
    for season in sorted(ratings["season"].astype(str).unique()):
        player_ids = set(
            ratings.loc[ratings["season"].astype(str).eq(season), "player_id"]
            .astype(int)
            .tolist()
        )
        stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        for side in ("home", "away"):
            side_rows = stints.loc[
                :,
                [
                    "game_id",
                    "game_time_utc",
                    f"{side}_team_id",
                    f"{side}_team_tricode",
                    f"{side}_player_ids",
                    "possessions",
                ],
            ].explode(f"{side}_player_ids", ignore_index=True)
            side_rows = side_rows.rename(
                columns={
                    f"{side}_team_id": "team_id",
                    f"{side}_team_tricode": "team",
                    f"{side}_player_ids": "player_id",
                }
            )
            side_rows["player_id"] = side_rows["player_id"].astype(int)
            frames.append(
                side_rows.loc[side_rows["player_id"].isin(player_ids)].assign(season=season)
            )
    if not frames:
        return pd.DataFrame(
            columns=[
                "season",
                "player_id",
                "team_id",
                "team",
                "possessions",
                "games",
                "last_game_time_utc",
                "is_primary_team",
                "is_latest_team",
            ]
        )
    rows = pd.concat(frames, ignore_index=True)
    splits = (
        rows.groupby(["season", "player_id", "team_id", "team"], as_index=False, sort=False)
        .agg(
            possessions=("possessions", "sum"),
            games=("game_id", "nunique"),
            last_game_time_utc=("game_time_utc", "max"),
        )
        .sort_values(
            ["season", "player_id", "possessions", "team_id"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    splits["is_primary_team"] = ~splits.duplicated(["season", "player_id"])
    latest = (
        splits.sort_values(
            ["season", "player_id", "last_game_time_utc", "possessions", "team_id"],
            ascending=[True, True, False, False, True],
            kind="stable",
        )
        .drop_duplicates(["season", "player_id"], keep="first")
        .loc[:, ["season", "player_id", "team_id"]]
        .assign(is_latest_team=True)
    )
    splits = splits.merge(
        latest,
        on=["season", "player_id", "team_id"],
        how="left",
        validate="many_to_one",
    )
    splits["is_latest_team"] = splits["is_latest_team"].eq(True)
    splits["player_id"] = splits["player_id"].astype("int64")
    splits["team_id"] = splits["team_id"].astype("int64")
    splits["games"] = splits["games"].astype("int64")
    return splits


def _player_team_splits_by_season(
    splits: pd.DataFrame,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Package persisted team splits for the browser's player-season history."""

    required = {"season", "player_id", "team_id", "team", "possessions", "games"}
    missing = required - set(splits.columns)
    if missing:
        raise LineupEvaluationError(
            "Player-team split artifact is missing required columns: "
            + ", ".join(sorted(missing))
        )
    ordered = splits.sort_values(
        ["season", "player_id", "possessions", "team_id"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    if "is_primary_team" not in ordered:
        ordered["is_primary_team"] = ~ordered.duplicated(["season", "player_id"])
    if "is_latest_team" not in ordered:
        ordered["is_latest_team"] = ordered["is_primary_team"]
    return {
        (str(season), int(player_id)): [
            {
                "team_id": int(row.team_id),
                "team": str(row.team),
                "possessions": float(row.possessions),
                "games": int(row.games),
                "is_primary_team": bool(row.is_primary_team),
                "is_latest_team": bool(row.is_latest_team),
            }
            for row in rows.itertuples(index=False)
        ]
        for (season, player_id), rows in ordered.groupby(["season", "player_id"], sort=False)
    }


def _player_latest_teams_by_season(
    splits: pd.DataFrame,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Return the team from each player's final observed game in a season."""

    required = {"season", "player_id", "team_id", "team"}
    missing = required - set(splits.columns)
    if missing:
        raise LineupEvaluationError(
            "Player-team split artifact is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if "is_latest_team" in splits:
        latest = splits.loc[splits["is_latest_team"].astype(bool)].copy()
    elif "last_game_time_utc" in splits:
        latest = (
            splits.sort_values(
                ["season", "player_id", "last_game_time_utc", "possessions", "team_id"],
                ascending=[True, True, False, False, True],
                kind="stable",
            )
            .drop_duplicates(["season", "player_id"], keep="first")
        )
    else:
        # Older caches only preserve exposure order. Treat their primary team as
        # the fallback until the cache is rematerialized from dated stints.
        latest = (
            splits.sort_values(
                ["season", "player_id", "possessions", "team_id"],
                ascending=[True, True, False, True],
                kind="stable",
            )
            .drop_duplicates(["season", "player_id"], keep="first")
        )
    if latest.duplicated(["season", "player_id"]).any():
        raise LineupEvaluationError("Player-team splits identify multiple latest teams")
    return {
        (str(row.season), int(row.player_id)): {
            "team_id": int(row.team_id),
            "team": str(row.team),
        }
        for row in latest.itertuples(index=False)
    }


def _rookie_seasons(ratings: pd.DataFrame) -> dict[int, str]:
    """Return each player's explicit rookie season, falling back to first observed season."""

    required = {"season", "player_id", "is_rookie"}
    missing = required - set(ratings.columns)
    if missing:
        raise LineupEvaluationError(
            "Player-season rating artifact is missing required columns: "
            + ", ".join(sorted(missing))
        )
    ordered = ratings.loc[:, ["season", "player_id", "is_rookie"]].sort_values(
        ["player_id", "season"], kind="stable"
    )
    seasons: dict[int, str] = {}
    for player_id, rows in ordered.groupby("player_id", sort=False):
        rookie_rows = rows.loc[rows["is_rookie"].astype(bool)]
        seasons[int(player_id)] = str(
            (rookie_rows if not rookie_rows.empty else rows).iloc[0]["season"]
        )
    return seasons


def _antisymmetric_feature_contributions(
    context_model: MatchupContextualModel, features: pd.DataFrame
) -> np.ndarray:
    """Return relative feature contributions under the exact antisymmetric total."""

    forward = _feature_contributions(context_model.pipeline, features)
    reverse = _feature_contributions(context_model.pipeline, -features)
    return 0.5 * (forward - reverse)


def _is_compiled_linear_x3(context_model: MatchupContextualModel) -> bool:
    """Return whether this is the exact additive-compilable NAIL-RAPM contract."""

    return (
        context_model.feature_set
        in {
            CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE,
            CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
            CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT,
            CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
        }
        and tuple(context_model.pipeline.named_steps) == ("scale", "ridge")
    )


def _compiled_linear_x3_nonadditive_side_scores(
    context_model: MatchupContextualModel,
    unit_features: pd.DataFrame,
    opponent_features: pd.DataFrame,
) -> tuple[float, float]:
    """Return reference-centered non-additive side scores for compiled NAIL-RAPM."""

    side_columns = side_context_feature_columns(context_model.feature_set)
    relative_columns = contextual_feature_columns(context_model.feature_set)
    additive_ids = {
        f"home_minus_away_{column}"
        for column in _linear_x3_additive_feature_map(context_model.feature_set)
    }
    nonadditive_columns = [
        side_column
        for feature_id, side_column in zip(
            relative_columns, side_columns, strict=True
        )
        if feature_id not in additive_ids
    ]
    if not nonadditive_columns:
        return 0.0, 0.0

    scale = context_model.pipeline.named_steps["scale"]
    ridge = context_model.pipeline.named_steps["ridge"]
    raw_coefficients = pd.Series(
        np.asarray(ridge.coef_, dtype=float) / np.asarray(scale.scale_, dtype=float),
        index=side_columns,
    )
    reference = context_model.reference_features.loc[:, nonadditive_columns]
    weights = np.asarray(context_model.reference_weights, dtype=float)
    weights = weights / weights.sum()
    reference_mean = np.average(reference.to_numpy(dtype=float), axis=0, weights=weights)
    coefficients = raw_coefficients.loc[nonadditive_columns].to_numpy(dtype=float)

    def score(side: pd.DataFrame) -> float:
        values = side.loc[:, nonadditive_columns].to_numpy(dtype=float)[0]
        return float((values - reference_mean) @ coefficients)

    return score(unit_features), score(opponent_features)


def _compiled_linear_x3_coefficients(
    coefficients: pd.DataFrame,
    profiles: pd.DataFrame,
    context_model: MatchupContextualModel,
    *,
    center: float = 0.0,
) -> pd.DataFrame:
    """Add exact linear additive-profile context credit to player coefficients."""

    if not _is_compiled_linear_x3(context_model):
        return coefficients.copy()
    additive_feature_map = _linear_x3_additive_feature_map(context_model.feature_set)
    required = {"player_id", "rapm", *additive_feature_map.values()}
    missing = required - set(coefficients.columns) - set(profiles.columns)
    if missing:
        raise LineupEvaluationError(
            "NAIL-RAPM lacks profile columns: " + ", ".join(sorted(missing))
        )
    scale = context_model.pipeline.named_steps["scale"]
    ridge = context_model.pipeline.named_steps["ridge"]
    raw_coefficients = np.asarray(ridge.coef_, dtype=float) / np.asarray(
        scale.scale_, dtype=float
    )
    side_columns = side_context_feature_columns(context_model.feature_set)
    raw_by_feature = dict(zip(side_columns, raw_coefficients, strict=True))
    adjustment = np.zeros(len(profiles), dtype=float)
    for feature, profile_column in additive_feature_map.items():
        adjustment += raw_by_feature[feature] * profiles[profile_column].to_numpy(dtype=float)
    adjustments = pd.DataFrame(
        {
            "player_id": profiles["player_id"].astype(int),
            "compiled_profile_adjustment": adjustment,
        }
    )
    output = coefficients.merge(adjustments, on="player_id", how="left", validate="one_to_one")
    output["rapm"] = (
        output["rapm"] + output["compiled_profile_adjustment"].fillna(0.0) - center
    )
    return output.drop(columns="compiled_profile_adjustment")


def compiled_linear_x3_additive_profile_breakdown(
    profiles: pd.DataFrame,
    context_model: MatchupContextualModel,
    reference_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Return exact player-versus-reference additive-profile contributions.

    The reference is the possession-weighted player profile in the applicable
    forecast pool.  Keeping this calculation beside the compiled-rating logic
    ensures that the web explanation uses the same raw Ridge coefficients as
    the published rating.
    """

    if not _is_compiled_linear_x3(context_model):
        return pd.DataFrame(
            columns=[
                "player_id",
                "feature",
                "profile_column",
                "player_value",
                "reference_value",
                "coefficient",
                "contribution",
            ]
        )

    additive_feature_map = _linear_x3_additive_feature_map(context_model.feature_set)
    required = {"player_id", *additive_feature_map.values()}
    missing = required - set(profiles.columns)
    if missing:
        raise LineupEvaluationError(
            "NAIL-RAPM lacks profile columns: " + ", ".join(sorted(missing))
        )
    if {"player_id", "possessions"} - set(reference_weights.columns):
        raise LineupEvaluationError(
            "Additive-profile reference requires player possession weights"
        )

    scale = context_model.pipeline.named_steps["scale"]
    ridge = context_model.pipeline.named_steps["ridge"]
    raw_coefficients = np.asarray(ridge.coef_, dtype=float) / np.asarray(
        scale.scale_, dtype=float
    )
    side_columns = side_context_feature_columns(context_model.feature_set)
    raw_by_feature = dict(zip(side_columns, raw_coefficients, strict=True))
    values = profiles.loc[:, ["player_id", *additive_feature_map.values()]].copy()
    weights = values.loc[:, ["player_id"]].merge(
        reference_weights.loc[:, ["player_id", "possessions"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )["possessions"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)

    output: list[pd.DataFrame] = []
    for feature, profile_column in additive_feature_map.items():
        player_values = values[profile_column].to_numpy(dtype=float)
        if float(weights.sum()) > 0.0:
            reference_value = float(np.average(player_values, weights=weights))
        else:
            reference_value = float(np.mean(player_values))
        coefficient = float(raw_by_feature[feature])
        output.append(
            pd.DataFrame(
                {
                    "player_id": values["player_id"].astype(int),
                    "feature": feature,
                    "profile_column": profile_column,
                    "player_value": player_values,
                    "reference_value": reference_value,
                    "coefficient": coefficient,
                    "contribution": coefficient * (player_values - reference_value),
                }
            )
        )
    return pd.concat(output, ignore_index=True)


def _player_rating_center(coefficients: pd.DataFrame, players: pd.DataFrame) -> float:
    """Return the possession-weighted display center for one player pool."""

    values = coefficients.loc[:, ["player_id", "rapm"]].merge(
        players.loc[:, ["player_id", "possessions"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    weights = values["possessions"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    ratings = values["rapm"].to_numpy(dtype=float)
    if float(weights.sum()) > 0.0:
        return float(np.average(ratings, weights=weights))
    return float(np.mean(ratings))


def _feature_contributions(context_model: object, features: pd.DataFrame) -> np.ndarray:
    """Return exact per-original-feature Ridge contributions, excluding intercept."""

    try:
        scale = context_model.named_steps["scale"]
        ridge = context_model.named_steps["ridge"]
    except AttributeError as error:
        raise TypeError(
            "Context attribution requires the published Ridge pipeline"
        ) from error
    columns = list(features.columns)
    if "spline" not in context_model.named_steps:
        scaled = scale.transform(features.loc[:, columns])
        coefficients = np.asarray(ridge.coef_, dtype=float)
        if scaled.shape[1] != len(columns) or len(coefficients) != len(columns):
            raise ValueError("Linear feature layout is incompatible with contextual attribution")
        return scaled * coefficients
    spline = context_model.named_steps["spline"]
    basis = spline.transform(features.loc[:, columns])
    scaled = scale.transform(basis)
    coefficients = np.asarray(ridge.coef_, dtype=float)
    feature_count = len(columns)
    if scaled.shape[1] % feature_count or len(coefficients) != scaled.shape[1]:
        raise ValueError("Spline feature layout is incompatible with contextual attribution")
    basis_count = scaled.shape[1] // feature_count
    return (scaled * coefficients).reshape(len(features), feature_count, basis_count).sum(axis=2)


def _feature_label(column: str) -> str:
    """Return the user-facing label for one fixed contextual feature."""

    try:
        return _FEATURE_LABELS[column]
    except KeyError as error:
        raise ValueError(f"No display label is defined for contextual feature {column}") from error


def _model_feature_inputs(
    context_model: MatchupContextualModel,
    unit_features: pd.DataFrame,
    opponent_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the exact side and relative inputs used by the fitted model."""

    side_columns = side_context_feature_columns(context_model.feature_set)
    relative_columns = contextual_feature_columns(context_model.feature_set)
    unit = unit_features.loc[:, side_columns].copy()
    opponent = opponent_features.loc[:, side_columns].copy()
    if isinstance(context_model, BoundedMatchupContextualModel):
        unit = unit.clip(context_model.side_lower, context_model.side_upper, axis=1)
        opponent = opponent.clip(context_model.side_lower, context_model.side_upper, axis=1)
    relative = pd.DataFrame(
        {
            column: (
                unit[side_columns[index]].to_numpy(dtype=float)
                - opponent[side_columns[index]].to_numpy(dtype=float)
            )
            for index, column in enumerate(relative_columns)
        },
        columns=relative_columns,
    )
    return unit, opponent, _model_relative_features(context_model, relative)


def _model_relative_features(
    context_model: MatchupContextualModel, relative: pd.DataFrame
) -> pd.DataFrame:
    """Apply the model's symmetric relative support, when present."""

    if not isinstance(context_model, BoundedMatchupContextualModel):
        return relative
    return relative.clip(-context_model.relative_cap, context_model.relative_cap, axis=1)


def _side_feature_contributions(
    context_model: MatchupContextualModel, side_features: pd.DataFrame
) -> np.ndarray:
    """Attribute h(U) feature-by-feature against the frozen reference field."""

    side_columns = side_context_feature_columns(context_model.feature_set)
    reference = context_model.reference_features.loc[:, side_columns].to_numpy(dtype=float)
    weights = context_model.reference_weights
    side = side_features.loc[:, side_columns].to_numpy(dtype=float)
    relative_columns = contextual_feature_columns(context_model.feature_set)
    output = np.empty((len(side), len(relative_columns)), dtype=float)
    for index, values in enumerate(side):
        relative = pd.DataFrame(
            values - reference,
            columns=relative_columns,
        )
        contributions = _antisymmetric_feature_contributions(
            context_model,
            _model_relative_features(context_model, relative),
        )
        output[index] = (contributions * weights[:, np.newaxis]).sum(axis=0)
    return output


def _compiled_linear_feature_details(
    context_model: MatchupContextualModel,
    unit_features: pd.DataFrame,
    opponent_features: pd.DataFrame,
    unit_player_ids: list[int],
    opponent_player_ids: list[int],
    unit_state: SeasonLineupState,
    opponent_state: SeasonLineupState,
) -> dict[str, dict[str, Any]]:
    """Expose the exact inputs and Ridge scaling behind linear context cards."""

    try:
        scale = context_model.pipeline.named_steps["scale"]
        ridge = context_model.pipeline.named_steps["ridge"]
    except (AttributeError, KeyError) as error:
        raise LineupEvaluationError(
            "Compiled NAIL-RAPM context details require a scaled Ridge model"
        ) from error

    feature_columns = contextual_feature_columns(context_model.feature_set)
    side_columns = side_context_feature_columns(context_model.feature_set)
    coefficients = np.asarray(ridge.coef_, dtype=float)
    scales = np.asarray(scale.scale_, dtype=float)
    if len(feature_columns) != len(side_columns) or len(coefficients) != len(feature_columns):
        raise LineupEvaluationError("Compiled NAIL-RAPM context feature layout is invalid")

    details: dict[str, dict[str, Any]] = {}
    for index, (feature_id, side_column) in enumerate(
        zip(feature_columns, side_columns, strict=True)
    ):
        unit_value = float(unit_features.iloc[0][side_column])
        opponent_value = float(opponent_features.iloc[0][side_column])
        difference = unit_value - opponent_value
        feature_scale = float(scales[index])
        standardized_coefficient = float(coefficients[index])
        detail: dict[str, Any] = {
            "kind": "generic",
            "unit_value": unit_value,
            "opponent_value": opponent_value,
            "difference": difference,
            "standard_deviation": feature_scale,
            "standardized_difference": difference / feature_scale if feature_scale else 0.0,
            "standardized_coefficient": standardized_coefficient,
            "raw_coefficient": standardized_coefficient / feature_scale if feature_scale else 0.0,
        }
        if side_column == "usage_concentration":
            detail.update(
                {
                    "kind": "usage_concentration",
                    "unit_top_players": _profile_leaders(
                        unit_player_ids, unit_state, "usage_per_100"
                    ),
                    "opponent_top_players": _profile_leaders(
                        opponent_player_ids, opponent_state, "usage_per_100"
                    ),
                    "unit_total": _profile_total(
                        unit_player_ids, unit_state, "usage_per_100"
                    ),
                    "opponent_total": _profile_total(
                        opponent_player_ids, opponent_state, "usage_per_100"
                    ),
                }
            )
        elif side_column == "top_two_assists":
            detail.update(
                {
                    "kind": "top_two_assists",
                    "unit_top_players": _profile_leaders(
                        unit_player_ids, unit_state, "assists_per_100"
                    ),
                    "opponent_top_players": _profile_leaders(
                        opponent_player_ids, opponent_state, "assists_per_100"
                    ),
                }
            )
        details[feature_id] = detail
    return details


def _profile_leaders(
    player_ids: list[int], state: SeasonLineupState, profile_column: str
) -> list[dict[str, Any]]:
    """Return the two player-profile values that define a top-two feature."""

    profile_rows = state.profiles.set_index("player_id")
    player_rows = state.players.set_index("player_id")
    leaders = []
    for player_id in player_ids:
        if player_id not in profile_rows.index or player_id not in player_rows.index:
            raise LineupEvaluationError("A selected player is missing a published profile")
        leaders.append(
            {
                "player_name": str(player_rows.at[player_id, "player_name"]),
                "value": float(profile_rows.at[player_id, profile_column]),
            }
        )
    return sorted(leaders, key=lambda row: float(row["value"]), reverse=True)[:2]


def _profile_total(
    player_ids: list[int], state: SeasonLineupState, profile_column: str
) -> float:
    """Return the five-player total for one profile coordinate."""

    profile_rows = state.profiles.set_index("player_id")
    missing = set(player_ids) - set(profile_rows.index.astype(int))
    if missing:
        raise LineupEvaluationError("A selected player is missing a published profile")
    return float(profile_rows.loc[player_ids, profile_column].sum())


def _feature_rows(
    features: pd.DataFrame,
    contributions: np.ndarray,
    *,
    feature_set: str = CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT,
    include_ids: set[str] | None = None,
    details: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Package sortable original-feature attributions for the browser response."""

    rows = []
    for column, contribution in zip(
        contextual_feature_columns(feature_set), contributions, strict=True
    ):
        if include_ids is not None and column not in include_ids:
            continue
        row = {
            "id": column,
            "label": _feature_label(column),
            "value": float(features.iloc[0][column]),
            "contribution": float(contribution),
        }
        if details is not None and column in details:
            row["detail"] = details[column]
        rows.append(row)
    rows.sort(key=lambda row: abs(float(row["contribution"])), reverse=True)
    return rows


def _composition_response_curves(
    context_model: MatchupContextualModel,
    response_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    unit_features: pd.DataFrame,
    opponent_features: pd.DataFrame,
    unit_contributions: np.ndarray,
    opponent_contributions: np.ndarray,
    *,
    feature_ids: set[str],
) -> list[dict[str, Any]]:
    """Return per-side portable curves for the visible composition factors."""

    curves: list[dict[str, Any]] = []
    for index, column in enumerate(contextual_feature_columns()):
        if column not in feature_ids:
            continue
        side_column = side_context_feature_columns()[index]
        raw_unit_value = float(unit_features.iloc[0][side_column])
        raw_opponent_value = float(opponent_features.iloc[0][side_column])
        support_low, support_high = _reference_feature_support(context_model, index)
        unit_value = _clip_to_support(raw_unit_value, support_low, support_high)
        opponent_value = _clip_to_support(raw_opponent_value, support_low, support_high)
        values = _curve_values(support_low, support_high)
        curves.append(
            {
                "id": column,
                "support_low": support_low,
                "support_high": support_high,
                "unit_value": unit_value,
                "unit_contribution": float(
                    _cached_portable_component(response_cache, index, np.asarray([unit_value]))[0]
                ),
                "opponent_value": opponent_value,
                "opponent_contribution": float(
                    _cached_portable_component(
                        response_cache, index, np.asarray([opponent_value])
                    )[0]
                ),
                "unit_clamped": bool(unit_value != raw_unit_value),
                "opponent_clamped": bool(opponent_value != raw_opponent_value),
                "points": _curve_points(
                    values, _cached_portable_component(response_cache, index, values)
                ),
            }
        )
    return curves


def _matchup_response_curves(
    context_model: MatchupContextualModel,
    response_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    unit_features: pd.DataFrame,
    opponent_features: pd.DataFrame,
    opponent_contributions: np.ndarray,
    *,
    feature_ids: set[str],
) -> list[dict[str, Any]]:
    """Return opponent-fixed matchup-residual curves for visible matchup factors."""

    curves: list[dict[str, Any]] = []
    for index, column in enumerate(contextual_feature_columns()):
        if column not in feature_ids:
            continue
        side_column = side_context_feature_columns()[index]
        raw_unit_value = float(unit_features.iloc[0][side_column])
        raw_opponent_value = float(opponent_features.iloc[0][side_column])
        support_low, support_high = _reference_feature_support(context_model, index)
        unit_value = _clip_to_support(raw_unit_value, support_low, support_high)
        opponent_value = _clip_to_support(raw_opponent_value, support_low, support_high)
        values = _curve_values(support_low, support_high)
        portable_values = _cached_portable_component(response_cache, index, values)
        portable_opponent_contribution = _cached_portable_component(
            response_cache, index, np.asarray([opponent_value])
        )[0]
        total_values = isolated_feature_component(
            context_model, index, values - opponent_value
        )
        matchup_values = total_values - portable_values + portable_opponent_contribution
        current_value = isolated_feature_component(
            context_model, index, np.asarray([unit_value - opponent_value])
        )[0] - _cached_portable_component(
            response_cache, index, np.asarray([unit_value])
        )[0] + portable_opponent_contribution
        curves.append(
            {
                "id": column,
                "support_low": support_low,
                "support_high": support_high,
                "unit_value": unit_value,
                "unit_contribution": float(current_value),
                "opponent_value": opponent_value,
                "opponent_contribution": 0.0,
                "unit_clamped": bool(unit_value != raw_unit_value),
                "opponent_clamped": bool(opponent_value != raw_opponent_value),
                "points": _curve_points(values, matchup_values),
            }
        )
    return curves


def _portable_feature_component(
    context_model: MatchupContextualModel, feature_index: int, values: np.ndarray
) -> np.ndarray:
    """Return h_k(x) by averaging an isolated feature response over the reference field."""

    reference = context_model.reference_features.iloc[:, feature_index].to_numpy(dtype=float)
    weights = context_model.reference_weights
    output = np.empty(len(values), dtype=float)
    for start in range(0, len(values), 4):
        chunk = values[start : start + 4]
        difference = (chunk[:, np.newaxis] - reference[np.newaxis, :]).reshape(-1)
        response = isolated_feature_component(context_model, feature_index, difference)
        output[start : start + len(chunk)] = response.reshape(len(chunk), len(reference)) @ weights
    return output


def _warm_response_cache(
    context_model: MatchupContextualModel,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Materialize every portable feature response once per loaded artifact."""

    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index in range(len(contextual_feature_columns())):
        low, high = _reference_feature_support(context_model, index)
        values = np.linspace(low, high, WARM_RESPONSE_CURVE_POINTS)
        cache[index] = (values, _portable_feature_component(context_model, index, values))
    return cache


def response_cache_path(
    model_artifact: str, run_id: str, season: str | None = None
) -> Path:
    """Return the immutable response cache location for one completed model season."""

    suffix = f"-{season}" if season is not None else ""
    return DEFAULT_RESPONSE_CACHE_DIR / model_artifact / f"{run_id}{suffix}.joblib"


def player_context_exposure_path(model_artifact: str, run_id: str) -> Path:
    """Return the cached annual unit-context exposure table for one model run."""

    return DEFAULT_RESPONSE_CACHE_DIR / model_artifact / f"{run_id}-player-context.parquet"


def player_team_splits_path(model_artifact: str, run_id: str) -> Path:
    """Return the persisted player-season team exposure table for one model run."""

    return DEFAULT_PLAYER_TEAM_SPLITS_CACHE_DIR / model_artifact / f"{run_id}.parquet"


def lineup_rankings_path(model_artifact: str, run_id: str, season: str) -> Path:
    """Return the immutable observed-five lineup table for one completed model run."""

    return DEFAULT_LINEUP_RANKINGS_CACHE_DIR / model_artifact / run_id / f"{season}.parquet"


def exposure_cohort_path(model_artifact: str, run_id: str) -> Path:
    """Return the compact historical exposure cache needed by the public API."""

    return DEFAULT_EXPOSURE_COHORT_CACHE_DIR / model_artifact / f"{run_id}.parquet"


def historical_profiles_path(model_artifact: str, run_id: str) -> Path:
    """Return the compact completed-season player-profile cache for the public API."""

    return DEFAULT_HISTORICAL_PROFILE_CACHE_DIR / model_artifact / f"{run_id}.parquet"


def historical_realized_profiles_path(model_artifact: str, run_id: str) -> Path:
    """Return the realized profile cache used by retrospective Lab evaluations."""

    return (
        DEFAULT_HISTORICAL_REALIZED_PROFILE_CACHE_DIR / model_artifact / f"{run_id}.parquet"
    )


def published_player_ratings_path(model_artifact: str, run_id: str) -> Path:
    """Return the web cache containing compiled completed-fit player ratings."""

    return DEFAULT_LINEUP_RANKINGS_CACHE_DIR / model_artifact / run_id / "player_ratings.parquet"


def preseason_rankings_path(model_artifact: str, run_id: str, season: str) -> Path:
    """Return the immutable cached preseason ranking catalog for one model run."""

    return (
        DEFAULT_PRESEASON_RANKINGS_CACHE_DIR
        / model_artifact
        / run_id
        / f"{season}.parquet"
    )


def team_roster_path(season: str) -> Path:
    """Return the normalized active-roster snapshot for one preseason."""

    return DEFAULT_TEAM_ROSTERS_DIR / season / "part-00000.parquet"


def forward_draft_cold_start_rankings_path(season: str) -> Path:
    """Return the latest forward draft-cold-start ranking artifact, when published."""

    root = DEFAULT_FORWARD_DRAFT_COLD_START_DIR / season
    latest = root / "latest.json"
    if not latest.is_file():
        return root / "drafted_rookie_rankings.parquet"
    run_id = str(json.loads(latest.read_text()).get("run_id", ""))
    return root / run_id / "drafted_rookie_rankings.parquet"


def build_observed_lineup_rankings(
    *,
    season: str,
    profiles: pd.DataFrame,
    players: pd.DataFrame,
    coefficients: pd.DataFrame,
    context_model: MatchupContextualModel,
    analytical_dir: Path | str = Path("data/analytical"),
) -> pd.DataFrame:
    """Materialize regular-season five-man ratings against actual opponents.

    Each row represents one observed team-unit. Player totals are sums of the
    five completed-fit coefficients. Edges and GESTALT scores use the opponents
    that unit actually faced, weighted by stint possessions.
    """

    stints = read_rapm_stints(season, analytical_dir=analytical_dir)
    required = {
        "game_id",
        "home_team_id",
        "away_team_id",
        "home_team_tricode",
        "away_team_tricode",
        "home_player_ids",
        "away_player_ids",
        "possessions",
        "target_home_net_rating",
    }
    missing = required - set(stints)
    if missing:
        raise LineupEvaluationError(
            "Observed lineup source is missing required columns: " + ", ".join(sorted(missing))
        )
    coefficient_map = dict(
        zip(
            coefficients["player_id"].astype(int),
            coefficients["rapm"].astype(float),
            strict=True,
        )
    )
    home_lineups = [_lineup_key(lineup) for lineup in stints["home_player_ids"]]
    away_lineups = [_lineup_key(lineup) for lineup in stints["away_player_ids"]]
    all_lineups = list(dict.fromkeys([*home_lineups, *away_lineups]))
    feature_kwargs = {
        "feature_set": context_model.feature_set,
        "rebound_model": getattr(context_model, "rebound_model", None),
        "usage_model": getattr(context_model, "usage_model", None),
    }
    raw_side_features = lineup_side_context_features(all_lineups, profiles, **feature_kwargs)
    home_raw = lineup_side_context_features(home_lineups, profiles, **feature_kwargs)
    away_raw = lineup_side_context_features(away_lineups, profiles, **feature_kwargs)
    home_features, away_features, _ = _model_feature_inputs(context_model, home_raw, away_raw)
    if _is_compiled_linear_x3(context_model):
        side_columns = side_context_feature_columns(context_model.feature_set)
        additive_ids = {
            f"home_minus_away_{column}"
            for column in _linear_x3_additive_feature_map(context_model.feature_set)
        }
        nonadditive_columns = [
            side_column
            for feature_id, side_column in zip(
                contextual_feature_columns(context_model.feature_set), side_columns, strict=True
            )
            if feature_id not in additive_ids
        ]
        scale = context_model.pipeline.named_steps["scale"]
        ridge = context_model.pipeline.named_steps["ridge"]
        raw_coefficients = pd.Series(
            np.asarray(ridge.coef_, dtype=float) / np.asarray(scale.scale_, dtype=float),
            index=side_columns,
        )
        raw_side_scores = raw_side_features.loc[:, nonadditive_columns].to_numpy(dtype=float) @ (
            raw_coefficients.loc[nonadditive_columns].to_numpy(dtype=float)
        )
        side_score_map = dict(zip(all_lineups, raw_side_scores, strict=True))
        home_scores = np.asarray([side_score_map[lineup] for lineup in home_lineups], dtype=float)
        away_scores = np.asarray([side_score_map[lineup] for lineup in away_lineups], dtype=float)
        total_context = home_scores - away_scores
        matchup_bonus = np.zeros(len(stints), dtype=float)
    else:
        side_features, _, _ = _model_feature_inputs(
            context_model, raw_side_features, raw_side_features
        )
        side_scores = _sampled_side_scores(context_model, side_features)
        side_score_map = dict(zip(all_lineups, side_scores, strict=True))
        total_context = context_model.predict_side_pairs(home_features, away_features)
        home_scores = np.asarray([side_score_map[lineup] for lineup in home_lineups], dtype=float)
        away_scores = np.asarray([side_score_map[lineup] for lineup in away_lineups], dtype=float)
        matchup_bonus = total_context - home_scores + away_scores
    home_player_ratings = np.asarray(
        [_lineup_rating(lineup, coefficient_map) for lineup in home_lineups], dtype=float
    )
    away_player_ratings = np.asarray(
        [_lineup_rating(lineup, coefficient_map) for lineup in away_lineups], dtype=float
    )
    possessions = stints["possessions"].to_numpy(dtype=float)
    actual_home = stints["target_home_net_rating"].to_numpy(dtype=float)

    home = _observed_lineup_side_rows(
        team_ids=stints["home_team_id"].to_numpy(),
        teams=stints["home_team_tricode"].astype(str).to_numpy(),
        lineups=home_lineups,
        game_ids=stints["game_id"].astype(str).to_numpy(),
        possessions=possessions,
        player_rating=home_player_ratings,
        opponent_player_rating=away_player_ratings,
        composition_rating=home_scores,
        opponent_composition_rating=away_scores,
        matchup_bonus=matchup_bonus,
        actual_net_rating=actual_home,
    )
    away = _observed_lineup_side_rows(
        team_ids=stints["away_team_id"].to_numpy(),
        teams=stints["away_team_tricode"].astype(str).to_numpy(),
        lineups=away_lineups,
        game_ids=stints["game_id"].astype(str).to_numpy(),
        possessions=possessions,
        player_rating=away_player_ratings,
        opponent_player_rating=home_player_ratings,
        composition_rating=away_scores,
        opponent_composition_rating=home_scores,
        matchup_bonus=-matchup_bonus,
        actual_net_rating=-actual_home,
    )
    names = dict(
        zip(players["player_id"].astype(int), players["player_name"].astype(str), strict=True)
    )
    return _aggregate_observed_lineups(pd.concat([home, away], ignore_index=True), names)


def _observed_lineup_side_rows(
    *,
    team_ids: np.ndarray,
    teams: np.ndarray,
    lineups: list[tuple[int, ...]],
    game_ids: np.ndarray,
    possessions: np.ndarray,
    player_rating: np.ndarray,
    opponent_player_rating: np.ndarray,
    composition_rating: np.ndarray,
    opponent_composition_rating: np.ndarray,
    matchup_bonus: np.ndarray,
    actual_net_rating: np.ndarray,
) -> pd.DataFrame:
    """Return one orientation's observed-unit contributions in the unit frame."""

    composition_edge = composition_rating - opponent_composition_rating
    player_edge = player_rating - opponent_player_rating
    context_edge = composition_edge + matchup_bonus
    return pd.DataFrame(
        {
            "team_id": team_ids.astype(int),
            "team": teams,
            "lineup_key": ["|".join(str(player_id) for player_id in lineup) for lineup in lineups],
            "game_id": game_ids,
            "possessions": possessions,
            "player_rating": player_rating,
            "player_edge": player_edge,
            "composition_rating": composition_rating,
            "composition_edge": composition_edge,
            "matchup_bonus": matchup_bonus,
            "context_edge": context_edge,
            "gestalt_score": player_edge + context_edge,
            "actual_net_rating": actual_net_rating,
        }
    )


def _aggregate_observed_lineups(
    rows: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """Possession-weight aggregate stint-level ratings into an observed five-man table."""

    metrics = (
        "player_rating",
        "player_edge",
        "composition_rating",
        "composition_edge",
        "matchup_bonus",
        "context_edge",
        "gestalt_score",
        "actual_net_rating",
    )
    weighted = rows.copy()
    for metric in metrics:
        values = weighted[metric].to_numpy(dtype=float)
        weighted[metric] = values * weighted["possessions"].to_numpy(dtype=float)
    aggregated = (
        weighted.groupby(["team_id", "team", "lineup_key"], as_index=False, sort=False)
        .agg(
            possessions=("possessions", "sum"),
            games=("game_id", "nunique"),
            **{metric: (metric, "sum") for metric in metrics},
        )
        .reset_index(drop=True)
    )
    for metric in metrics:
        aggregated[metric] /= aggregated["possessions"]
    aggregated["player_ids"] = aggregated["lineup_key"].map(
        lambda key: [int(player_id) for player_id in key.split("|")]
    )
    aggregated["player_names"] = aggregated["player_ids"].map(
        lambda lineup: [names.get(player_id, f"Player {player_id}") for player_id in lineup]
    )
    aggregated["lineup_label"] = aggregated["player_names"].map(", ".join)
    return aggregated.sort_values(
        ["context_edge", "possessions", "team", "lineup_label"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _lineup_key(lineup: list[int] | tuple[int, ...] | np.ndarray) -> tuple[int, ...]:
    """Return an order-invariant five-player identity key."""

    return tuple(sorted(int(player_id) for player_id in lineup))


def _lineup_rating(lineup: tuple[int, ...], coefficients: dict[int, float]) -> float:
    """Sum completed-fit player ratings, failing rather than silently dropping a player."""

    missing = [player_id for player_id in lineup if player_id not in coefficients]
    if missing:
        raise LineupEvaluationError(f"Observed lineup contains players without ratings: {missing}")
    return float(sum(coefficients[player_id] for player_id in lineup))


def _sampled_side_scores(
    context_model: MatchupContextualModel,
    side_features: pd.DataFrame,
    *,
    sample_size: int = LINEUP_REFERENCE_SAMPLE_SIZE,
) -> np.ndarray:
    """Approximate h(U) with a deterministic weighted reference subsample.

    The observed-lineup table may include thousands of rare units. Exact h(U)
    for every one would create hundreds of millions of spline evaluations. The
    total C(U,O) remains exact; this numerical approximation is used only to
    partition that total into composition and matchup components.
    """

    side_columns = side_context_feature_columns(context_model.feature_set)
    reference = context_model.reference_features.loc[:, side_columns]
    weights = np.asarray(context_model.reference_weights, dtype=float)
    count = min(sample_size, len(reference))
    cumulative = np.cumsum(weights)
    sample_positions = (np.arange(count, dtype=float) + 0.5) / count
    indices = np.searchsorted(cumulative, sample_positions, side="left")
    sample = reference.iloc[indices].reset_index(drop=True)
    output = np.empty(len(side_features), dtype=float)
    for start in range(0, len(side_features), 64):
        chunk = side_features.iloc[start : start + 64].reset_index(drop=True)
        home = pd.DataFrame(
            np.repeat(chunk.to_numpy(dtype=float), count, axis=0),
            columns=side_columns,
        )
        away = pd.DataFrame(
            np.tile(sample.to_numpy(dtype=float), (len(chunk), 1)),
            columns=side_columns,
        )
        output[start : start + len(chunk)] = context_model.predict_side_pairs(home, away).reshape(
            len(chunk), count
        ).mean(axis=1)
    return output


def warm_player_context_exposure(
    models: dict[str, MatchupContextualModel],
    ratings: pd.DataFrame,
    *,
    panel_path: Path,
    padding_contract: ProfilePaddingContract | None = None,
    use_last_observed_profile: bool = False,
) -> pd.DataFrame:
    """Aggregate each player's observed lineup-shape context over regular stints."""

    target = str(ratings["season"].max())
    print("Preparing context-exposure cohort", flush=True)
    panel = pd.read_parquet(panel_path)
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(target)],
        through_season=target,
    )
    rows: list[dict[str, object]] = []
    for season in sorted(ratings["season"].astype(str).unique()):
        model = models.get(season)
        if model is None or not _is_compiled_linear_x3(model):
            continue
        print(f"[{season}] calculating observed lineup-shape context exposure", flush=True)
        stints = read_rapm_stints(season)
        participants = set().union(*stints["home_player_ids"], *stints["away_player_ids"])
        profiles = build_contextual_player_profiles(
            panel,
            target_season=season,
            target_player_ids=participants,
            exposure_cohort=exposure_cohort,
            **(
                {"padding_contract": padding_contract}
                if padding_contract is not None
                else {}
            ),
            use_last_observed_profile=use_last_observed_profile,
        )
        additive_ids = {
            f"home_minus_away_{column}"
            for column in _linear_x3_additive_feature_map(model.feature_set)
        }
        side_columns = side_context_feature_columns(model.feature_set)
        nonadditive_columns = [
            side_column
            for feature_id, side_column in zip(
                contextual_feature_columns(model.feature_set), side_columns, strict=True
            )
            if feature_id not in additive_ids
        ]
        scale = model.pipeline.named_steps["scale"]
        ridge = model.pipeline.named_steps["ridge"]
        raw_coefficients = pd.Series(
            np.asarray(ridge.coef_, dtype=float) / np.asarray(scale.scale_, dtype=float),
            index=side_columns,
        )
        unique_lineups = sorted(
            {
                tuple(sorted(int(player_id) for player_id in lineup))
                for lineup in [*stints["home_player_ids"], *stints["away_player_ids"]]
            }
        )
        lineup_features = lineup_side_context_features(
            unique_lineups,
            profiles,
            feature_set=model.feature_set,
        )
        lineup_scores = dict(
            zip(
                unique_lineups,
                lineup_features.loc[:, nonadditive_columns].to_numpy(dtype=float)
                @ raw_coefficients.loc[nonadditive_columns].to_numpy(dtype=float),
                strict=True,
            )
        )
        home_keys = stints["home_player_ids"].map(
            lambda lineup: tuple(sorted(int(player_id) for player_id in lineup))
        )
        away_keys = stints["away_player_ids"].map(
            lambda lineup: tuple(sorted(int(player_id) for player_id in lineup))
        )
        offsets = home_keys.map(lineup_scores).to_numpy(dtype=float) - away_keys.map(
            lineup_scores
        ).to_numpy(dtype=float)
        exposure = pd.DataFrame(
            {
                "player_id": [
                    int(player_id)
                    for lineup in stints["home_player_ids"]
                    for player_id in lineup
                ]
                + [
                    int(player_id)
                    for lineup in stints["away_player_ids"]
                    for player_id in lineup
                ],
                "possessions": np.repeat(
                    stints["possessions"].to_numpy(dtype=float), 5
                ).tolist()
                * 2,
                "weighted_context": np.repeat(offsets * stints["possessions"], 5).tolist()
                + np.repeat(-offsets * stints["possessions"], 5).tolist(),
            }
        )
        totals = exposure.groupby("player_id", as_index=False).agg(
            weighted_context=("weighted_context", "sum"),
            context_exposure_possessions=("possessions", "sum"),
        )
        rows.extend(
            {
                "season": season,
                "player_id": int(row.player_id),
                "observed_context_exposure": (
                    float(row.weighted_context) / float(row.context_exposure_possessions)
                ),
                "context_exposure_possessions": float(row.context_exposure_possessions),
                "context_model_season": season,
                "context_profile_information_cutoff": _previous_season(season),
            }
            for row in totals.itertuples(index=False)
            if float(row.context_exposure_possessions) > 0.0
        )
    return pd.DataFrame(rows)


def _cached_portable_component(
    cache: dict[int, tuple[np.ndarray, np.ndarray]], index: int, values: np.ndarray
) -> np.ndarray:
    """Interpolate a warmed portable response inside its stored support."""

    if index not in cache:
        raise LineupEvaluationError("Response cache is unavailable for this artifact")
    grid, response = cache[index]
    return np.interp(values, grid, response)


def _reference_feature_support(
    context_model: MatchupContextualModel, feature_index: int
) -> tuple[float, float]:
    """Return the possession-weighted central support of one side feature."""

    if isinstance(context_model, BoundedMatchupContextualModel):
        column = side_context_feature_columns()[feature_index]
        return float(context_model.side_lower[column]), float(context_model.side_upper[column])
    values = context_model.reference_features.iloc[:, feature_index].to_numpy(dtype=float)
    return _weighted_quantiles(values, context_model.reference_weights, [0.05, 0.95])


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: list[float]
) -> tuple[float, float]:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative_weights = np.cumsum(weights[order])
    return tuple(
        float(np.interp(quantile, cumulative_weights, sorted_values)) for quantile in quantiles
    )


def _curve_values(support_low: float, support_high: float) -> np.ndarray:
    """Evaluate response curves only inside their central observed support."""

    return np.linspace(support_low, support_high, RESPONSE_CURVE_POINTS)


def _clip_to_support(value: float, support_low: float, support_high: float) -> float:
    """Keep visual markers inside the published 5th-95th percentile support."""

    return float(np.clip(value, support_low, support_high))


def _curve_points(values: np.ndarray, contributions: np.ndarray) -> list[dict[str, float]]:
    return [
        {"value": float(value), "contribution": float(contribution)}
        for value, contribution in zip(values, contributions, strict=True)
    ]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace({np.nan: None})
    records = [dict(row) for row in clean.to_dict(orient="records")]
    for record in records:
        serialized_breakdown = record.pop("additive_profile_breakdown_json", None)
        if serialized_breakdown:
            record["additive_profile_breakdown"] = json.loads(serialized_breakdown)
    return records


def _normalize_draft_metadata(frame: pd.DataFrame) -> None:
    """Expose draft fields with stable nullable API types."""

    for column in ("draft_year", "draft_round", "draft_number"):
        values = pd.to_numeric(
            frame.get(column, pd.Series(index=frame.index, dtype="float64")),
            errors="coerce",
        )
        frame[column] = pd.Series(
            [int(value) if pd.notna(value) else None for value in values],
            index=frame.index,
            dtype="object",
        )
    undrafted = frame.get("is_undrafted", pd.Series(index=frame.index, dtype="object"))
    frame["is_undrafted"] = pd.Series(
        [bool(value) if pd.notna(value) else None for value in undrafted],
        index=frame.index,
        dtype="object",
    )


def _assign_draft_class_year(frame: pd.DataFrame) -> None:
    """Attach a shared entering class for drafted and undrafted players."""

    draft_year = pd.to_numeric(frame["draft_year"], errors="coerce")
    rookie_year = pd.to_numeric(
        frame.get("rookie_season", pd.Series(index=frame.index, dtype="object"))
        .astype("string")
        .str.slice(0, 4),
        errors="coerce",
    )
    undrafted = frame["is_undrafted"].eq(True)
    class_year = draft_year.where(draft_year.notna(), rookie_year.where(undrafted))
    frame["draft_class_year"] = pd.Series(
        [int(value) if pd.notna(value) else None for value in class_year],
        index=frame.index,
        dtype="object",
    )


def _optional_int(value: object) -> int | None:
    return int(value) if pd.notna(value) else None


def _optional_float(value: object) -> float | None:
    return float(value) if pd.notna(value) else None


def _optional_bool(value: object) -> bool | None:
    return bool(value) if pd.notna(value) else None


def _normalize_search_text(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def _validate_lineup(name: str, player_ids: list[int]) -> None:
    if len(player_ids) != 5:
        raise LineupEvaluationError(f"The {name} must contain exactly five players")
    if len(set(player_ids)) != 5:
        raise LineupEvaluationError(f"The {name} must contain five distinct players")
