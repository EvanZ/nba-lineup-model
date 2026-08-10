"""Compact in-memory inference for the NBA GESTALT Lineup Lab."""

from __future__ import annotations

import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling import (
    forward_aging_bounded_hierarchical_portable_matchup_contextual_rapm as portable_model,
)
from nba_lineup_model.modeling.contextual_features import (
    contextual_feature_columns,
    lineup_side_context_features,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.lineup_context_case_study import (
    _feature_contributions,
    _feature_label,
)
from nba_lineup_model.modeling.matchup_contextual import (
    BoundedMatchupContextualModel,
    MatchupContextualModel,
    isolated_feature_component,
)

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_RESPONSE_CACHE_DIR = Path("artifacts/web/response_curve_cache")
MODEL_ARTIFACT = "forward_aging_bounded_hierarchical_portable_matchup_contextual_rapm"
DISPLAY_SEASON = "2025-26"
RESPONSE_CURVE_POINTS = 33
# The client renders the same 33-point polyline, so a denser server cache only
# adds warmup cost without increasing the published curve resolution.
WARM_RESPONSE_CURVE_POINTS = RESPONSE_CURVE_POINTS


class LineupEvaluationError(ValueError):
    """Raised when an input cannot be evaluated against the loaded model state."""


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

    @classmethod
    def from_latest_artifact(
        cls,
        *,
        artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
        season: str = DISPLAY_SEASON,
        panel_path: Path | str = "data/analytical/player_season_panel/player_seasons.parquet",
    ) -> LineupEvaluator:
        """Load the completed seasonal model state used by the local Lineup Lab."""

        root = Path(artifacts_dir) / MODEL_ARTIFACT / season
        latest_path = root / "latest.json"
        if not latest_path.is_file():
            raise FileNotFoundError(
                "No published aging bounded hierarchical portable-matchup artifact for "
                f"{season}: {latest_path}"
            )
        run_id = str(json.loads(latest_path.read_text())["run_id"])
        run_dir = root / run_id
        metadata = json.loads((run_dir / "metadata.json").read_text())
        if metadata.get("model") != portable_model.MODEL_NAME:
            raise LineupEvaluationError(
                "The selected artifact is not forward aging bounded hierarchical portable-matchup "
                "contextual RAPM"
            )
        if str(metadata.get("target_season")) != season:
            raise LineupEvaluationError("The selected artifact has an unexpected target season")

        coefficients = pd.read_parquet(run_dir / "historical_player_coefficients.parquet")
        coefficients = coefficients.loc[
            coefficients["season"].eq(season), ["player_id", "rapm"]
        ].copy()
        if coefficients.empty or coefficients["player_id"].duplicated().any():
            raise LineupEvaluationError("The artifact has invalid completed player coefficients")
        profiles = pd.read_parquet(run_dir / "target_player_profiles.parquet")
        if profiles["player_id"].duplicated().any():
            raise LineupEvaluationError("The artifact has duplicate player profiles")
        players = _player_catalog(
            coefficients, profiles, panel_path=Path(panel_path), season=season
        )
        models = joblib.load(run_dir / "season_context_models.joblib")
        context_model = models.get(season)
        if context_model is None:
            raise LineupEvaluationError(
                "The artifact has no completed aging hierarchical portable-matchup function"
            )
        if not isinstance(context_model, MatchupContextualModel):
            raise LineupEvaluationError("The artifact has an incompatible contextual model")
        cache_path = response_cache_path(MODEL_ARTIFACT, run_id)
        if not cache_path.is_file():
            raise FileNotFoundError(
                "Response cache is not materialized for the selected artifact: "
                f"{cache_path}"
            )
        response_cache = joblib.load(cache_path)
        return cls(
            season=season,
            run_id=run_id,
            coefficients=coefficients,
            profiles=profiles,
            players=players,
            context_model=context_model,
            response_cache=response_cache,
        )

    def search_players(self, query: str, *, limit: int = 12) -> list[dict[str, Any]]:
        """Return a short, deterministic player-name search result list."""

        normalized = _normalize_search_text(query)
        if not normalized:
            return []
        names = self.players["player_name"].map(_normalize_search_text)
        matches = self.players.loc[names.str.contains(normalized, regex=False)]
        return _records(matches.head(max(1, min(limit, 25))))

    def player(self, player_id: int) -> dict[str, Any]:
        """Return the display profile for one available player."""

        row = self.players.loc[self.players["player_id"].eq(player_id)]
        if row.empty:
            raise LineupEvaluationError(f"Player {player_id} is not available in {self.season}")
        return _records(row)[0]

    def default_opponent(
        self, *, excluded_player_ids: set[int] | None = None
    ) -> list[dict[str, Any]]:
        """Sample a plausible baseline opponent from high-possession players."""

        excluded = excluded_player_ids or set()
        eligible = self.players.loc[
            self.players["possessions"].ge(self.players["possessions"].quantile(0.75))
            & ~self.players["player_id"].isin(excluded)
        ]
        if len(eligible) < 5:
            raise LineupEvaluationError(
                "The loaded player catalog has too few high-possession players"
            )
        sample = eligible.sample(
            n=5,
            replace=False,
            random_state=random.SystemRandom().randrange(2**32),
        )
        return _records(sample.sort_values("player_name", kind="stable"))

    def evaluate(
        self,
        unit_player_ids: list[int],
        opponent_player_ids: list[int],
        *,
        include_response_curves: bool = False,
        response_curve_feature_id: str | None = None,
        response_curve_kind: str | None = None,
    ) -> dict[str, Any]:
        """Score a neutral-court five-player unit against a five-player opponent."""

        _validate_lineup("unit", unit_player_ids)
        _validate_lineup("opponent", opponent_player_ids)
        overlap = set(unit_player_ids) & set(opponent_player_ids)
        if overlap:
            raise LineupEvaluationError("A player cannot appear on both sides of a matchup")
        available = set(self.players["player_id"].astype(int))
        missing = sorted((set(unit_player_ids) | set(opponent_player_ids)) - available)
        if missing:
            raise LineupEvaluationError(f"Players are unavailable in {self.season}: {missing}")

        coefficient_map = dict(
            zip(
                self.coefficients["player_id"].astype(int),
                self.coefficients["rapm"].astype(float),
                strict=True,
            )
        )
        raw_unit_features = lineup_side_context_features([unit_player_ids], self.profiles)
        raw_opponent_features = lineup_side_context_features(
            [opponent_player_ids], self.profiles
        )
        unit_features, opponent_features, features = _model_feature_inputs(
            self.context_model,
            raw_unit_features,
            raw_opponent_features,
        )
        components = self.context_model.decompose_side_pairs(
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
        total_contributions = _antisymmetric_feature_contributions(self.context_model, features)[0]
        unit_composition_contributions = _side_feature_contributions(
            self.context_model, unit_features
        )[0]
        opponent_composition_contributions = _side_feature_contributions(
            self.context_model, opponent_features
        )[0]
        composition_contributions = (
            unit_composition_contributions - opponent_composition_contributions
        )
        matchup_contributions = total_contributions - composition_contributions
        additive_unit = float(sum(coefficient_map[player_id] for player_id in unit_player_ids))
        additive_opponent = float(
            sum(coefficient_map[player_id] for player_id in opponent_player_ids)
        )
        additive_margin = additive_unit - additive_opponent
        feature_rows = _feature_rows(features, total_contributions)
        composition_feature_rows = _feature_rows(features, composition_contributions)
        matchup_feature_rows = _feature_rows(features, matchup_contributions)
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
            self.context_model,
            self.response_cache,
            unit_features,
            opponent_features,
            unit_composition_contributions,
            opponent_composition_contributions,
            feature_ids=composition_feature_ids,
        ) if include_response_curves else []
        matchup_response_curves = _matchup_response_curves(
            self.context_model,
            self.response_cache,
            unit_features,
            opponent_features,
            opponent_composition_contributions,
            feature_ids=matchup_feature_ids,
        ) if include_response_curves else []
        response = {
            "season": self.season,
            "run_id": self.run_id,
            "retrospective": True,
            "unit": self._side(unit_player_ids, coefficient_map),
            "opponent": self._side(opponent_player_ids, coefficient_map),
            "additive_margin": additive_margin,
            "contextual_adjustment": contextual_adjustment,
            "unit_composition_rating": unit_composition_rating,
            "opponent_composition_rating": opponent_composition_rating,
            "portable_composition_margin": portable_composition_margin,
            "matchup_adjustment": matchup_adjustment,
            "predicted_net_rating": additive_margin + contextual_adjustment,
            "feature_contributions": feature_rows,
            "composition_feature_contributions": composition_feature_rows,
            "matchup_feature_contributions": matchup_feature_rows,
        }
        if include_response_curves:
            response["composition_response_curves"] = composition_response_curves
            response["matchup_response_curves"] = matchup_response_curves
        return response

    def _side(self, player_ids: list[int], coefficient_map: dict[int, float]) -> dict[str, Any]:
        rows = self.players.set_index("player_id").loc[player_ids].reset_index()
        players = _records(rows)
        for player in players:
            player["rapm"] = coefficient_map[int(player["player_id"])]
        return {"additive_rating": sum(player["rapm"] for player in players), "players": players}


def _player_catalog(
    coefficients: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    panel_path: Path,
    season: str,
) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    panel_columns = [
        "player_id",
        "player_name",
        "primary_team_tricode",
        "listed_position",
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
        "three_pm_per_100",
        "assists_per_100",
        "usage_per_100",
        "offensive_rebounds_per_100",
        "defensive_rebounds_per_100",
    ]
    catalog = coefficients.merge(profiles.loc[:, profile_columns], on="player_id", how="inner")
    catalog = catalog.merge(
        season_panel.drop(columns=["player_name"], errors="ignore"), on="player_id", how="left"
    )
    catalog["team"] = catalog.get(
        "primary_team_tricode", pd.Series(index=catalog.index)
    ).fillna("-")
    catalog["position"] = catalog.get(
        "listed_position", pd.Series(index=catalog.index)
    ).fillna("-")
    catalog["possessions"] = catalog.get(
        "rapm_possessions", pd.Series(index=catalog.index)
    ).fillna(0.0)
    catalog["games"] = catalog.get("games", pd.Series(index=catalog.index)).fillna(0).astype(int)
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
            "rapm",
            "possessions",
            "games",
            "profile_source",
            "profile_imputed",
            "profile_replacement_weight",
            "three_pm_per_100",
            "assists_per_100",
            "usage_per_100",
            "offensive_rebounds_per_100",
            "defensive_rebounds_per_100",
        ],
    ]


def _antisymmetric_feature_contributions(
    context_model: MatchupContextualModel, features: pd.DataFrame
) -> np.ndarray:
    """Return relative feature contributions under the exact antisymmetric total."""

    forward = _feature_contributions(context_model.pipeline, features)
    reverse = _feature_contributions(context_model.pipeline, -features)
    return 0.5 * (forward - reverse)


def _model_feature_inputs(
    context_model: MatchupContextualModel,
    unit_features: pd.DataFrame,
    opponent_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the exact side and relative inputs used by the fitted model."""

    unit = unit_features.loc[:, side_context_feature_columns()].copy()
    opponent = opponent_features.loc[:, side_context_feature_columns()].copy()
    if isinstance(context_model, BoundedMatchupContextualModel):
        unit = unit.clip(context_model.side_lower, context_model.side_upper, axis=1)
        opponent = opponent.clip(context_model.side_lower, context_model.side_upper, axis=1)
    relative = pd.DataFrame(
        {
            column: (
                unit[side_context_feature_columns()[index]].to_numpy(dtype=float)
                - opponent[side_context_feature_columns()[index]].to_numpy(dtype=float)
            )
            for index, column in enumerate(contextual_feature_columns())
        },
        columns=contextual_feature_columns(),
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

    side_columns = side_context_feature_columns()
    reference = context_model.reference_features.loc[:, side_columns].to_numpy(dtype=float)
    weights = context_model.reference_weights
    side = side_features.loc[:, side_columns].to_numpy(dtype=float)
    output = np.empty((len(side), len(contextual_feature_columns())), dtype=float)
    for index, values in enumerate(side):
        relative = pd.DataFrame(
            values - reference,
            columns=contextual_feature_columns(),
        )
        contributions = _antisymmetric_feature_contributions(
            context_model,
            _model_relative_features(context_model, relative),
        )
        output[index] = (contributions * weights[:, np.newaxis]).sum(axis=0)
    return output


def _feature_rows(features: pd.DataFrame, contributions: np.ndarray) -> list[dict[str, Any]]:
    """Package sortable original-feature attributions for the browser response."""

    rows = [
        {
            "id": column,
            "label": _feature_label(column),
            "value": float(features.iloc[0][column]),
            "contribution": float(contribution),
        }
        for column, contribution in zip(
            contextual_feature_columns(), contributions, strict=True
        )
    ]
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


def response_cache_path(model_artifact: str, run_id: str) -> Path:
    """Return the immutable web cache location for one completed model run."""

    return DEFAULT_RESPONSE_CACHE_DIR / model_artifact / f"{run_id}.joblib"


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
    return [dict(row) for row in clean.to_dict(orient="records")]


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
