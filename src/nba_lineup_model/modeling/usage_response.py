"""Pregame usage-state and O/D residual-response experiment.

The production NAIL-RAPM release and its Constrained Split companion remain
fixed.  This module asks a narrower forecasting question: once a player has
shifted into a higher or lower usage role than their preseason expectation,
does that information improve the next game's offensive or defensive scoring
prediction?  A response is fitted only to the frozen split baseline's residual
and is never folded back into scalar player value here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.constrained_split_nail import (
    DISPLAY_SEASON,
    PRODUCTION_RUN_ID,
    PRODUCTION_MODEL_ARTIFACT,
    ConstrainedSplitState,
    ProductionState,
    _forecast_mean_coordinates,
    _load_production_state,
    _raw_coefficients,
    _seasons_through,
    _state_specialization_for_design,
)
from nba_lineup_model.modeling.forward_contextual_rapm import _previous_season
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import _Tee
from nba_lineup_model.modeling.split_nail import SplitNailDesign

MODEL_NAME = "usage_response_screen"
RUN_PREFIX = "usage-response-screen"
FIRST_SEASON = "1996-97"
DEVELOPMENT_TARGET_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23")
FROZEN_TARGET_SEASONS = ("2023-24", "2024-25", "2025-26")
USAGE_PSEUDO_OPPORTUNITIES = 300.0
DEFAULT_LEAGUE_USAGE_RATE = 0.20
RESPONSE_REGULARIZATION_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_CONSTRAINED_RUN_DIR = Path(
    "artifacts/models/constrained_split_nail/2025-26/"
    "constrained-split-nail-2025-26-20260829T130012Z-81a5fad3"
)
DEFAULT_PRODUCTION_RUN_DIR = (
    DEFAULT_ARTIFACTS_DIR
    / PRODUCTION_MODEL_ARTIFACT
    / DISPLAY_SEASON
    / PRODUCTION_RUN_ID
)
DEFAULT_DESIGN_CACHE_PATH = Path(
    "artifacts/cache/constrained_split_nail/2025-26/season_designs.joblib"
)


@dataclass(frozen=True)
class UsageResponseModel:
    """A zero-at-zero two-side residual correction.

    ``offense_coefficient`` and ``defense_coefficient`` are in rating points
    per 100 possessions for one percentage-point usage shock from one player.
    The latter is a defensive *rating* effect: negative means that higher usage
    is associated with worse defense.
    """

    regularization: float
    feature_scales: np.ndarray
    scaled_coefficients: np.ndarray

    @property
    def raw_coefficients(self) -> np.ndarray:
        return self.scaled_coefficients / self.feature_scales

    @property
    def offense_coefficient(self) -> float:
        return float(self.raw_coefficients[0])

    @property
    def defense_coefficient(self) -> float:
        return float(self.raw_coefficients[1])

    @property
    def total_coefficient(self) -> float:
        return self.offense_coefficient + self.defense_coefficient

    def predict_response(self, features: np.ndarray) -> np.ndarray:
        values = _validate_response_features(features)
        return (values / self.feature_scales) @ self.scaled_coefficients


@dataclass(frozen=True)
class UsageResponseRun:
    """Immutable location and identity for a completed response screen."""

    run_dir: Path
    run_id: str
    selected_regularization: float


def game_usage_from_player_boxscores(players: pd.DataFrame) -> pd.DataFrame:
    """Return conventional-USG numerator and opportunity rows for played games.

    This is deliberately the same numerator and team-minute opportunity
    construction used by the NAIL additive ``usage_pct`` profile.  The returned
    rows are one listed player/team/game observation, not a completed season
    aggregate.  Zero-minute listed players retain their pregame state so a
    lineup reconstruction can still be matched; they contribute zero current
    usage exposure and therefore do not update that state after the game.
    """

    required = {
        "game_id",
        "team_id",
        "personId",
        "statistics_minutes",
        "statistics_fieldGoalsAttempted",
        "statistics_freeThrowsAttempted",
        "statistics_turnovers",
        "game_time_utc",
    }
    missing = sorted(required - set(players))
    if missing:
        raise ValueError(f"Player boxscores missing usage inputs: {missing}")
    frame = players.copy()
    frame["minutes"] = (
        pd.to_timedelta(frame["statistics_minutes"], errors="coerce").dt.total_seconds()
        / 60.0
    ).fillna(0.0)
    if frame["minutes"].lt(0.0).any():
        raise ValueError("Usage state requires non-negative parsed minutes")
    frame["player_id"] = pd.to_numeric(frame["personId"], errors="raise").astype("int64")
    frame["usage_events"] = (
        pd.to_numeric(frame["statistics_fieldGoalsAttempted"], errors="raise").astype(float)
        + 0.44
        * pd.to_numeric(frame["statistics_freeThrowsAttempted"], errors="raise").astype(float)
        + pd.to_numeric(frame["statistics_turnovers"], errors="raise").astype(float)
    )
    frame.loc[frame["minutes"].eq(0.0), "usage_events"] = 0.0
    if frame["usage_events"].lt(0.0).any():
        raise ValueError("Usage events must be non-negative")
    played = frame.loc[frame["minutes"].gt(0.0)].copy()
    if played.empty:
        raise ValueError("Usage state requires at least one positive-minute player row")
    team = played.groupby(["game_id", "team_id"], as_index=False, sort=False).agg(
        team_minutes=("minutes", "sum"),
        team_usage_events=("usage_events", "sum"),
    )
    frame = frame.merge(team, on=["game_id", "team_id"], validate="many_to_one")
    frame["usage_opportunities"] = (
        frame["minutes"] / (frame["team_minutes"] / 5.0) * frame["team_usage_events"]
    )
    if frame["usage_opportunities"].lt(0.0).any():
        raise ValueError("Usage opportunities must be non-negative")
    frame["game_time_utc"] = pd.to_datetime(frame["game_time_utc"], utc=True)
    if frame["game_time_utc"].isna().any():
        raise ValueError("Usage state requires a game timestamp for every player row")
    columns = [
        "game_id",
        "team_id",
        "player_id",
        "game_time_utc",
        "minutes",
        "usage_events",
        "usage_opportunities",
    ]
    for optional in ("name", "team_tricode", "season"):
        if optional in frame:
            columns.append(optional)
    output = frame.loc[:, columns].copy()
    output["game_id"] = output["game_id"].astype(str)
    if output.duplicated(["game_id", "team_id", "player_id"]).any():
        raise ValueError("Usage game rows must be unique by game/team/player")
    return output.sort_values(
        ["player_id", "game_time_utc", "game_id", "team_id"], kind="stable"
    ).reset_index(drop=True)


def build_pregame_usage_states_from_games(
    current_games: pd.DataFrame,
    *,
    prior_summary: pd.DataFrame | None = None,
    pseudo_opportunities: float = USAGE_PSEUDO_OPPORTUNITIES,
    default_usage_rate: float = DEFAULT_LEAGUE_USAGE_RATE,
) -> pd.DataFrame:
    """Build a strictly pregame empirical-Bayes usage state.

    The first game starts at the prior-season shrunken role.  Every later game
    sees its own past target-season box scores, but never its current-game
    FGA, FTA, or turnover total.  This is the leakage boundary of the study.
    """

    if pseudo_opportunities <= 0.0:
        raise ValueError("Usage pseudo opportunities must be positive")
    if not 0.0 < default_usage_rate < 1.0:
        raise ValueError("Default usage rate must be between zero and one")
    required = {
        "game_id",
        "team_id",
        "player_id",
        "game_time_utc",
        "usage_events",
        "usage_opportunities",
    }
    missing = sorted(required - set(current_games))
    if missing:
        raise ValueError(f"Current usage games missing columns: {missing}")
    frame = current_games.copy()
    frame["game_time_utc"] = pd.to_datetime(frame["game_time_utc"], utc=True)
    frame["game_id"] = frame["game_id"].astype(str)
    for column in ("usage_events", "usage_opportunities"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if frame["usage_events"].lt(0.0).any() or frame["usage_opportunities"].lt(0.0).any():
        raise ValueError("Usage state inputs must have non-negative events and exposure")
    if frame.duplicated(["game_id", "team_id", "player_id"]).any():
        raise ValueError("Current usage games must be unique by game/team/player")

    prior = _normalized_usage_prior(
        prior_summary,
        pseudo_opportunities=pseudo_opportunities,
        default_usage_rate=default_usage_rate,
    )
    frame = frame.merge(prior, on="player_id", how="left", validate="many_to_one")
    frame["preseason_usage_rate"] = frame["preseason_usage_rate"].fillna(
        prior.attrs["league_usage_rate"]
    )
    ordered = frame.sort_values(
        ["player_id", "game_time_utc", "game_id", "team_id"], kind="stable"
    ).copy()
    grouped = ordered.groupby("player_id", sort=False)
    ordered["prior_target_usage_events"] = grouped["usage_events"].cumsum() - ordered[
        "usage_events"
    ]
    ordered["prior_target_usage_opportunities"] = grouped["usage_opportunities"].cumsum() - ordered[
        "usage_opportunities"
    ]
    ordered["usage_state_opportunities"] = (
        pseudo_opportunities + ordered["prior_target_usage_opportunities"]
    )
    ordered["pregame_usage_rate"] = (
        pseudo_opportunities * ordered["preseason_usage_rate"]
        + ordered["prior_target_usage_events"]
    ) / ordered["usage_state_opportunities"]
    ordered["preseason_usage_pct"] = 100.0 * ordered["preseason_usage_rate"]
    ordered["pregame_usage_pct"] = 100.0 * ordered["pregame_usage_rate"]
    ordered["usage_shock_pct"] = (
        ordered["pregame_usage_pct"] - ordered["preseason_usage_pct"]
    )
    return ordered.drop(columns=["preseason_usage_rate", "pregame_usage_rate"]).reset_index(
        drop=True
    )


def build_pregame_usage_states(
    season: str,
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    pseudo_opportunities: float = USAGE_PSEUDO_OPPORTUNITIES,
) -> pd.DataFrame:
    """Read one season and materialize its leakage-safe player-game states."""

    current = _read_usage_games(season, curated_dir=curated_dir)
    year = int(str(season)[:4])
    prior_summary: pd.DataFrame | None = None
    if year > int(FIRST_SEASON[:4]):
        prior = _read_usage_games(_previous_season(season), curated_dir=curated_dir)
        prior_summary = _season_usage_summary(prior)
    output = build_pregame_usage_states_from_games(
        current,
        prior_summary=prior_summary,
        pseudo_opportunities=pseudo_opportunities,
    )
    if "season" in output:
        output["season"] = str(season)
        columns = ["season", *[column for column in output if column != "season"]]
        output = output.loc[:, columns]
    else:
        output.insert(0, "season", str(season))
    return output


def lineup_usage_shock_features(
    stints: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    """Sum player pregame shocks for each observed lineup scoring side."""

    required_stints = {
        "game_id",
        "home_player_ids",
        "away_player_ids",
        "home_offensive_possessions",
        "away_offensive_possessions",
    }
    missing_stints = sorted(required_stints - set(stints))
    if missing_stints:
        raise ValueError(f"RAPM stints missing lineup inputs: {missing_stints}")
    required_states = {"game_id", "player_id", "usage_shock_pct"}
    missing_states = sorted(required_states - set(states))
    if missing_states:
        raise ValueError(f"Usage states missing columns: {missing_states}")
    values = states.loc[:, ["game_id", "player_id", "usage_shock_pct"]].copy()
    values["game_id"] = values["game_id"].astype(str)
    values["player_id"] = pd.to_numeric(values["player_id"], errors="raise").astype(int)
    if values.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Usage states must be unique by game and player")
    lookup = {
        (str(game_id), int(player_id)): float(shock)
        for game_id, player_id, shock in values.itertuples(index=False, name=None)
    }
    rows: list[dict[str, object]] = []
    for game in stints.loc[
        :,
        [
            "game_id",
            "home_player_ids",
            "away_player_ids",
            "home_offensive_possessions",
            "away_offensive_possessions",
        ],
    ].itertuples(index=False):
        game_id = str(game.game_id)
        home = tuple(int(player_id) for player_id in game.home_player_ids)
        away = tuple(int(player_id) for player_id in game.away_player_ids)
        missing = [
            (game_id, player_id)
            for player_id in (*home, *away)
            if (game_id, player_id) not in lookup
        ]
        if missing:
            preview = ", ".join(f"{game_id}/{player_id}" for game_id, player_id in missing[:8])
            raise ValueError(f"Pregame usage state missing lineup participants: {preview}")
        home_sum = sum(lookup[(game_id, player)] for player in home)
        away_sum = sum(lookup[(game_id, player)] for player in away)
        if float(game.home_offensive_possessions) > 0.0:
            rows.append(
                {
                    "game_id": game_id,
                    "home_offense": True,
                    "own_usage_shock_sum": home_sum,
                    "opponent_usage_shock_sum": away_sum,
                    "own_player_count": len(home),
                    "opponent_player_count": len(away),
                }
            )
        if float(game.away_offensive_possessions) > 0.0:
            rows.append(
                {
                    "game_id": game_id,
                    "home_offense": False,
                    "own_usage_shock_sum": away_sum,
                    "opponent_usage_shock_sum": home_sum,
                    "own_player_count": len(away),
                    "opponent_player_count": len(home),
                }
            )
    return pd.DataFrame(rows)


def build_usage_response_features(
    design: SplitNailDesign,
    lineup_shocks: pd.DataFrame,
) -> np.ndarray:
    """Return own-offense and opponent-defense usage-response columns.

    The second column is negated because a positive defensive rating lowers
    opponent scoring.  Its fitted coefficient can be negative, which means a
    higher usage shock is associated with worse defense.
    """

    required = {
        "game_id",
        "home_offense",
        "own_usage_shock_sum",
        "opponent_usage_shock_sum",
    }
    missing = sorted(required - set(lineup_shocks))
    if missing:
        raise ValueError(f"Lineup shock features missing columns: {missing}")
    ordered = lineup_shocks.reset_index(drop=True)
    if len(ordered) != len(design.game_ids):
        raise ValueError("Lineup shock scoring rows do not align with the split design")
    expected_game_ids = ordered["game_id"].astype(str).to_numpy()
    if not np.array_equal(expected_game_ids, design.game_ids.astype(str)):
        raise ValueError("Lineup shock scoring-row game order does not match the split design")
    if not np.array_equal(
        ordered["home_offense"].to_numpy(dtype=bool), design.home_offense.astype(bool)
    ):
        raise ValueError("Lineup shock scoring sides do not match the split design")
    own = ordered["own_usage_shock_sum"].to_numpy(dtype=float)
    opponent = ordered["opponent_usage_shock_sum"].to_numpy(dtype=float)
    return np.column_stack([own, -opponent])


def fit_usage_response(
    features: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    *,
    regularization: float,
) -> UsageResponseModel:
    """Fit a no-intercept, sample-size-normalized two-side ridge response."""

    if regularization < 0.0:
        raise ValueError("Usage response regularization must be non-negative")
    matrix = _validate_response_features(features)
    target = np.asarray(residual, dtype=float)
    sample_weight = _validated_weights(weights, len(matrix))
    if target.shape != (len(matrix),) or not np.isfinite(target).all():
        raise ValueError("Usage response residuals must be finite and match features")
    normalized_weight = sample_weight / sample_weight.mean()
    scales = np.sqrt(np.average(np.square(matrix), axis=0, weights=normalized_weight))
    if (scales <= 0.0).any() or not np.isfinite(scales).all():
        raise ValueError("Usage response features must have positive weighted RMS scales")
    scaled = matrix / scales
    gram = scaled.T @ (normalized_weight[:, None] * scaled)
    right_hand = scaled.T @ (normalized_weight * target)
    system = gram + regularization * len(scaled) * np.eye(scaled.shape[1])
    coefficients = np.linalg.solve(system, right_hand)
    return UsageResponseModel(
        regularization=float(regularization),
        feature_scales=scales,
        scaled_coefficients=coefficients,
    )


def usage_response_metrics(
    rows: pd.DataFrame,
    *,
    reference_column: str = "baseline_prediction",
    candidate_column: str = "candidate_prediction",
) -> pd.DataFrame:
    """Report possession-weighted side scoring and complete-game margin RMSE."""

    required = {
        "game_id",
        "target",
        "weight",
        "home_offense",
        reference_column,
        candidate_column,
    }
    missing = sorted(required - set(rows))
    if missing:
        raise ValueError(f"Usage response metrics missing columns: {missing}")
    target = rows["target"].to_numpy(dtype=float)
    weights = rows["weight"].to_numpy(dtype=float)
    reference = rows[reference_column].to_numpy(dtype=float)
    candidate = rows[candidate_column].to_numpy(dtype=float)
    side_rows = [
        {
            "metric": "side_scoring_rmse",
            "reference_value": _weighted_rmse(target, reference, weights),
            "candidate_value": _weighted_rmse(target, candidate, weights),
            "observation_count": len(rows),
            "exposure": float(weights.sum()),
        }
    ]
    games = _game_margin_frame(rows, reference_column, candidate_column)
    side_rows.append(
        {
            "metric": "game_margin_rmse",
            "reference_value": float(
                np.sqrt(np.mean(np.square(games["actual_margin"] - games["reference_margin"])))
            ),
            "candidate_value": float(
                np.sqrt(np.mean(np.square(games["actual_margin"] - games["candidate_margin"])))
            ),
            "observation_count": len(games),
            "exposure": float(len(games)),
        }
    )
    output = pd.DataFrame(side_rows)
    output["candidate_minus_reference"] = (
        output["candidate_value"] - output["reference_value"]
    )
    return output


def paired_usage_response_bootstrap(
    rows: pd.DataFrame,
    *,
    draws: int = 2_000,
    seed: int = 20_260_926,
) -> pd.DataFrame:
    """Resample complete games for paired O/D side and margin uncertainty."""

    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    game_frame = _game_margin_frame(rows, "baseline_prediction", "candidate_prediction")
    game_ids = pd.Index(sorted(rows["game_id"].astype(str).unique()))
    groups = rows.assign(game_id=rows["game_id"].astype(str)).groupby("game_id", sort=True)
    side_weight = np.array([groups.get_group(game_id)["weight"].sum() for game_id in game_ids])
    side_reference_sse = np.array(
        [
            np.dot(
                groups.get_group(game_id)["weight"],
                np.square(
                    groups.get_group(game_id)["target"]
                    - groups.get_group(game_id)["baseline_prediction"]
                ),
            )
            for game_id in game_ids
        ]
    )
    side_candidate_sse = np.array(
        [
            np.dot(
                groups.get_group(game_id)["weight"],
                np.square(
                    groups.get_group(game_id)["target"]
                    - groups.get_group(game_id)["candidate_prediction"]
                ),
            )
            for game_id in game_ids
        ]
    )
    game_frame = game_frame.set_index("game_id").loc[game_ids]
    margin_reference_sse = np.square(
        game_frame["actual_margin"].to_numpy(dtype=float)
        - game_frame["reference_margin"].to_numpy(dtype=float)
    )
    margin_candidate_sse = np.square(
        game_frame["actual_margin"].to_numpy(dtype=float)
        - game_frame["candidate_margin"].to_numpy(dtype=float)
    )
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(game_ids), size=(draws, len(game_ids)))
    sampled_weights = side_weight[samples].sum(axis=1)
    side_differences = np.sqrt(side_candidate_sse[samples].sum(axis=1) / sampled_weights) - np.sqrt(
        side_reference_sse[samples].sum(axis=1) / sampled_weights
    )
    game_differences = np.sqrt(margin_candidate_sse[samples].mean(axis=1)) - np.sqrt(
        margin_reference_sse[samples].mean(axis=1)
    )
    point = usage_response_metrics(rows).set_index("metric")["candidate_minus_reference"]
    return pd.DataFrame(
        [
            _bootstrap_row("side_scoring_rmse", float(point["side_scoring_rmse"]), side_differences, draws, seed),
            _bootstrap_row("game_margin_rmse", float(point["game_margin_rmse"]), game_differences, draws, seed),
        ]
    )


def run_usage_response_screen(
    *,
    constrained_run_dir: Path | str = DEFAULT_CONSTRAINED_RUN_DIR,
    production_run_dir: Path | str = DEFAULT_PRODUCTION_RUN_DIR,
    design_cache_path: Path | str = DEFAULT_DESIGN_CACHE_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    bootstrap_draws: int = 2_000,
) -> UsageResponseRun:
    """Select globally shared O/D usage response slopes and freeze the replay."""

    print("[usage-response] loading production and constrained O/D states", flush=True)
    production = _load_production_state(Path(production_run_dir))
    states: dict[str, ConstrainedSplitState] = joblib.load(
        Path(constrained_run_dir) / "season_states.joblib"
    )
    designs: dict[str, SplitNailDesign] = joblib.load(Path(design_cache_path))
    all_seasons = _seasons_through(FROZEN_TARGET_SEASONS[-1])
    expected_designs = set(all_seasons)
    if set(designs) != expected_designs:
        raise ValueError("Constrained Split design cache does not cover the required seasons")
    records: dict[str, pd.DataFrame] = {}
    state_frames: list[pd.DataFrame] = []
    shock_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    for index, season in enumerate(all_seasons, start=1):
        print(f"[usage-response] pregame state and baseline {season} ({index}/{len(all_seasons)})", flush=True)
        player_states = build_pregame_usage_states(season, curated_dir=curated_dir)
        state_frames.append(player_states)
        if season == FIRST_SEASON:
            continue
        design = designs[season]
        stints = _design_stints(design, season)
        shocks = lineup_usage_shock_features(stints, player_states)
        shocks.insert(0, "season", season)
        shock_frames.append(shocks)
        response_features = build_usage_response_features(design, shocks)
        baseline = _constrained_baseline_prediction(
            season,
            design=design,
            states=states,
            production=production,
        )
        rows = pd.DataFrame(
            {
                "season": season,
                "game_id": design.game_ids.astype(str),
                "home_offense": design.home_offense.astype(bool),
                "target": design.target,
                "weight": design.weights,
                "baseline_prediction": baseline,
                "own_usage_shock_sum": shocks["own_usage_shock_sum"].to_numpy(dtype=float),
                "opponent_usage_shock_sum": shocks["opponent_usage_shock_sum"].to_numpy(
                    dtype=float
                ),
            }
        )
        records[season] = rows
        coverage_rows.append(
            {
                "season": season,
                "player_game_states": len(player_states),
                "distinct_players": player_states["player_id"].nunique(),
                "side_scoring_rows": len(shocks),
                "side_rows": len(rows),
                "side_possessions": float(rows["weight"].sum()),
                "usage_shock_sd": float(player_states["usage_shock_pct"].std(ddof=0)),
            }
        )

    selection_rows: list[dict[str, object]] = []
    for regularization in RESPONSE_REGULARIZATION_GRID:
        for target_season in DEVELOPMENT_TARGET_SEASONS:
            source = _source_response_rows(records, target_season)
            target = records[target_season]
            model = fit_usage_response(
                source.loc[:, ["own_usage_shock_sum", "opponent_usage_shock_sum"]].to_numpy(
                    dtype=float
                )
                * np.array([1.0, -1.0]),
                source["target"].to_numpy(dtype=float)
                - source["baseline_prediction"].to_numpy(dtype=float),
                source["weight"].to_numpy(dtype=float),
                regularization=regularization,
            )
            prediction = target["baseline_prediction"].to_numpy(dtype=float) + model.predict_response(
                target.loc[:, ["own_usage_shock_sum", "opponent_usage_shock_sum"]].to_numpy(
                    dtype=float
                )
                * np.array([1.0, -1.0])
            )
            scored = target.assign(candidate_prediction=prediction)
            metrics = usage_response_metrics(scored)
            for metric in metrics.itertuples(index=False):
                selection_rows.append(
                    {
                        "regularization": regularization,
                        "target_season": target_season,
                        "metric": metric.metric,
                        "reference_value": metric.reference_value,
                        "candidate_value": metric.candidate_value,
                        "candidate_minus_reference": metric.candidate_minus_reference,
                        "offense_coefficient": model.offense_coefficient,
                        "defense_coefficient": model.defense_coefficient,
                        "total_coefficient": model.total_coefficient,
                    }
                )
    selection = pd.DataFrame(selection_rows)
    summary = (
        selection.groupby(["regularization", "metric"], as_index=False)
        .agg(
            reference_value=("reference_value", "mean"),
            candidate_value=("candidate_value", "mean"),
            candidate_minus_reference=("candidate_minus_reference", "mean"),
            median_candidate_minus_reference=("candidate_minus_reference", "median"),
        )
        .sort_values(["metric", "candidate_value", "regularization"], kind="stable")
        .reset_index(drop=True)
    )
    side_selection = summary.loc[summary["metric"].eq("side_scoring_rmse")].copy()
    selected_regularization = float(side_selection.iloc[0]["regularization"])
    selected_interior = selected_regularization not in {
        RESPONSE_REGULARIZATION_GRID[0],
        RESPONSE_REGULARIZATION_GRID[-1],
    }
    print(
        "[usage-response] selected development regularization "
        f"lambda={selected_regularization:g}; interior={selected_interior}",
        flush=True,
    )

    frozen_rows: list[pd.DataFrame] = []
    frozen_metrics: list[pd.DataFrame] = []
    bootstrap_rows: list[pd.DataFrame] = []
    for frozen_season in FROZEN_TARGET_SEASONS:
        print(f"[usage-response] frozen replay {frozen_season}", flush=True)
        source = _source_response_rows(records, frozen_season)
        model = fit_usage_response(
            source.loc[:, ["own_usage_shock_sum", "opponent_usage_shock_sum"]].to_numpy(
                dtype=float
            )
            * np.array([1.0, -1.0]),
            source["target"].to_numpy(dtype=float)
            - source["baseline_prediction"].to_numpy(dtype=float),
            source["weight"].to_numpy(dtype=float),
            regularization=selected_regularization,
        )
        target = records[frozen_season].copy()
        target_features = target.loc[
            :, ["own_usage_shock_sum", "opponent_usage_shock_sum"]
        ].to_numpy(dtype=float) * np.array([1.0, -1.0])
        target["candidate_prediction"] = target["baseline_prediction"] + model.predict_response(
            target_features
        )
        target["response_prediction"] = target["candidate_prediction"] - target[
            "baseline_prediction"
        ]
        frozen_rows.append(target.assign(regularization=selected_regularization))
        metrics = usage_response_metrics(target).assign(
            season=frozen_season,
            regularization=selected_regularization,
            offense_coefficient=model.offense_coefficient,
            defense_coefficient=model.defense_coefficient,
            total_coefficient=model.total_coefficient,
        )
        frozen_metrics.append(metrics)
        bootstrap_rows.append(
            paired_usage_response_bootstrap(target, draws=bootstrap_draws).assign(
                scope=frozen_season
            )
        )
    frozen = pd.concat(frozen_rows, ignore_index=True)
    frozen_metrics_frame = pd.concat(frozen_metrics, ignore_index=True)
    pooled_bootstrap = paired_usage_response_bootstrap(frozen, draws=bootstrap_draws).assign(
        scope="pooled"
    )
    bootstrap = pd.concat([pooled_bootstrap, *bootstrap_rows], ignore_index=True)
    frozen_games = _game_margin_frame(frozen, "baseline_prediction", "candidate_prediction")
    curve_source = _source_response_rows(records, FROZEN_TARGET_SEASONS[0])
    curve_model = fit_usage_response(
        curve_source.loc[:, ["own_usage_shock_sum", "opponent_usage_shock_sum"]].to_numpy(
            dtype=float
        )
        * np.array([1.0, -1.0]),
        curve_source["target"].to_numpy(dtype=float)
        - curve_source["baseline_prediction"].to_numpy(dtype=float),
        curve_source["weight"].to_numpy(dtype=float),
        regularization=selected_regularization,
    )
    state_history = pd.concat(state_frames, ignore_index=True)
    source_state_history = state_history.loc[
        state_history["season"].astype(str).lt(FROZEN_TARGET_SEASONS[0])
        & state_history["minutes"].gt(0.0)
    ]
    shock_sd = float(source_state_history["usage_shock_pct"].std(ddof=0))
    response_curve = _response_curve(curve_model, shock_sd)
    run = _write_usage_response_run(
        state_history=state_history,
        lineup_shocks=pd.concat(shock_frames, ignore_index=True),
        coverage=pd.DataFrame(coverage_rows),
        selection=selection,
        selection_summary=summary,
        frozen_side_predictions=frozen,
        frozen_game_predictions=frozen_games,
        frozen_metrics=frozen_metrics_frame,
        bootstrap=bootstrap,
        response_curve=response_curve,
        selected_regularization=selected_regularization,
        selected_interior=selected_interior,
        constrained_run_dir=Path(constrained_run_dir),
        production_run_dir=Path(production_run_dir),
        artifacts_dir=Path(artifacts_dir),
        bootstrap_draws=bootstrap_draws,
    )
    print(f"[usage-response] complete: {run.run_dir}", flush=True)
    return run


def _read_usage_games(season: str, *, curated_dir: Path | str) -> pd.DataFrame:
    path = Path(curated_dir) / "players" / str(season) / "regular"
    if not path.is_dir():
        raise FileNotFoundError(f"Player boxscore partition is missing: {path}")
    return game_usage_from_player_boxscores(pd.read_parquet(path))


def _season_usage_summary(games: pd.DataFrame) -> pd.DataFrame:
    played = games.loc[games["usage_opportunities"].gt(0.0)].copy()
    return played.groupby("player_id", as_index=False, sort=True).agg(
        usage_events=("usage_events", "sum"),
        usage_opportunities=("usage_opportunities", "sum"),
    )


def _normalized_usage_prior(
    prior_summary: pd.DataFrame | None,
    *,
    pseudo_opportunities: float,
    default_usage_rate: float,
) -> pd.DataFrame:
    if prior_summary is None or prior_summary.empty:
        output = pd.DataFrame(
            {
                "player_id": pd.Series(dtype="int64"),
                "preseason_usage_rate": pd.Series(dtype=float),
            }
        )
        output.attrs["league_usage_rate"] = default_usage_rate
        return output
    required = {"player_id", "usage_events", "usage_opportunities"}
    missing = sorted(required - set(prior_summary))
    if missing:
        raise ValueError(f"Prior usage summary missing columns: {missing}")
    frame = prior_summary.loc[:, list(required)].copy()
    for column in ("usage_events", "usage_opportunities"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if frame["usage_events"].lt(0.0).any() or frame["usage_opportunities"].le(0.0).any():
        raise ValueError("Prior usage summary has invalid events or opportunities")
    league_rate = float(frame["usage_events"].sum() / frame["usage_opportunities"].sum())
    frame["preseason_usage_rate"] = (
        frame["usage_events"] + pseudo_opportunities * league_rate
    ) / (frame["usage_opportunities"] + pseudo_opportunities)
    output = frame.loc[:, ["player_id", "preseason_usage_rate"]]
    output.attrs["league_usage_rate"] = league_rate
    return output


def _constrained_baseline_prediction(
    season: str,
    *,
    design: SplitNailDesign,
    states: dict[str, ConstrainedSplitState],
    production: ProductionState,
) -> np.ndarray:
    source = _previous_season(season)
    if source not in states:
        raise ValueError(f"Constrained O/D source state is missing for {season}")
    state = states[source]
    mean = _forecast_mean_coordinates(design, season, production)
    specialization = _state_specialization_for_design(design, state)
    raw = _raw_coefficients(design, mean, specialization)
    return np.asarray(design.features @ raw, dtype=float).reshape(-1) + state.intercept


def _design_stints(design: SplitNailDesign, season: str) -> pd.DataFrame:
    """Load exactly the stint rows represented by a cached split design."""

    from nba_lineup_model.modeling.stints import read_rapm_stints

    stints = read_rapm_stints(season)
    expected = set(design.game_ids.astype(str))
    stints = stints.loc[stints["game_id"].astype(str).isin(expected)].copy()
    scoring_rows = int(
        stints["home_offensive_possessions"].gt(0.0).sum()
        + stints["away_offensive_possessions"].gt(0.0).sum()
    )
    if scoring_rows != len(design.target):
        raise ValueError(f"Split design and source stints differ for {season}")
    return stints


def _source_response_rows(records: dict[str, pd.DataFrame], target_season: str) -> pd.DataFrame:
    source = [
        rows
        for season, rows in records.items()
        if int(season[:4]) < int(target_season[:4])
    ]
    if not source:
        raise ValueError(f"Usage response has no source rows before {target_season}")
    return pd.concat(source, ignore_index=True)


def _response_curve(model: UsageResponseModel, shock_sd: float) -> pd.DataFrame:
    if shock_sd <= 0.0 or not np.isfinite(shock_sd):
        raise ValueError("Usage shock standard deviation must be positive")
    rows = []
    for standard_deviations in (-1.0, 0.0, 1.0, 2.0):
        shock = standard_deviations * shock_sd
        offense = shock * model.offense_coefficient
        defense = shock * model.defense_coefficient
        rows.append(
            {
                "usage_shock_standard_deviations": standard_deviations,
                "usage_shock_percentage_points": shock,
                "offense_response": offense,
                "defense_response": defense,
                "total_response": offense + defense,
            }
        )
    return pd.DataFrame(rows)


def _game_margin_frame(
    rows: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
) -> pd.DataFrame:
    frame = rows.copy()
    sign = np.where(frame["home_offense"].to_numpy(dtype=bool), 1.0, -1.0)
    frame["actual_points_signed"] = sign * frame["target"] * frame["weight"] / 100.0
    frame["reference_points_signed"] = (
        sign * frame[reference_column] * frame["weight"] / 100.0
    )
    frame["candidate_points_signed"] = (
        sign * frame[candidate_column] * frame["weight"] / 100.0
    )
    return (
        frame.groupby("game_id", as_index=False, sort=True)
        .agg(
            actual_margin=("actual_points_signed", "sum"),
            reference_margin=("reference_points_signed", "sum"),
            candidate_margin=("candidate_points_signed", "sum"),
        )
        .sort_values("game_id", kind="stable")
        .reset_index(drop=True)
    )


def _weighted_rmse(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.average(np.square(target - prediction), weights=weights)))


def _bootstrap_row(
    metric: str,
    point: float,
    differences: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, object]:
    return {
        "metric": metric,
        "candidate_minus_reference": point,
        "ci_lower": float(np.quantile(differences, 0.025)),
        "ci_upper": float(np.quantile(differences, 0.975)),
        "probability_candidate_better": float(np.mean(differences < 0.0)),
        "bootstrap_unit": "complete_game",
        "bootstrap_draws": draws,
        "random_seed": seed,
    }


def _validate_response_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("Usage response features must be a non-empty two-column matrix")
    if not np.isfinite(values).all():
        raise ValueError("Usage response features must be finite")
    return values


def _validated_weights(weights: np.ndarray, expected_length: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.shape != (expected_length,) or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("Usage response weights must be finite, positive, and row-aligned")
    return values


def _write_usage_response_run(
    *,
    state_history: pd.DataFrame,
    lineup_shocks: pd.DataFrame,
    coverage: pd.DataFrame,
    selection: pd.DataFrame,
    selection_summary: pd.DataFrame,
    frozen_side_predictions: pd.DataFrame,
    frozen_game_predictions: pd.DataFrame,
    frozen_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    response_curve: pd.DataFrame,
    selected_regularization: float,
    selected_interior: bool,
    constrained_run_dir: Path,
    production_run_dir: Path,
    artifacts_dir: Path,
    bootstrap_draws: int,
) -> UsageResponseRun:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{RUN_PREFIX}-{timestamp}-{uuid4().hex[:8]}"
    root = artifacts_dir / MODEL_NAME
    staging = root / f".{run_id}.tmp"
    run_dir = root / run_id
    staging.mkdir(parents=True, exist_ok=False)
    try:
        state_history.to_parquet(staging / "pregame_usage_states.parquet", index=False)
        lineup_shocks.to_parquet(staging / "lineup_usage_shocks.parquet", index=False)
        coverage.to_parquet(staging / "coverage.parquet", index=False)
        selection.to_parquet(staging / "selection_by_season.parquet", index=False)
        selection_summary.to_parquet(staging / "selection_summary.parquet", index=False)
        frozen_side_predictions.to_parquet(staging / "frozen_side_predictions.parquet", index=False)
        frozen_game_predictions.to_parquet(staging / "frozen_game_predictions.parquet", index=False)
        frozen_metrics.to_parquet(staging / "frozen_metrics.parquet", index=False)
        bootstrap.to_parquet(staging / "paired_bootstrap_metrics.parquet", index=False)
        response_curve.to_parquet(staging / "response_curve.parquet", index=False)
        metadata = {
            "model": MODEL_NAME,
            "contract": "pregame_usage_state_fixed_constrained_split_residual_response",
            "production_run_dir": str(production_run_dir),
            "constrained_split_run_dir": str(constrained_run_dir),
            "usage_pseudo_opportunities": USAGE_PSEUDO_OPPORTUNITIES,
            "development_target_seasons": list(DEVELOPMENT_TARGET_SEASONS),
            "frozen_target_seasons": list(FROZEN_TARGET_SEASONS),
            "regularization_grid": list(RESPONSE_REGULARIZATION_GRID),
            "selected_regularization": selected_regularization,
            "selected_regularization_is_interior": selected_interior,
            "bootstrap_draws": bootstrap_draws,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest = {"run_id": run_id, "files": sorted(path.name for path in staging.iterdir())}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.replace(run_dir)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    except Exception:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return UsageResponseRun(
        run_dir=run_dir,
        run_id=run_id,
        selected_regularization=selected_regularization,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pregame O/D usage-response screen")
    parser.add_argument("--log-path")
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    args = parser.parse_args()
    if args.log_path:
        path = Path(args.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            stdout = sys.stdout
            sys.stdout = _Tee(stdout, handle)  # type: ignore[assignment]
            try:
                run = run_usage_response_screen(bootstrap_draws=args.bootstrap_draws)
                print(f"Usage response screen: run={run.run_dir}", flush=True)
            finally:
                sys.stdout = stdout
        return
    run = run_usage_response_screen(bootstrap_draws=args.bootstrap_draws)
    print(f"Usage response screen: run={run.run_dir}")


if __name__ == "__main__":
    main()
