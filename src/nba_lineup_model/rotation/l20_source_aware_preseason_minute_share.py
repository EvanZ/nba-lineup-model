"""Source-aware preseason first-20-game minute-share allocation.

L20-NAIL-MSP v0.3 preserves the existing opening-roster target but distinguishes
continuous players, injury-gap returners, and true NBA cold starts.  It learns
one roster-relative softmax allocation from strictly preseason inputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from nba_lineup_model.rotation.l0_opening_minute_share_persistence import (
    read_preseason_roster_candidates,
)
from nba_lineup_model.rotation.l1_minute_share_persistence import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    read_regular_game_minutes,
)
from nba_lineup_model.rotation.l20_nail_minute_share import (
    RatingAwareSeasonInputs,
    evaluate_l20_nail_minute_share,
    select_l20_nail_parameters,
)
from nba_lineup_model.rotation.l20_preseason_minute_share_persistence import (
    DEFAULT_TEAM_GAMES,
    _assert_distribution,
    _first_n_team_game_allocations,
    _validate_candidates,
    _validate_game_minutes,
    previous_season,
    summarize_l20_metrics,
)
from nba_lineup_model.rotation.l20_preseason_nail_forecast_minute_share import (
    DEFAULT_MODEL_RUN_DIR,
    DEFAULT_PANEL_PATH,
    build_preseason_nail_forecast_ratings,
)

MODEL_NAME = "l20_source_aware_preseason_minute_share"
MODEL_VERSION = "v0.3"
DEFAULT_SEASONS = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2026))
DEFAULT_ROLE_SHRINKAGE_GAMES_GRID = (1.0, 2.0, 3.0, 4.0, 5.0, 15.0, 30.0)
DEFAULT_GAP_DECAY_GRID = (0.25, 0.5, 0.75)
DEFAULT_ALPHA_GRID = (0.01, 0.02, 0.05, 0.1, 1.0, 10.0)
DEFAULT_HISTORY_SEASONS = 5
REGULATION_TEAM_MINUTES = 240.0
_FEATURE_COLUMNS = (
    "log_total_minutes",
    "shrunken_mpg",
    "availability_rate",
    "shrunken_start_rate",
    "preseason_nail",
    "is_gap_returner",
    "gap_seasons",
    "is_cold_start",
    "cold_draft_capital",
    "cold_undrafted",
    "cold_age_offset",
    "cold_guard",
    "cold_forward",
    "cold_center",
)


@dataclass(frozen=True)
class SourceAwareConfig:
    """Hyperparameters selected only from earlier target seasons."""

    role_shrinkage_games: float
    gap_decay: float
    alpha: float


@dataclass(frozen=True)
class SourceAwareSeasonInputs:
    """All pre-target role evidence plus a first-20-game realized target."""

    season: str
    target_game_minutes: pd.DataFrame
    opening_candidates: pd.DataFrame
    role_state: pd.DataFrame
    preseason_nail_ratings: pd.Series


@dataclass(frozen=True)
class FittedSourceAwareMinuteModel:
    """Standardized softmax model used for one frozen target season."""

    feature_columns: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    config: SourceAwareConfig
    replacement_token: bool = False
    replacement_intercept: float = 0.0


@dataclass(frozen=True)
class L20SourceAwareMinuteShareRun:
    """Immutable location and identity for one rolling L20 v0.3 evaluation."""

    run_dir: Path
    run_id: str


def build_l20_source_aware_season_inputs(
    season: str,
    *,
    panel: pd.DataFrame,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
    history_seasons: int = DEFAULT_HISTORY_SEASONS,
) -> SourceAwareSeasonInputs:
    """Build a target-season input without reading target-season player outcomes."""

    if history_seasons <= 0:
        raise ValueError("Source-aware role history must be positive")
    candidates = read_preseason_roster_candidates(season, curated_dir=curated_dir)
    prior_season = previous_season(season)
    immediate = _season_role_summary(
        prior_season,
        panel=panel,
        curated_dir=curated_dir,
    )
    history: list[pd.DataFrame] = []
    source = previous_season(prior_season)
    for _ in range(history_seasons):
        try:
            history.append(_season_role_summary(source, panel=panel, curated_dir=curated_dir))
        except FileNotFoundError:
            break
        source = previous_season(source)
    nail = build_preseason_nail_forecast_ratings(
        season,
        candidates,
        panel=panel,
        run_dir=run_dir,
    )
    return SourceAwareSeasonInputs(
        season=season,
        target_game_minutes=read_regular_game_minutes(season, curated_dir=curated_dir),
        opening_candidates=candidates,
        role_state=_build_role_state(
            season,
            candidates=candidates,
            immediate=immediate,
            history=history,
            panel=panel,
            preseason_nail=nail,
        ),
        preseason_nail_ratings=nail,
    )


def build_current_source_aware_role_state(
    season: str,
    *,
    roster: pd.DataFrame,
    panel: pd.DataFrame,
    preseason_nail: pd.Series,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    history_seasons: int = DEFAULT_HISTORY_SEASONS,
) -> pd.DataFrame:
    """Build a roster-snapshot role state without reading target-season outcomes.

    Unlike the historical evaluator, this accepts the current published roster
    directly. ``panel`` must include roster-only target biographies for the
    requested season, but no target-season box score, minutes, or RAPM values.
    """

    required = {"team_id", "team_abbreviation", "player_id", "player_name"}
    missing = sorted(required - set(roster))
    if missing:
        raise ValueError(f"Current roster is missing source-aware columns: {missing}")
    if history_seasons <= 0:
        raise ValueError("Source-aware role history must be positive")
    candidates = roster.loc[
        :, ["team_id", "team_abbreviation", "player_id", "player_name"]
    ].rename(columns={"team_abbreviation": "team"})
    candidates["player_id"] = pd.to_numeric(candidates["player_id"], errors="raise").astype(int)
    if candidates["player_id"].duplicated().any():
        raise ValueError("Current roster contains duplicate player IDs")
    prior_season = previous_season(season)
    immediate = _season_role_summary(prior_season, panel=panel, curated_dir=curated_dir)
    history: list[pd.DataFrame] = []
    source = previous_season(prior_season)
    for _ in range(history_seasons):
        try:
            history.append(_season_role_summary(source, panel=panel, curated_dir=curated_dir))
        except FileNotFoundError:
            break
        source = previous_season(source)
    return _build_role_state(
        season,
        candidates=candidates,
        immediate=immediate,
        history=history,
        panel=panel,
        preseason_nail=preseason_nail,
    )


def predict_source_aware_minute_shares(
    role_state: pd.DataFrame,
    *,
    model: FittedSourceAwareMinuteModel,
) -> pd.DataFrame:
    """Predict one normalized opening allocation for each rostered team."""

    features = build_source_aware_features(role_state, config=model.config)
    output = features.copy()
    shares = pd.Series(0.0, index=output.index, dtype=float)
    for _, group in output.groupby("team_id", sort=True):
        predicted = _predict_shares(group, model)
        shares.loc[group.index] = group["player_id"].map(predicted).astype(float)
    output["baseline_minute_share"] = shares
    output["baseline_minutes_per_game"] = REGULATION_TEAM_MINUTES * shares
    return output


def build_source_aware_features(
    role_state: pd.DataFrame,
    *,
    config: SourceAwareConfig,
) -> pd.DataFrame:
    """Apply source-specific evidence weighting without pulling toward star norms."""

    if config.role_shrinkage_games < 0.0:
        raise ValueError("Role shrinkage games must be non-negative")
    if not 0.0 <= config.gap_decay <= 1.0:
        raise ValueError("Gap decay must be between zero and one")
    output = role_state.copy()
    state = output["player_state"].astype(str)
    gap_years = output["gap_seasons"].astype(float)
    evidence_weight = np.where(
        state.eq("continuous"),
        1.0,
        np.where(state.eq("gap_returner"), np.power(config.gap_decay, gap_years), 0.0),
    )
    games = output["source_games"].astype(float).to_numpy()
    minutes = output["source_minutes"].astype(float).to_numpy()
    starts = output["source_starts"].astype(float).to_numpy()
    reliability = games / (games + config.role_shrinkage_games)
    reliability = np.where(games > 0.0, reliability, 0.0)
    output["log_total_minutes"] = evidence_weight * np.log1p(minutes)
    output["shrunken_mpg"] = evidence_weight * (minutes / np.maximum(games, 1.0)) * reliability
    output["availability_rate"] = evidence_weight * games / 82.0
    output["shrunken_start_rate"] = evidence_weight * starts / (
        games + config.role_shrinkage_games
    )
    output["preseason_nail"] = output["preseason_nail"].astype(float)
    output["is_gap_returner"] = state.eq("gap_returner").astype(float)
    output["gap_seasons"] = np.where(state.eq("gap_returner"), gap_years, 0.0)
    output["is_cold_start"] = state.eq("cold_start").astype(float)
    output["cold_draft_capital"] = output["is_cold_start"] * output["draft_capital"]
    output["cold_undrafted"] = output["is_cold_start"] * output["is_undrafted"].astype(float)
    output["cold_age_offset"] = output["is_cold_start"] * output["age_offset"]
    output["cold_guard"] = output["is_cold_start"] * output["is_guard"]
    output["cold_forward"] = output["is_cold_start"] * output["is_forward"]
    output["cold_center"] = output["is_cold_start"] * output["is_center"]
    output.loc[:, list(_FEATURE_COLUMNS)] = output.loc[:, list(_FEATURE_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return output


def fit_source_aware_minute_model(
    inputs: list[SourceAwareSeasonInputs],
    *,
    config: SourceAwareConfig,
    team_games: int = DEFAULT_TEAM_GAMES,
    replacement_token: bool = False,
) -> FittedSourceAwareMinuteModel:
    """Fit ridge-regularized cross-entropy over one opening-roster allocation.

    When ``replacement_token`` is enabled, every team allocation also has one
    unpenalized logit representing first-window minutes consumed by players who
    were not on the transaction-derived opening roster.
    """

    if not inputs:
        raise ValueError("Source-aware minute model requires at least one source season")
    raw_features, targets, team_slices = _training_problem(
        inputs,
        config=config,
        team_games=team_games,
        replacement_token=replacement_token,
    )
    feature_mean = raw_features.mean(axis=0)
    feature_scale = raw_features.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    features = (raw_features - feature_mean) / feature_scale
    penalty = float(config.alpha)

    coefficient_count = features.shape[1]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = parameters[:coefficient_count]
        replacement_intercept = (
            float(parameters[-1]) if replacement_token else 0.0
        )
        value = 0.5 * penalty * float(np.dot(coefficients, coefficients))
        gradient = np.zeros_like(parameters)
        gradient[:coefficient_count] = penalty * coefficients
        for (start, end), target in zip(team_slices, targets, strict=True):
            x = features[start:end]
            logits = x @ coefficients
            if replacement_token:
                logits = np.append(logits, replacement_intercept)
            probabilities = np.exp(logits - logsumexp(logits))
            value -= float(np.dot(target, logits - logsumexp(logits)))
            gradient[:coefficient_count] += x.T @ (probabilities[: len(x)] - target[: len(x)])
            if replacement_token:
                # The replacement token is an intercept, conventionally left
                # unpenalized so it estimates the pooled outside-roster mass.
                gradient[-1] += probabilities[-1] - target[-1]
        return value, gradient

    result = minimize(
        fun=lambda parameters: objective(parameters)[0],
        x0=np.zeros(coefficient_count + int(replacement_token), dtype=float),
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Source-aware minute fit failed: {result.message}")
    return FittedSourceAwareMinuteModel(
        feature_columns=_FEATURE_COLUMNS,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=np.asarray(result.x[:coefficient_count], dtype=float),
        config=config,
        replacement_token=replacement_token,
        replacement_intercept=float(result.x[-1]) if replacement_token else 0.0,
    )


def evaluate_source_aware_minute_share(
    inputs: SourceAwareSeasonInputs,
    *,
    model: FittedSourceAwareMinuteModel,
    team_games: int = DEFAULT_TEAM_GAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one frozen source-aware allocation against first-20-game shares."""

    _validate_game_minutes(inputs.target_game_minutes, label="target")
    _validate_candidates(inputs.opening_candidates)
    features = build_source_aware_features(inputs.role_state, config=model.config)
    targets = _first_n_team_game_allocations(inputs.target_game_minutes, team_games=team_games)
    feature_groups = {
        int(team_id): frame.set_index("player_id", drop=False)
        for team_id, frame in features.groupby("team_id", sort=True)
    }
    records: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for target in targets:
        candidates = feature_groups.get(int(target.team_id))
        if candidates is None or candidates.empty:
            raise ValueError(f"Role state has no candidates for team {target.team_id}")
        predicted_map, replacement_prediction = _predict_allocation(candidates, model)
        player_ids = sorted(set(predicted_map) | set(target.player_shares))
        player_actual_values = np.asarray(
            [float(target.player_shares.get(player_id, 0.0)) for player_id in player_ids]
        )
        player_predicted_values = np.asarray(
            [float(predicted_map.get(player_id, 0.0)) for player_id in player_ids]
        )
        outside_candidate_actual_share = float(
            player_actual_values[[player_id not in predicted_map for player_id in player_ids]].sum()
        )
        if getattr(model, "replacement_token", False):
            actual_values = np.append(
                [float(target.player_shares.get(player_id, 0.0)) for player_id in predicted_map],
                outside_candidate_actual_share,
            )
            predicted_values = np.append(
                [float(predicted_map[player_id]) for player_id in predicted_map],
                replacement_prediction,
            )
        else:
            actual_values = player_actual_values
            predicted_values = player_predicted_values
        _assert_distribution(actual_values, label=f"actual for {target.team}")
        _assert_distribution(predicted_values, label=f"prediction for {target.team}")
        absolute_error = np.abs(actual_values - predicted_values)
        squared_error = np.square(actual_values - predicted_values)
        zero_probability_active = (player_actual_values > 0.0) & (
            player_predicted_values <= 0.0
        )
        states = {
            int(row.player_id): str(row.player_state)
            for row in candidates.itertuples(index=False)
        }
        names = {
            int(row.player_id): str(row.player_name)
            for row in candidates.itertuples(index=False)
        }
        metrics.append(
            {
                "season": inputs.season,
                "team_id": target.team_id,
                "team": target.team,
                "allocation_total_variation": float(0.5 * absolute_error.sum()),
                "brier_score": float(squared_error.sum()),
                "player_share_mae": float(
                    np.abs(player_actual_values - player_predicted_values).mean()
                ),
                "player_share_mse": float(
                    np.square(player_actual_values - player_predicted_values).mean()
                ),
                "allocation_entity_mae": float(absolute_error.mean()),
                "cross_entropy": float(
                    -(actual_values * np.log(np.maximum(predicted_values, 1e-15))).sum()
                ),
                "has_zero_probability_active_player": bool(zero_probability_active.any()),
                "outside_candidate_actual_share": outside_candidate_actual_share,
                "replacement_token_actual_share": outside_candidate_actual_share,
                "replacement_token_predicted_share": replacement_prediction,
            }
        )
        for player_id, actual_share, predicted_share in zip(
            player_ids, player_actual_values, player_predicted_values, strict=True
        ):
            records.append(
                {
                    "season": inputs.season,
                    "team_id": target.team_id,
                    "team": target.team,
                    "player_id": player_id,
                    "player_name": names.get(player_id, str(player_id)),
                    "player_state": states.get(player_id, "outside_opening_roster"),
                    "actual_minute_share": float(actual_share),
                    "predicted_minute_share": float(predicted_share),
                    "absolute_error": float(abs(actual_share - predicted_share)),
                    "was_opening_candidate": player_id in predicted_map,
                    "was_in_first_n_games": player_id in target.player_shares,
                    "entity_type": "player",
                }
            )
        if getattr(model, "replacement_token", False):
            records.append(
                {
                    "season": inputs.season,
                    "team_id": target.team_id,
                    "team": target.team,
                    "player_id": -1,
                    "player_name": "Replacement token",
                    "player_state": "replacement_token",
                    "actual_minute_share": outside_candidate_actual_share,
                    "predicted_minute_share": replacement_prediction,
                    "absolute_error": float(
                        abs(outside_candidate_actual_share - replacement_prediction)
                    ),
                    "was_opening_candidate": False,
                    "was_in_first_n_games": outside_candidate_actual_share > 0.0,
                    "entity_type": "replacement_token",
                }
            )
    return pd.DataFrame(records), pd.DataFrame(metrics)


def select_source_aware_parameters(
    source_inputs: list[SourceAwareSeasonInputs],
    *,
    role_shrinkage_games_grid: tuple[float, ...] = DEFAULT_ROLE_SHRINKAGE_GAMES_GRID,
    gap_decay_grid: tuple[float, ...] = DEFAULT_GAP_DECAY_GRID,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    team_games: int = DEFAULT_TEAM_GAMES,
    replacement_token: bool = False,
) -> tuple[SourceAwareConfig, pd.DataFrame]:
    """Select parameters by rolling internal season validation.

    A one-season source has no earlier validation split, so it uses the same
    season only to establish an initial hyperparameter choice. All later
    selections validate every source season on a fit using strictly earlier
    sources.
    """

    if not source_inputs:
        raise ValueError("Source-aware parameter selection requires source seasons")
    rows: list[dict[str, float | int]] = []
    for shrinkage in sorted(set(role_shrinkage_games_grid)):
        for decay in sorted(set(gap_decay_grid)):
            for alpha in sorted(set(alpha_grid)):
                config = SourceAwareConfig(shrinkage, decay, alpha)
                metrics: list[pd.DataFrame] = []
                if len(source_inputs) == 1:
                    model = fit_source_aware_minute_model(
                        source_inputs,
                        config=config,
                        team_games=team_games,
                        replacement_token=replacement_token,
                    )
                    metrics.append(
                        evaluate_source_aware_minute_share(
                            source_inputs[0], model=model, team_games=team_games
                        )[1]
                    )
                else:
                    for index in range(1, len(source_inputs)):
                        model = fit_source_aware_minute_model(
                            source_inputs[:index],
                            config=config,
                            team_games=team_games,
                            replacement_token=replacement_token,
                        )
                        metrics.append(
                            evaluate_source_aware_minute_share(
                                source_inputs[index], model=model, team_games=team_games
                            )[1]
                        )
                combined = pd.concat(metrics, ignore_index=True)
                rows.append(
                    {
                        "role_shrinkage_games": shrinkage,
                        "gap_decay": decay,
                        "alpha": alpha,
                        "validation_season_count": len(metrics),
                        "mean_allocation_total_variation": float(
                            combined["allocation_total_variation"].mean()
                        ),
                        "mean_brier_score": float(combined["brier_score"].mean()),
                        "player_share_mae": float(combined["player_share_mae"].mean()),
                    }
                )
    grid = pd.DataFrame(rows).sort_values(
        ["mean_allocation_total_variation", "alpha", "gap_decay", "role_shrinkage_games"],
        kind="stable",
    ).reset_index(drop=True)
    winner = grid.iloc[0]
    return (
        SourceAwareConfig(
            role_shrinkage_games=float(winner["role_shrinkage_games"]),
            gap_decay=float(winner["gap_decay"]),
            alpha=float(winner["alpha"]),
        ),
        grid,
    )


def run_l20_source_aware_preseason_minute_share(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
    role_shrinkage_games_grid: tuple[float, ...] = DEFAULT_ROLE_SHRINKAGE_GAMES_GRID,
    gap_decay_grid: tuple[float, ...] = DEFAULT_GAP_DECAY_GRID,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    history_seasons: int = DEFAULT_HISTORY_SEASONS,
    team_games: int = DEFAULT_TEAM_GAMES,
    replacement_token: bool = False,
) -> L20SourceAwareMinuteShareRun:
    """Run rolling frozen v0.3 evaluation and its v0.2 comparison control."""

    ordered_seasons = tuple(sorted(set(seasons)))
    if len(ordered_seasons) < 2:
        raise ValueError("Source-aware rolling evaluation requires at least two seasons")
    panel = pd.read_parquet(panel_path)
    label = "L20-v0.3+replacement" if replacement_token else "L20-v0.3"
    print(f"[{label}] loading {len(ordered_seasons)} preseason states", flush=True)
    inputs = [
        build_l20_source_aware_season_inputs(
            season,
            panel=panel,
            curated_dir=curated_dir,
            run_dir=run_dir,
            history_seasons=history_seasons,
        )
        for season in ordered_seasons
    ]
    artifact_model_name = (
        f"{MODEL_NAME}_replacement_token" if replacement_token else MODEL_NAME
    )
    artifact_root = Path(artifacts_dir) / artifact_model_name
    run_stem = "l20-source-aware-replacement-token" if replacement_token else "l20-source-aware"
    run_id = f"{run_stem}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    output_dir = artifact_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    grids: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    prediction_outputs: list[pd.DataFrame] = []
    metric_outputs: list[pd.DataFrame] = []
    state_outputs: list[pd.DataFrame] = []
    for index, holdout in enumerate(inputs[1:], start=1):
        source_inputs = inputs[:index]
        print(
            f"[{label}] tuning {holdout.season} from {source_inputs[0].season} "
            f"through {source_inputs[-1].season}",
            flush=True,
        )
        config, grid = select_source_aware_parameters(
            source_inputs,
            role_shrinkage_games_grid=role_shrinkage_games_grid,
            gap_decay_grid=gap_decay_grid,
            alpha_grid=alpha_grid,
            team_games=team_games,
            replacement_token=replacement_token,
        )
        model = fit_source_aware_minute_model(
            source_inputs,
            config=config,
            team_games=team_games,
            replacement_token=replacement_token,
        )
        candidate_predictions, candidate_metrics = evaluate_source_aware_minute_share(
            holdout, model=model, team_games=team_games
        )
        base_inputs = [
            RatingAwareSeasonInputs(
                season=source.season,
                prior_game_minutes=read_regular_game_minutes(
                    previous_season(source.season), curated_dir=curated_dir
                ),
                target_game_minutes=source.target_game_minutes,
                opening_candidates=source.opening_candidates,
                completed_nail_ratings=source.preseason_nail_ratings,
            )
            for source in source_inputs
        ]
        baseline_epsilon, baseline_weight, _baseline_grid = select_l20_nail_parameters(
            base_inputs,
            team_games=team_games,
        )
        baseline_predictions, baseline_metrics = evaluate_l20_nail_minute_share(
            RatingAwareSeasonInputs(
                season=holdout.season,
                prior_game_minutes=read_regular_game_minutes(
                    previous_season(holdout.season), curated_dir=curated_dir
                ),
                target_game_minutes=holdout.target_game_minutes,
                opening_candidates=holdout.opening_candidates,
                completed_nail_ratings=holdout.preseason_nail_ratings,
            ),
            epsilon=baseline_epsilon,
            nail_weight=baseline_weight,
            team_games=team_games,
        )
        grid["holdout_season"] = holdout.season
        grids.append(grid)
        selections.append(
            {
                "holdout_season": holdout.season,
                "source_first_season": source_inputs[0].season,
                "source_last_season": source_inputs[-1].season,
                "source_season_count": len(source_inputs),
                "role_shrinkage_games": config.role_shrinkage_games,
                "gap_decay": config.gap_decay,
                "alpha": config.alpha,
                "baseline_epsilon": baseline_epsilon,
                "baseline_nail_weight": baseline_weight,
            }
        )
        print(
            f"[{label}] selected q={config.role_shrinkage_games:.0f}, "
            f"gap={config.gap_decay:.2f}, alpha={config.alpha:.2f} for {holdout.season}",
            flush=True,
        )
        candidate_variant = (
            "source_aware_v03_replacement_token"
            if replacement_token
            else "source_aware_v03"
        )
        candidate_predictions["variant"] = candidate_variant
        candidate_metrics["variant"] = candidate_variant
        baseline_predictions["variant"] = "l20_nail_msp_v02"
        baseline_metrics["variant"] = "l20_nail_msp_v02"
        prediction_outputs.extend((candidate_predictions, baseline_predictions))
        metric_outputs.extend((candidate_metrics, baseline_metrics))
        state_outputs.append(_summarize_state_metrics(candidate_predictions))
    grids_frame = pd.concat(grids, ignore_index=True)
    selection_frame = pd.DataFrame(selections)
    prediction_frame = pd.concat(prediction_outputs, ignore_index=True)
    metrics_frame = pd.concat(metric_outputs, ignore_index=True)
    summary = pd.concat(
        [
            summarize_l20_metrics(group, season=str(season)).assign(variant=variant)
            for (season, variant), group in metrics_frame.groupby(["season", "variant"], sort=True)
        ],
        ignore_index=True,
    )
    grids_frame.to_parquet(output_dir / "parameter_grid.parquet", index=False)
    selection_frame.to_parquet(output_dir / "selected_parameters.parquet", index=False)
    prediction_frame.to_parquet(output_dir / "predictions.parquet", index=False)
    metrics_frame.to_parquet(output_dir / "team_metrics.parquet", index=False)
    summary.to_parquet(output_dir / "metrics.parquet", index=False)
    pd.concat(state_outputs, ignore_index=True).to_parquet(
        output_dir / "player_state_metrics.parquet", index=False
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "seasons": list(ordered_seasons),
                "team_games": team_games,
                "role_shrinkage_games_grid": list(role_shrinkage_games_grid),
                "gap_decay_grid": list(gap_decay_grid),
                "alpha_grid": list(alpha_grid),
                "history_seasons": history_seasons,
                "replacement_token": replacement_token,
                "contract": (
                    "A roster-relative ridge-softmax allocation uses total minutes, "
                    "evidence-shrunken MPG/GP/GS, exact preseason NAIL, decayed "
                    "last-observed gap-returner role evidence, and a conservative "
                    "draft/age/position cold-start pathway."
                    + (
                        " One unpenalized team-level replacement token absorbs "
                        "first-window minute share allocated to players outside the "
                        "opening roster."
                        if replacement_token
                        else ""
                    )
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return L20SourceAwareMinuteShareRun(run_dir=output_dir, run_id=run_id)


def fit_l20_source_aware_production_model(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
    history_seasons: int = DEFAULT_HISTORY_SEASONS,
    team_games: int = DEFAULT_TEAM_GAMES,
) -> L20SourceAwareMinuteShareRun:
    """Select and persist one next-season v0.3 model from all completed seasons."""

    ordered_seasons = tuple(sorted(set(seasons)))
    panel = pd.read_parquet(panel_path)
    print(f"[L20-v0.3] loading {len(ordered_seasons)} production source states", flush=True)
    inputs = [
        build_l20_source_aware_season_inputs(
            season,
            panel=panel,
            curated_dir=curated_dir,
            run_dir=run_dir,
            history_seasons=history_seasons,
        )
        for season in ordered_seasons
    ]
    config, grid = select_source_aware_parameters(inputs, team_games=team_games)
    model = fit_source_aware_minute_model(inputs, config=config, team_games=team_games)
    run_id = (
        f"l20-source-aware-production-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:7]}"
    )
    output_dir = Path(artifacts_dir) / MODEL_NAME / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    grid.to_parquet(output_dir / "production_parameter_grid.parquet", index=False)
    joblib.dump(model, output_dir / "production_model.joblib")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "source_seasons": list(ordered_seasons),
                "team_games": team_games,
                "selected_role_shrinkage_games": config.role_shrinkage_games,
                "selected_gap_decay": config.gap_decay,
                "selected_alpha": config.alpha,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[L20-v0.3] production q={config.role_shrinkage_games:.0f}, "
        f"gap={config.gap_decay:.2f}, alpha={config.alpha:.2f}",
        flush=True,
    )
    return L20SourceAwareMinuteShareRun(run_dir=output_dir, run_id=run_id)


def _season_role_summary(
    season: str,
    *,
    panel: pd.DataFrame,
    curated_dir: Path | str,
) -> pd.DataFrame:
    game_minutes = read_regular_game_minutes(season, curated_dir=curated_dir)
    minutes = game_minutes.groupby("player_id", as_index=False).agg(
        source_minutes=("minutes", "sum"),
        observed_games=("game_id", "nunique"),
    )
    panel_rows = panel.loc[
        panel["season"].astype(str).eq(season),
        ["player_id", "games", "games_started"],
    ].copy()
    panel_rows["player_id"] = pd.to_numeric(panel_rows["player_id"], errors="raise").astype(int)
    panel_rows = panel_rows.drop_duplicates("player_id", keep="last")
    output = minutes.merge(panel_rows, on="player_id", how="left", validate="one_to_one")
    output["source_games"] = pd.to_numeric(output["games"], errors="coerce").fillna(
        output["observed_games"]
    )
    output["source_starts"] = pd.to_numeric(output["games_started"], errors="coerce").fillna(0.0)
    output = output.loc[:, ["player_id", "source_minutes", "source_games", "source_starts"]]
    return output.astype({"player_id": int})


def _build_role_state(
    season: str,
    *,
    candidates: pd.DataFrame,
    immediate: pd.DataFrame,
    history: list[pd.DataFrame],
    panel: pd.DataFrame,
    preseason_nail: pd.Series,
) -> pd.DataFrame:
    output = candidates.copy()
    output["player_id"] = output["player_id"].astype(int)
    immediate_values = immediate.set_index("player_id")
    output = _merge_role_values(output, immediate_values, prefix="immediate")
    output["gap_seasons"] = 0.0
    output["last_minutes"] = 0.0
    output["last_games"] = 0.0
    output["last_starts"] = 0.0
    missing_immediate = output["immediate_minutes"].le(0.0)
    for gap_seasons, role in enumerate(history, start=1):
        values = role.set_index("player_id")
        fallback_minutes = output["player_id"].map(values["source_minutes"]).fillna(0.0)
        use = missing_immediate & output["last_minutes"].le(0.0) & fallback_minutes.gt(0.0)
        output.loc[use, "last_minutes"] = fallback_minutes.loc[use]
        output.loc[use, "last_games"] = output.loc[use, "player_id"].map(
            values["source_games"]
        ).fillna(0.0)
        output.loc[use, "last_starts"] = output.loc[use, "player_id"].map(
            values["source_starts"]
        ).fillna(0.0)
        output.loc[use, "gap_seasons"] = float(gap_seasons)
    output["player_state"] = np.where(
        output["immediate_minutes"].gt(0.0),
        "continuous",
        np.where(output["last_minutes"].gt(0.0), "gap_returner", "cold_start"),
    )
    output["source_minutes"] = np.where(
        output["player_state"].eq("continuous"),
        output["immediate_minutes"],
        output["last_minutes"],
    )
    output["source_games"] = np.where(
        output["player_state"].eq("continuous"),
        output["immediate_games"],
        output["last_games"],
    )
    output["source_starts"] = np.where(
        output["player_state"].eq("continuous"),
        output["immediate_starts"],
        output["last_starts"],
    )
    bios = _target_bios(panel, season=season, player_ids=output["player_id"])
    output = output.merge(bios, on="player_id", how="left", validate="one_to_one")
    draft_number = pd.to_numeric(output["draft_number"], errors="coerce")
    output["draft_capital"] = np.where(
        draft_number.notna(), np.clip((61.0 - draft_number) / 60.0, 0.0, 1.0), 0.0
    )
    output["is_undrafted"] = output["is_undrafted"].astype("boolean").fillna(True).astype(bool)
    output["age_offset"] = pd.to_numeric(output["age"], errors="coerce").fillna(22.0) - 22.0
    position = output["listed_position"].fillna("").astype(str)
    output["is_guard"] = position.str.contains("G", regex=False).astype(float)
    output["is_forward"] = position.str.contains("F", regex=False).astype(float)
    output["is_center"] = position.str.contains("C", regex=False).astype(float)
    output["preseason_nail"] = output["player_id"].map(preseason_nail).fillna(0.0)
    return output


def _merge_role_values(
    frame: pd.DataFrame, values: pd.DataFrame, *, prefix: str
) -> pd.DataFrame:
    output = frame.copy()
    for source, destination in (
        ("source_minutes", f"{prefix}_minutes"),
        ("source_games", f"{prefix}_games"),
        ("source_starts", f"{prefix}_starts"),
    ):
        output[destination] = output["player_id"].map(values[source]).fillna(0.0)
    return output


def _target_bios(panel: pd.DataFrame, *, season: str, player_ids: pd.Series) -> pd.DataFrame:
    columns = ["player_id", "draft_number", "is_undrafted", "age", "listed_position"]
    target = panel.loc[
        panel["season"].astype(str).eq(season) & panel["player_id"].isin(player_ids), columns
    ].copy()
    target["player_id"] = target["player_id"].astype(int)
    target = target.drop_duplicates("player_id", keep="last")
    missing_ids = sorted(set(player_ids.astype(int)) - set(target["player_id"]))
    if missing_ids:
        historical = panel.loc[
            panel["season"].astype(str).lt(season) & panel["player_id"].isin(missing_ids),
            [*columns, "season_start_year"],
        ].copy()
        historical = historical.sort_values(
            ["player_id", "season_start_year"], kind="stable"
        ).drop_duplicates(
            "player_id", keep="last"
        )
        historical = historical.drop(columns="season_start_year")
        target = pd.concat([target, historical], ignore_index=True)
    return target.drop_duplicates("player_id", keep="last")


def _training_problem(
    inputs: list[SourceAwareSeasonInputs],
    *,
    config: SourceAwareConfig,
    team_games: int,
    replacement_token: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int]]]:
    feature_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    slices: list[tuple[int, int]] = []
    cursor = 0
    for inputs_for_season in inputs:
        features = build_source_aware_features(inputs_for_season.role_state, config=config)
        groups = {
            int(team_id): frame.set_index("player_id", drop=False)
            for team_id, frame in features.groupby("team_id", sort=True)
        }
        for target in _first_n_team_game_allocations(
            inputs_for_season.target_game_minutes, team_games=team_games
        ):
            candidates = groups.get(target.team_id)
            if candidates is None or candidates.empty:
                continue
            player_ids = candidates["player_id"].astype(int).tolist()
            outcome = np.asarray(
                [float(target.player_shares.get(player_id, 0.0)) for player_id in player_ids]
            )
            covered_share = float(outcome.sum())
            if replacement_token:
                outcome = np.append(outcome, max(0.0, 1.0 - covered_share))
            else:
                if covered_share <= 0.0:
                    continue
                # The baseline fits the conditional allocation among known
                # candidates. The replacement-token experiment instead keeps
                # the uncovered mass as an explicit team-level outcome.
                outcome /= covered_share
            values = candidates.loc[:, list(_FEATURE_COLUMNS)].to_numpy(dtype=float)
            feature_rows.append(values)
            targets.append(outcome)
            slices.append((cursor, cursor + len(values)))
            cursor += len(values)
    if not feature_rows:
        raise ValueError("Source-aware minute training has no fully covered opening-roster teams")
    return np.vstack(feature_rows), targets, slices


def _predict_allocation(
    candidates: pd.DataFrame, model: FittedSourceAwareMinuteModel
) -> tuple[dict[int, float], float]:
    values = candidates.loc[:, list(model.feature_columns)].to_numpy(dtype=float)
    standardized = (values - model.feature_mean) / model.feature_scale
    logits = standardized @ model.coefficients
    replacement_token = getattr(model, "replacement_token", False)
    if replacement_token:
        logits = np.append(logits, getattr(model, "replacement_intercept", 0.0))
    shares = np.exp(logits - logsumexp(logits))
    player_shares = shares[:-1] if replacement_token else shares
    return (
        dict(zip(candidates["player_id"].astype(int), player_shares, strict=True)),
        float(shares[-1]) if replacement_token else 0.0,
    )


def _predict_shares(
    candidates: pd.DataFrame, model: FittedSourceAwareMinuteModel
) -> dict[int, float]:
    """Return opening-roster player shares for legacy production callers."""

    return _predict_allocation(candidates, model)[0]


def _summarize_state_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if "entity_type" in predictions:
        predictions = predictions.loc[predictions["entity_type"].eq("player")]
    rows: list[dict[str, object]] = []
    for (season, state), group in predictions.groupby(["season", "player_state"], sort=True):
        rows.append(
            {
                "season": str(season),
                "player_state": str(state),
                "player_count": int(len(group)),
                "mean_absolute_error": float(group["absolute_error"].mean()),
                "mean_actual_share": float(group["actual_minute_share"].mean()),
                "mean_predicted_share": float(group["predicted_minute_share"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the rolling L20-NAIL-MSP v0.3 frozen evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate source-aware preseason minutes")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--team-games", type=int, default=DEFAULT_TEAM_GAMES)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_MODEL_RUN_DIR)
    parser.add_argument("--production-only", action="store_true")
    parser.add_argument("--replacement-token", action="store_true")
    args = parser.parse_args()
    if args.production_only:
        run = fit_l20_source_aware_production_model(
            seasons=tuple(args.seasons),
            panel_path=args.panel_path,
            curated_dir=args.curated_dir,
            artifacts_dir=args.artifacts_dir,
            run_dir=args.run_dir,
            team_games=args.team_games,
        )
    else:
        run = run_l20_source_aware_preseason_minute_share(
            seasons=tuple(args.seasons),
            panel_path=args.panel_path,
            curated_dir=args.curated_dir,
            artifacts_dir=args.artifacts_dir,
            run_dir=args.run_dir,
            team_games=args.team_games,
            replacement_token=args.replacement_token,
        )
    print(run.run_dir)
