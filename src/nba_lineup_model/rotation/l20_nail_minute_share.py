"""Rating-aware preseason first-20-game minute-share allocation baseline.

This model intentionally nests L20-MSP v0.0.  With ``nail_weight == 0``, it
uses precisely the same prior-minute allocation and uniform cold-start mass.
Positive NAIL weight only rebalances that allocation among an opening roster's
known candidates.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.rotation.l0_opening_minute_share_persistence import (
    read_preseason_roster_candidates,
)
from nba_lineup_model.rotation.l1_minute_share_persistence import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    read_regular_game_minutes,
)
from nba_lineup_model.rotation.l20_preseason_minute_share_persistence import (
    DEFAULT_EPSILON_GRID,
    DEFAULT_TEAM_GAMES,
    _assert_distribution,
    _first_n_team_game_allocations,
    _smoothed_prior_minutes,
    _validate_candidates,
    _validate_game_minutes,
    previous_season,
    summarize_l20_metrics,
)

MODEL_NAME = "l20_nail_minute_share"
MODEL_VERSION = "v0.1"
DEFAULT_SEASONS = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2026))
DEFAULT_NAIL_WEIGHT_GRID = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0)
DEFAULT_PUBLISHED_NAIL_RATINGS_PATH = Path(
    "artifacts/web/lineup_rankings/forward_nail_rapm_v1212_residualized_lambda/"
    "forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25/"
    "player_ratings.parquet"
)


@dataclass(frozen=True)
class RatingAwareSeasonInputs:
    """All strictly preseason inputs and the realized first-20 target for one season."""

    season: str
    prior_game_minutes: pd.DataFrame
    target_game_minutes: pd.DataFrame
    opening_candidates: pd.DataFrame
    completed_nail_ratings: pd.Series


@dataclass(frozen=True)
class L20NailMinuteShareRun:
    """Immutable location and identity for one rolling L20 NAIL evaluation."""

    run_dir: Path
    run_id: str


def read_completed_nail_ratings(
    completed_season: str,
    *,
    ratings_path: Path | str = DEFAULT_PUBLISHED_NAIL_RATINGS_PATH,
) -> pd.Series:
    """Return completed NAIL-RAPM available before the following season begins."""

    ratings = pd.read_parquet(ratings_path)
    required = {"season", "player_id", "rapm"}
    missing = sorted(required - set(ratings))
    if missing:
        raise ValueError("Published NAIL ratings lack required columns: " + ", ".join(missing))
    selected = ratings.loc[
        ratings["season"].astype(str).eq(completed_season), ["player_id", "rapm"]
    ].copy()
    if selected.empty:
        raise ValueError(f"Published NAIL ratings have no completed season {completed_season}")
    selected["player_id"] = selected["player_id"].astype(int)
    if selected["player_id"].duplicated().any():
        raise ValueError(f"Published NAIL ratings duplicate players in {completed_season}")
    return selected.set_index("player_id")["rapm"].astype(float)


def build_rating_aware_shares(
    candidate_ids: list[int],
    prior_minutes: pd.Series,
    completed_nail_ratings: pd.Series,
    *,
    epsilon: float,
    nail_weight: float,
    rating_scale: float,
) -> tuple[dict[int, float], dict[int, float]]:
    """Tilt an L20-MSP allocation by preseason-available completed NAIL ratings.

    ``rating_scale`` is computed once per target season over all opening
    candidates.  Centering is unnecessary because the team-level softmax
    removes a common shift; the scale makes ``nail_weight`` comparable across
    eras.  Ratings unavailable in the completed cache are neutral (zero).
    """

    if nail_weight < 0.0:
        raise ValueError("NAIL minute weight must be non-negative")
    if rating_scale <= 0.0 or not np.isfinite(rating_scale):
        raise ValueError("NAIL rating scale must be finite and positive")
    base = _smoothed_prior_minutes(candidate_ids, prior_minutes, epsilon=epsilon)
    z_scores = {
        player_id: float(completed_nail_ratings.get(player_id, 0.0)) / rating_scale
        for player_id in candidate_ids
    }
    base_values = np.asarray([base[player_id] for player_id in candidate_ids], dtype=float)
    z_values = np.asarray([z_scores[player_id] for player_id in candidate_ids], dtype=float)
    tilted = base_values * np.exp(np.clip(nail_weight * z_values, -30.0, 30.0))
    values = tilted / tilted.sum()
    return dict(zip(candidate_ids, values, strict=True)), z_scores


def evaluate_l20_nail_minute_share(
    inputs: RatingAwareSeasonInputs,
    *,
    epsilon: float,
    nail_weight: float,
    team_games: int = DEFAULT_TEAM_GAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one preseason-safe rating-aware allocation against first 20 games."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("L20-MSP epsilon must be between zero and one")
    if team_games <= 0:
        raise ValueError("L20 team-games target must be positive")
    _validate_game_minutes(inputs.prior_game_minutes, label="prior")
    _validate_game_minutes(inputs.target_game_minutes, label="target")
    _validate_candidates(inputs.opening_candidates)

    prior_minutes = inputs.prior_game_minutes.groupby("player_id", as_index=True)["minutes"].sum()
    candidate_ids_all = (
        inputs.opening_candidates["player_id"].astype(int).drop_duplicates().tolist()
    )
    candidate_ratings = np.asarray(
        [
            float(inputs.completed_nail_ratings.get(player_id, 0.0))
            for player_id in candidate_ids_all
        ],
        dtype=float,
    )
    rating_scale = float(np.std(candidate_ratings, ddof=0))
    if rating_scale <= 1e-12:
        rating_scale = 1.0

    targets = _first_n_team_game_allocations(inputs.target_game_minutes, team_games=team_games)
    candidates_by_team = {
        int(team_id): group.copy()
        for team_id, group in inputs.opening_candidates.groupby("team_id", sort=True)
    }
    records: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for target in targets:
        candidates = candidates_by_team.get(target.team_id)
        if candidates is None or candidates.empty:
            raise ValueError(f"Opening candidates have no rows for team {target.team_id}")
        candidate_ids = candidates["player_id"].astype(int).tolist()
        predicted_map, z_scores = build_rating_aware_shares(
            candidate_ids,
            prior_minutes,
            inputs.completed_nail_ratings,
            epsilon=epsilon,
            nail_weight=nail_weight,
            rating_scale=rating_scale,
        )
        player_ids = sorted(set(predicted_map) | set(target.player_shares))
        actual_values = np.asarray(
            [float(target.player_shares.get(player_id, 0.0)) for player_id in player_ids]
        )
        predicted_values = np.asarray(
            [float(predicted_map.get(player_id, 0.0)) for player_id in player_ids]
        )
        _assert_distribution(actual_values, label=f"actual for {target.team}")
        _assert_distribution(predicted_values, label=f"prediction for {target.team}")
        names = candidates.set_index("player_id")["player_name"].to_dict()
        absolute_error = np.abs(actual_values - predicted_values)
        squared_error = np.square(actual_values - predicted_values)
        positive_actual = actual_values > 0.0
        zero_probability_active = positive_actual & (predicted_values == 0.0)
        cross_entropy = float("inf")
        if not zero_probability_active.any():
            cross_entropy = float(
                -(actual_values[positive_actual] * np.log(predicted_values[positive_actual])).sum()
            )
        candidate_set = set(candidate_ids)
        metrics.append(
            {
                "season": inputs.season,
                "team_id": target.team_id,
                "team": target.team,
                "epsilon": epsilon,
                "nail_weight": nail_weight,
                "rating_scale": rating_scale,
                "allocation_total_variation": float(0.5 * absolute_error.sum()),
                "brier_score": float(squared_error.sum()),
                "player_share_mae": float(absolute_error.mean()),
                "player_share_mse": float(squared_error.mean()),
                "cross_entropy": cross_entropy,
                "has_zero_probability_active_player": bool(zero_probability_active.any()),
                "outside_candidate_actual_share": float(
                    actual_values[
                        [player_id not in candidate_set for player_id in player_ids]
                    ].sum()
                ),
            }
        )
        for player_id, actual_share, predicted_share in zip(
            player_ids, actual_values, predicted_values, strict=True
        ):
            records.append(
                {
                    "season": inputs.season,
                    "team_id": target.team_id,
                    "team": target.team,
                    "player_id": player_id,
                    "player_name": names.get(player_id, str(player_id)),
                    "actual_minute_share": float(actual_share),
                    "predicted_minute_share": float(predicted_share),
                    "absolute_error": float(abs(actual_share - predicted_share)),
                    "completed_nail_rating": float(
                        inputs.completed_nail_ratings.get(player_id, 0.0)
                    ),
                    "completed_nail_z_score": float(z_scores.get(player_id, 0.0)),
                    "was_opening_candidate": player_id in candidate_set,
                    "was_in_first_n_games": player_id in target.player_shares,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(metrics)


def select_l20_nail_parameters(
    source_inputs: list[RatingAwareSeasonInputs],
    *,
    epsilon_grid: tuple[float, ...] = DEFAULT_EPSILON_GRID,
    nail_weight_grid: tuple[float, ...] = DEFAULT_NAIL_WEIGHT_GRID,
    team_games: int = DEFAULT_TEAM_GAMES,
) -> tuple[float, float, pd.DataFrame]:
    """Choose nested L20 and NAIL parameters using only earlier seasons."""

    if not source_inputs:
        raise ValueError("L20 NAIL tuning requires at least one source season")
    rows: list[dict[str, object]] = []
    for epsilon in sorted(set(epsilon_grid)):
        for nail_weight in sorted(set(nail_weight_grid)):
            metrics = [
                evaluate_l20_nail_minute_share(
                    inputs,
                    epsilon=epsilon,
                    nail_weight=nail_weight,
                    team_games=team_games,
                )[1]
                for inputs in source_inputs
            ]
            all_metrics = pd.concat(metrics, ignore_index=True)
            rows.append(
                {
                    "epsilon": epsilon,
                    "nail_weight": nail_weight,
                    "source_season_count": len(source_inputs),
                    "source_first_season": source_inputs[0].season,
                    "source_last_season": source_inputs[-1].season,
                    "mean_allocation_total_variation": float(
                        all_metrics["allocation_total_variation"].mean()
                    ),
                    "mean_brier_score": float(all_metrics["brier_score"].mean()),
                    "player_share_mae": float(all_metrics["player_share_mae"].mean()),
                }
            )
    grid = pd.DataFrame(rows).sort_values(
        ["mean_allocation_total_variation", "nail_weight", "epsilon"], kind="stable"
    ).reset_index(drop=True)
    winner = grid.iloc[0]
    return float(winner["epsilon"]), float(winner["nail_weight"]), grid


def build_l20_nail_season_inputs(
    season: str,
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    ratings_path: Path | str = DEFAULT_PUBLISHED_NAIL_RATINGS_PATH,
) -> RatingAwareSeasonInputs:
    """Load one season's opening inputs and strictly prior completed NAIL ratings."""

    return RatingAwareSeasonInputs(
        season=season,
        prior_game_minutes=read_regular_game_minutes(
            previous_season(season), curated_dir=curated_dir
        ),
        target_game_minutes=read_regular_game_minutes(season, curated_dir=curated_dir),
        opening_candidates=read_preseason_roster_candidates(season, curated_dir=curated_dir),
        completed_nail_ratings=read_completed_nail_ratings(
            previous_season(season), ratings_path=ratings_path
        ),
    )


def run_l20_nail_minute_share(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    ratings_path: Path | str = DEFAULT_PUBLISHED_NAIL_RATINGS_PATH,
    epsilon_grid: tuple[float, ...] = DEFAULT_EPSILON_GRID,
    nail_weight_grid: tuple[float, ...] = DEFAULT_NAIL_WEIGHT_GRID,
    team_games: int = DEFAULT_TEAM_GAMES,
    input_loader: Callable[[str], RatingAwareSeasonInputs] | None = None,
    model_name: str = MODEL_NAME,
    model_version: str = MODEL_VERSION,
    contract: str | None = None,
) -> L20NailMinuteShareRun:
    """Run expanding-window frozen tests of rating-aware preseason allocation."""

    ordered_seasons = tuple(sorted(set(seasons)))
    if len(ordered_seasons) < 2:
        raise ValueError("L20 NAIL rolling evaluation requires at least two seasons")
    loader = input_loader or (
        lambda season: build_l20_nail_season_inputs(
            season, curated_dir=curated_dir, ratings_path=ratings_path
        )
    )
    inputs = [loader(season) for season in ordered_seasons]
    run_id = f"l20-nail-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    run_dir = Path(artifacts_dir) / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    grids: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    prediction_outputs: list[pd.DataFrame] = []
    metric_outputs: list[pd.DataFrame] = []
    for index, holdout in enumerate(inputs[1:], start=1):
        print(
            f"[L20-NAIL] tuning {holdout.season} from "
            f"{inputs[0].season} through {inputs[index - 1].season} "
            f"({index} source seasons)",
            flush=True,
        )
        epsilon, nail_weight, grid = select_l20_nail_parameters(
            inputs[:index],
            epsilon_grid=epsilon_grid,
            nail_weight_grid=nail_weight_grid,
            team_games=team_games,
        )
        control_winner = grid.loc[grid["nail_weight"].eq(0.0)].sort_values(
            ["mean_allocation_total_variation", "epsilon"], kind="stable"
        ).iloc[0]
        control_epsilon = float(control_winner["epsilon"])
        grid["holdout_season"] = holdout.season
        grids.append(grid)
        selections.append(
            {
                "holdout_season": holdout.season,
                "source_first_season": inputs[0].season,
                "source_last_season": inputs[index - 1].season,
                "source_season_count": index,
                "selected_epsilon": epsilon,
                "selected_nail_weight": nail_weight,
                "selected_control_epsilon": control_epsilon,
            }
        )
        print(
            f"[L20-NAIL] selected epsilon={epsilon:.2f}, beta={nail_weight:.2f} "
            f"for {holdout.season}",
            flush=True,
        )
        variants = (
            ("rating_aware", epsilon, nail_weight),
            ("l20_prior_minutes_control", control_epsilon, 0.0),
        )
        for variant, variant_epsilon, variant_weight in variants:
            predictions, metrics = evaluate_l20_nail_minute_share(
                holdout,
                epsilon=variant_epsilon,
                nail_weight=variant_weight,
                team_games=team_games,
            )
            predictions["variant"] = variant
            metrics["variant"] = variant
            metrics["selected_epsilon"] = epsilon
            metrics["selected_control_epsilon"] = control_epsilon
            metrics["selected_nail_weight"] = nail_weight
            prediction_outputs.append(predictions)
            metric_outputs.append(metrics)

    grid_frame = pd.concat(grids, ignore_index=True)
    selection_frame = pd.DataFrame(selections)
    prediction_frame = pd.concat(prediction_outputs, ignore_index=True)
    metric_frame = pd.concat(metric_outputs, ignore_index=True)
    summary = pd.concat(
        [
            summarize_l20_metrics(group, season=str(season)).assign(variant=variant)
            for (season, variant), group in metric_frame.groupby(["season", "variant"], sort=True)
        ],
        ignore_index=True,
    )
    grid_frame.to_parquet(run_dir / "parameter_grid.parquet", index=False)
    selection_frame.to_parquet(run_dir / "selected_parameters.parquet", index=False)
    prediction_frame.to_parquet(run_dir / "predictions.parquet", index=False)
    metric_frame.to_parquet(run_dir / "team_metrics.parquet", index=False)
    summary.to_parquet(run_dir / "metrics.parquet", index=False)
    production_epsilon, production_nail_weight, production_grid = select_l20_nail_parameters(
        inputs,
        epsilon_grid=epsilon_grid,
        nail_weight_grid=nail_weight_grid,
        team_games=team_games,
    )
    production_grid.to_parquet(run_dir / "production_parameter_grid.parquet", index=False)
    pd.DataFrame(
        [
            {
                "source_first_season": inputs[0].season,
                "source_last_season": inputs[-1].season,
                "source_season_count": len(inputs),
                "selected_epsilon": production_epsilon,
                "selected_nail_weight": production_nail_weight,
            }
        ]
    ).to_parquet(run_dir / "production_selected_parameters.parquet", index=False)
    print(
        f"[L20-NAIL] production selection epsilon={production_epsilon:.2f}, "
        f"beta={production_nail_weight:.2f}",
        flush=True,
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "version": model_version,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "seasons": list(ordered_seasons),
                "team_games": team_games,
                "epsilon_grid": list(epsilon_grid),
                "nail_weight_grid": list(nail_weight_grid),
                "ratings_path": str(ratings_path),
                "contract": contract
                or (
                    "For target season t, use t-1 completed NAIL-RAPM to tilt the "
                    "L20-MSP prior-minute allocation. Parameters for each holdout "
                    "season use only earlier completed target seasons."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return L20NailMinuteShareRun(run_dir=run_dir, run_id=run_id)


def main() -> None:
    """Run rolling rating-aware preseason first-20 minute-share evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate NAIL-aware preseason minutes")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--team-games", type=int, default=DEFAULT_TEAM_GAMES)
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--ratings-path", type=Path, default=DEFAULT_PUBLISHED_NAIL_RATINGS_PATH)
    args = parser.parse_args()
    run = run_l20_nail_minute_share(
        seasons=tuple(args.seasons),
        team_games=args.team_games,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        ratings_path=args.ratings_path,
    )
    print(run.run_dir)
