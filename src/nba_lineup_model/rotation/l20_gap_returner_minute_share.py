"""Gap-returner extension of the static preseason first-20-games allocation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.rotation.l0_opening_minute_share_persistence import (
    DEFAULT_EPSILON_GRID,
    read_preseason_roster_candidates,
)
from nba_lineup_model.rotation.l1_minute_share_persistence import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    read_regular_game_minutes,
)
from nba_lineup_model.rotation.l20_preseason_minute_share_persistence import (
    DEFAULT_HOLDOUT_SEASON,
    DEFAULT_TEAM_GAMES,
    DEFAULT_TUNE_SEASON,
    evaluate_l20_preseason_minute_share_persistence,
    previous_season,
    summarize_l20_metrics,
)

MODEL_NAME = "l20_gap_returner_minute_share"
MODEL_VERSION = "v0.1"
DEFAULT_RETURNER_WEIGHT_GRID = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_LOOKBACK_SEASONS = 3


@dataclass(frozen=True)
class L20GapReturnerMinuteShareRun:
    """Immutable location and identity for one L20-MSP gap-returner evaluation."""

    run_dir: Path
    run_id: str


def build_gap_returner_scores(
    prior_game_minutes: pd.DataFrame,
    historical_game_minutes: list[pd.DataFrame],
    opening_candidates: pd.DataFrame,
    *,
    returner_weight: float,
) -> tuple[pd.Series, pd.Series]:
    """Create allocation scores and flag zero-minute players with prior NBA roles.

    The immediate prior season is always authoritative. Only an opening-roster
    player with zero immediate-prior minutes can receive a score from the most
    recent earlier season in which the player logged minutes. This is a
    preseason availability proxy, not an injury label.
    """

    if returner_weight < 0.0:
        raise ValueError("Gap-returner weight must be non-negative")
    immediate = _total_minutes(prior_game_minutes)
    last_active = pd.Series(dtype=float)
    for historical in historical_game_minutes:
        candidate = _total_minutes(historical)
        candidate = candidate.loc[candidate.gt(0.0)]
        last_active = last_active.combine_first(candidate)

    candidate_ids = opening_candidates["player_id"].astype(int).drop_duplicates()
    scores = immediate.copy()
    returner_flags = pd.Series(False, index=candidate_ids.to_numpy(dtype=int), dtype=bool)
    for player_id in returner_flags.index:
        if float(immediate.get(player_id, 0.0)) > 0.0:
            continue
        fallback_minutes = float(last_active.get(player_id, 0.0))
        if fallback_minutes <= 0.0:
            continue
        scores.loc[player_id] = returner_weight * fallback_minutes
        returner_flags.loc[player_id] = True
    return scores, returner_flags


def select_gap_returner_parameters(
    prior_game_minutes: pd.DataFrame,
    historical_game_minutes: list[pd.DataFrame],
    target_game_minutes: pd.DataFrame,
    opening_candidates: pd.DataFrame,
    *,
    epsilon_grid: tuple[float, ...] = DEFAULT_EPSILON_GRID,
    returner_weight_grid: tuple[float, ...] = DEFAULT_RETURNER_WEIGHT_GRID,
    team_games: int = DEFAULT_TEAM_GAMES,
) -> tuple[float, float, pd.DataFrame]:
    """Tune static cold-start and gap-returner weights on the source season."""

    rows: list[dict[str, float]] = []
    for returner_weight in sorted(set(returner_weight_grid)):
        scores, _returners = build_gap_returner_scores(
            prior_game_minutes,
            historical_game_minutes,
            opening_candidates,
            returner_weight=returner_weight,
        )
        for epsilon in sorted(set(epsilon_grid)):
            _predictions, metrics = evaluate_l20_preseason_minute_share_persistence(
                prior_game_minutes,
                target_game_minutes,
                opening_candidates,
                epsilon=epsilon,
                team_games=team_games,
                prior_minutes=scores,
            )
            rows.append(
                {
                    "returner_weight": returner_weight,
                    "epsilon": epsilon,
                    "mean_allocation_total_variation": float(
                        metrics["allocation_total_variation"].mean()
                    ),
                    "mean_brier_score": float(metrics["brier_score"].mean()),
                }
            )
    grid = pd.DataFrame(rows).sort_values(
        ["mean_allocation_total_variation", "returner_weight", "epsilon"],
        kind="stable",
    ).reset_index(drop=True)
    winner = grid.loc[0]
    return float(winner["epsilon"]), float(winner["returner_weight"]), grid


def run_l20_gap_returner_minute_share(
    *,
    tune_season: str = DEFAULT_TUNE_SEASON,
    holdout_season: str = DEFAULT_HOLDOUT_SEASON,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    epsilon_grid: tuple[float, ...] = DEFAULT_EPSILON_GRID,
    returner_weight_grid: tuple[float, ...] = DEFAULT_RETURNER_WEIGHT_GRID,
    lookback_seasons: int = DEFAULT_LOOKBACK_SEASONS,
    team_games: int = DEFAULT_TEAM_GAMES,
) -> L20GapReturnerMinuteShareRun:
    """Tune on one season and freeze a gap-returner allocation for the next."""

    if lookback_seasons <= 0:
        raise ValueError("Gap-returner lookback must be positive")
    tune_prior, tune_history = _prior_and_history(
        tune_season,
        curated_dir=curated_dir,
        lookback_seasons=lookback_seasons,
    )
    tune_target = read_regular_game_minutes(tune_season, curated_dir=curated_dir)
    tune_candidates = read_preseason_roster_candidates(tune_season, curated_dir=curated_dir)
    epsilon, returner_weight, grid = select_gap_returner_parameters(
        tune_prior,
        tune_history,
        tune_target,
        tune_candidates,
        epsilon_grid=epsilon_grid,
        returner_weight_grid=returner_weight_grid,
        team_games=team_games,
    )

    outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series]] = {}
    for label, season in (("tune", tune_season), ("holdout", holdout_season)):
        prior, history = (tune_prior, tune_history) if label == "tune" else _prior_and_history(
            season,
            curated_dir=curated_dir,
            lookback_seasons=lookback_seasons,
        )
        target = tune_target if label == "tune" else read_regular_game_minutes(
            season,
            curated_dir=curated_dir,
        )
        candidates = tune_candidates if label == "tune" else read_preseason_roster_candidates(
            season,
            curated_dir=curated_dir,
        )
        scores, returners = build_gap_returner_scores(
            prior,
            history,
            candidates,
            returner_weight=returner_weight,
        )
        predictions, metrics = evaluate_l20_preseason_minute_share_persistence(
            prior,
            target,
            candidates,
            epsilon=epsilon,
            team_games=team_games,
            prior_minutes=scores,
        )
        predictions["is_gap_returner"] = predictions["player_id"].map(returners).eq(True)
        outputs[label] = predictions, metrics, returners

    run_id = f"l20-gap-returner-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    run_dir = Path(artifacts_dir) / MODEL_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    grid.to_parquet(run_dir / "parameter_grid.parquet", index=False)
    summaries: list[pd.DataFrame] = []
    for season, label in ((tune_season, "tune"), (holdout_season, "holdout")):
        predictions, metrics, returners = outputs[label]
        predictions.to_parquet(run_dir / f"{season}_predictions.parquet", index=False)
        metrics.to_parquet(run_dir / f"{season}_team_metrics.parquet", index=False)
        pd.DataFrame(
            {"player_id": returners.index, "is_gap_returner": returners.to_numpy()}
        ).to_parquet(run_dir / f"{season}_gap_returners.parquet", index=False)
        summaries.append(summarize_l20_metrics(metrics, season=season))
    pd.concat(summaries, ignore_index=True).to_parquet(run_dir / "metrics.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "tune_season": tune_season,
                "holdout_season": holdout_season,
                "team_games": team_games,
                "selected_epsilon": epsilon,
                "selected_returner_weight": returner_weight,
                "lookback_seasons": lookback_seasons,
                "contract": (
                    "static prior-season total minutes with a most-recent-active-season "
                    "score only for zero-minute opening-roster returners"
                ),
                "reconciliation_input": "excluded from opening candidates",
            },
            indent=2,
        )
        + "\n"
    )
    return L20GapReturnerMinuteShareRun(run_dir=run_dir, run_id=run_id)


def _prior_and_history(
    season: str,
    *,
    curated_dir: Path | str,
    lookback_seasons: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    prior_season = previous_season(season)
    history: list[pd.DataFrame] = []
    historical_season = previous_season(prior_season)
    for _ in range(lookback_seasons):
        history.append(read_regular_game_minutes(historical_season, curated_dir=curated_dir))
        historical_season = previous_season(historical_season)
    return read_regular_game_minutes(prior_season, curated_dir=curated_dir), history


def _total_minutes(game_minutes: pd.DataFrame) -> pd.Series:
    return game_minutes.groupby("player_id", as_index=True)["minutes"].sum().astype(float)


def main() -> None:
    """Tune and freeze the L20 gap-returner minute-share experiment."""

    parser = argparse.ArgumentParser(description="Evaluate L20 gap-returner minute shares")
    parser.add_argument("--tune-season", default=DEFAULT_TUNE_SEASON)
    parser.add_argument("--holdout-season", default=DEFAULT_HOLDOUT_SEASON)
    parser.add_argument("--team-games", type=int, default=DEFAULT_TEAM_GAMES)
    parser.add_argument("--lookback-seasons", type=int, default=DEFAULT_LOOKBACK_SEASONS)
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()
    run = run_l20_gap_returner_minute_share(
        tune_season=args.tune_season,
        holdout_season=args.holdout_season,
        team_games=args.team_games,
        lookback_seasons=args.lookback_seasons,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    print(run.run_dir)
