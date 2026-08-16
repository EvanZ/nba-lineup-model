"""Sequential paired-bootstrap promotion test for frozen model candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_BACKTEST_ROOT = DEFAULT_ARTIFACTS_DIR / "frozen_multiseason_backtest" / "2023-24_to_2025-26"
DEFAULT_COMPLETE_ROOT = (
    DEFAULT_ARTIFACTS_DIR / "forward_complete_player_prior_rapm" / "2023-24_to_2025-26"
)
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20_260_814


@dataclass(frozen=True)
class Candidate:
    model: str
    label: str


CANDIDATES = (
    Candidate("forward_complete_player_prior_rapm", "Complete player-prior RAPM"),
    Candidate(
        "forward_centered_value_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm",
        "Original HPM v1",
    ),
    Candidate("forward_hpm_v2_depth_aware_shooting", "HPM v2 depth-aware shooting"),
    Candidate("forward_hpm_v21_empirical_rebound_capacity", "HPM v2.1 rebound capacity"),
    Candidate("forward_hpm_v22_usage_allocation", "HPM v2.2 usage allocation"),
    Candidate("forward_hpm_v23_shot_portfolio", "HPM v2.3 shot portfolio"),
    Candidate("forward_hpm_x1_orb_claim_total", "HPM x1 ORB claim context"),
    Candidate("forward_hpm_x2_orb_per_100_total", "HPM x2 raw OREB/100 context"),
)


def run_frozen_model_tournament(
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    backtest_root: Path | str = DEFAULT_BACKTEST_ROOT,
    complete_root: Path | str = DEFAULT_COMPLETE_ROOT,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> Path:
    """Promote only challengers with a one-sided 95% full-game RMSE interval."""

    if draws < 1:
        raise ValueError("draws must be positive")
    sources = _read_sources(Path(backtest_root), Path(complete_root))
    incumbent = CANDIDATES[0]
    rows: list[dict[str, object]] = []
    for match_index, challenger in enumerate(CANDIDATES[1:], start=1):
        result = _paired_metrics(
            sources[incumbent.model],
            sources[challenger.model],
            draws=draws,
            seed=seed + match_index,
        )
        primary = result.loc[result["metric"].eq("full_game_margin_rmse")].iloc[0]
        promoted = bool(primary["ci_upper"] < 0.0)
        result["match_index"] = match_index
        result["incumbent_model"] = incumbent.model
        result["incumbent_label"] = incumbent.label
        result["challenger_model"] = challenger.model
        result["challenger_label"] = challenger.label
        result["challenger_promoted"] = promoted
        rows.extend(result.to_dict("records"))
        if promoted:
            incumbent = challenger
    tournament = pd.DataFrame(rows)
    root = Path(artifacts_dir) / "frozen_model_tournament" / "2023-24_to_2025-26"
    run_id = f"frozen-model-tournament-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    tournament.to_parquet(run_dir / "paired_bootstrap_metrics.parquet", index=False)
    winners = tournament.loc[tournament["metric"].eq("full_game_margin_rmse")].copy()
    winners.to_parquet(run_dir / "promotion_matches.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "draws": draws,
                "seed": seed,
                "promotion_metric": "full_game_margin_rmse",
                "promotion_rule": "candidate_minus_incumbent ci_upper < 0",
                "final_winner_model": incumbent.model,
                "final_winner_label": incumbent.label,
            },
            indent=2,
        )
        + "\n"
    )
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return run_dir


def _read_sources(backtest_root: Path, complete_root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    backtest = _latest_run(backtest_root)
    complete = _latest_run(complete_root)
    regular_games = pd.read_parquet(backtest / "regular_game_predictions.parquet")
    regular_possessions = pd.read_parquet(backtest / "possession_predictions.parquet")
    complete_games = pd.read_parquet(complete / "regular_game_predictions.parquet")
    complete_possessions = pd.read_parquet(complete / "possession_predictions.parquet")
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for candidate in CANDIDATES:
        if candidate.model == "forward_complete_player_prior_rapm":
            games, possessions = complete_games, complete_possessions
        else:
            games, possessions = regular_games, regular_possessions
        result[candidate.model] = {
            "games": games.loc[games["model"].eq(candidate.model)].copy(),
            "possessions": possessions.loc[
                possessions["model"].eq(candidate.model)
                & possessions["cohort"].eq("regular_season")
            ].copy(),
        }
    return result


def _paired_metrics(
    incumbent: dict[str, pd.DataFrame], candidate: dict[str, pd.DataFrame], *, draws: int, seed: int
) -> pd.DataFrame:
    games = _paired_games(incumbent["games"], candidate["games"])
    possessions = _paired_possessions(incumbent["possessions"], candidate["possessions"])
    outputs = [
        _bootstrap_game_metric(games, "full_game_margin_rmse", draws=draws, seed=seed),
        _bootstrap_game_metric(games, "winner_accuracy", draws=draws, seed=seed + 1),
        _bootstrap_possession_metric(possessions, "possession_rmse", draws=draws, seed=seed + 2),
        _bootstrap_possession_metric(possessions, "possession_mae", draws=draws, seed=seed + 3),
    ]
    return pd.DataFrame(outputs)


def _paired_games(incumbent: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "game_id"]
    merged = incumbent.merge(
        candidate,
        on=keys,
        suffixes=("_incumbent", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(incumbent) or len(merged) != len(candidate):
        raise ValueError("Tournament models must contain identical regular-season games")
    return merged


def _paired_possessions(incumbent: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "game_id", "possession_id"]
    merged = incumbent.merge(
        candidate,
        on=keys,
        suffixes=("_incumbent", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(incumbent) or len(merged) != len(candidate):
        raise ValueError("Tournament models must contain identical regular-season possessions")
    return merged


def _bootstrap_game_metric(
    frame: pd.DataFrame, metric: str, *, draws: int, seed: int
) -> dict[str, object]:
    if metric == "full_game_margin_rmse":
        incumbent = np.square(frame["margin_error_incumbent"].to_numpy(float))
        candidate = np.square(frame["margin_error_candidate"].to_numpy(float))
        transform = np.sqrt
        lower_is_better = True
    elif metric == "winner_accuracy":
        incumbent = (
            frame["predicted_home_win_incumbent"] == frame["actual_home_win_incumbent"]
        ).to_numpy(float)
        candidate = (
            frame["predicted_home_win_candidate"] == frame["actual_home_win_candidate"]
        ).to_numpy(float)
        transform = _identity
        lower_is_better = False
    else:
        raise ValueError(metric)
    return _stratified_bootstrap(
        frame["season"], incumbent, candidate, transform, lower_is_better, draws, seed, metric
    )


def _bootstrap_possession_metric(
    frame: pd.DataFrame, metric: str, *, draws: int, seed: int
) -> dict[str, object]:
    residual_i = frame["residual_offense_margin_incumbent"].to_numpy(float)
    residual_c = frame["residual_offense_margin_candidate"].to_numpy(float)
    values_i = np.square(residual_i) if metric == "possession_rmse" else np.abs(residual_i)
    values_c = np.square(residual_c) if metric == "possession_rmse" else np.abs(residual_c)
    grouped = (
        pd.DataFrame(
            {"season": frame["season"], "game_id": frame["game_id"], "i": values_i, "c": values_c}
        )
        .groupby(["season", "game_id"], as_index=False)
        .agg(i=("i", "sum"), c=("c", "sum"), n=("i", "size"))
    )
    transform = np.sqrt if metric == "possession_rmse" else lambda value: value
    return _stratified_bootstrap(
        grouped["season"],
        grouped["i"].to_numpy(),
        grouped["c"].to_numpy(),
        transform,
        True,
        draws,
        seed,
        metric,
        weights=grouped["n"].to_numpy(dtype=float),
    )


def _stratified_bootstrap(
    seasons: pd.Series,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    transform,
    lower_is_better: bool,
    draws: int,
    seed: int,
    metric: str,
    weights: np.ndarray | None = None,
) -> dict[str, object]:
    generator = np.random.default_rng(seed)
    incumbent_draws = np.zeros(draws)
    candidate_draws = np.zeros(draws)
    denominator = np.zeros(draws)
    season_values = np.asarray(seasons.astype(str))
    for season in np.unique(season_values):
        indices = np.flatnonzero(season_values == season)
        sampled = generator.integers(0, len(indices), size=(draws, len(indices)))
        selected = indices[sampled]
        incumbent_draws += incumbent[selected].sum(axis=1)
        candidate_draws += candidate[selected].sum(axis=1)
        denominator += weights[selected].sum(axis=1) if weights is not None else len(indices)
    incumbent_score = transform(incumbent_draws / denominator)
    candidate_score = transform(candidate_draws / denominator)
    total_weight = weights.sum() if weights is not None else len(incumbent)
    point_i = transform(incumbent.sum() / total_weight)
    point_c = transform(candidate.sum() / total_weight)
    difference = candidate_score - incumbent_score
    better = difference < 0.0 if lower_is_better else difference > 0.0
    return {
        "metric": metric,
        "incumbent_value": float(point_i),
        "challenger_value": float(point_c),
        "difference_candidate_minus_incumbent": float(point_c - point_i),
        "ci_lower": float(np.quantile(difference, 0.025)),
        "ci_upper": float(np.quantile(difference, 0.975)),
        "probability_challenger_better": float(np.mean(better)),
        "bootstrap_draws": draws,
    }


def _latest_run(root: Path) -> Path:
    pointer = root / "latest.json"
    if pointer.is_file():
        return root / str(json.loads(pointer.read_text())["run_id"])
    runs = sorted(
        (path for path in root.iterdir() if (path / "metadata.json").is_file()),
        key=lambda path: path.name,
    )
    if not runs:
        raise ValueError(f"No immutable runs found under {root}")
    return runs[-1]


def _identity(value: np.ndarray | float) -> np.ndarray | float:
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen sequential model tournament")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    args = parser.parse_args()
    run = run_frozen_model_tournament(draws=args.draws)
    print(f"Frozen model tournament: run={run}")


if __name__ == "__main__":
    main()
