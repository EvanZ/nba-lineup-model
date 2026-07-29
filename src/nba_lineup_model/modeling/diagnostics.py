from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse

from nba_lineup_model.evaluation.metrics import (
    game_margin_rmse,
    mean_absolute_error,
    mean_squared_error,
    rmse,
    skill_score,
)
from nba_lineup_model.modeling.allocation import (
    POSSESSION_ALLOCATION_POLICIES,
    PossessionAllocationPolicy,
    allocation_policy_stints,
    possession_allocation_summary,
)
from nba_lineup_model.modeling.schema import (
    ArtifactRecord,
    BaselineRunManifest,
    RapmDiagnosticsManifest,
)
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.modeling.train import validate_baseline_run
from nba_lineup_model.models.baselines import (
    FittedMeanModel,
    RidgeLineupModel,
    signed_entity_matrix,
)

DEFAULT_SENSITIVITY_LAMBDAS = (0.003, 0.01, 0.03, 0.1, 0.3)


def diagnostics_code_fingerprint(
    source_paths: Sequence[Path | str] | None = None,
) -> str:
    """Hash diagnostics-owned source files for reproducible reports."""

    if source_paths is None:
        package_root = Path(__file__).parents[1]
        paths = (
            package_root / "evaluation" / "metrics.py",
            package_root / "modeling" / "allocation.py",
            package_root / "modeling" / "diagnostics.py",
            package_root / "modeling" / "schema.py",
            package_root / "modeling" / "stints.py",
            package_root / "models" / "baselines.py",
        )
    else:
        paths = tuple(Path(path) for path in source_paths)
    ordered = sorted(paths)
    if not ordered:
        raise ValueError("At least one diagnostics source path is required")
    digest = hashlib.sha256()
    for path in ordered:
        if not path.is_file():
            raise ValueError(f"Diagnostics source does not exist: {path}")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def lambda_sensitivity(
    stints: pd.DataFrame,
    player_matrix: sparse.csr_matrix,
    player_ids: tuple[int, ...],
    rankings: pd.DataFrame,
    lambdas: tuple[float, ...],
    selected_lambda: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit full-season RAPM over a lambda path and compare player ranks."""

    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    eligible_ids = set(rankings.loc[rankings["exposure_eligible"], "player_id"].astype(int))
    rows: list[pd.DataFrame] = []
    for regularization in lambdas:
        model = RidgeLineupModel(regularization).fit(
            player_matrix,
            target,
            weights,
        )
        frame = _coefficient_frame(
            player_ids,
            model.coef_,
            eligible_ids,
        )
        frame.insert(0, "regularization", regularization)
        rows.append(frame)
    coefficients = pd.concat(rows, ignore_index=True)
    names = rankings.loc[
        :,
        ["player_id", "player_name", "primary_team_tricode", "possessions"],
    ]
    coefficients = coefficients.merge(
        names,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    reference = coefficients.loc[
        coefficients["regularization"].eq(selected_lambda) & coefficients["eligible_rank"].notna(),
        ["player_id", "coefficient", "eligible_rank"],
    ].rename(
        columns={
            "coefficient": "selected_coefficient",
            "eligible_rank": "selected_eligible_rank",
        }
    )
    summary_rows: list[dict[str, Any]] = []
    reference_top_25 = set(reference.nsmallest(25, "selected_eligible_rank")["player_id"])
    reference_top_50 = set(reference.nsmallest(50, "selected_eligible_rank")["player_id"])
    for regularization in lambdas:
        current = coefficients.loc[
            coefficients["regularization"].eq(regularization)
            & coefficients["eligible_rank"].notna(),
            ["player_id", "coefficient", "eligible_rank"],
        ]
        comparison = current.merge(
            reference,
            on="player_id",
            how="inner",
            validate="one_to_one",
        )
        current_top_25 = set(current.nsmallest(25, "eligible_rank")["player_id"])
        current_top_50 = set(current.nsmallest(50, "eligible_rank")["player_id"])
        summary_rows.append(
            {
                "regularization": regularization,
                "eligible_player_count": len(comparison),
                "coefficient_correlation": comparison["coefficient"].corr(
                    comparison["selected_coefficient"]
                ),
                "rank_spearman": comparison["eligible_rank"].corr(
                    comparison["selected_eligible_rank"],
                    method="spearman",
                ),
                "mean_absolute_rank_change": (
                    comparison["eligible_rank"] - comparison["selected_eligible_rank"]
                )
                .abs()
                .mean(),
                "top_25_overlap": len(current_top_25 & reference_top_25),
                "top_50_overlap": len(current_top_50 & reference_top_50),
                "is_selected": regularization == selected_lambda,
            }
        )
    return coefficients, pd.DataFrame(summary_rows)


def chronological_stability(
    stints: pd.DataFrame,
    player_matrix: sparse.csr_matrix,
    player_ids: tuple[int, ...],
    rankings: pd.DataFrame,
    game_splits: pd.DataFrame,
    selected_lambda: float,
    minimum_possessions: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare fixed-lambda coefficients across expanding season windows."""

    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    game_ids = stints["game_id"].astype(str).to_numpy()
    windows: list[tuple[str, tuple[str, ...]]] = []
    cv_splits = sorted(
        value for value in game_splits["split"].unique() if str(value).startswith("cv_")
    )
    for split_name in cv_splits:
        identifiers = tuple(
            game_splits.loc[
                game_splits["split"].eq(split_name) & game_splits["role"].eq("train"),
                "game_id",
            ].astype(str)
        )
        windows.append((f"{split_name}_train", identifiers))
    final_train = tuple(
        game_splits.loc[
            game_splits["split"].eq("final") & game_splits["role"].eq("train"),
            "game_id",
        ].astype(str)
    )
    windows.extend(
        [
            ("final_train", final_train),
            ("full_season", tuple(pd.unique(game_ids))),
        ]
    )

    rows: list[pd.DataFrame] = []
    for window_order, (window, identifiers) in enumerate(windows):
        mask = np.isin(game_ids, identifiers)
        model = RidgeLineupModel(selected_lambda).fit(
            player_matrix[mask],
            target[mask],
            weights[mask],
        )
        exposure = np.asarray(abs(player_matrix[mask]).T @ weights[mask]).reshape(-1)
        eligible_ids = {
            player_ids[index] for index in np.flatnonzero(exposure >= minimum_possessions)
        }
        frame = _coefficient_frame(player_ids, model.coef_, eligible_ids)
        frame["possessions"] = exposure
        frame.insert(0, "window_order", window_order)
        frame.insert(1, "window", window)
        frame["game_count"] = len(set(identifiers))
        rows.append(frame)
    coefficients = pd.concat(rows, ignore_index=True)
    coefficients = coefficients.merge(
        rankings.loc[
            :,
            ["player_id", "player_name", "primary_team_tricode"],
        ],
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    all_summary = coefficients.groupby("player_id", as_index=False).agg(
        coefficient_min=("coefficient", "min"),
        coefficient_max=("coefficient", "max"),
        coefficient_mean=("coefficient", "mean"),
        coefficient_std=("coefficient", lambda values: values.std(ddof=0)),
        window_count=("window", "size"),
    )
    eligible = coefficients.loc[coefficients["eligible_rank"].notna()]
    eligible_summary = eligible.groupby("player_id", as_index=False).agg(
        eligible_window_count=("window", "size"),
        eligible_rank_min=("eligible_rank", "min"),
        eligible_rank_max=("eligible_rank", "max"),
        eligible_rank_mean=("eligible_rank", "mean"),
        eligible_rank_std=(
            "eligible_rank",
            lambda values: values.std(ddof=0),
        ),
    )
    summary = all_summary.merge(
        eligible_summary,
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    summary["coefficient_range"] = summary["coefficient_max"] - summary["coefficient_min"]
    summary["eligible_rank_range"] = summary["eligible_rank_max"] - summary["eligible_rank_min"]
    summary = summary.merge(
        rankings.loc[
            :,
            ["player_id", "player_name", "primary_team_tricode", "rapm", "rank"],
        ],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    return coefficients, summary


def bootstrap_stability(
    stints: pd.DataFrame,
    player_matrix: sparse.csr_matrix,
    player_ids: tuple[int, ...],
    rankings: pd.DataFrame,
    selected_lambda: float,
    *,
    samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit RAPM on complete-game bootstrap samples."""

    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    game_ids = stints["game_id"].astype(str).to_numpy()
    unique_games = np.asarray(pd.unique(game_ids))
    row_indices = {game_id: np.flatnonzero(game_ids == game_id) for game_id in unique_games}
    rng = np.random.default_rng(seed)
    coefficient_samples = np.empty((samples, len(player_ids)), dtype=float)
    rank_samples = np.empty((samples, len(player_ids)), dtype=np.int64)
    eligible_mask = np.asarray(
        [
            bool(
                rankings.loc[
                    rankings["player_id"].eq(player_id),
                    "exposure_eligible",
                ].item()
            )
            for player_id in player_ids
        ]
    )
    eligible_indices = np.flatnonzero(eligible_mask)
    eligible_rank_samples = np.full(
        (samples, len(player_ids)),
        np.nan,
        dtype=float,
    )
    for sample in range(samples):
        sampled_games = rng.choice(
            unique_games,
            size=len(unique_games),
            replace=True,
        )
        sampled_rows = np.concatenate([row_indices[game_id] for game_id in sampled_games])
        model = RidgeLineupModel(selected_lambda).fit(
            player_matrix[sampled_rows],
            target[sampled_rows],
            weights[sampled_rows],
        )
        coefficient_samples[sample] = model.coef_
        rank_samples[sample] = _ranks(model.coef_)
        eligible_rank_samples[sample, eligible_indices] = _ranks(model.coef_[eligible_indices])

    coefficients = pd.DataFrame(
        {
            "bootstrap_sample": np.repeat(np.arange(samples), len(player_ids)),
            "player_id": np.tile(np.asarray(player_ids), samples),
            "coefficient": coefficient_samples.reshape(-1),
            "rank": rank_samples.reshape(-1),
        }
    )
    coefficients["eligible_rank"] = pd.array(
        eligible_rank_samples.reshape(-1),
        dtype="Int64",
    )
    names = rankings.loc[
        :,
        ["player_id", "player_name", "primary_team_tricode", "possessions"],
    ]
    coefficients = coefficients.merge(
        names,
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    top_25_probability = np.full(len(player_ids), np.nan)
    top_50_probability = np.full(len(player_ids), np.nan)
    median_eligible_rank = np.full(len(player_ids), np.nan)
    eligible_rank_p10 = np.full(len(player_ids), np.nan)
    eligible_rank_p90 = np.full(len(player_ids), np.nan)
    top_25_probability[eligible_indices] = np.mean(
        eligible_rank_samples[:, eligible_indices] <= 25,
        axis=0,
    )
    top_50_probability[eligible_indices] = np.mean(
        eligible_rank_samples[:, eligible_indices] <= 50,
        axis=0,
    )
    median_eligible_rank[eligible_indices] = np.median(
        eligible_rank_samples[:, eligible_indices],
        axis=0,
    )
    eligible_rank_p10[eligible_indices] = np.quantile(
        eligible_rank_samples[:, eligible_indices],
        0.10,
        axis=0,
    )
    eligible_rank_p90[eligible_indices] = np.quantile(
        eligible_rank_samples[:, eligible_indices],
        0.90,
        axis=0,
    )
    summary = pd.DataFrame(
        {
            "player_id": player_ids,
            "bootstrap_median": np.median(coefficient_samples, axis=0),
            "bootstrap_mean": np.mean(coefficient_samples, axis=0),
            "bootstrap_std": np.std(coefficient_samples, axis=0),
            "bootstrap_p05": np.quantile(coefficient_samples, 0.05, axis=0),
            "bootstrap_p10": np.quantile(coefficient_samples, 0.10, axis=0),
            "bootstrap_p90": np.quantile(coefficient_samples, 0.90, axis=0),
            "bootstrap_p95": np.quantile(coefficient_samples, 0.95, axis=0),
            "positive_probability": np.mean(coefficient_samples > 0, axis=0),
            "top_25_probability": top_25_probability,
            "top_50_probability": top_50_probability,
            "median_eligible_rank": median_eligible_rank,
            "eligible_rank_p10": eligible_rank_p10,
            "eligible_rank_p90": eligible_rank_p90,
        }
    )
    summary["bootstrap_interval_width_90"] = summary["bootstrap_p95"] - summary["bootstrap_p05"]
    summary = summary.merge(
        rankings.loc[
            :,
            [
                "player_id",
                "player_name",
                "primary_team_tricode",
                "rapm",
                "rank",
                "eligible_rank",
                "possessions",
                "exposure_eligible",
            ],
        ],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    return coefficients, summary


def context_concentration(
    stints: pd.DataFrame,
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """Measure how broadly each player is connected to teammates and lineups."""

    total: Counter[int] = Counter()
    teammate_exposure: dict[int, Counter[int]] = defaultdict(Counter)
    lineup_exposure: dict[int, Counter[tuple[int, ...]]] = defaultdict(Counter)
    top_25_exposure: Counter[int] = Counter()
    top_25 = set(
        rankings.loc[rankings["exposure_eligible"]]
        .nsmallest(25, "eligible_rank")["player_id"]
        .astype(int)
    )
    for row in stints.itertuples(index=False):
        weight = float(row.possessions)
        for lineup in (tuple(row.home_player_ids), tuple(row.away_player_ids)):
            normalized_lineup = tuple(sorted(int(player) for player in lineup))
            for player in normalized_lineup:
                teammates = tuple(teammate for teammate in normalized_lineup if teammate != player)
                total[player] += weight
                lineup_exposure[player][normalized_lineup] += weight
                for teammate in teammates:
                    teammate_exposure[player][teammate] += weight
                if any(teammate in top_25 for teammate in teammates):
                    top_25_exposure[player] += weight

    rows = []
    for player_id, possessions in total.items():
        teammate_counts = teammate_exposure[player_id]
        lineup_counts = lineup_exposure[player_id]
        rows.append(
            {
                "player_id": player_id,
                "possessions": possessions,
                "distinct_teammates": len(teammate_counts),
                "distinct_lineups": len(lineup_counts),
                "most_common_teammate_id": teammate_counts.most_common(1)[0][0],
                "most_common_teammate_share": (teammate_counts.most_common(1)[0][1] / possessions),
                "most_common_lineup_share": (lineup_counts.most_common(1)[0][1] / possessions),
                "effective_lineup_count": _effective_count(lineup_counts.values()),
                "top_25_teammate_share": (top_25_exposure[player_id] / possessions),
            }
        )
    result = pd.DataFrame(rows)
    names = rankings.loc[
        :,
        [
            "player_id",
            "player_name",
            "primary_team_tricode",
            "rapm",
            "rank",
            "eligible_rank",
            "exposure_eligible",
        ],
    ]
    teammate_names = rankings.loc[:, ["player_id", "player_name"]].rename(
        columns={
            "player_id": "most_common_teammate_id",
            "player_name": "most_common_teammate_name",
        }
    )
    return (
        result.merge(names, on="player_id", how="left", validate="one_to_one")
        .merge(
            teammate_names,
            on="most_common_teammate_id",
            how="left",
            validate="many_to_one",
        )
        .sort_values("rank", kind="stable")
        .reset_index(drop=True)
    )


def raw_adjusted_comparison(rankings: pd.DataFrame) -> pd.DataFrame:
    """Compare raw on-court net rating with the adjusted RAPM coefficient."""

    result = rankings.copy()
    result["raw_on_court_rank"] = (
        result["raw_on_court_net_rating"].rank(method="first", ascending=False).astype(int)
    )
    result["rapm_adjustment"] = result["rapm"] - result["raw_on_court_net_rating"]
    result["absolute_rapm_adjustment"] = result["rapm_adjustment"].abs()
    result["rank_change_from_raw"] = result["raw_on_court_rank"] - result["rank"]
    return result.sort_values("rank", kind="stable").reset_index(drop=True)


def influence_diagnostics(
    stints: pd.DataFrame,
    player_matrix: sparse.csr_matrix,
    player_ids: tuple[int, ...],
    rankings: pd.DataFrame,
    selected_lambda: float,
    *,
    influence_player_count: int,
    stints_per_player: int,
    delete_games_per_player: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Screen influential stints and verify top games with exact deletions."""

    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    model = RidgeLineupModel(selected_lambda).fit(
        player_matrix,
        target,
        weights,
    )
    predicted = model.predict(player_matrix)
    residual = target - predicted
    normalized_weights = weights / np.mean(weights)
    leverage = _ridge_leverage(
        player_matrix,
        normalized_weights,
        model.sklearn_alpha,
    )
    influence_score = np.abs(normalized_weights * residual) * np.sqrt(
        leverage / np.maximum(1.0 - leverage, 1e-12)
    )
    long = _player_stint_long(stints)
    row_index = long["row_index"].to_numpy(dtype=int)
    long["actual_net_rating"] = target[row_index]
    long["predicted_net_rating"] = predicted[row_index]
    long["residual"] = residual[row_index]
    long["normalized_weight"] = normalized_weights[row_index]
    long["ridge_leverage"] = leverage[row_index]
    long["influence_score"] = influence_score[row_index]
    long["signed_gradient"] = (
        normalized_weights[row_index] * residual[row_index] * long["sign"].to_numpy(dtype=float)
    )
    names = rankings.loc[:, ["player_id", "player_name", "primary_team_tricode"]]
    long = long.merge(
        names,
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    influential_stints = (
        long.sort_values(
            ["player_id", "influence_score", "game_id", "stint_index"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        .groupby("player_id", sort=False)
        .head(stints_per_player)
        .reset_index(drop=True)
    )
    games = long.groupby(
        [
            "player_id",
            "player_name",
            "primary_team_tricode",
            "game_id",
        ],
        as_index=False,
    ).agg(
        stint_count=("stint_index", "size"),
        possessions=("possessions", "sum"),
        signed_gradient=("signed_gradient", "sum"),
        total_influence=("influence_score", "sum"),
        max_stint_leverage=("ridge_leverage", "max"),
    )
    games["absolute_signed_gradient"] = games["signed_gradient"].abs()
    influential_games = (
        games.sort_values(
            ["player_id", "absolute_signed_gradient", "game_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("player_id", sort=False)
        .head(max(delete_games_per_player, 5))
        .reset_index(drop=True)
    )

    selected_players = set(
        rankings.loc[
            rankings["exposure_eligible"],
            "player_id",
        ]
        .head(influence_player_count)
        .astype(int)
    )
    delete_candidates = (
        influential_games.loc[influential_games["player_id"].isin(selected_players)]
        .sort_values(
            ["player_id", "absolute_signed_gradient", "game_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("player_id", sort=False)
        .head(delete_games_per_player)
    )
    player_to_column = {player_id: column for column, player_id in enumerate(player_ids)}
    game_ids = stints["game_id"].astype(str).to_numpy()
    deleted_coefficients: dict[str, np.ndarray] = {}
    for game_id in sorted(delete_candidates["game_id"].unique()):
        mask = game_ids != game_id
        deleted_model = RidgeLineupModel(selected_lambda).fit(
            player_matrix[mask],
            target[mask],
            weights[mask],
        )
        deleted_coefficients[str(game_id)] = deleted_model.coef_
    delete_rows = []
    for row in delete_candidates.itertuples(index=False):
        column = player_to_column[int(row.player_id)]
        without = float(deleted_coefficients[str(row.game_id)][column])
        full = float(model.coef_[column])
        delete_rows.append(
            {
                "player_id": int(row.player_id),
                "player_name": row.player_name,
                "primary_team_tricode": row.primary_team_tricode,
                "game_id": str(row.game_id),
                "full_season_coefficient": full,
                "without_game_coefficient": without,
                "coefficient_change_when_deleted": without - full,
                "absolute_coefficient_change": abs(without - full),
                "game_signed_gradient": float(row.signed_gradient),
                "game_total_influence": float(row.total_influence),
                "game_possessions": float(row.possessions),
            }
        )
    return (
        influential_stints,
        influential_games,
        pd.DataFrame(delete_rows),
    )


def allocation_sensitivity(
    lineup_stints: pd.DataFrame,
    possession_segments: pd.DataFrame,
    rankings: pd.DataFrame,
    player_ids: tuple[int, ...],
    player_columns: Mapping[int, int],
    game_splits: pd.DataFrame,
    selected_lambda: float,
    policies: tuple[PossessionAllocationPolicy, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit RAPM under alternative multi-lineup possession policies."""

    final_train_ids = set(
        game_splits.loc[
            game_splits["split"].eq("final") & game_splits["role"].eq("train"),
            "game_id",
        ].astype(str)
    )
    final_test_ids = set(
        game_splits.loc[
            game_splits["split"].eq("final") & game_splits["role"].eq("test"),
            "game_id",
        ].astype(str)
    )
    reference = rankings.loc[
        :,
        ["player_id", "rapm", "rank", "eligible_rank", "exposure_eligible"],
    ].rename(
        columns={
            "rapm": "reference_rapm",
            "rank": "reference_rank",
            "eligible_rank": "reference_eligible_rank",
        }
    )
    eligible_ids = set(rankings.loc[rankings["exposure_eligible"], "player_id"].astype(int))
    coefficient_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for policy in policies:
        policy_stints = allocation_policy_stints(
            lineup_stints,
            possession_segments,
            policy,
        )
        matrix = signed_entity_matrix(
            policy_stints,
            "home_player_ids",
            "away_player_ids",
            player_columns,
            multiple=True,
        )
        target = policy_stints["target_home_net_rating"].to_numpy(dtype=float)
        weights = policy_stints["possessions"].to_numpy(dtype=float)
        game_ids = policy_stints["game_id"].astype(str).to_numpy()
        train_mask = np.isin(game_ids, tuple(final_train_ids))
        test_mask = np.isin(game_ids, tuple(final_test_ids))
        mean_model = FittedMeanModel.fit(target[train_mask], weights[train_mask])
        rapm_model = RidgeLineupModel(selected_lambda).fit(
            matrix[train_mask],
            target[train_mask],
            weights[train_mask],
        )
        mean_predictions = mean_model.predict(int(test_mask.sum()))
        rapm_predictions = rapm_model.predict(matrix[test_mask])
        mean_mse = mean_squared_error(
            target[test_mask],
            mean_predictions,
            weights[test_mask],
        )
        rapm_mse = mean_squared_error(
            target[test_mask],
            rapm_predictions,
            weights[test_mask],
        )
        for model_name, predictions, mse in (
            ("mean", mean_predictions, mean_mse),
            ("rapm", rapm_predictions, rapm_mse),
        ):
            metric_rows.append(
                {
                    "allocation_policy": policy,
                    "model": model_name,
                    "train_game_count": len(final_train_ids),
                    "test_game_count": len(final_test_ids),
                    "test_stint_count": int(test_mask.sum()),
                    "test_possessions": float(weights[test_mask].sum()),
                    "weighted_mse": mse,
                    "weighted_rmse": rmse(
                        target[test_mask],
                        predictions,
                        weights[test_mask],
                    ),
                    "weighted_mae": mean_absolute_error(
                        target[test_mask],
                        predictions,
                        weights[test_mask],
                    ),
                    "game_margin_rmse": game_margin_rmse(
                        policy_stints.loc[test_mask, "game_id"],
                        target[test_mask],
                        predictions,
                        weights[test_mask],
                    ),
                    "skill_vs_mean": skill_score(mse, mean_mse),
                }
            )
        full_model = RidgeLineupModel(selected_lambda).fit(
            matrix,
            target,
            weights,
        )
        exposure = np.asarray(abs(matrix).T @ weights).reshape(-1)
        frame = _coefficient_frame(
            player_ids,
            full_model.coef_,
            eligible_ids,
        )
        frame = frame.drop(columns="eligible")
        frame.insert(0, "allocation_policy", policy)
        frame["possessions"] = exposure
        frame = frame.merge(
            reference,
            on="player_id",
            how="left",
            validate="one_to_one",
        )
        frame["coefficient_change"] = frame["coefficient"] - frame["reference_rapm"]
        frame["rank_change"] = frame["rank"] - frame["reference_rank"]
        frame["eligible_rank_change"] = frame["eligible_rank"] - frame["reference_eligible_rank"]
        coefficient_rows.append(frame)
    coefficients = pd.concat(coefficient_rows, ignore_index=True).merge(
        rankings.loc[
            :,
            ["player_id", "player_name", "primary_team_tricode"],
        ],
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    return coefficients, pd.DataFrame(metric_rows)


def player_diagnostics_summary(
    rankings: pd.DataFrame,
    lambda_coefficients: pd.DataFrame,
    chronological_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    raw_adjusted: pd.DataFrame,
    allocation_coefficients: pd.DataFrame,
    delete_game: pd.DataFrame,
) -> pd.DataFrame:
    """Combine review-oriented player diagnostics into one wide table."""

    base = rankings.loc[
        :,
        [
            "player_id",
            "player_name",
            "primary_team_tricode",
            "rapm",
            "rank",
            "eligible_rank",
            "possessions",
            "exposure_eligible",
        ],
    ].copy()
    lambda_summary = lambda_coefficients.groupby("player_id", as_index=False).agg(
        lambda_coefficient_min=("coefficient", "min"),
        lambda_coefficient_max=("coefficient", "max"),
        lambda_eligible_rank_min=("eligible_rank", "min"),
        lambda_eligible_rank_max=("eligible_rank", "max"),
    )
    lambda_summary["lambda_coefficient_range"] = (
        lambda_summary["lambda_coefficient_max"] - lambda_summary["lambda_coefficient_min"]
    )
    lambda_summary["lambda_eligible_rank_range"] = (
        lambda_summary["lambda_eligible_rank_max"] - lambda_summary["lambda_eligible_rank_min"]
    )
    chronology = chronological_summary.loc[
        :,
        [
            "player_id",
            "coefficient_std",
            "coefficient_range",
            "eligible_window_count",
            "eligible_rank_range",
        ],
    ].rename(
        columns={
            "coefficient_std": "chronological_coefficient_std",
            "coefficient_range": "chronological_coefficient_range",
            "eligible_window_count": "chronological_eligible_window_count",
            "eligible_rank_range": "chronological_eligible_rank_range",
        }
    )
    bootstrap = bootstrap_summary.loc[
        :,
        [
            "player_id",
            "bootstrap_median",
            "bootstrap_std",
            "bootstrap_p05",
            "bootstrap_p95",
            "bootstrap_interval_width_90",
            "positive_probability",
            "top_25_probability",
            "top_50_probability",
            "median_eligible_rank",
            "eligible_rank_p10",
            "eligible_rank_p90",
        ],
    ]
    context = concentration.loc[
        :,
        [
            "player_id",
            "distinct_teammates",
            "distinct_lineups",
            "most_common_teammate_id",
            "most_common_teammate_name",
            "most_common_teammate_share",
            "most_common_lineup_share",
            "effective_lineup_count",
            "top_25_teammate_share",
        ],
    ]
    adjustment = raw_adjusted.loc[
        :,
        [
            "player_id",
            "raw_on_court_net_rating",
            "raw_on_court_rank",
            "rapm_adjustment",
            "absolute_rapm_adjustment",
            "rank_change_from_raw",
        ],
    ]
    allocation = allocation_coefficients.assign(
        absolute_coefficient_change=lambda frame: frame["coefficient_change"].abs(),
        absolute_eligible_rank_change=lambda frame: frame["eligible_rank_change"].abs(),
    )
    allocation_coefficient_rows = allocation.loc[
        allocation.groupby("player_id")["absolute_coefficient_change"].idxmax(),
        [
            "player_id",
            "allocation_policy",
            "absolute_coefficient_change",
        ],
    ].rename(
        columns={
            "allocation_policy": "max_coefficient_change_allocation_policy",
            "absolute_coefficient_change": ("max_allocation_absolute_coefficient_change"),
        }
    )
    allocation_rank_rows = allocation.loc[
        allocation.loc[allocation["exposure_eligible"]]
        .groupby("player_id")["absolute_eligible_rank_change"]
        .idxmax(),
        [
            "player_id",
            "allocation_policy",
            "absolute_eligible_rank_change",
        ],
    ].rename(
        columns={
            "allocation_policy": ("max_eligible_rank_change_allocation_policy"),
            "absolute_eligible_rank_change": ("max_allocation_absolute_eligible_rank_change"),
        }
    )
    if delete_game.empty:
        deletion = pd.DataFrame(
            columns=[
                "player_id",
                "max_delete_game_absolute_coefficient_change",
                "max_delete_game_id",
            ]
        )
    else:
        deletion = delete_game.loc[
            delete_game.groupby("player_id")["absolute_coefficient_change"].idxmax(),
            ["player_id", "game_id", "absolute_coefficient_change"],
        ].rename(
            columns={
                "game_id": "max_delete_game_id",
                "absolute_coefficient_change": ("max_delete_game_absolute_coefficient_change"),
            }
        )

    result = base
    for diagnostic in (
        lambda_summary,
        chronology,
        bootstrap,
        context,
        adjustment,
        allocation_coefficient_rows,
        allocation_rank_rows,
        deletion,
    ):
        result = result.merge(
            diagnostic,
            on="player_id",
            how="left",
            validate="one_to_one",
        )
    return result.sort_values("rank", kind="stable").reset_index(drop=True)


def run_rapm_diagnostics(
    season: str,
    *,
    source_run_id: str | None = None,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    model_artifacts_dir: Path | str = Path("artifacts/models"),
    reports_dir: Path | str = Path("artifacts/reports"),
    sensitivity_lambdas: tuple[float, ...] = DEFAULT_SENSITIVITY_LAMBDAS,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 7,
    influence_player_count: int = 25,
    influence_stints_per_player: int = 5,
    delete_games_per_player: int = 3,
    allocation_policies: tuple[
        PossessionAllocationPolicy,
        ...,
    ] = POSSESSION_ALLOCATION_POLICIES,
) -> tuple[RapmDiagnosticsManifest, Path]:
    """Run every RAPM stability diagnostic and publish immutable reports."""

    source_dir = _resolve_source_run(
        season,
        Path(model_artifacts_dir),
        source_run_id,
    )
    source_manifest = validate_baseline_run(source_dir)
    selected_lambda = source_manifest.selected_rapm_lambda
    if selected_lambda <= 0:
        raise ValueError("RAPM diagnostics require positive regularization")
    if selected_lambda not in sensitivity_lambdas:
        raise ValueError("Sensitivity lambdas must include the selected RAPM lambda")
    stints = read_rapm_stints(season, analytical_dir)
    dataset_part = Path(analytical_dir) / "rapm_stints" / season / "regular" / "part-00000.parquet"
    if _sha256_file(dataset_part) != source_manifest.dataset_part_sha256:
        raise ValueError("Current RAPM stints do not match the source model run")
    rankings = pd.read_parquet(source_dir / "player_rankings.parquet")
    game_splits = pd.read_parquet(source_dir / "game_splits.parquet")
    player_columns = {
        int(identifier): int(column)
        for identifier, column in json.loads(
            (source_dir / "player_columns.json").read_text()
        ).items()
    }
    player_ids = tuple(
        identifier
        for identifier, _ in sorted(
            player_columns.items(),
            key=lambda item: item[1],
        )
    )
    player_matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        player_columns,
        multiple=True,
    )

    print("lambda sensitivity", flush=True)
    lambda_coefficients, lambda_summary = lambda_sensitivity(
        stints,
        player_matrix,
        player_ids,
        rankings,
        sensitivity_lambdas,
        selected_lambda,
    )
    print("chronological stability", flush=True)
    chronological_coefficients, chronological_summary = chronological_stability(
        stints,
        player_matrix,
        player_ids,
        rankings,
        game_splits,
        selected_lambda,
        source_manifest.minimum_ranking_possessions,
    )
    print(f"game-block bootstrap ({bootstrap_samples} samples)", flush=True)
    bootstrap_coefficients, bootstrap_summary = bootstrap_stability(
        stints,
        player_matrix,
        player_ids,
        rankings,
        selected_lambda,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    print("context concentration and raw adjustment", flush=True)
    concentration = context_concentration(stints, rankings)
    raw_adjusted = raw_adjusted_comparison(rankings)
    print("stint and delete-game influence", flush=True)
    influential_stints, influential_games, delete_game = influence_diagnostics(
        stints,
        player_matrix,
        player_ids,
        rankings,
        selected_lambda,
        influence_player_count=influence_player_count,
        stints_per_player=influence_stints_per_player,
        delete_games_per_player=delete_games_per_player,
    )
    print("possession-allocation sensitivity", flush=True)
    lineup_stints = pd.read_parquet(Path(curated_dir) / "lineup_stints" / season / "regular")
    possession_segments = pd.read_parquet(
        Path(curated_dir) / "possession_segments" / season / "regular"
    )
    allocation_coefficients, allocation_metrics = allocation_sensitivity(
        lineup_stints,
        possession_segments,
        rankings,
        player_ids,
        player_columns,
        game_splits,
        selected_lambda,
        allocation_policies,
    )
    allocation_summary = possession_allocation_summary(possession_segments)
    player_diagnostics = player_diagnostics_summary(
        rankings,
        lambda_coefficients,
        chronological_summary,
        bootstrap_summary,
        concentration,
        raw_adjusted,
        allocation_coefficients,
        delete_game,
    )
    outputs = {
        "player_diagnostics.parquet": player_diagnostics,
        "lambda_coefficients.parquet": lambda_coefficients,
        "lambda_summary.parquet": lambda_summary,
        "chronological_coefficients.parquet": chronological_coefficients,
        "chronological_summary.parquet": chronological_summary,
        "bootstrap_coefficients.parquet": bootstrap_coefficients,
        "bootstrap_summary.parquet": bootstrap_summary,
        "context_concentration.parquet": concentration,
        "raw_adjusted.parquet": raw_adjusted,
        "influential_stints.parquet": influential_stints,
        "influential_games.parquet": influential_games,
        "delete_game_influence.parquet": delete_game,
        "allocation_coefficients.parquet": allocation_coefficients,
        "allocation_metrics.parquet": allocation_metrics,
        "allocation_summary.parquet": allocation_summary,
    }
    return _write_diagnostics(
        season,
        source_dir,
        source_manifest,
        outputs,
        sensitivity_lambdas,
        bootstrap_samples,
        bootstrap_seed,
        influence_player_count,
        influence_stints_per_player,
        delete_games_per_player,
        allocation_policies,
        reports_dir,
    )


def validate_diagnostics_run(
    run_dir: Path | str,
) -> RapmDiagnosticsManifest:
    """Require every diagnostics artifact to match its manifest."""

    root = Path(run_dir)
    manifest = RapmDiagnosticsManifest.model_validate_json((root / "manifest.json").read_text())
    actual = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    expected = {artifact.filename for artifact in manifest.artifacts}
    if actual != expected:
        raise ValueError("Diagnostics files do not match the manifest")
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if path.stat().st_size != artifact.byte_count:
            raise ValueError(f"Diagnostics artifact byte count changed: {artifact.filename}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"Diagnostics artifact hash changed: {artifact.filename}")
        if artifact.row_count is not None:
            if len(pd.read_parquet(path)) != artifact.row_count:
                raise ValueError(f"Diagnostics artifact rows changed: {artifact.filename}")
    return manifest


def _coefficient_frame(
    player_ids: tuple[int, ...],
    coefficients: np.ndarray,
    eligible_ids: set[int],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "player_id": player_ids,
            "coefficient": coefficients,
            "rank": _ranks(coefficients),
        }
    )
    frame["eligible"] = frame["player_id"].isin(eligible_ids)
    eligible = frame.loc[frame["eligible"]].sort_values(
        ["coefficient", "player_id"],
        ascending=[False, True],
        kind="stable",
    )
    eligible_ranks = pd.Series(
        np.arange(1, len(eligible) + 1),
        index=eligible.index,
        dtype="Int64",
    )
    frame["eligible_rank"] = eligible_ranks
    return frame


def _ranks(coefficients: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(coefficients), kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _effective_count(exposures: Any) -> float:
    values = np.asarray(list(exposures), dtype=float)
    probabilities = values / values.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _ridge_leverage(
    matrix: sparse.csr_matrix,
    normalized_weights: np.ndarray,
    sklearn_alpha: float | None,
    *,
    batch_size: int = 2_000,
) -> np.ndarray:
    if sklearn_alpha is None or sklearn_alpha <= 0:
        raise ValueError("Ridge leverage requires positive regularization")
    weighted_matrix = matrix.multiply(normalized_weights[:, None])
    coefficient_gram = (matrix.T @ weighted_matrix).toarray()
    coefficient_gram.flat[:: coefficient_gram.shape[0] + 1] += sklearn_alpha
    intercept_cross = np.asarray(matrix.T @ normalized_weights).reshape(-1)
    gram = np.empty(
        (matrix.shape[1] + 1, matrix.shape[1] + 1),
        dtype=float,
    )
    gram[0, 0] = normalized_weights.sum()
    gram[0, 1:] = intercept_cross
    gram[1:, 0] = intercept_cross
    gram[1:, 1:] = coefficient_gram
    inverse = np.linalg.pinv(gram, hermitian=True)
    leverage = np.empty(matrix.shape[0], dtype=float)
    for start in range(0, matrix.shape[0], batch_size):
        stop = min(start + batch_size, matrix.shape[0])
        block = matrix[start:stop]
        linear = np.asarray(block @ inverse[1:, 0]).reshape(-1)
        quadratic = np.sum(
            np.asarray(block @ inverse[1:, 1:]) * block.toarray(),
            axis=1,
        )
        leverage[start:stop] = normalized_weights[start:stop] * (
            inverse[0, 0] + 2.0 * linear + quadratic
        )
    return np.clip(leverage, 0.0, 1.0 - 1e-12)


def _player_stint_long(stints: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side, sign in (("home", 1.0), ("away", -1.0)):
        player_lists = stints[f"{side}_player_ids"]
        frame = pd.DataFrame(
            {
                "row_index": np.repeat(np.arange(len(stints)), 5),
                "player_id": np.concatenate(
                    [np.asarray(players, dtype=np.int64) for players in player_lists]
                ),
                "sign": sign,
            }
        )
        for column in (
            "game_id",
            "stint_index",
            "possessions",
            "home_team_tricode",
            "away_team_tricode",
        ):
            frame[column] = np.repeat(stints[column].to_numpy(), 5)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _resolve_source_run(
    season: str,
    model_artifacts_dir: Path,
    source_run_id: str | None,
) -> Path:
    season_dir = model_artifacts_dir / "rapm" / season
    if source_run_id is None:
        source_run_id = json.loads((season_dir / "latest.json").read_text())["run_id"]
    source_dir = season_dir / source_run_id
    if not source_dir.is_dir():
        raise ValueError(f"RAPM source run does not exist: {source_dir}")
    return source_dir


def _write_diagnostics(
    season: str,
    source_dir: Path,
    source_manifest: BaselineRunManifest,
    outputs: dict[str, pd.DataFrame],
    sensitivity_lambdas: tuple[float, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
    influence_player_count: int,
    influence_stints_per_player: int,
    delete_games_per_player: int,
    allocation_policies: tuple[PossessionAllocationPolicy, ...],
    reports_dir: Path | str,
) -> tuple[RapmDiagnosticsManifest, Path]:
    now = datetime.now(UTC)
    run_id = f"diagnostics-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    season_dir = Path(reports_dir) / "rapm" / season
    run_dir = season_dir / run_id
    temporary_dir = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        for filename, frame in outputs.items():
            frame.to_parquet(temporary_dir / filename, index=False)
        artifacts = tuple(
            ArtifactRecord(
                filename=path.name,
                row_count=len(outputs[path.name]),
                byte_count=path.stat().st_size,
                sha256=_sha256_file(path),
            )
            for path in sorted(temporary_dir.iterdir())
        )
        manifest = RapmDiagnosticsManifest(
            run_id=run_id,
            created_at=now,
            season=season,
            diagnostics_code_version=diagnostics_code_fingerprint(),
            source_model_run_id=source_manifest.run_id,
            source_model_manifest_sha256=_sha256_file(source_dir / "manifest.json"),
            dataset_part_sha256=source_manifest.dataset_part_sha256,
            selected_rapm_lambda=source_manifest.selected_rapm_lambda,
            sensitivity_lambdas=sensitivity_lambdas,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            minimum_ranking_possessions=(source_manifest.minimum_ranking_possessions),
            influence_player_count=influence_player_count,
            influence_stints_per_player=influence_stints_per_player,
            delete_games_per_player=delete_games_per_player,
            allocation_policies=allocation_policies,
            player_count=source_manifest.player_count,
            game_count=source_manifest.game_count,
            stint_count=source_manifest.stint_count,
            artifacts=artifacts,
        )
        (temporary_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        temporary_dir.replace(run_dir)
        validate_diagnostics_run(run_dir)
        latest_path = season_dir / "latest.json"
        temporary_latest = latest_path.with_suffix(".json.tmp")
        temporary_latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        temporary_latest.replace(latest_path)
        return manifest, run_dir
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Values must be comma-separated numbers") from exc
    if len(values) < 3 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Provide at least three positive values")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Values must be unique")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run stability and sensitivity diagnostics for one RAPM run."
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument("--source-run-id")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--model-artifacts-dir", default="artifacts/models")
    parser.add_argument("--reports-dir", default="artifacts/reports")
    parser.add_argument(
        "--sensitivity-lambdas",
        type=_parse_float_tuple,
        default=DEFAULT_SENSITIVITY_LAMBDAS,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--influence-player-count", type=int, default=25)
    parser.add_argument("--influence-stints-per-player", type=int, default=5)
    parser.add_argument("--delete-games-per-player", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, run_dir = run_rapm_diagnostics(
        args.season,
        source_run_id=args.source_run_id,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        model_artifacts_dir=args.model_artifacts_dir,
        reports_dir=args.reports_dir,
        sensitivity_lambdas=args.sensitivity_lambdas,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        influence_player_count=args.influence_player_count,
        influence_stints_per_player=args.influence_stints_per_player,
        delete_games_per_player=args.delete_games_per_player,
    )
    print(
        f"{manifest.season} RAPM diagnostics: "
        f"players={manifest.player_count}, games={manifest.game_count}, "
        f"bootstrap_samples={manifest.bootstrap_samples}; run={run_dir}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
