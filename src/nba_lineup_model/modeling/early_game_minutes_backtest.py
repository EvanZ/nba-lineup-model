#!/usr/bin/env python3
"""Tune and evaluate an early-game, cutoff-normalized minutes updater."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.in_season_minutes import (
    CONDITIONAL_OUTCOMES,
    HISTORICAL_AVAILABILITY_MAP,
    REGULATION_TEAM_MINUTES,
    UNKNOWN,
)

DEFAULT_CUTOFFS = (36, 40, 44, 48)
DEFAULT_LOOKBACKS: tuple[int | None, ...] = (1, 2, 3, 5, 10, None)
DEFAULT_PRIOR_WEIGHTS = (0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0)
TEAM_SECONDS_TOLERANCE = 2.0


def _read_parts(root: Path, columns: list[str] | None = None) -> pd.DataFrame:
    paths = sorted(root.glob("part-*.parquet"))
    if not paths:
        raise ValueError(f"No parquet parts found under {root}")
    return pd.concat(
        [pd.read_parquet(path, columns=columns) for path in paths],
        ignore_index=True,
    )


def _digest_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _normalize_availability(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id",
        "game_date",
        "team",
        "player_id",
        "player_name",
        "minutes",
        "availability_state",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError("Availability frame lacks: " + ", ".join(missing))
    output = frame.copy()
    states = output["availability_state"].astype("string")
    output["availability_state"] = (
        states.map(HISTORICAL_AVAILABILITY_MAP).fillna(states).fillna(UNKNOWN)
    )
    output["minutes"] = pd.to_numeric(output["minutes"], errors="raise").astype(float)
    output["game_id"] = output["game_id"].astype(str)
    output["game_date"] = pd.to_datetime(output["game_date"], utc=True)
    return output


def _side_player_seconds(
    stints: pd.DataFrame,
    *,
    side: str,
    cutoff_minutes: int,
) -> pd.DataFrame:
    cutoff_seconds = float(cutoff_minutes * 60)
    start = pd.to_numeric(stints["start_elapsed_game_seconds"], errors="raise").astype(float)
    end = pd.to_numeric(stints["end_elapsed_game_seconds"], errors="raise").astype(float)
    clipped_seconds = (np.minimum(end, cutoff_seconds) - start).clip(lower=0.0)
    team_column = f"catalog_{side}_team_tricode"
    player_column = f"{side}_player_ids"
    output = stints[["game_id", team_column, player_column]].copy()
    output["game_id"] = output["game_id"].astype(str)
    output["stint_seconds"] = clipped_seconds
    output = output.loc[output["stint_seconds"].gt(0.0)].explode(player_column)
    output = output.rename(columns={team_column: "team", player_column: "player_id"})
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    return (
        output.groupby(["game_id", "team", "player_id"], as_index=False)["stint_seconds"]
        .sum()
        .assign(cutoff_minutes=cutoff_minutes)
    )


def _actual_player_seconds(stints: pd.DataFrame, *, side: str) -> pd.DataFrame:
    start = pd.to_numeric(stints["start_elapsed_game_seconds"], errors="raise").astype(
        float
    )
    end = pd.to_numeric(stints["end_elapsed_game_seconds"], errors="raise").astype(float)
    team_column = f"catalog_{side}_team_tricode"
    player_column = f"{side}_player_ids"
    output = stints[["game_id", team_column, player_column]].copy()
    output["game_id"] = output["game_id"].astype(str)
    output["actual_seconds"] = (end - start).clip(lower=0.0)
    output = output.loc[output["actual_seconds"].gt(0.0)].explode(player_column)
    output = output.rename(columns={team_column: "team", player_column: "player_id"})
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    return output.groupby(
        ["game_id", "team", "player_id"], as_index=False
    )["actual_seconds"].sum()


def cutoff_role_observations(
    stints: pd.DataFrame,
    availability: pd.DataFrame,
    *,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one cutoff-normalized role for every player-game availability row."""

    required = {
        "game_id",
        "start_elapsed_game_seconds",
        "end_elapsed_game_seconds",
        "home_player_ids",
        "away_player_ids",
        "catalog_home_team_tricode",
        "catalog_away_team_tricode",
    }
    missing = sorted(required - set(stints))
    if missing:
        raise ValueError("Lineup stints lack: " + ", ".join(missing))
    normalized = _normalize_availability(availability)
    role_frames: list[pd.DataFrame] = []
    valid_keys: set[tuple[str, str]] | None = None
    coverage_rows: list[dict[str, object]] = []
    for cutoff in cutoffs:
        sides = pd.concat(
            [
                _side_player_seconds(stints, side="home", cutoff_minutes=cutoff),
                _side_player_seconds(stints, side="away", cutoff_minutes=cutoff),
            ],
            ignore_index=True,
        )
        totals = sides.groupby(["game_id", "team"], as_index=False)["stint_seconds"].sum()
        expected = 5.0 * cutoff * 60.0
        totals["coverage_error_seconds"] = (totals["stint_seconds"] - expected).abs()
        cutoff_valid = {
            (str(row.game_id), str(row.team))
            for row in totals.itertuples(index=False)
            if float(row.coverage_error_seconds) <= TEAM_SECONDS_TOLERANCE
        }
        valid_keys = cutoff_valid if valid_keys is None else valid_keys & cutoff_valid
        coverage_rows.append(
            {
                "cutoff_minutes": cutoff,
                "game_teams": int(len(totals)),
                "complete_game_teams": int(len(cutoff_valid)),
            }
        )
        sides["observed_role_minutes"] = (
            sides["stint_seconds"] / (cutoff * 60.0) * 48.0
        )
        role_frames.append(
            sides[["game_id", "team", "player_id", "cutoff_minutes", "observed_role_minutes"]]
        )
    common = valid_keys or set()
    if not common:
        raise ValueError("No game-teams have complete stint coverage at every cutoff")
    key_frame = pd.DataFrame(sorted(common), columns=["game_id", "team"])
    base = normalized.merge(key_frame, on=["game_id", "team"], how="inner")
    actual = pd.concat(
        [
            _actual_player_seconds(stints, side="home"),
            _actual_player_seconds(stints, side="away"),
        ],
        ignore_index=True,
    ).merge(key_frame, on=["game_id", "team"], how="inner")
    actual_keys = actual[["game_id", "team", "player_id"]]
    missing_actual_players = actual_keys.merge(
        base[["game_id", "team", "player_id"]],
        on=["game_id", "team", "player_id"],
        how="left",
        indicator=True,
    )
    missing_actual_players = missing_actual_players.loc[
        missing_actual_players["_merge"].eq("left_only")
    ]
    if not missing_actual_players.empty:
        examples = missing_actual_players.head(5).to_dict("records")
        raise ValueError(f"Stint players missing from availability: {examples}")
    base = base.merge(
        actual,
        on=["game_id", "team", "player_id"],
        how="left",
        validate="one_to_one",
    )
    base["actual_seconds"] = base["actual_seconds"].fillna(0.0)
    played_state_overrides = int(
        (
            base["actual_seconds"].gt(0.0)
            & ~base["availability_state"].eq("played")
        ).sum()
    )
    base.loc[base["actual_seconds"].gt(0.0), "availability_state"] = "played"
    team_actual_seconds = base.groupby(["game_id", "team"], sort=False)[
        "actual_seconds"
    ].transform("sum")
    if team_actual_seconds.le(0.0).any():
        raise ValueError("Stints contain a game-team with zero actual seconds")
    base["target_minutes"] = (
        base["actual_seconds"] * REGULATION_TEAM_MINUTES / team_actual_seconds
    )
    roles = pd.concat(role_frames, ignore_index=True)
    roles = roles.merge(key_frame, on=["game_id", "team"], how="inner")
    # Build every cutoff row explicitly so coach DNPs and players used only
    # after an early cutoff receive zero.
    cutoff_frame = pd.DataFrame({"cutoff_minutes": list(cutoffs)})
    complete = base.merge(cutoff_frame, how="cross").merge(
        roles,
        on=["game_id", "team", "player_id", "cutoff_minutes"],
        how="left",
        validate="one_to_one",
    )
    complete["observed_role_minutes"] = complete["observed_role_minutes"].fillna(0.0)
    eligible = complete["availability_state"].isin(CONDITIONAL_OUTCOMES)
    team_totals = (
        complete.loc[eligible]
        .groupby(["game_id", "team", "cutoff_minutes"])["observed_role_minutes"]
        .sum()
    )
    if not np.allclose(team_totals.to_numpy(), REGULATION_TEAM_MINUTES, atol=1e-6):
        deviations = team_totals.loc[
            ~np.isclose(team_totals.to_numpy(), REGULATION_TEAM_MINUTES, atol=1e-6)
        ]
        examples = ", ".join(
            f"{game_id}/{team}/{cutoff}={value:.1f}"
            for (game_id, team, cutoff), value in deviations.head(5).items()
        )
        raise ValueError(
            "Eligible cutoff roles do not conserve 240 team minutes; "
            f"{len(deviations)} failures (examples: {examples})"
        )
    audit = {
        "cutoff_coverage": coverage_rows,
        "common_complete_game_teams": len(common),
        "common_complete_games": int(key_frame["game_id"].nunique()),
        "availability_rows": int(len(base)),
        "eligible_rows": int(
            base["availability_state"].isin(CONDITIONAL_OUTCOMES).sum()
        ),
        "played_state_overrides_from_stints": played_state_overrides,
    }
    return complete, audit


def _priors(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"player_id", "player_name", "predicted_minutes_per_available_game"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError("Preseason priors lack: " + ", ".join(missing))
    output = frame[
        ["player_id", "player_name", "predicted_minutes_per_available_game"]
    ].copy()
    output = output.rename(
        columns={"predicted_minutes_per_available_game": "preseason_role_minutes"}
    )
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    output["preseason_role_minutes"] = pd.to_numeric(
        output["preseason_role_minutes"], errors="raise"
    ).astype(float)
    if output["player_id"].duplicated().any():
        raise ValueError("Preseason priors contain duplicate player IDs")
    return output


def chronological_evaluation(
    panel: pd.DataFrame,
    priors: pd.DataFrame,
    *,
    cutoff_minutes: int,
    lookback_games: int | None,
    prior_weight: float,
) -> pd.DataFrame:
    """Score one configuration with strict prior-day information."""

    if cutoff_minutes not in DEFAULT_CUTOFFS:
        raise ValueError(f"Unsupported cutoff: {cutoff_minutes}")
    if lookback_games is not None and lookback_games <= 0:
        raise ValueError("Lookback games must be positive or None")
    if not np.isfinite(prior_weight) or prior_weight <= 0.0:
        raise ValueError("Preseason prior weight must be finite and positive")
    frame = panel.loc[
        panel["cutoff_minutes"].eq(cutoff_minutes)
        & panel["availability_state"].isin(CONDITIONAL_OUTCOMES)
    ].merge(
        _priors(priors)[["player_id", "preseason_role_minutes"]],
        on="player_id",
        how="inner",
        validate="many_to_one",
    )
    frame = frame.sort_values(["game_date", "game_id", "player_id"], kind="stable")
    histories: dict[int, list[float]] = defaultdict(list)
    records: list[dict[str, object]] = []
    for game_date, day in frame.groupby("game_date", sort=True):
        for row in day.itertuples(index=False):
            player_id = int(row.player_id)
            history = histories[player_id]
            selected = history if lookback_games is None else history[-lookback_games:]
            observed_games = len(selected)
            updated = (
                prior_weight * float(row.preseason_role_minutes) + sum(selected)
            ) / (prior_weight + observed_games)
            records.append(
                {
                    "game_id": str(row.game_id),
                    "game_date": game_date,
                    "team": str(row.team),
                    "player_id": player_id,
                    "player_name": str(row.player_name),
                    "availability_state": str(row.availability_state),
                    "target_minutes": float(row.target_minutes),
                    "observed_role_minutes": float(row.observed_role_minutes),
                    "preseason_role_minutes": float(row.preseason_role_minutes),
                    "updated_role_minutes": float(updated),
                    "prior_observations": observed_games,
                    "cutoff_minutes": cutoff_minutes,
                    "lookback_games": lookback_games,
                    "prior_weight": prior_weight,
                }
            )
        for row in day.itertuples(index=False):
            histories[int(row.player_id)].append(float(row.observed_role_minutes))
    output = pd.DataFrame(records)
    if output.empty:
        raise ValueError("Configuration produced no evaluation rows")
    output["frozen_absolute_error"] = (
        output["preseason_role_minutes"] - output["target_minutes"]
    ).abs()
    output["updated_absolute_error"] = (
        output["updated_role_minutes"] - output["target_minutes"]
    ).abs()
    return output


def evaluation_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    frozen_error = frame["preseason_role_minutes"] - frame["target_minutes"]
    updated_error = frame["updated_role_minutes"] - frame["target_minutes"]
    frozen_mae = float(frozen_error.abs().mean())
    updated_mae = float(updated_error.abs().mean())
    return {
        "rows": int(len(frame)),
        "games": int(frame["game_id"].nunique()),
        "players": int(frame["player_id"].nunique()),
        "frozen_mae": frozen_mae,
        "updated_mae": updated_mae,
        "mae_improvement": frozen_mae - updated_mae,
        "mae_improvement_pct": 100.0 * (frozen_mae - updated_mae) / frozen_mae,
        "frozen_rmse": float(np.sqrt(np.square(frozen_error).mean())),
        "updated_rmse": float(np.sqrt(np.square(updated_error).mean())),
        "frozen_bias": float(frozen_error.mean()),
        "updated_bias": float(updated_error.mean()),
    }


def _paired_game_bootstrap(
    frame: pd.DataFrame,
    *,
    samples: int = 2000,
    seed: int = 20261026,
) -> dict[str, float | int]:
    by_game = (
        frame.assign(
            improvement=lambda value: value["frozen_absolute_error"]
            - value["updated_absolute_error"]
        )
        .groupby("game_id", as_index=False)
        .agg(improvement_sum=("improvement", "sum"), rows=("improvement", "size"))
    )
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    game_count = len(by_game)
    sums = by_game["improvement_sum"].to_numpy(dtype=float)
    counts = by_game["rows"].to_numpy(dtype=float)
    for index in range(samples):
        selected = rng.integers(0, game_count, size=game_count)
        values[index] = sums[selected].sum() / counts[selected].sum()
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "samples": samples,
        "seed": seed,
        "mean_mae_improvement": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "probability_improvement": float(np.mean(values > 0.0)),
    }


def _paired_candidate_bootstrap(
    selected: pd.DataFrame,
    control: pd.DataFrame,
    *,
    samples: int = 2000,
    seed: int = 20261026,
) -> dict[str, float | int]:
    keys = ["game_id", "player_id"]
    paired = selected[keys + ["updated_absolute_error"]].merge(
        control[keys + ["updated_absolute_error"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_selected", "_control"),
    )
    if len(paired) != len(selected) or len(paired) != len(control):
        raise ValueError("Selected and control evaluations do not contain identical rows")
    paired["improvement"] = (
        paired["updated_absolute_error_control"]
        - paired["updated_absolute_error_selected"]
    )
    by_game = paired.groupby("game_id", as_index=False).agg(
        improvement_sum=("improvement", "sum"), rows=("improvement", "size")
    )
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    game_count = len(by_game)
    sums = by_game["improvement_sum"].to_numpy(dtype=float)
    counts = by_game["rows"].to_numpy(dtype=float)
    for index in range(samples):
        chosen = rng.integers(0, game_count, size=game_count)
        values[index] = sums[chosen].sum() / counts[chosen].sum()
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "rows": int(len(paired)),
        "samples": samples,
        "seed": seed,
        "selected_mae_advantage": float(paired["improvement"].mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "probability_selected_better": float(np.mean(values > 0.0)),
    }


def _segment_metrics(
    frame: pd.DataFrame,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    month_rows: list[dict[str, object]] = []
    months = frame.assign(month=frame["game_date"].dt.strftime("%Y-%m"))
    for month, group in months.groupby("month", sort=True):
        month_rows.append({"month": str(month), **evaluation_metrics(group)})
    bins = [-1, 0, 1, 2, 5, 10, np.inf]
    labels = ["0", "1", "2", "3-5", "6-10", "11+"]
    observed = frame.assign(
        observation_band=pd.cut(
            frame["prior_observations"], bins=bins, labels=labels
        ).astype("string")
    )
    observation_rows: list[dict[str, object]] = []
    for band in labels:
        group = observed.loc[observed["observation_band"].eq(band)]
        if not group.empty:
            observation_rows.append(
                {"prior_observations": band, **evaluation_metrics(group)}
            )
    state_rows: list[dict[str, object]] = []
    for state, group in frame.groupby("availability_state", sort=True):
        state_rows.append({"availability_state": str(state), **evaluation_metrics(group)})
    minute_bins = [-np.inf, 8.0, 16.0, 24.0, 32.0, np.inf]
    minute_labels = ["0-8", "8-16", "16-24", "24-32", "32+"]
    prior_bands = frame.assign(
        preseason_minutes_band=pd.cut(
            frame["preseason_role_minutes"],
            bins=minute_bins,
            labels=minute_labels,
            right=False,
        ).astype("string")
    )
    prior_minute_rows: list[dict[str, object]] = []
    for band in minute_labels:
        group = prior_bands.loc[prior_bands["preseason_minutes_band"].eq(band)]
        if not group.empty:
            prior_minute_rows.append(
                {"preseason_minutes_band": band, **evaluation_metrics(group)}
            )
    return month_rows, observation_rows, state_rows, prior_minute_rows


def _player_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["player_id", "player_name"], as_index=False)
        .agg(
            team=("team", "last"),
            games=("game_id", "nunique"),
            target_minutes=("target_minutes", "mean"),
            frozen_minutes=("preseason_role_minutes", "mean"),
            updated_minutes=("updated_role_minutes", "mean"),
            frozen_mae=("frozen_absolute_error", "mean"),
            updated_mae=("updated_absolute_error", "mean"),
        )
        .assign(improvement=lambda value: value["frozen_mae"] - value["updated_mae"])
        .sort_values(["improvement", "games"], ascending=[False, False], kind="stable")
    )


def _grid(
    panel: pd.DataFrame,
    priors: pd.DataFrame,
    *,
    cutoffs: tuple[int, ...],
    lookbacks: tuple[int | None, ...],
    prior_weights: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cutoff in cutoffs:
        for lookback in lookbacks:
            for weight in prior_weights:
                evaluation = chronological_evaluation(
                    panel,
                    priors,
                    cutoff_minutes=cutoff,
                    lookback_games=lookback,
                    prior_weight=weight,
                )
                rows.append(
                    {
                        "cutoff_minutes": cutoff,
                        "lookback_games": lookback,
                        "lookback_label": "all" if lookback is None else str(lookback),
                        "prior_weight": weight,
                        **evaluation_metrics(evaluation),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["updated_mae", "updated_rmse", "cutoff_minutes", "lookback_label", "prior_weight"],
        kind="stable",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    cutoffs = tuple(int(value) for value in args.cutoffs)
    lookbacks = tuple(
        None if str(value).casefold() == "all" else int(value) for value in args.lookbacks
    )
    weights = tuple(float(value) for value in args.prior_weights)
    train_stints_root = Path(args.train_stints_dir)
    holdout_stints_root = Path(args.holdout_stints_dir)
    train_availability_path = Path(args.train_availability_path)
    holdout_availability_path = Path(args.holdout_availability_path)
    train_priors_path = Path(args.train_priors_path)
    holdout_priors_path = Path(args.holdout_priors_path)
    stint_columns = [
        "game_id",
        "start_elapsed_game_seconds",
        "end_elapsed_game_seconds",
        "home_player_ids",
        "away_player_ids",
        "catalog_home_team_tricode",
        "catalog_away_team_tricode",
    ]
    train_panel, train_audit = cutoff_role_observations(
        _read_parts(train_stints_root, stint_columns),
        pd.read_parquet(train_availability_path),
        cutoffs=cutoffs,
    )
    holdout_panel, holdout_audit = cutoff_role_observations(
        _read_parts(holdout_stints_root, stint_columns),
        pd.read_parquet(holdout_availability_path),
        cutoffs=cutoffs,
    )
    train_priors = pd.read_parquet(train_priors_path)
    holdout_priors = pd.read_parquet(holdout_priors_path)
    grid = _grid(
        train_panel,
        train_priors,
        cutoffs=cutoffs,
        lookbacks=lookbacks,
        prior_weights=weights,
    )
    selected = grid.iloc[0]
    selected_lookback = (
        None if str(selected["lookback_label"]) == "all" else int(selected["lookback_games"])
    )
    holdout = chronological_evaluation(
        holdout_panel,
        holdout_priors,
        cutoff_minutes=int(selected["cutoff_minutes"]),
        lookback_games=selected_lookback,
        prior_weight=float(selected["prior_weight"]),
    )
    holdout_metrics = evaluation_metrics(holdout)
    bootstrap = _paired_game_bootstrap(holdout)
    by_month, by_observations, by_state, by_prior_minutes = _segment_metrics(holdout)
    players = _player_summary(holdout)
    minimum_player_games = int(args.minimum_player_games)
    eligible_players = players.loc[players["games"].ge(minimum_player_games)]
    cutoff_controls: list[dict[str, object]] = []
    control_48: pd.DataFrame | None = None
    for cutoff in cutoffs:
        evaluation = chronological_evaluation(
            holdout_panel,
            holdout_priors,
            cutoff_minutes=cutoff,
            lookback_games=selected_lookback,
            prior_weight=float(selected["prior_weight"]),
        )
        cutoff_controls.append({"cutoff_minutes": cutoff, **evaluation_metrics(evaluation)})
        if cutoff == 48:
            control_48 = evaluation
    selected_vs_48 = (
        _paired_candidate_bootstrap(holdout, control_48)
        if control_48 is not None and int(selected["cutoff_minutes"]) != 48
        else None
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "study": "empirical_early_game_minutes_v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "train_season": str(args.train_season),
        "holdout_season": str(args.holdout_season),
        "grid": {
            "cutoffs": list(cutoffs),
            "lookbacks": ["all" if value is None else value for value in lookbacks],
            "prior_weights": list(weights),
            "combinations": int(len(grid)),
        },
        "selection_rule": "lowest train updated MAE, then lowest train updated RMSE",
        "selected": {
            "cutoff_minutes": int(selected["cutoff_minutes"]),
            "lookback_games": "all" if selected_lookback is None else selected_lookback,
            "prior_weight": float(selected["prior_weight"]),
            "train_metrics": {
                key: selected[key]
                for key in (
                    "rows",
                    "games",
                    "players",
                    "frozen_mae",
                    "updated_mae",
                    "mae_improvement",
                    "mae_improvement_pct",
                    "frozen_rmse",
                    "updated_rmse",
                )
            },
        },
        "holdout_metrics": holdout_metrics,
        "holdout_paired_game_bootstrap": bootstrap,
        "holdout_by_month": by_month,
        "holdout_by_prior_observations": by_observations,
        "holdout_by_availability_state": by_state,
        "holdout_by_preseason_minutes": by_prior_minutes,
        "holdout_cutoff_controls": cutoff_controls,
        "holdout_selected_vs_48_minute_control": selected_vs_48,
        "representative_players": {
            "minimum_games": minimum_player_games,
            "most_improved": eligible_players.head(12).to_dict("records"),
            "most_regressed": eligible_players.tail(12)
            .sort_values("improvement")
            .to_dict("records"),
        },
        "coverage": {"train": train_audit, "holdout": holdout_audit},
        "input_lineage": {
            "train_stints_sha256": _digest_paths(
                sorted(train_stints_root.glob("part-*.parquet"))
            ),
            "holdout_stints_sha256": _digest_paths(
                sorted(holdout_stints_root.glob("part-*.parquet"))
            ),
            "train_availability_sha256": hashlib.sha256(
                train_availability_path.read_bytes()
            ).hexdigest(),
            "holdout_availability_sha256": hashlib.sha256(
                holdout_availability_path.read_bytes()
            ).hexdigest(),
            "train_priors_sha256": hashlib.sha256(train_priors_path.read_bytes()).hexdigest(),
            "holdout_priors_sha256": hashlib.sha256(
                holdout_priors_path.read_bytes()
            ).hexdigest(),
        },
    }
    output = Path(args.output_dir)
    output_files = (
        output / "tuning_grid.csv",
        output / "holdout_evaluation.parquet",
        output / "holdout_player_summary.csv",
        output / "summary.json",
    )
    existing = [str(path) for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite immutable study artifacts: " + ", ".join(existing)
        )
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "tuning_grid.csv", index=False)
    holdout.to_parquet(output / "holdout_evaluation.parquet", index=False)
    players.to_csv(output / "holdout_player_summary.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    return result


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-season", default="2024-25")
    parser.add_argument("--holdout-season", default="2025-26")
    parser.add_argument("--train-stints-dir", type=Path, required=True)
    parser.add_argument("--holdout-stints-dir", type=Path, required=True)
    parser.add_argument("--train-availability-path", type=Path, required=True)
    parser.add_argument("--holdout-availability-path", type=Path, required=True)
    parser.add_argument("--train-priors-path", type=Path, required=True)
    parser.add_argument("--holdout-priors-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoffs", nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument(
        "--lookbacks", nargs="+", default=("1", "2", "3", "5", "10", "all")
    )
    parser.add_argument("--prior-weights", nargs="+", default=DEFAULT_PRIOR_WEIGHTS)
    parser.add_argument("--minimum-player-games", type=int, default=20)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
