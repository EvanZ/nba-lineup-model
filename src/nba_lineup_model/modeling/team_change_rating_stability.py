"""Audit whether completed player-RAPM changes are less stable after a team move."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_PANEL_PATH
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import (
    MODEL_NAME,
)

DEFAULT_ARTIFACT_SEASON = "2025-26"
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/team_change_rating_stability")
DEFAULT_CHART_PATH = Path("docs/assets/images/team-change-rating-stability/absolute-change.svg")
MINIMUM_POSSESSIONS = 1_000.0
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20260908


@dataclass(frozen=True)
class TeamChangeRatingStabilityRun:
    """Persisted result of the descriptive team-change stability audit."""

    run_dir: Path
    source_run_dir: Path
    chart_path: Path


def build_player_transitions(
    ratings: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    minimum_possessions: float = MINIMUM_POSSESSIONS,
) -> pd.DataFrame:
    """Return clean consecutive player-season transitions with a team-change flag.

    A clean transition means the player has one primary team in both seasons. A
    switch is then a change in that primary team. Multi-team seasons are omitted
    rather than being arbitrarily assigned to either state.
    """

    if minimum_possessions <= 0:
        raise ValueError("minimum_possessions must be positive")
    rating_columns = {
        "season",
        "player_id",
        "player_name",
        "rapm",
        "prior_rapm",
        "rapm_adjustment_from_prior",
    }
    panel_columns = {
        "season",
        "season_start_year",
        "player_id",
        "primary_team_tricode",
        "team_count",
        "rapm_possessions",
        "age",
    }
    missing_ratings = rating_columns - set(ratings)
    missing_panel = panel_columns - set(panel)
    if missing_ratings or missing_panel:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_ratings | missing_panel))
        )

    metadata = panel.loc[:, list(panel_columns)].copy()
    if metadata.duplicated(["season", "player_id"]).any():
        raise ValueError("player-season panel must have one row per player and season")
    merged = ratings.loc[:, list(rating_columns)].merge(
        metadata,
        on=["season", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values(["player_id", "season_start_year"], kind="stable").copy()
    grouped = merged.groupby("player_id", sort=False)
    merged["previous_season_start_year"] = grouped["season_start_year"].shift()
    merged["previous_team"] = grouped["primary_team_tricode"].shift()
    merged["previous_team_count"] = grouped["team_count"].shift()
    merged["previous_possessions"] = grouped["rapm_possessions"].shift()
    merged["previous_rapm"] = grouped["rapm"].shift()
    merged["previous_prior_rapm"] = grouped["prior_rapm"].shift()
    merged["previous_season_update"] = grouped["rapm_adjustment_from_prior"].shift()

    transitions = merged.loc[
        merged["season_start_year"].eq(merged["previous_season_start_year"] + 1)
    ].copy()
    transitions["minimum_transition_possessions"] = np.minimum(
        transitions["rapm_possessions"].astype(float),
        transitions["previous_possessions"].astype(float),
    )
    clean = (
        transitions["team_count"].eq(1)
        & transitions["previous_team_count"].eq(1)
        & transitions["primary_team_tricode"].notna()
        & transitions["previous_team"].notna()
        & transitions["minimum_transition_possessions"].ge(minimum_possessions)
    )
    transitions = transitions.loc[clean].copy()
    transitions["team_changed"] = transitions["primary_team_tricode"].ne(
        transitions["previous_team"]
    )
    transitions["transition_group"] = np.where(
        transitions["team_changed"], "changed_team", "same_team"
    )
    transitions["rating_change"] = transitions["rapm"] - transitions["previous_rapm"]
    transitions["absolute_rating_change"] = transitions["rating_change"].abs()
    transitions["prior_change"] = (
        transitions["prior_rapm"] - transitions["previous_prior_rapm"]
    )
    transitions["season_update_change"] = (
        transitions["rapm_adjustment_from_prior"] - transitions["previous_season_update"]
    )
    return transitions.sort_values(
        ["season_start_year", "player_name", "player_id"], kind="stable"
    ).reset_index(drop=True)


def _group_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    grouped = transitions.groupby("transition_group", sort=True)["absolute_rating_change"]
    summary = grouped.agg(count="size", mean="mean", median="median", std="std").reset_index()
    return summary


def _cluster_bootstrap_mean_difference(
    transitions: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Resample players, preserving each player's serially correlated transitions."""

    if draws < 1:
        raise ValueError("draws must be positive")
    player_codes, players = pd.factorize(transitions["player_id"], sort=True)
    changed = transitions["team_changed"].to_numpy(bool)
    values = transitions["absolute_rating_change"].to_numpy(float)
    player_count = len(players)
    same_counts = np.bincount(player_codes, weights=(~changed), minlength=player_count)
    changed_counts = np.bincount(player_codes, weights=changed, minlength=player_count)
    same_sums = np.bincount(player_codes, weights=values * (~changed), minlength=player_count)
    changed_sums = np.bincount(player_codes, weights=values * changed, minlength=player_count)
    if not same_counts.sum() or not changed_counts.sum():
        raise ValueError("Both same-team and changed-team transitions are required")

    rng = np.random.default_rng(seed)
    effects = np.empty(draws, dtype=float)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        sampled = rng.integers(0, player_count, size=(stop - start, player_count))
        multiplicities = np.apply_along_axis(
            np.bincount, 1, sampled, minlength=player_count
        )
        same_denominator = multiplicities @ same_counts
        changed_denominator = multiplicities @ changed_counts
        effects[start:stop] = (multiplicities @ changed_sums) / changed_denominator - (
            multiplicities @ same_sums
        ) / same_denominator
    return effects


def _adjusted_switch_effect(transitions: pd.DataFrame) -> float:
    """OLS switch coefficient conditional on target season, age, and prior exposure."""

    years = transitions["season_start_year"].astype(int)
    year_levels = sorted(years.unique())
    age = transitions["age"].astype(float) - 27.0
    design = [
        np.ones(len(transitions)),
        transitions["team_changed"].astype(float).to_numpy(),
        np.log1p(transitions["minimum_transition_possessions"].astype(float).to_numpy()),
        age.to_numpy(),
        np.square(age.to_numpy()),
    ]
    design.extend(years.eq(year).astype(float).to_numpy() for year in year_levels[1:])
    coefficients, *_ = np.linalg.lstsq(
        np.column_stack(design),
        transitions["absolute_rating_change"].to_numpy(float),
        rcond=None,
    )
    return float(coefficients[1])


def _season_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season, group in transitions.groupby("season", sort=True):
        same = group.loc[~group["team_changed"], "absolute_rating_change"]
        changed = group.loc[group["team_changed"], "absolute_rating_change"]
        rows.append(
            {
                "season": season,
                "season_start_year": int(group["season_start_year"].iloc[0]),
                "same_team_count": len(same),
                "changed_team_count": len(changed),
                "same_team_mean_absolute_change": same.mean(),
                "changed_team_mean_absolute_change": changed.mean(),
                "same_team_median_absolute_change": same.median(),
                "changed_team_median_absolute_change": changed.median(),
                "mean_difference_changed_minus_same": changed.mean() - same.mean(),
                "median_difference_changed_minus_same": changed.median() - same.median(),
            }
        )
    return pd.DataFrame(rows)


def _render_chart(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    colors = {"same": "#2f6ea8", "changed": "#e8793e"}

    sample = summary.iloc[0]
    axes[0].bar(
        ["Same team", "Changed team"],
        [sample["same_team_mean"], sample["changed_team_mean"]],
        color=[colors["same"], colors["changed"]],
        width=0.58,
    )
    axes[0].errorbar(
        [0, 1],
        [sample["same_team_mean"], sample["changed_team_mean"]],
        yerr=[
            [
                sample["same_team_mean"] - sample["same_team_ci_lower"],
                sample["changed_team_mean"] - sample["changed_team_ci_lower"],
            ],
            [
                sample["same_team_ci_upper"] - sample["same_team_mean"],
                sample["changed_team_ci_upper"] - sample["changed_team_mean"],
            ],
        ],
        fmt="none",
        color="#1c2522",
        capsize=5,
    )
    axes[0].set_title("Mean absolute year-to-year RAPM change")
    axes[0].set_ylabel("Net rating per 100 possessions")
    axes[0].set_ylim(bottom=0)

    for column, label, color in (
        ("same_team_median_absolute_change", "Same team", colors["same"]),
        ("changed_team_median_absolute_change", "Changed team", colors["changed"]),
    ):
        axes[1].plot(
            summary["season_start_year"], summary[column], marker="o", linewidth=1.7,
            markersize=3.5, label=label, color=color,
        )
    axes[1].axhline(0, color="#8a938d", linewidth=0.8)
    axes[1].set_title("Season-by-season median change")
    axes[1].set_xlabel("Target season start year")
    axes[1].set_ylabel("Absolute RAPM change")
    axes[1].legend(frameon=False)
    figure.savefig(path, format="svg", dpi=160)
    plt.close(figure)


def _latest_run(root: Path) -> Path:
    pointer = root / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"No latest artifact pointer: {pointer}")
    return root / str(json.loads(pointer.read_text())["run_id"])


def run_team_change_rating_stability_study(
    *,
    source_run_dir: Path | str | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
    minimum_possessions: float = MINIMUM_POSSESSIONS,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> TeamChangeRatingStabilityRun:
    """Build the descriptive audit from the completed production NAIL fit."""

    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    ratings = pd.read_parquet(source / "player_season_ratings.parquet")
    panel = pd.read_parquet(player_season_panel_path)
    transitions = build_player_transitions(
        ratings, panel, minimum_possessions=minimum_possessions
    )
    group_summary = _group_summary(transitions).set_index("transition_group")
    effects = _cluster_bootstrap_mean_difference(transitions, draws=draws, seed=seed)
    same_mean = float(group_summary.loc["same_team", "mean"])
    changed_mean = float(group_summary.loc["changed_team", "mean"])
    summary = pd.DataFrame(
        [
            {
                "same_team_count": int(group_summary.loc["same_team", "count"]),
                "changed_team_count": int(group_summary.loc["changed_team", "count"]),
                "same_team_mean": same_mean,
                "changed_team_mean": changed_mean,
                "same_team_median": float(group_summary.loc["same_team", "median"]),
                "changed_team_median": float(group_summary.loc["changed_team", "median"]),
                "mean_difference_changed_minus_same": changed_mean - same_mean,
                "bootstrap_ci_lower": float(np.quantile(effects, 0.025)),
                "bootstrap_ci_upper": float(np.quantile(effects, 0.975)),
                "bootstrap_probability_changed_larger": float(np.mean(effects > 0)),
                "adjusted_switch_effect": _adjusted_switch_effect(transitions),
                "minimum_possessions": minimum_possessions,
                "draws": draws,
            }
        ]
    )
    season_summary = _season_summary(transitions)
    summary_for_chart = summary.copy()
    summary_for_chart["same_team_ci_lower"] = same_mean
    summary_for_chart["same_team_ci_upper"] = same_mean
    summary_for_chart["changed_team_ci_lower"] = same_mean + float(
        np.quantile(effects, 0.025)
    )
    summary_for_chart["changed_team_ci_upper"] = same_mean + float(
        np.quantile(effects, 0.975)
    )
    chart = Path(chart_path)
    _render_chart(pd.concat([summary_for_chart, season_summary], axis=1), chart)

    root = Path(output_root)
    run_id = f"team-change-rating-stability-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    transitions.to_parquet(run_dir / "player_transitions.parquet", index=False)
    summary.to_parquet(run_dir / "summary.parquet", index=False)
    season_summary.to_parquet(run_dir / "season_summary.parquet", index=False)
    pd.DataFrame({"bootstrap_mean_difference": effects}).to_parquet(
        run_dir / "cluster_bootstrap.parquet", index=False
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source),
                "rating": "completed raw player RAPM before display-only additive compilation",
                "transition_definition": (
                    "consecutive seasons; one primary team in each; changed means primary "
                    "team differs"
                ),
                "minimum_possessions": minimum_possessions,
                "controls": (
                    "target-season fixed effects, age, age squared, and log minimum "
                    "two-season possessions"
                ),
                "bootstrap": "player-cluster resampling",
                "draws": draws,
                "seed": seed,
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return TeamChangeRatingStabilityRun(run_dir=run_dir, source_run_dir=source, chart_path=chart)


def main() -> None:
    parser = argparse.ArgumentParser(description="Study team-change RAPM stability")
    parser.add_argument("--minimum-possessions", type=float, default=MINIMUM_POSSESSIONS)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run = run_team_change_rating_stability_study(
        minimum_possessions=args.minimum_possessions,
        draws=args.draws,
        seed=args.seed,
    )
    print(f"Team-change rating stability study: run={run.run_dir}")


if __name__ == "__main__":
    main()
