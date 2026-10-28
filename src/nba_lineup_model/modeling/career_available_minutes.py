#!/usr/bin/env python3
"""Evaluate career-available-game shrinkage for the in-season minutes updater."""

# ruff: noqa: E501 -- The portable HTML report is intentionally inline.

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.early_game_minutes_backtest import (
    _priors,
    _read_parts,
    cutoff_role_observations,
)
from nba_lineup_model.modeling.in_season_minutes import (
    CONDITIONAL_OUTCOMES,
    HISTORICAL_AVAILABILITY_MAP,
    PLAYED,
    UNKNOWN,
)

CONTROL_PRIOR_WEIGHT = 0.2
ROLE_CUTOFF_MINUTES = 44
ROLE_LOOKBACK_GAMES = 3
DEFAULT_DEVELOPMENT_SEASONS = ("2023-24", "2024-25")
DEFAULT_HOLDOUT_SEASON = "2025-26"
DEFAULT_W_MIN = (0.05, 0.1, 0.2)
DEFAULT_BETA = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
DEFAULT_W_MAX = (0.2, 0.5, 1.0, 2.0, 4.0)
WEIGHT_AUDIT_GAMES = (0, 10, 25, 50, 82, 164, 410, 820)
CAREER_BINS = (-1, 81, 163, 327, np.inf)
CAREER_LABELS = ("First year", "Second year", "Years 3–4", "Veteran")


@dataclass(frozen=True)
class CareerWeightConfig:
    """Monotonic log shrinkage configuration."""

    w_min: float
    beta: float
    w_max: float

    def validate(self) -> None:
        if not np.isfinite(self.w_min) or self.w_min <= 0.0:
            raise ValueError("w_min must be finite and positive")
        if not np.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        if not np.isfinite(self.w_max) or self.w_max < self.w_min:
            raise ValueError("w_max must be finite and at least w_min")

    def label(self) -> str:
        return f"wmin={self.w_min:g},beta={self.beta:g},wmax={self.w_max:g}"


def career_prior_weight(
    career_available_games: pd.Series | np.ndarray | float | int,
    config: CareerWeightConfig,
) -> pd.Series | np.ndarray | float:
    """Return the clipped monotonic log prior weight."""

    config.validate()
    values = np.asarray(career_available_games, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Career available games must be finite and non-negative")
    weights = np.clip(
        config.w_min + config.beta * np.log1p(values / 82.0),
        config.w_min,
        config.w_max,
    )
    if isinstance(career_available_games, pd.Series):
        return pd.Series(weights, index=career_available_games.index, dtype=float)
    if np.ndim(values) == 0:
        return float(weights)
    return weights


def _normalized_availability_state(frame: pd.DataFrame) -> pd.Series:
    states = frame["availability_state"].astype("string")
    return states.map(HISTORICAL_AVAILABILITY_MAP).fillna(states).fillna(UNKNOWN)


def build_career_available_feature(availability: pd.DataFrame) -> pd.DataFrame:
    """Build strict pre-day cumulative available games for every player-game.

    Every game on the same date receives the count entering that date. A played
    game, known healthy coach DNP, or G League/two-way assignment increments
    the count only for later dates.
    """

    required = {"player_id", "game_id", "game_date", "availability_state"}
    missing = sorted(required - set(availability))
    if missing:
        raise ValueError("Availability history lacks: " + ", ".join(missing))
    frame = availability[list(required)].copy()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype(int)
    frame["game_id"] = frame["game_id"].astype(str)
    frame["game_date"] = pd.to_datetime(frame["game_date"], utc=True).dt.normalize()
    frame["availability_state"] = _normalized_availability_state(frame)
    duplicate = frame.duplicated(["player_id", "game_id"], keep=False)
    if duplicate.any():
        conflicts = (
            frame.loc[duplicate].groupby(["player_id", "game_id"])["availability_state"].nunique()
        )
        if conflicts.gt(1).any():
            raise ValueError("Availability history has conflicting player-game states")
        frame = frame.drop_duplicates(["player_id", "game_id"], keep="last")
    frame["available_increment"] = (
        frame["availability_state"].isin(CONDITIONAL_OUTCOMES).astype(int)
    )
    daily = (
        frame.groupby(["player_id", "game_date"], as_index=False)
        .agg(daily_available_games=("available_increment", "sum"))
        .sort_values(["player_id", "game_date"], kind="stable")
    )
    daily["career_available_games"] = (
        daily.groupby("player_id", sort=False)["daily_available_games"].cumsum()
        - daily["daily_available_games"]
    )
    output = frame.merge(
        daily[["player_id", "game_date", "career_available_games"]],
        on=["player_id", "game_date"],
        how="left",
        validate="many_to_one",
    )
    return output[["player_id", "game_id", "game_date", "career_available_games"]].sort_values(
        ["game_date", "game_id", "player_id"], kind="stable"
    )


def career_available_totals(availability: pd.DataFrame) -> pd.Series:
    """Return available-game exposure through the end of a history frame."""

    required = {"player_id", "game_id", "availability_state"}
    missing = sorted(required - set(availability))
    if missing:
        raise ValueError("Availability history lacks: " + ", ".join(missing))
    frame = availability[list(required)].copy()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype(int)
    frame["game_id"] = frame["game_id"].astype(str)
    frame["availability_state"] = _normalized_availability_state(frame)
    frame = frame.drop_duplicates(["player_id", "game_id"], keep="last")
    return (
        frame["availability_state"]
        .isin(CONDITIONAL_OUTCOMES)
        .astype(int)
        .groupby(frame["player_id"])
        .sum()
        .astype(int)
    )


def _played_rows_from_boxscores(players_root: Path, seasons: Iterable[str]) -> pd.DataFrame:
    """Return conservative played-only availability before the full mart begins."""

    frames: list[pd.DataFrame] = []
    columns = ["game_id", "game_date", "personId", "statistics_minutes"]
    for season in seasons:
        root = players_root / season / "regular"
        paths = sorted(root.glob("part-*.parquet"))
        if not paths:
            continue
        frame = _read_parts(root, columns)
        minutes = pd.to_timedelta(frame["statistics_minutes"], errors="coerce").dt.total_seconds()
        frame = frame.loc[minutes.fillna(0.0).gt(0.0)].copy()
        frames.append(
            frame.rename(columns={"personId": "player_id"}).assign(availability_state=PLAYED)[
                ["player_id", "game_id", "game_date", "availability_state"]
            ]
        )
    if not frames:
        return pd.DataFrame(columns=["player_id", "game_id", "game_date", "availability_state"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["player_id", "game_id"], keep="last"
    )


def _season_start(season: str) -> int:
    return int(season.split("-", maxsplit=1)[0])


def _load_availability_history(
    availability_root: Path,
    players_root: Path | None,
    *,
    through_season: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    seasons = sorted(
        path.name
        for path in availability_root.iterdir()
        if path.is_dir() and _season_start(path.name) <= _season_start(through_season)
    )
    if not seasons:
        raise ValueError(f"No availability seasons found under {availability_root}")
    frames = [
        pd.read_parquet(
            availability_root / season / "part-00000.parquet",
            columns=["player_id", "game_id", "game_date", "availability_state"],
        )
        for season in seasons
    ]
    precoverage_seasons: list[str] = []
    precoverage = pd.DataFrame()
    if players_root is not None and players_root.is_dir():
        first_coverage = _season_start(seasons[0])
        precoverage_seasons = sorted(
            path.name
            for path in players_root.iterdir()
            if path.is_dir() and _season_start(path.name) < first_coverage
        )
        precoverage = _played_rows_from_boxscores(players_root, precoverage_seasons)
    history = pd.concat([precoverage, *frames], ignore_index=True)
    audit = {
        "classified_availability_seasons": seasons,
        "precoverage_played_only_seasons": precoverage_seasons,
        "precoverage_played_rows": int(len(precoverage)),
        "classified_rows": int(sum(len(frame) for frame in frames)),
        "boundary": (
            "Before the classified availability mart begins, only positive-minute "
            "box-score appearances count as available; unobserved DNPs add nothing."
        ),
    }
    return history, audit


def chronological_components(
    panel: pd.DataFrame,
    priors: pd.DataFrame,
    career_feature: pd.DataFrame,
    *,
    cutoff_minutes: int = ROLE_CUTOFF_MINUTES,
    lookback_games: int = ROLE_LOOKBACK_GAMES,
) -> pd.DataFrame:
    """Build configuration-independent, leakage-safe scoring components."""

    if lookback_games <= 0:
        raise ValueError("Lookback games must be positive")
    frame = panel.loc[
        panel["cutoff_minutes"].eq(cutoff_minutes)
        & panel["availability_state"].isin(CONDITIONAL_OUTCOMES)
    ].merge(
        _priors(priors)[["player_id", "preseason_role_minutes"]],
        on="player_id",
        how="inner",
        validate="many_to_one",
    )
    feature = career_feature[["player_id", "game_id", "career_available_games"]].copy()
    feature["game_id"] = feature["game_id"].astype(str)
    frame = frame.merge(
        feature,
        on=["player_id", "game_id"],
        how="left",
        validate="one_to_one",
    )
    if frame["career_available_games"].isna().any():
        examples = frame.loc[
            frame["career_available_games"].isna(),
            ["game_id", "player_id", "player_name"],
        ].head(5)
        raise ValueError(f"Career feature missing target rows: {examples.to_dict('records')}")
    frame = frame.sort_values(["game_date", "game_id", "player_id"], kind="stable")
    histories: dict[int, list[float]] = defaultdict(list)
    history_sum: list[float] = []
    history_count: list[int] = []
    for _game_date, day in frame.groupby("game_date", sort=True):
        for row in day.itertuples(index=False):
            selected = histories[int(row.player_id)][-lookback_games:]
            history_sum.append(float(sum(selected)))
            history_count.append(len(selected))
        for row in day.itertuples(index=False):
            histories[int(row.player_id)].append(float(row.observed_role_minutes))
    frame["history_sum"] = history_sum
    frame["prior_observations"] = history_count
    return frame.reset_index(drop=True)


def score_components(
    components: pd.DataFrame,
    config: CareerWeightConfig,
    *,
    control_weight: float = CONTROL_PRIOR_WEIGHT,
) -> pd.DataFrame:
    """Score one career-weight candidate and the constant-weight control."""

    output = components.copy()
    weight = career_prior_weight(output["career_available_games"], config)
    observed = output["prior_observations"].astype(float)
    output["candidate_prior_weight"] = weight
    output["control_role_minutes"] = (
        control_weight * output["preseason_role_minutes"] + output["history_sum"]
    ) / (control_weight + observed)
    output["candidate_role_minutes"] = (
        output["candidate_prior_weight"] * output["preseason_role_minutes"] + output["history_sum"]
    ) / (output["candidate_prior_weight"] + observed)
    output["control_absolute_error"] = (
        output["control_role_minutes"] - output["target_minutes"]
    ).abs()
    output["candidate_absolute_error"] = (
        output["candidate_role_minutes"] - output["target_minutes"]
    ).abs()
    return output


def _population(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "played_only":
        return frame.loc[frame["availability_state"].eq(PLAYED)]
    if name == "played_plus_coach_dnp":
        return frame
    raise ValueError(f"Unknown evaluation population: {name}")


def comparison_metrics(frame: pd.DataFrame, population: str) -> dict[str, object]:
    selected = _population(frame, population)
    target = selected["target_minutes"].astype(float)
    control_error = selected["control_role_minutes"].astype(float) - target
    candidate_error = selected["candidate_role_minutes"].astype(float) - target
    control_mae = float(control_error.abs().mean())
    candidate_mae = float(candidate_error.abs().mean())
    return {
        "population": population,
        "rows": int(len(selected)),
        "games": int(selected["game_id"].nunique()),
        "players": int(selected["player_id"].nunique()),
        "control_mae": control_mae,
        "candidate_mae": candidate_mae,
        "candidate_mae_advantage": control_mae - candidate_mae,
        "candidate_mae_advantage_pct": 100.0 * (control_mae - candidate_mae) / control_mae,
        "control_rmse": float(np.sqrt(np.square(control_error).mean())),
        "candidate_rmse": float(np.sqrt(np.square(candidate_error).mean())),
        "control_bias": float(control_error.mean()),
        "candidate_bias": float(candidate_error.mean()),
    }


def _grid_configs(
    w_min: Iterable[float], beta: Iterable[float], w_max: Iterable[float]
) -> list[CareerWeightConfig]:
    configs: list[CareerWeightConfig] = []
    seen: set[tuple[float, ...]] = set()
    for minimum in w_min:
        for slope in beta:
            for maximum in w_max:
                if maximum < minimum:
                    continue
                config = CareerWeightConfig(float(minimum), float(slope), float(maximum))
                signature = tuple(
                    round(float(career_prior_weight(value, config)), 12)
                    for value in WEIGHT_AUDIT_GAMES
                )
                if signature in seen:
                    continue
                seen.add(signature)
                configs.append(config)
    return configs


def select_config(
    development: pd.DataFrame,
    configs: Iterable[CareerWeightConfig],
) -> tuple[CareerWeightConfig, pd.DataFrame]:
    """Select from development rows without accessing the locked holdout."""

    rows: list[dict[str, object]] = []
    for config in configs:
        scored = score_components(development, config)
        all_metrics = comparison_metrics(scored, "played_plus_coach_dnp")
        played_metrics = comparison_metrics(scored, "played_only")
        rows.append(
            {
                "w_min": config.w_min,
                "beta": config.beta,
                "w_max": config.w_max,
                "conditional_mae": all_metrics["candidate_mae"],
                "conditional_rmse": all_metrics["candidate_rmse"],
                "played_mae": played_metrics["candidate_mae"],
                "conditional_control_mae": all_metrics["control_mae"],
            }
        )
    grid = pd.DataFrame(rows).sort_values(
        ["conditional_mae", "conditional_rmse", "played_mae", "w_min", "beta", "w_max"],
        kind="stable",
    )
    selected = grid.iloc[0]
    return (
        CareerWeightConfig(
            float(selected["w_min"]),
            float(selected["beta"]),
            float(selected["w_max"]),
        ),
        grid,
    )


def _bootstrap(
    frame: pd.DataFrame,
    population: str,
    *,
    samples: int = 2000,
    seed: int = 20261026,
) -> dict[str, object]:
    selected = _population(frame, population).copy()
    selected["advantage"] = (
        selected["control_absolute_error"] - selected["candidate_absolute_error"]
    )
    by_game = selected.groupby("game_id", as_index=False).agg(
        advantage_sum=("advantage", "sum"), rows=("advantage", "size")
    )
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    sums = by_game["advantage_sum"].to_numpy(dtype=float)
    counts = by_game["rows"].to_numpy(dtype=float)
    for index in range(samples):
        chosen = rng.integers(0, len(by_game), size=len(by_game))
        values[index] = sums[chosen].sum() / counts[chosen].sum()
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "population": population,
        "samples": samples,
        "seed": seed,
        "mean_candidate_mae_advantage": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "probability_candidate_better": float(np.mean(values > 0.0)),
    }


def _cohort_metrics(frame: pd.DataFrame, population: str) -> list[dict[str, object]]:
    selected = _population(frame, population).copy()
    selected["career_exposure_band"] = pd.cut(
        selected["career_available_games"],
        bins=CAREER_BINS,
        labels=CAREER_LABELS,
    ).astype("string")
    rows: list[dict[str, object]] = []
    for label in CAREER_LABELS:
        cohort = selected.loc[selected["career_exposure_band"].eq(label)]
        if not cohort.empty:
            rows.append({"career_exposure_band": label, **comparison_metrics(cohort, population)})
    return rows


def _state_metrics(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {"availability_state": str(state), **comparison_metrics(group, "played_plus_coach_dnp")}
        for state, group in frame.groupby("availability_state", sort=True)
    ]


def _player_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["player_id", "player_name"], as_index=False)
        .agg(
            team=("team", "last"),
            rows=("game_id", "size"),
            career_available_games=("career_available_games", "median"),
            actual_minutes=("target_minutes", "mean"),
            control_minutes=("control_role_minutes", "mean"),
            candidate_minutes=("candidate_role_minutes", "mean"),
            control_mae=("control_absolute_error", "mean"),
            candidate_mae=("candidate_absolute_error", "mean"),
        )
        .assign(advantage=lambda value: value["control_mae"] - value["candidate_mae"])
        .sort_values(["advantage", "rows"], ascending=[False, False], kind="stable")
    )


def score_current_season_rows(
    evaluation: pd.DataFrame,
    starting_career_available_games: pd.Series,
    config: CareerWeightConfig,
    preseason_conditional_minutes: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Replay the selected curve over current-season shadow rows."""

    required = {
        "game_id",
        "game_date",
        "team",
        "player_id",
        "outcome_classification",
        "early_game_role_minutes",
        "frozen_conditional_minutes",
        "updated_conditional_minutes",
        "updated_projected_minutes",
    }
    missing = sorted(required - set(evaluation))
    if missing:
        raise ValueError("Current-season evaluation lacks: " + ", ".join(missing))
    frame = evaluation.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["game_date"] = pd.to_datetime(frame["game_date"], utc=True).dt.normalize()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype(int)
    frame = frame.sort_values(["game_date", "game_id", "player_id"], kind="stable")
    start = starting_career_available_games.to_dict()
    preseason = (
        {int(key): float(value) for key, value in preseason_conditional_minutes.to_dict().items()}
        if preseason_conditional_minutes is not None
        else None
    )
    exposure = defaultdict(int, {int(key): int(value) for key, value in start.items()})
    histories: dict[int, list[tuple[float, float]]] = defaultdict(list)
    candidate: list[float] = []
    reconstructed_control: list[float] = []
    career_counts: list[int] = []
    weights: list[float] = []
    for _game_date, day in frame.groupby("game_date", sort=True):
        for row in day.itertuples(index=False):
            player_id = int(row.player_id)
            history = histories[player_id][-ROLE_LOOKBACK_GAMES:]
            history_sum = sum(minutes * weight for minutes, weight in history)
            history_weight = sum(weight for _minutes, weight in history)
            prior = (
                float(preseason.get(player_id, 0.0))
                if preseason is not None
                else float(row.frozen_conditional_minutes)
            )
            count = int(exposure[player_id])
            weight = float(career_prior_weight(count, config))
            if preseason is not None and player_id not in preseason:
                candidate.append(0.0)
                reconstructed_control.append(0.0)
            else:
                candidate.append((weight * prior + history_sum) / (weight + history_weight))
                reconstructed_control.append(
                    (CONTROL_PRIOR_WEIGHT * prior + history_sum)
                    / (CONTROL_PRIOR_WEIGHT + history_weight)
                )
            career_counts.append(count)
            weights.append(weight)
        for row in day.itertuples(index=False):
            if str(row.outcome_classification) not in CONDITIONAL_OUTCOMES:
                continue
            player_id = int(row.player_id)
            if hasattr(row, "conditional_observation_weight"):
                observation_weight = float(row.conditional_observation_weight)
            else:
                actual_minutes = float(getattr(row, "regulation_equivalent_minutes", 0.0))
                observation_weight = (
                    0.5
                    if str(row.outcome_classification) == PLAYED and 0.0 < actual_minutes < 4.0
                    else 1.0
                )
            histories[player_id].append((float(row.early_game_role_minutes), observation_weight))
            exposure[player_id] += 1
    frame["career_available_games"] = career_counts
    frame["candidate_prior_weight"] = weights
    frame["reconstructed_control_conditional_minutes"] = reconstructed_control
    frame["candidate_conditional_minutes"] = candidate
    denominator = frame["reconstructed_control_conditional_minutes"].astype(float)
    ratio = np.divide(
        frame["candidate_conditional_minutes"].astype(float),
        denominator,
        out=np.zeros(len(frame), dtype=float),
        where=denominator.gt(0.0),
    )
    frame["candidate_projected_minutes"] = frame["updated_projected_minutes"].astype(float) * ratio
    team_total = frame.groupby(["game_id", "team"], sort=False)[
        "candidate_projected_minutes"
    ].transform("sum")
    frame["candidate_projected_minutes"] *= 240.0 / team_total
    reproduced = frame["reconstructed_control_conditional_minutes"].astype(float)
    published = frame["updated_conditional_minutes"].astype(float)
    absolute = (reproduced - published).abs()
    audit = {
        "rows": int(len(frame)),
        "games": int(frame["game_id"].nunique()),
        "control_reproduction_mae": float(absolute.mean()),
        "control_reproduction_max_absolute_error": float(absolute.max()),
        "candidate_team_minutes_max_error": float(
            (frame.groupby(["game_id", "team"])["candidate_projected_minutes"].sum() - 240.0)
            .abs()
            .max()
        ),
    }
    return frame, audit


def downstream_game_ab(
    evaluation: pd.DataFrame,
    games: pd.DataFrame,
    summaries: list[dict[str, Any]],
    priors: pd.DataFrame,
    starting_career_available_games: pd.Series,
    config: CareerWeightConfig,
    *,
    rating_cache_root: Path,
    rating_coverage_path: Path,
    rating_prior_possessions: float = 6000.0,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float | int]]:
    """Compare current control and candidate minutes under identical NAIL ratings."""

    from scipy.special import expit, logit

    from nba_lineup_model.modeling.in_season_minutes_monitor import (
        _ats,
        _game_metrics,
        _rating_state_by_date,
    )
    from nba_lineup_model.season.slate_time import summary_slate_date

    scored, replay_audit = score_current_season_rows(
        evaluation,
        starting_career_available_games,
        config,
        priors.set_index("player_id")["conditional_minutes_per_game"],
    )
    games = games.copy()
    games["game_id"] = games["game_id"].astype(str)
    summary_by_game = {str(row["game_id"]): row for row in summaries}
    summary_subset = [
        summary_by_game[game_id] for game_id in games["game_id"] if game_id in summary_by_game
    ]
    if len(summary_subset) != len(games):
        missing = sorted(set(games["game_id"]) - set(summary_by_game))
        raise ValueError(f"Current-season summaries missing games: {missing}")
    rating_by_date = _rating_state_by_date(
        summary_subset,
        rating_cache_root=rating_cache_root,
        rating_coverage_path=rating_coverage_path,
        prior_possessions=rating_prior_possessions,
    )
    preseason_rating = dict(
        zip(
            priors["player_id"].astype(int),
            priors["projected_nail"].astype(float),
            strict=True,
        )
    )
    valid_scale = games.loc[
        games["updated_updated_margin"].abs().gt(1e-8)
        & games["updated_updated_probability"].between(1e-9, 1 - 1e-9)
    ]
    probability_scale = float(
        np.median(
            logit(valid_scale["updated_updated_probability"].astype(float))
            / valid_scale["updated_updated_margin"].astype(float)
        )
    )
    candidate_margins: list[float] = []
    candidate_probabilities: list[float] = []
    candidate_ats: list[str | None] = []
    for game in games.itertuples(index=False):
        game_id = str(game.game_id)
        summary = summary_by_game[game_id]
        game_date = summary_slate_date(summary)
        adjustments = rating_by_date.get(game_date, {})
        ratings = {
            player_id: value + adjustments.get(player_id, 0.0)
            for player_id, value in preseason_rating.items()
        }
        rows = scored.loc[scored["game_id"].eq(game_id)]
        sign = np.where(rows["team"].astype(str).eq(str(game.home_team)), 1.0, -1.0)
        player_rating = rows["player_id"].map(ratings).fillna(0.0).to_numpy(dtype=float)
        control_strength = float(
            np.sum(
                sign
                * rows["updated_projected_minutes"].to_numpy(dtype=float)
                / 48.0
                * player_rating
            )
        )
        candidate_strength = float(
            np.sum(
                sign
                * rows["candidate_projected_minutes"].to_numpy(dtype=float)
                / 48.0
                * player_rating
            )
        )
        margin = float(game.updated_updated_margin) + candidate_strength - control_strength
        candidate_margins.append(margin)
        candidate_probabilities.append(float(expit(probability_scale * margin)))
        candidate_ats.append(_ats(summary, margin))
    games["career_candidate_margin"] = candidate_margins
    games["career_candidate_probability"] = candidate_probabilities
    games["career_candidate_ats"] = candidate_ats
    metrics = {
        "control_updated_minutes": _game_metrics(games, "updated_updated"),
        "career_candidate_minutes": _game_metrics(games, "career_candidate"),
    }
    return games, metrics, replay_audit


def _digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _season_components(
    season: str,
    *,
    stints_root: Path,
    availability_root: Path,
    priors_root: Path,
    career_feature: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    stint_columns = [
        "game_id",
        "start_elapsed_game_seconds",
        "end_elapsed_game_seconds",
        "home_player_ids",
        "away_player_ids",
        "catalog_home_team_tricode",
        "catalog_away_team_tricode",
    ]
    stint_dir = stints_root / season / "regular"
    availability_path = availability_root / season / "part-00000.parquet"
    priors_path = priors_root / f"{season}_predictions.parquet"
    panel, audit = cutoff_role_observations(
        _read_parts(stint_dir, stint_columns),
        pd.read_parquet(availability_path),
        cutoffs=(ROLE_CUTOFF_MINUTES,),
    )
    components = chronological_components(
        panel,
        pd.read_parquet(priors_path),
        career_feature,
    )
    components["season"] = season
    audit["scored_rows"] = int(len(components))
    audit["stints_sha256"] = _digest(stint_dir.glob("part-*.parquet"))
    audit["availability_sha256"] = hashlib.sha256(availability_path.read_bytes()).hexdigest()
    audit["priors_sha256"] = hashlib.sha256(priors_path.read_bytes()).hexdigest()
    return components, audit


def _weight_rows(config: CareerWeightConfig) -> list[dict[str, object]]:
    return [
        {
            "career_available_games": games,
            "prior_weight": float(career_prior_weight(games, config)),
        }
        for games in WEIGHT_AUDIT_GAMES
    ]


def _canonical_debut_audit(
    history: pd.DataFrame, career_feature: pd.DataFrame
) -> list[dict[str, object]]:
    examples = ((201933, "Blake Griffin"), (1631096, "Chet Holmgren"))
    rows: list[dict[str, object]] = []
    normalized = history.copy()
    normalized["player_id"] = pd.to_numeric(normalized["player_id"], errors="raise").astype(int)
    normalized["game_date"] = pd.to_datetime(normalized["game_date"], utc=True).dt.normalize()
    normalized["availability_state"] = _normalized_availability_state(normalized)
    for player_id, player_name in examples:
        player = normalized.loc[normalized["player_id"].eq(player_id)].sort_values(
            ["game_date", "game_id"], kind="stable"
        )
        debut = player.loc[player["availability_state"].eq(PLAYED)].head(1)
        if debut.empty:
            continue
        game_id = str(debut.iloc[0]["game_id"])
        feature = career_feature.loc[
            career_feature["player_id"].eq(player_id)
            & career_feature["game_id"].astype(str).eq(game_id)
        ].iloc[0]
        predebut = player.loc[player["game_date"].lt(debut.iloc[0]["game_date"])]
        rows.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "first_played_game_id": game_id,
                "first_played_date": debut.iloc[0]["game_date"].date().isoformat(),
                "career_available_games_at_debut": int(feature["career_available_games"]),
                "observed_predebut_rows": int(len(predebut)),
                "predebut_available_increments": int(
                    predebut["availability_state"].isin(CONDITIONAL_OUTCOMES).sum()
                ),
            }
        )
    return rows


def _fmt(value: object, digits: int = 3) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def _render_report(summary: dict[str, Any], players: pd.DataFrame, output: Path) -> None:
    metrics = {row["population"]: row for row in summary["holdout_metrics"]}
    played = metrics["played_only"]
    all_available = metrics["played_plus_coach_dnp"]
    winner = "improved" if all_available["candidate_mae_advantage"] > 0 else "regressed"
    weight_rows = "".join(
        f"<tr><td>{row['career_available_games']}</td><td>{_fmt(row['prior_weight'], 3)}</td></tr>"
        for row in summary["selected_weight_audit"]
    )

    def cohort_rows(population: str) -> str:
        return "".join(
            "<tr>"
            f"<td>{html.escape(str(row['career_exposure_band']))}</td>"
            f"<td>{int(row['rows']):,}</td>"
            f"<td>{_fmt(row['control_mae'])}</td>"
            f"<td>{_fmt(row['candidate_mae'])}</td>"
            f"<td>{float(row['candidate_mae_advantage']):+.3f}</td>"
            f"<td>{_fmt(row['control_rmse'])}</td>"
            f"<td>{_fmt(row['candidate_rmse'])}</td>"
            "</tr>"
            for row in summary["holdout_by_career_exposure"][population]
        )

    state_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['availability_state']))}</td>"
        f"<td>{int(row['rows']):,}</td>"
        f"<td>{_fmt(row['control_mae'])}</td>"
        f"<td>{_fmt(row['candidate_mae'])}</td>"
        f"<td>{float(row['candidate_mae_advantage']):+.3f}</td>"
        "</tr>"
        for row in summary["holdout_by_availability_state"]
    )

    def player_rows(frame: pd.DataFrame) -> str:
        return "".join(
            "<tr>"
            f"<td>{html.escape(str(row.player_name))}</td>"
            f"<td>{html.escape(str(row.team))}</td>"
            f"<td>{int(row.rows)}</td>"
            f"<td>{int(row.career_available_games)}</td>"
            f"<td>{_fmt(row.control_mae)}</td>"
            f"<td>{_fmt(row.candidate_mae)}</td>"
            f"<td>{float(row.advantage):+.3f}</td>"
            "</tr>"
            for row in frame.itertuples(index=False)
        )

    eligible = players.loc[players["rows"].ge(20)]
    top = eligible.head(12)
    bottom = eligible.tail(12).sort_values("advantage")
    bootstrap = summary["holdout_bootstrap"]["played_plus_coach_dnp"]
    generated = str(summary["generated_at"])
    downstream = summary.get("downstream_current_season")
    downstream_section = ""
    if downstream:
        game_metrics = downstream["metrics"]
        control_game = game_metrics["control_updated_minutes"]
        candidate_game = game_metrics["career_candidate_minutes"]
        downstream_section = f"""
<h2>Current-season downstream check</h2>
<p>Secondary evidence through {html.escape(str(downstream["through_date"]))}. Both rows use the same date-specific possession-updated NAIL ratings, archived injury assumptions, and game context.</p>
<div class="table-wrap"><table><thead><tr><th>Minutes arm</th><th>Games</th><th>Margin MAE</th><th>RMSE</th><th>Brier</th><th>Log loss</th><th>Winner</th><th>ATS</th></tr></thead><tbody>
<tr><td>Constant 0.2 control</td><td>{int(control_game["games"])}</td><td>{_fmt(control_game["margin_mae"])}</td><td>{_fmt(control_game["margin_rmse"])}</td><td>{_fmt(control_game["brier_score"], 5)}</td><td>{_fmt(control_game["log_loss"], 5)}</td><td>{100 * float(control_game["winner_accuracy"]):.1f}%</td><td>{int(control_game["ats_wins"])}-{int(control_game["ats_losses"])}-{int(control_game["ats_pushes"])}</td></tr>
<tr><td>Career-weight candidate</td><td>{int(candidate_game["games"])}</td><td>{_fmt(candidate_game["margin_mae"])}</td><td>{_fmt(candidate_game["margin_rmse"])}</td><td>{_fmt(candidate_game["brier_score"], 5)}</td><td>{_fmt(candidate_game["log_loss"], 5)}</td><td>{100 * float(candidate_game["winner_accuracy"]):.1f}%</td><td>{int(candidate_game["ats_wins"])}-{int(candidate_game["ats_losses"])}-{int(candidate_game["ats_pushes"])}</td></tr>
</tbody></table></div>"""
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NBA GESTALT · Career-available-game minutes study</title><style>
:root{{--paper:#f5f2e9;--ink:#18231f;--muted:#68736d;--line:#d6d2c5;--green:#175c48;--red:#a84a36;--amber:#b77a18}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.45}}
main{{max-width:1420px;margin:auto;padding:32px}}h1,h2{{font-family:Inter,system-ui,sans-serif;letter-spacing:-.025em}}h1{{font-size:34px;margin:0 0 6px}}h2{{font-size:20px;margin:34px 0 12px}}p{{color:var(--muted)}}
.eyebrow{{color:var(--green);font-weight:800;text-transform:uppercase;letter-spacing:.08em;font-size:12px}}.note{{border-left:4px solid var(--amber);background:#fff5de;padding:12px 16px;margin:18px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:22px 0}}.card{{background:#fffdf7;border:1px solid var(--line);padding:16px}}.label{{color:var(--muted);font-size:11px;text-transform:uppercase;font-weight:800}}.value{{font:800 27px/1.2 Inter,system-ui,sans-serif;margin-top:6px}}.sub{{font-size:12px;color:var(--muted);margin-top:4px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.table-wrap{{overflow:auto;border:1px solid var(--line);background:#fffdf7}}table{{border-collapse:collapse;width:100%;font-size:12px}}th{{background:#ece8dc;text-transform:uppercase;font-size:10px}}td,th{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}.positive{{color:var(--green)}}.negative{{color:var(--red)}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}main{{padding:18px}}}}
</style></head><body><main><div class="eyebrow">NBA GESTALT · Locked holdout study</div>
<h1>Does experience-aware shrinkage improve minutes?</h1>
<p>Selected on {", ".join(summary["development_seasons"])}; evaluated once on {summary["holdout_season"]}. Generated {html.escape(generated)}.</p>
<div class="note"><strong>Finding:</strong> The career-weighted candidate {winner} available-opportunity MAE by <strong>{float(all_available["candidate_mae_advantage"]):+.3f} minutes</strong> ({float(all_available["candidate_mae_advantage_pct"]):+.2f}%). The paired game bootstrap 95% interval is [{float(bootstrap["ci95_lower"]):+.3f}, {float(bootstrap["ci95_upper"]):+.3f}]. Played-only MAE changed by {float(played["candidate_mae_advantage"]):+.3f} minutes. Zero-minute healthy opportunities include coach DNP and G League/two-way assignment rows. This is a shadow result; public projections are unchanged.</div>
<section class="cards"><div class="card"><div class="label">Selected curve</div><div class="value">{summary["selected_config"]["w_min"]:g} + {summary["selected_config"]["beta"]:g} log</div><div class="sub">cap {summary["selected_config"]["w_max"]:g} equivalent games</div></div>
<div class="card"><div class="label">All available rows</div><div class="value">{_fmt(all_available["candidate_mae"])}</div><div class="sub">control {_fmt(all_available["control_mae"])} · {int(all_available["rows"]):,} rows</div></div>
<div class="card"><div class="label">Played-only rows</div><div class="value">{_fmt(played["candidate_mae"])}</div><div class="sub">control {_fmt(played["control_mae"])} · {int(played["rows"]):,} rows</div></div>
<div class="card"><div class="label">Probability candidate wins</div><div class="value">{100 * float(bootstrap["probability_candidate_better"]):.1f}%</div><div class="sub">paired game bootstrap · 2,000 draws</div></div></section>
<div class="grid"><section><h2>Selected prior-weight curve</h2><p>Equivalent preseason games by career available games entering the date.</p><div class="table-wrap"><table><thead><tr><th>Career available games</th><th>Prior weight</th></tr></thead><tbody>{weight_rows}</tbody></table></div></section>
<section><h2>Availability-state decomposition</h2><p>This reveals whether a gain comes from played games or healthy zero-minute opportunities.</p><div class="table-wrap"><table><thead><tr><th>State</th><th>Rows</th><th>Control MAE</th><th>Candidate MAE</th><th>Advantage</th></tr></thead><tbody>{state_rows}</tbody></table></div></section></div>
<div class="grid"><section><h2>Career cohorts · available opportunities</h2><p>Includes played games, coach DNPs, and G League/two-way assignments. Positive advantage means the candidate is more accurate.</p><div class="table-wrap"><table><thead><tr><th>Band</th><th>Rows</th><th>Control MAE</th><th>Candidate MAE</th><th>Advantage</th><th>Control RMSE</th><th>Candidate RMSE</th></tr></thead><tbody>{cohort_rows("played_plus_coach_dnp")}</tbody></table></div></section>
<section><h2>Career cohorts · played only</h2><p>This conditions on the player actually entering the game.</p><div class="table-wrap"><table><thead><tr><th>Band</th><th>Rows</th><th>Control MAE</th><th>Candidate MAE</th><th>Advantage</th><th>Control RMSE</th><th>Candidate RMSE</th></tr></thead><tbody>{cohort_rows("played_only")}</tbody></table></div></section></div>
<div class="grid"><section><h2>Most improved players</h2><div class="table-wrap"><table><thead><tr><th>Player</th><th>Team</th><th>Rows</th><th>Career available</th><th>Control MAE</th><th>Candidate MAE</th><th>Advantage</th></tr></thead><tbody>{player_rows(top)}</tbody></table></div></section>
<section><h2>Largest regressions</h2><div class="table-wrap"><table><thead><tr><th>Player</th><th>Team</th><th>Rows</th><th>Career available</th><th>Control MAE</th><th>Candidate MAE</th><th>Advantage</th></tr></thead><tbody>{player_rows(bottom)}</tbody></table></div></section></div>
{downstream_section}
<h2>Design and guardrails</h2><p>The control is the current constant 0.2-game preseason weight. Both arms use the same first-44-minute normalized roles and three eligible-game lookback. Played games, known healthy coach DNPs, and G League/two-way assignments increase career exposure; medical, unavailable inactive, suspension/ineligible, and unknown rows do not. All games on a date are scored before that date updates history or exposure. Before classified availability coverage, only positive-minute box-score appearances count. The locked holdout was not used for parameter selection.</p>
</main></body></html>"""
    output.write_text(document)


def run(args: argparse.Namespace) -> dict[str, Any]:
    development_seasons = tuple(args.development_seasons)
    holdout_season = str(args.holdout_season)
    availability_root = Path(args.availability_root)
    stints_root = Path(args.stints_root)
    priors_root = Path(args.priors_root)
    players_root = Path(args.players_root) if args.players_root else None
    history, source_audit = _load_availability_history(
        availability_root,
        players_root,
        through_season=holdout_season,
    )
    career = build_career_available_feature(history)

    development_frames: list[pd.DataFrame] = []
    coverage: dict[str, object] = {}
    for season in development_seasons:
        components, audit = _season_components(
            season,
            stints_root=stints_root,
            availability_root=availability_root,
            priors_root=priors_root,
            career_feature=career,
        )
        development_frames.append(components)
        coverage[season] = audit
    development = pd.concat(development_frames, ignore_index=True)
    configs = _grid_configs(args.w_min, args.beta, args.w_max)
    selected, grid = select_config(development, configs)

    # The locked holdout is loaded only after the curve has been selected.
    holdout_components, holdout_audit = _season_components(
        holdout_season,
        stints_root=stints_root,
        availability_root=availability_root,
        priors_root=priors_root,
        career_feature=career,
    )
    coverage[holdout_season] = holdout_audit
    holdout = score_components(holdout_components, selected)
    populations = ("played_only", "played_plus_coach_dnp")
    metrics = [comparison_metrics(holdout, population) for population in populations]
    bootstrap = {population: _bootstrap(holdout, population) for population in populations}
    cohorts = {population: _cohort_metrics(holdout, population) for population in populations}
    players = _player_summary(holdout)
    selected_grid = grid.iloc[0]
    result: dict[str, Any] = {
        "schema_version": 1,
        "study": "career_available_game_minutes_shrinkage_v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "development_seasons": list(development_seasons),
        "holdout_season": holdout_season,
        "selection_rule": (
            "lowest pooled development available-opportunity MAE, then RMSE, "
            "then played-only MAE; holdout loaded after selection"
        ),
        "control_prior_weight": CONTROL_PRIOR_WEIGHT,
        "role_cutoff_minutes": ROLE_CUTOFF_MINUTES,
        "role_lookback_games": ROLE_LOOKBACK_GAMES,
        "grid": {
            "w_min": list(args.w_min),
            "beta": list(args.beta),
            "w_max": list(args.w_max),
            "unique_curves": int(len(grid)),
        },
        "selected_config": {
            "w_min": selected.w_min,
            "beta": selected.beta,
            "w_max": selected.w_max,
            "development_conditional_mae": float(selected_grid["conditional_mae"]),
            "development_control_mae": float(selected_grid["conditional_control_mae"]),
        },
        "selected_weight_audit": _weight_rows(selected),
        "availability_classification_audit": {
            "increments": sorted(CONDITIONAL_OUTCOMES),
            "does_not_increment": [
                "medical_absence",
                "inactive_or_assignment",
                "suspension_or_ineligible",
                "unknown",
            ],
            "same_date_rule": (
                "All games on a date receive the career count entering that date; "
                "the date's observations increment only later dates."
            ),
            "canonical_lost_rookie_examples": _canonical_debut_audit(history, career),
        },
        "holdout_metrics": metrics,
        "holdout_bootstrap": bootstrap,
        "holdout_by_career_exposure": cohorts,
        "holdout_by_availability_state": _state_metrics(holdout),
        "representative_players": {
            "minimum_rows": 20,
            "most_improved": players.loc[players["rows"].ge(20)].head(12).to_dict("records"),
            "most_regressed": players.loc[players["rows"].ge(20)]
            .tail(12)
            .sort_values("advantage")
            .to_dict("records"),
        },
        "career_feature_source": source_audit,
        "coverage": coverage,
        "production_status": "shadow_only_no_public_projection_change",
    }
    downstream_games: pd.DataFrame | None = None
    downstream_rows: pd.DataFrame | None = None
    current_paths = (
        getattr(args, "current_evaluation_path", None),
        getattr(args, "current_games_path", None),
        getattr(args, "current_summary_dir", None),
        getattr(args, "current_rating_cache_root", None),
        getattr(args, "current_rating_coverage_path", None),
        getattr(args, "release_root", None),
    )
    if any(current_paths) and not all(current_paths):
        raise ValueError("All current-season downstream A/B paths must be provided together")
    if all(current_paths):
        release_root = Path(args.release_root)
        cache_paths = sorted(release_root.glob("artifacts/web/win_projections/*/*.json"))
        if len(cache_paths) != 1:
            raise ValueError(f"Expected one current minutes cache, found {len(cache_paths)}")
        cache = json.loads(cache_paths[0].read_text())
        current_priors = pd.DataFrame(cache["minutes"]["players"])
        summaries = [
            json.loads(path.read_text())
            for path in sorted(Path(args.current_summary_dir).glob("*.summary.json"))
        ]
        current_evaluation = pd.read_parquet(args.current_evaluation_path)
        downstream_games, downstream_metrics, downstream_audit = downstream_game_ab(
            current_evaluation,
            pd.read_csv(args.current_games_path, dtype={"game_id": str}),
            summaries,
            current_priors,
            career_available_totals(history),
            selected,
            rating_cache_root=Path(args.current_rating_cache_root),
            rating_coverage_path=Path(args.current_rating_coverage_path),
        )
        downstream_rows, _ = score_current_season_rows(
            current_evaluation,
            career_available_totals(history),
            selected,
            current_priors.set_index("player_id")["conditional_minutes_per_game"],
        )
        result["downstream_current_season"] = {
            "through_date": str(downstream_games["game_date"].max()),
            "metrics": downstream_metrics,
            "replay_audit": downstream_audit,
            "input_lineage": {
                "player_game_evaluation_sha256": hashlib.sha256(
                    Path(args.current_evaluation_path).read_bytes()
                ).hexdigest(),
                "game_evaluation_sha256": hashlib.sha256(
                    Path(args.current_games_path).read_bytes()
                ).hexdigest(),
                "summary_files_sha256": _digest(
                    Path(args.current_summary_dir).glob("*.summary.json")
                ),
                "rating_coverage_sha256": hashlib.sha256(
                    Path(args.current_rating_coverage_path).read_bytes()
                ).hexdigest(),
                "rating_contributions_sha256": _digest(
                    Path(args.current_rating_cache_root).glob(
                        "contributions/**/contribution_sha256=*/*"
                    )
                ),
                "release_minutes_cache_sha256": hashlib.sha256(
                    cache_paths[0].read_bytes()
                ).hexdigest(),
            },
            "interpretation": (
                "Secondary only: both minutes arms use the same date-specific "
                "possession-updated NAIL ratings and archived injury-conditioned inputs."
            ),
        }
    output = Path(args.output_dir)
    artifacts: tuple[Path, ...] = (
        output / "summary.json",
        output / "tuning_grid.parquet",
        output / "holdout_evaluation.parquet",
        output / "holdout_player_summary.parquet",
        output / "report.html",
    )
    if downstream_games is not None:
        artifacts += (
            output / "current_season_game_ab.csv",
            output / "current_season_player_evaluation.parquet",
        )
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite immutable artifacts: " + ", ".join(existing))
    output.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(output / "tuning_grid.parquet", index=False)
    holdout.to_parquet(output / "holdout_evaluation.parquet", index=False)
    players.to_parquet(output / "holdout_player_summary.parquet", index=False)
    if downstream_games is not None and downstream_rows is not None:
        downstream_games.to_csv(output / "current_season_game_ab.csv", index=False)
        downstream_rows.to_parquet(output / "current_season_player_evaluation.parquet", index=False)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    _render_report(result, players, output / "report.html")
    return result


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-seasons", nargs="+", default=DEFAULT_DEVELOPMENT_SEASONS)
    parser.add_argument("--holdout-season", default=DEFAULT_HOLDOUT_SEASON)
    parser.add_argument("--availability-root", type=Path, required=True)
    parser.add_argument("--stints-root", type=Path, required=True)
    parser.add_argument("--priors-root", type=Path, required=True)
    parser.add_argument("--players-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--w-min", nargs="+", type=float, default=DEFAULT_W_MIN)
    parser.add_argument("--beta", nargs="+", type=float, default=DEFAULT_BETA)
    parser.add_argument("--w-max", nargs="+", type=float, default=DEFAULT_W_MAX)
    parser.add_argument("--current-evaluation-path", type=Path)
    parser.add_argument("--current-games-path", type=Path)
    parser.add_argument("--current-summary-dir", type=Path)
    parser.add_argument("--current-rating-cache-root", type=Path)
    parser.add_argument("--current-rating-coverage-path", type=Path)
    parser.add_argument("--release-root", type=Path)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
