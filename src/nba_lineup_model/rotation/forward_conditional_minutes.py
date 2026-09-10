"""Forward player-state prior for conditional NBA minutes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_PLAYER_PANEL_PATH,
    ForwardAvailabilityConfig,
    build_availability_season_summary,
    predict_availability_season,
)
from nba_lineup_model.rotation.l0_opening_minute_share_persistence import (
    read_preseason_roster_candidates,
)

DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/rotation/forward_conditional_minutes")
DEFAULT_TUNING_SEASONS = ("2020-21", "2021-22", "2022-23")
DEFAULT_FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_PERSISTENCE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_UPDATE_STRENGTH_GRID = (5.0, 15.0, 30.0, 60.0)
DEFAULT_INITIAL_STRENGTH_GRID = (5.0, 15.0, 30.0, 60.0)
PROMOTED_AVAILABILITY_CONFIG = ForwardAvailabilityConfig(0.5, 60.0, 15.0, -0.25)
REGULATION_SEASON_TEAM_MINUTES = 82.0 * 240.0


@dataclass(frozen=True)
class ForwardConditionalMinutesConfig:
    """State persistence and available-game pseudo-exposure strengths."""

    persistence: float
    update_strength: float
    initial_strength: float


AGE_ONLY_CONFIG = ForwardConditionalMinutesConfig(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class AgeConditionalMinutesModel:
    """Pooled quadratic age baseline on log(minutes per available game)."""

    coefficients: np.ndarray
    population_log_minutes: float

    def predict_log_minutes(self, ages: np.ndarray) -> np.ndarray:
        values = np.asarray(ages, dtype=float)
        output = np.full(len(values), self.population_log_minutes, dtype=float)
        known = np.isfinite(values)
        if known.any():
            centered = values[known] - 28.0
            design = np.column_stack((np.ones(len(centered)), centered, np.square(centered)))
            output[known] = design @ self.coefficients
        return output


@dataclass(frozen=True)
class ForwardConditionalMinutesRun:
    """Immutable artifact location for one frozen forward-minutes replay."""

    run_dir: Path
    run_id: str


def fit_age_conditional_minutes_model(summary: pd.DataFrame) -> AgeConditionalMinutesModel:
    """Fit an exposure-weighted age baseline from completed player-seasons."""

    _validate_summary(summary)
    observed = summary.loc[summary["available_games"].gt(0)].copy()
    target = np.log1p(observed["minutes_per_available_game"].to_numpy(dtype=float))
    weights = observed["available_games"].to_numpy(dtype=float)
    population = float(np.average(target, weights=weights))
    known = observed.loc[observed["age"].notna()].copy()
    if known.empty:
        return AgeConditionalMinutesModel(
            coefficients=np.array((population, 0.0, 0.0)),
            population_log_minutes=population,
        )
    centered = known["age"].to_numpy(dtype=float) - 28.0
    design = np.column_stack((np.ones(len(known)), centered, np.square(centered)))
    y = np.log1p(known["minutes_per_available_game"].to_numpy(dtype=float))
    w = known["available_games"].to_numpy(dtype=float)
    penalty = np.diag((0.0, 1e-3, 1e-3))
    coefficients = np.linalg.solve(
        design.T @ (w[:, None] * design) + penalty,
        design.T @ (w * y),
    )
    return AgeConditionalMinutesModel(
        coefficients=np.asarray(coefficients, dtype=float),
        population_log_minutes=population,
    )


def predict_conditional_minutes_season(
    summary: pd.DataFrame,
    *,
    target_season: str,
    config: ForwardConditionalMinutesConfig,
) -> tuple[pd.DataFrame, AgeConditionalMinutesModel]:
    """Forecast target conditional minutes using only completed prior seasons."""

    _validate_summary(summary)
    target_year = _season_year(target_season)
    history = summary.loc[summary["season_start_year"].lt(target_year)].copy()
    target = summary.loc[summary["season"].eq(target_season)].copy()
    if history.empty or target.empty:
        raise ValueError(f"Target {target_season} requires non-empty history and observations")
    age_model = fit_age_conditional_minutes_model(history)
    state = _filtered_minutes_state(history, config=config)
    output = target.loc[
        :,
        [
            "season",
            "season_start_year",
            "player_id",
            "player_name",
            "age",
            "available_games",
            "known_player_games",
            "available_share",
            "total_nba_minutes",
            "minutes_per_available_game",
        ],
    ].copy()
    predicted_log: list[float] = []
    gap_years: list[int] = []
    prior_state: list[bool] = []
    for row in output.itertuples(index=False):
        baseline = float(age_model.predict_log_minutes(np.array([row.age]))[0])
        prior = state.get(int(row.player_id))
        if prior is None:
            predicted_log.append(baseline)
            gap_years.append(-1)
            prior_state.append(False)
            continue
        gap = max(target_year - int(prior["season_start_year"]), 1)
        prior_baseline = float(age_model.predict_log_minutes(np.array([prior["age"]]))[0])
        prediction = baseline + config.persistence**gap * (
            float(prior["posterior_log_minutes"]) - prior_baseline
        )
        predicted_log.append(prediction)
        gap_years.append(gap)
        prior_state.append(True)
    output["predicted_log_minutes_per_available_game"] = predicted_log
    output["predicted_minutes_per_available_game"] = np.expm1(predicted_log).clip(0.0)
    output["prior_gap_years"] = gap_years
    output["has_prior_minutes_state"] = prior_state
    output["conditional_absolute_error"] = np.abs(
        output["minutes_per_available_game"] - output["predicted_minutes_per_available_game"]
    )
    output["conditional_squared_error"] = np.square(
        output["minutes_per_available_game"] - output["predicted_minutes_per_available_game"]
    )
    return output, age_model


def predict_conditional_minutes_roster(
    summary: pd.DataFrame,
    *,
    roster: pd.DataFrame,
    target_season: str,
    config: ForwardConditionalMinutesConfig,
) -> tuple[pd.DataFrame, AgeConditionalMinutesModel]:
    """Forecast conditional minutes for an opening roster without target outcomes."""

    _validate_summary(summary)
    required = {"player_id", "player_name", "age"}
    missing = sorted(required - set(roster))
    if missing:
        raise ValueError(f"Opening roster lacks required columns: {missing}")
    if roster["player_id"].astype(int).duplicated().any():
        raise ValueError("Opening roster contains duplicate player IDs")
    target_year = _season_year(target_season)
    history = summary.loc[summary["season_start_year"].lt(target_year)].copy()
    if history.empty:
        raise ValueError(f"Target {target_season} requires completed conditional-minutes history")
    age_model = fit_age_conditional_minutes_model(history)
    state = _filtered_minutes_state(history, config=config)
    output = roster.loc[:, ["player_id", "player_name", "age"]].copy()
    output.insert(0, "season_start_year", target_year)
    output.insert(0, "season", target_season)
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    output["age"] = pd.to_numeric(output["age"], errors="coerce")
    predicted_log: list[float] = []
    gap_years: list[int] = []
    prior_state: list[bool] = []
    for row in output.itertuples(index=False):
        baseline = float(age_model.predict_log_minutes(np.array([row.age]))[0])
        prior = state.get(int(row.player_id))
        if prior is None:
            predicted_log.append(baseline)
            gap_years.append(-1)
            prior_state.append(False)
            continue
        gap = max(target_year - int(prior["season_start_year"]), 1)
        prior_baseline = float(age_model.predict_log_minutes(np.array([prior["age"]]))[0])
        predicted_log.append(
            baseline
            + config.persistence**gap * (
                float(prior["posterior_log_minutes"]) - prior_baseline
            )
        )
        gap_years.append(gap)
        prior_state.append(True)
    output["predicted_log_minutes_per_available_game"] = predicted_log
    output["predicted_minutes_per_available_game"] = np.expm1(predicted_log).clip(0.0)
    output["prior_gap_years"] = gap_years
    output["has_prior_minutes_state"] = prior_state
    return output, age_model


def attach_expected_total_minutes(
    conditional_predictions: pd.DataFrame,
    availability_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Combine independent forward components before team normalization."""

    availability = availability_predictions.loc[
        :, ["player_id", "predicted_available_share"]
    ].copy()
    output = conditional_predictions.merge(
        availability, on="player_id", how="left", validate="one_to_one"
    )
    if output["predicted_available_share"].isna().any():
        raise ValueError("Conditional minutes lack matched availability predictions")
    output["raw_expected_total_minutes"] = (
        82.0 * output["predicted_available_share"] * output["predicted_minutes_per_available_game"]
    )
    if "total_nba_minutes" in output:
        output["total_minutes_absolute_error"] = np.abs(
            output["total_nba_minutes"] - output["raw_expected_total_minutes"]
        )
        output["total_minutes_squared_error"] = np.square(
            output["total_nba_minutes"] - output["raw_expected_total_minutes"]
        )
    return output


def normalize_opening_roster_minutes(
    predictions: pd.DataFrame,
    *,
    opening_roster: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize raw expected totals to exact regulation team-season minutes."""

    required = {"team_id", "team", "player_id", "player_name"}
    missing = sorted(required - set(opening_roster))
    if missing:
        raise ValueError(f"Opening roster lacks required columns: {missing}")
    source = opening_roster.loc[:, ["team_id", "team", "player_id", "player_name"]].copy()
    source["team_id"] = pd.to_numeric(source["team_id"], errors="raise").astype("int64")
    source["player_id"] = pd.to_numeric(source["player_id"], errors="raise").astype("int64")
    if source.duplicated(["team_id", "player_id"]).any():
        raise ValueError("Opening roster contains duplicate player-team rows")
    forecast = source.merge(
        predictions.loc[:, ["player_id", "raw_expected_total_minutes"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    if forecast["raw_expected_total_minutes"].isna().any():
        missing_ids = forecast.loc[forecast["raw_expected_total_minutes"].isna(), "player_id"]
        raise ValueError(
            f"Opening roster lacks minutes forecasts for players: {sorted(missing_ids)}"
        )
    forecast["minutes_weight"] = forecast["raw_expected_total_minutes"].clip(lower=0.0)
    totals = forecast.groupby("team_id", sort=False)["minutes_weight"].transform("sum")
    if totals.le(0.0).any():
        raise ValueError("Opening roster has no positive raw expected-minute weight")
    forecast["projected_minute_share"] = forecast["minutes_weight"] / totals
    forecast["projected_total_minutes"] = (
        REGULATION_SEASON_TEAM_MINUTES * forecast["projected_minute_share"]
    )
    if not np.isclose(
        forecast.groupby("team_id")["projected_total_minutes"].sum(),
        REGULATION_SEASON_TEAM_MINUTES,
        atol=1e-6,
    ).all():
        raise ValueError("Opening-roster normalization did not conserve team minutes")
    return forecast


def tune_forward_conditional_minutes(
    summary: pd.DataFrame,
    *,
    target_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    persistence_grid: tuple[float, ...] = DEFAULT_PERSISTENCE_GRID,
    update_strength_grid: tuple[float, ...] = DEFAULT_UPDATE_STRENGTH_GRID,
    initial_strength_grid: tuple[float, ...] = DEFAULT_INITIAL_STRENGTH_GRID,
) -> tuple[ForwardConditionalMinutesConfig, pd.DataFrame]:
    """Select state parameters from completed pre-frozen conditional outcomes."""

    rows: list[dict[str, float]] = []
    for persistence in sorted(set(persistence_grid)):
        for update_strength in sorted(set(update_strength_grid)):
            for initial_strength in sorted(set(initial_strength_grid)):
                config = ForwardConditionalMinutesConfig(
                    persistence=float(persistence),
                    update_strength=float(update_strength),
                    initial_strength=float(initial_strength),
                )
                predictions = [
                    predict_conditional_minutes_season(
                        summary, target_season=season, config=config
                    )[0]
                    for season in target_seasons
                ]
                metrics = summarize_conditional_minutes_metrics(
                    pd.concat(predictions, ignore_index=True)
                )
                rows.append({**asdict(config), **metrics})
    grid = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "available_game_weighted_rmse",
                "available_game_weighted_mae",
                "persistence",
                "update_strength",
                "initial_strength",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    winner = grid.loc[0]
    return (
        ForwardConditionalMinutesConfig(
            persistence=float(winner["persistence"]),
            update_strength=float(winner["update_strength"]),
            initial_strength=float(winner["initial_strength"]),
        ),
        grid,
    )


def summarize_conditional_minutes_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Report conditional and raw expected-total metrics before roster squashing."""

    required = {
        "available_games",
        "conditional_absolute_error",
        "conditional_squared_error",
    }
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError(f"Conditional predictions lack required columns: {missing}")
    weights = predictions["available_games"].to_numpy(dtype=float)
    absolute = predictions["conditional_absolute_error"].to_numpy(dtype=float)
    squared = predictions["conditional_squared_error"].to_numpy(dtype=float)
    if weights.sum() <= 0:
        raise ValueError("Conditional metrics require positive available-game exposure")
    output = {
        "conditional_mae": float(absolute.mean()),
        "conditional_rmse": float(np.sqrt(squared.mean())),
        "available_game_weighted_mae": float(np.average(absolute, weights=weights)),
        "available_game_weighted_rmse": float(np.sqrt(np.average(squared, weights=weights))),
        "player_season_count": float(len(predictions)),
    }
    if {"total_minutes_absolute_error", "total_minutes_squared_error"}.issubset(predictions):
        output |= {
            "total_minutes_mae": float(predictions["total_minutes_absolute_error"].mean()),
            "total_minutes_rmse": float(np.sqrt(predictions["total_minutes_squared_error"].mean())),
        }
    return output


def run_forward_conditional_minutes(
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
) -> ForwardConditionalMinutesRun:
    """Tune and evaluate the preseason conditional-minutes state model."""

    final_year = max(_season_year(season) for season in (*tuning_seasons, *frozen_seasons))
    all_seasons = tuple(
        f"{year}-{str(year + 1)[-2:]}"
        for year in range(_season_year(DEFAULT_TUNING_SEASONS[0]) - 5, final_year + 1)
    )
    print(f"Forward conditional minutes: summarizing {len(all_seasons)} seasons", flush=True)
    summary = build_availability_season_summary(
        all_seasons, curated_dir=curated_dir, player_panel_path=player_panel_path
    )
    print("Forward conditional minutes: tuning player-state filter", flush=True)
    config, tuning_grid = tune_forward_conditional_minutes(summary, target_seasons=tuning_seasons)
    print(f"Forward conditional minutes: selected {asdict(config)}", flush=True)
    run_id = (
        f"forward-conditional-minutes-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:7]}"
    )
    run_dir = Path(artifacts_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary.to_parquet(run_dir / "player_season_summary.parquet", index=False)
    tuning_grid.to_parquet(run_dir / "tuning_grid.parquet", index=False)

    metrics_rows: list[dict[str, float | str]] = []
    roster_rows: list[pd.DataFrame] = []
    for index, season in enumerate(frozen_seasons, start=1):
        print(
            f"Forward conditional minutes: frozen {index}/{len(frozen_seasons)} {season}",
            flush=True,
        )
        conditional, _age_model = predict_conditional_minutes_season(
            summary, target_season=season, config=config
        )
        age_only, _age_only_model = predict_conditional_minutes_season(
            summary, target_season=season, config=AGE_ONLY_CONFIG
        )
        availability, _availability_age, _metadata = predict_availability_season(
            summary, target_season=season, config=PROMOTED_AVAILABILITY_CONFIG
        )
        combined = attach_expected_total_minutes(conditional, availability)
        combined_age_only = attach_expected_total_minutes(age_only, availability)
        combined.to_parquet(run_dir / f"{season}_predictions.parquet", index=False)
        combined_age_only.to_parquet(
            run_dir / f"{season}_age_only_control_predictions.parquet", index=False
        )
        metrics_rows.extend(
            [
                {
                    "season": season,
                    "model": "Forward Conditional Minutes v0.1",
                    **summarize_conditional_minutes_metrics(combined),
                },
                {
                    "season": season,
                    "model": "Age-only conditional-minutes control",
                    **summarize_conditional_minutes_metrics(combined_age_only),
                },
            ]
        )
        roster = read_preseason_roster_candidates(season, curated_dir=curated_dir)
        normalized = normalize_opening_roster_minutes(combined, opening_roster=roster)
        normalized.insert(0, "season", season)
        normalized.to_parquet(run_dir / f"{season}_opening_roster_minutes.parquet", index=False)
        roster_rows.append(
            normalized.groupby("team_id", as_index=False)
            .agg(projected_team_minutes=("projected_total_minutes", "sum"))
            .assign(season=season)
        )
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_parquet(run_dir / "frozen_metrics.parquet", index=False)
    pd.concat(roster_rows, ignore_index=True).to_parquet(
        run_dir / "frozen_team_minute_conservation.parquet", index=False
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": "forward_conditional_minutes",
                "version": "v0.1",
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "selected_config": asdict(config),
                "tuning_seasons": list(tuning_seasons),
                "frozen_seasons": list(frozen_seasons),
                "target": "total NBA minutes / medically available games",
                "availability_component": {
                    "model": "Forward Availability v0.2",
                    "config": asdict(PROMOTED_AVAILABILITY_CONFIG),
                },
                "roster_normalization": "softmax(log raw expected total minutes) to 82 * 240",
            },
            indent=2,
        )
        + "\n"
    )
    return ForwardConditionalMinutesRun(run_dir=run_dir, run_id=run_id)


def _filtered_minutes_state(
    history: pd.DataFrame, *, config: ForwardConditionalMinutesConfig
) -> dict[int, dict[str, float]]:
    state: dict[int, dict[str, float]] = {}
    ordered = history.sort_values(["season_start_year", "player_id"], kind="stable")
    for year in sorted(ordered["season_start_year"].unique()):
        completed = ordered.loc[ordered["season_start_year"].lt(year)]
        current = ordered.loc[ordered["season_start_year"].eq(year)]
        age_model = (
            fit_age_conditional_minutes_model(completed)
            if not completed.empty
            else fit_age_conditional_minutes_model(current)
        )
        for row in current.itertuples(index=False):
            player_id = int(row.player_id)
            baseline = float(age_model.predict_log_minutes(np.array([row.age]))[0])
            prior = state.get(player_id)
            if prior is None:
                prior_log = baseline
                strength = config.initial_strength
            else:
                gap = max(int(row.season_start_year) - int(prior["season_start_year"]), 1)
                prior_baseline = float(age_model.predict_log_minutes(np.array([prior["age"]]))[0])
                prior_log = baseline + config.persistence**gap * (
                    float(prior["posterior_log_minutes"]) - prior_baseline
                )
                strength = config.update_strength
            exposure = float(row.available_games)
            observed = float(np.log1p(row.minutes_per_available_game))
            denominator = strength + exposure
            posterior = (
                (strength * prior_log + exposure * observed) / denominator
                if denominator > 0.0
                else prior_log
            )
            state[player_id] = {
                "season_start_year": float(row.season_start_year),
                "posterior_log_minutes": posterior,
                "age": float(row.age),
            }
    return state


def _validate_summary(summary: pd.DataFrame) -> None:
    required = {
        "season",
        "season_start_year",
        "player_id",
        "player_name",
        "age",
        "available_games",
        "known_player_games",
        "available_share",
        "total_nba_minutes",
        "minutes_per_available_game",
    }
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError(f"Availability summary lacks required columns: {missing}")
    if summary.duplicated(["season", "player_id"]).any():
        raise ValueError("Availability summary has duplicate player-season rows")
    if summary["available_games"].lt(0).any() or summary["minutes_per_available_game"].lt(0).any():
        raise ValueError("Availability summary has negative conditional-minute values")


def _season_year(season: str) -> int:
    return int(season[:4])


def main() -> None:
    """Run the forward conditional-minutes frozen replay."""

    parser = argparse.ArgumentParser(description="Evaluate forward conditional minutes")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--player-panel-path", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--tuning-seasons", nargs="+", default=list(DEFAULT_TUNING_SEASONS))
    parser.add_argument("--frozen-seasons", nargs="+", default=list(DEFAULT_FROZEN_SEASONS))
    args = parser.parse_args()
    run = run_forward_conditional_minutes(
        curated_dir=args.curated_dir,
        player_panel_path=args.player_panel_path,
        artifacts_dir=args.artifacts_dir,
        tuning_seasons=tuple(args.tuning_seasons),
        frozen_seasons=tuple(args.frozen_seasons),
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
