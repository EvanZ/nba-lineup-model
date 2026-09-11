"""Forward player-availability state model with lagged NBA workload.

This is a historical pilot.  It predicts a player's health availability among
known listed player-game rows; it does not yet alter the minutes pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_PLAYER_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ROSTER_DIR = Path("data/curated/team_rosters")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/rotation/forward_availability")
DEFAULT_TUNING_SEASONS = ("2020-21", "2021-22", "2022-23")
DEFAULT_FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_PERSISTENCE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_PRIOR_STRENGTH_GRID = (5.0, 15.0, 30.0, 60.0)
DEFAULT_INITIAL_PRIOR_STRENGTH_GRID = (5.0, 15.0, 30.0, 60.0)
DEFAULT_WORKLOAD_WEIGHT_GRID = (-1.0, -0.5, -0.25, 0.0, 0.25)
_EPSILON = 1e-6
_ROSTERED_PLAYER_GAME_CONTRACTS = frozenset(
    {"full_roster_status", "full_roster_membership"}
)


@dataclass(frozen=True)
class ForwardAvailabilityConfig:
    """Hyperparameters for the player-specific availability filter."""

    persistence: float
    prior_strength: float
    initial_prior_strength: float
    workload_weight: float
    use_age_baseline: bool = True


@dataclass(frozen=True)
class AgeAvailabilityModel:
    """Pooled age expectation on the logit availability scale."""

    coefficients: np.ndarray
    population_logit: float

    def predict_logit(self, ages: np.ndarray) -> np.ndarray:
        values = np.asarray(ages, dtype=float)
        output = np.full(len(values), self.population_logit, dtype=float)
        known = np.isfinite(values)
        if known.any():
            centered = values[known] - 28.0
            design = np.column_stack((np.ones(len(centered)), centered, np.square(centered)))
            output[known] = design @ self.coefficients
        return output


@dataclass(frozen=True)
class ForwardAvailabilityRun:
    """Immutable artifact location for one forward availability replay."""

    run_dir: Path
    run_id: str


def build_availability_season_summary(
    seasons: tuple[str, ...] | list[str],
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    roster_dir: Path | str = DEFAULT_ROSTER_DIR,
) -> pd.DataFrame:
    """Return one player-season availability observation and lagged workload.

    The denominator is only rostered player-games with a known state. An
    explicit historical inactive designation is unavailable under the approved
    binary contract; a player absent from both source tables is never inferred.
    """

    season_list = tuple(sorted(set(seasons), key=_season_year))
    root = Path(curated_dir) / "player_availability"
    rows: list[pd.DataFrame] = []
    for season in season_list:
        _require_full_roster_source_contract(root / season, season=season)
        path = root / season / "part-00000.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing availability mart: {path}")
        games = pd.read_parquet(path)
        known = games.loc[games["availability_state_known"].astype(bool)].copy()
        summary = (
            known.groupby(["season", "player_id"], as_index=False, sort=False)
            .agg(
                player_name=("player_name", "last"),
                known_player_games=("game_id", "size"),
                available_games=("available", "sum"),
                total_nba_minutes=("minutes", "sum"),
                injury_or_illness_games=(
                    "availability_state",
                    lambda values: int(values.eq("unavailable_injury_or_illness").sum()),
                ),
                rest_games=(
                    "availability_state",
                    lambda values: int(values.eq("unavailable_rest").sum()),
                ),
            )
            .assign(season_start_year=_season_year(season))
        )
        summary["available_games"] = summary["available_games"].astype(int)
        summary["known_player_games"] = summary["known_player_games"].astype(int)
        rows.append(summary)
    if not rows:
        raise ValueError("At least one availability season is required")
    summary = pd.concat(rows, ignore_index=True)
    summary["available_share"] = summary["available_games"] / summary["known_player_games"]
    summary["minutes_per_available_game"] = np.where(
        summary["available_games"].gt(0),
        summary["total_nba_minutes"] / summary["available_games"],
        0.0,
    )
    summary["log_workload"] = np.log1p(summary["minutes_per_available_game"])
    summary = _attach_ages(
        summary,
        player_panel_path=player_panel_path,
        roster_dir=roster_dir,
    )
    return summary.sort_values(["season_start_year", "player_id"], kind="stable").reset_index(
        drop=True
    )


def _require_full_roster_source_contract(season_dir: Path, *, season: str) -> None:
    """Reject a fit whose denominator is not rostered team-games.

    A listed-player box-score table cannot distinguish a player who was
    unavailable from one omitted from the table entirely. Fitting across that
    contract break would manufacture a time trend in availability.
    """

    coverage_path = season_dir / "game_coverage.parquet"
    if not coverage_path.exists():
        raise FileNotFoundError(f"Missing availability coverage audit: {coverage_path}")
    coverage = pd.read_parquet(coverage_path)
    if "source_player_table_contract" in coverage:
        contracts = coverage["source_player_table_contract"].fillna("missing")
    else:
        contracts = coverage["source_kind"].map(
            {
                "live_boxscore": "full_roster_status",
                "stats_v3_summary_v2": "full_roster_membership",
                "stats_v3": "boxscore_listed_players",
            }
        ).fillna("missing")
    invalid = coverage.loc[~contracts.isin(_ROSTERED_PLAYER_GAME_CONTRACTS)]
    if not invalid.empty:
        counts = contracts.value_counts().sort_index().to_dict()
        raise ValueError(
            f"{season} availability source contract is not a complete roster denominator: "
            f"{counts}. Recover historical inactive roster membership and reasons before "
            "running the forward availability model."
        )


def fit_age_availability_model(summary: pd.DataFrame) -> AgeAvailabilityModel:
    """Fit a pooled, exposure-weighted quadratic age availability curve."""

    _validate_summary(summary)
    available = summary["available_games"].to_numpy(dtype=float)
    known = summary["known_player_games"].to_numpy(dtype=float)
    population_logit = logit(
        np.clip(float(available.sum() / known.sum()), _EPSILON, 1.0 - _EPSILON)
    )
    observed_age = summary.loc[summary["age"].notna()].copy()
    if observed_age.empty:
        return AgeAvailabilityModel(
            coefficients=np.array((population_logit, 0.0, 0.0)),
            population_logit=float(population_logit),
        )
    ages = observed_age["age"].to_numpy(dtype=float)
    available = observed_age["available_games"].to_numpy(dtype=float)
    known = observed_age["known_player_games"].to_numpy(dtype=float)
    centered = ages - 28.0
    design = np.column_stack((np.ones(len(observed_age)), centered, np.square(centered)))

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ coefficients
        probabilities = expit(logits)
        value = float(
            -np.sum(available * logits - known * np.logaddexp(0.0, logits))
            + 0.5e-3 * np.dot(coefficients[1:], coefficients[1:])
        )
        residual = known * probabilities - available
        gradient = design.T @ residual
        gradient[1:] += 1e-3 * coefficients[1:]
        return value, gradient

    baseline = logit(np.clip(float(available.sum() / known.sum()), _EPSILON, 1.0 - _EPSILON))
    result = minimize(
        fun=lambda values: objective(values)[0],
        x0=np.array((baseline, 0.0, 0.0)),
        jac=lambda values: objective(values)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Age availability fit failed: {result.message}")
    return AgeAvailabilityModel(
        coefficients=np.asarray(result.x, dtype=float),
        population_logit=float(population_logit),
    )


def fit_availability_baseline(
    summary: pd.DataFrame,
    *,
    use_age_baseline: bool,
) -> AgeAvailabilityModel:
    """Fit the configured population baseline for the filtered player state."""

    if use_age_baseline:
        return fit_age_availability_model(summary)
    _validate_summary(summary)
    available = summary["available_games"].to_numpy(dtype=float)
    known = summary["known_player_games"].to_numpy(dtype=float)
    population_logit = logit(
        np.clip(float(available.sum() / known.sum()), _EPSILON, 1.0 - _EPSILON)
    )
    return AgeAvailabilityModel(
        coefficients=np.array((population_logit, 0.0, 0.0)),
        population_logit=float(population_logit),
    )


def predict_availability_season(
    summary: pd.DataFrame,
    *,
    target_season: str,
    config: ForwardAvailabilityConfig,
) -> tuple[pd.DataFrame, AgeAvailabilityModel, dict[str, float | bool]]:
    """Make strictly forward predictions for one completed target season."""

    target_year = _season_year(target_season)
    history = summary.loc[summary["season_start_year"].lt(target_year)].copy()
    target = summary.loc[summary["season"].eq(target_season)].copy()
    if history.empty or target.empty:
        raise ValueError(
            f"Target {target_season} requires non-empty history and target observations"
        )
    age_model = fit_availability_baseline(history, use_age_baseline=config.use_age_baseline)
    workload_mean = float(history["log_workload"].mean())
    workload_scale = float(history["log_workload"].std())
    if workload_scale <= 1e-12:
        workload_scale = 1.0
    state = _filtered_player_state(history, config=config)
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
            "injury_or_illness_games",
            "rest_games",
        ],
    ].copy()
    predicted: list[float] = []
    gap_years: list[int] = []
    has_history: list[bool] = []
    age_baseline_logits: list[float] = []
    carried_state_residuals: list[float] = []
    carried_workload_adjustments: list[float] = []
    for row in output.itertuples(index=False):
        player_id = int(row.player_id)
        age_baseline = float(age_model.predict_logit(np.array([row.age]))[0])
        age_baseline_logits.append(age_baseline)
        prior = state.get(player_id)
        if prior is None:
            predicted.append(float(expit(age_baseline)))
            gap_years.append(-1)
            has_history.append(False)
            carried_state_residuals.append(0.0)
            carried_workload_adjustments.append(0.0)
            continue
        gap = max(target_year - int(prior["season_start_year"]), 1)
        prior_age_baseline = float(age_model.predict_logit(np.array([prior["age"]]))[0])
        deviation = float(prior["posterior_logit"] - prior_age_baseline)
        workload_z = (float(prior["log_workload"]) - workload_mean) / workload_scale
        workload_effect = config.workload_weight * workload_z
        decay = config.persistence**gap
        carried_state = decay * deviation
        carried_workload = decay * workload_effect
        predicted_logit = age_baseline + carried_state + carried_workload
        predicted.append(float(expit(predicted_logit)))
        gap_years.append(gap)
        has_history.append(True)
        carried_state_residuals.append(carried_state)
        carried_workload_adjustments.append(carried_workload)
    output["predicted_available_share"] = predicted
    output["prior_gap_years"] = gap_years
    output["has_prior_availability_state"] = has_history
    output["age_baseline_logit"] = age_baseline_logits
    output["carried_state_residual_logit"] = carried_state_residuals
    output["carried_workload_adjustment_logit"] = carried_workload_adjustments
    output["absolute_error"] = np.abs(
        output["available_share"] - output["predicted_available_share"]
    )
    output["squared_error"] = np.square(
        output["available_share"] - output["predicted_available_share"]
    )
    metadata = {
        "workload_mean": workload_mean,
        "workload_scale": workload_scale,
        "age_intercept": float(age_model.coefficients[0]),
        "age_linear": float(age_model.coefficients[1]),
        "age_quadratic": float(age_model.coefficients[2]),
        "age_missing_population_logit": float(age_model.population_logit),
        "use_age_baseline": bool(config.use_age_baseline),
    }
    return output, age_model, metadata


def predict_availability_roster(
    summary: pd.DataFrame,
    *,
    roster: pd.DataFrame,
    target_season: str,
    config: ForwardAvailabilityConfig,
) -> tuple[pd.DataFrame, AgeAvailabilityModel, dict[str, float | bool]]:
    """Forecast target-season availability for an opening roster only.

    Unlike ``predict_availability_season``, this path deliberately has no
    target-season player-game observations. The roster supplies only player
    identity, name, and age, making it suitable for preseason inference.
    """

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
        raise ValueError(f"Target {target_season} requires completed availability history")
    age_model = fit_availability_baseline(history, use_age_baseline=config.use_age_baseline)
    workload_mean = float(history["log_workload"].mean())
    workload_scale = float(history["log_workload"].std())
    if workload_scale <= 1e-12:
        workload_scale = 1.0
    state = _filtered_player_state(history, config=config)
    output = roster.loc[:, ["player_id", "player_name", "age"]].copy()
    output.insert(0, "season_start_year", target_year)
    output.insert(0, "season", target_season)
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    output["age"] = pd.to_numeric(output["age"], errors="coerce")
    predicted: list[float] = []
    gap_years: list[int] = []
    has_history: list[bool] = []
    age_baseline_logits: list[float] = []
    carried_state_residuals: list[float] = []
    carried_workload_adjustments: list[float] = []
    for row in output.itertuples(index=False):
        age_baseline = float(age_model.predict_logit(np.array([row.age]))[0])
        age_baseline_logits.append(age_baseline)
        prior = state.get(int(row.player_id))
        if prior is None:
            predicted.append(float(expit(age_baseline)))
            gap_years.append(-1)
            has_history.append(False)
            carried_state_residuals.append(0.0)
            carried_workload_adjustments.append(0.0)
            continue
        gap = max(target_year - int(prior["season_start_year"]), 1)
        prior_age_baseline = float(age_model.predict_logit(np.array([prior["age"]]))[0])
        deviation = float(prior["posterior_logit"] - prior_age_baseline)
        workload_z = (float(prior["log_workload"]) - workload_mean) / workload_scale
        decay = config.persistence**gap
        carried_state = decay * deviation
        carried_workload = decay * config.workload_weight * workload_z
        predicted_logit = age_baseline + carried_state + carried_workload
        predicted.append(float(expit(predicted_logit)))
        gap_years.append(gap)
        has_history.append(True)
        carried_state_residuals.append(carried_state)
        carried_workload_adjustments.append(carried_workload)
    output["predicted_available_share"] = predicted
    output["prior_gap_years"] = gap_years
    output["has_prior_availability_state"] = has_history
    output["age_baseline_logit"] = age_baseline_logits
    output["carried_state_residual_logit"] = carried_state_residuals
    output["carried_workload_adjustment_logit"] = carried_workload_adjustments
    metadata = {
        "workload_mean": workload_mean,
        "workload_scale": workload_scale,
        "age_intercept": float(age_model.coefficients[0]),
        "age_linear": float(age_model.coefficients[1]),
        "age_quadratic": float(age_model.coefficients[2]),
        "age_missing_population_logit": float(age_model.population_logit),
        "use_age_baseline": bool(config.use_age_baseline),
    }
    return output, age_model, metadata


def tune_forward_availability(
    summary: pd.DataFrame,
    *,
    target_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    persistence_grid: tuple[float, ...] = DEFAULT_PERSISTENCE_GRID,
    prior_strength_grid: tuple[float, ...] = DEFAULT_PRIOR_STRENGTH_GRID,
    initial_prior_strength_grid: tuple[float, ...] = DEFAULT_INITIAL_PRIOR_STRENGTH_GRID,
    workload_weight_grid: tuple[float, ...] = DEFAULT_WORKLOAD_WEIGHT_GRID,
) -> tuple[ForwardAvailabilityConfig, pd.DataFrame]:
    """Select state parameters only on completed pre-frozen target seasons."""

    rows: list[dict[str, float]] = []
    for persistence in sorted(set(persistence_grid)):
        for prior_strength in sorted(set(prior_strength_grid)):
            for initial_prior_strength in sorted(set(initial_prior_strength_grid)):
                for workload_weight in sorted(set(workload_weight_grid)):
                    config = ForwardAvailabilityConfig(
                        persistence=float(persistence),
                        prior_strength=float(prior_strength),
                        initial_prior_strength=float(initial_prior_strength),
                        workload_weight=float(workload_weight),
                    )
                    predictions = [
                        predict_availability_season(summary, target_season=season, config=config)[0]
                        for season in target_seasons
                    ]
                    metrics = summarize_availability_metrics(
                        pd.concat(predictions, ignore_index=True)
                    )
                    rows.append({**asdict(config), **metrics})
    grid = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "weighted_brier",
                "binomial_log_loss",
                "persistence",
                "prior_strength",
                "initial_prior_strength",
                "workload_weight",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    winner = grid.loc[0]
    return (
        ForwardAvailabilityConfig(
            persistence=float(winner["persistence"]),
            prior_strength=float(winner["prior_strength"]),
            initial_prior_strength=float(winner["initial_prior_strength"]),
            workload_weight=float(winner["workload_weight"]),
        ),
        grid,
    )


def run_forward_availability(
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
) -> ForwardAvailabilityRun:
    """Tune a forward filter before the frozen three-season availability replay."""

    final_year = max(_season_year(season) for season in (*tuning_seasons, *frozen_seasons))
    all_seasons = tuple(
        f"{year}-{str(year + 1)[-2:]}"
        for year in range(_season_year(DEFAULT_TUNING_SEASONS[0]) - 5, final_year + 1)
    )
    summary = build_availability_season_summary(
        all_seasons,
        curated_dir=curated_dir,
        player_panel_path=player_panel_path,
    )
    config, grid = tune_forward_availability(summary, target_seasons=tuning_seasons)
    run_id = (
        f"forward-availability-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    )
    run_dir = Path(artifacts_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary.to_parquet(run_dir / "player_season_summary.parquet", index=False)
    grid.to_parquet(run_dir / "tuning_grid.parquet", index=False)
    metrics_rows: list[dict[str, float | str]] = []
    for season in frozen_seasons:
        predictions, _age_model, metadata = predict_availability_season(
            summary,
            target_season=season,
            config=config,
        )
        predictions.to_parquet(run_dir / f"{season}_predictions.parquet", index=False)
        metrics_rows.append({"season": season, **summarize_availability_metrics(predictions)})
        (run_dir / f"{season}_state_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_parquet(run_dir / "frozen_metrics.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": "forward_availability",
                "version": "v0.2",
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "selected_config": asdict(config),
                "tuning_seasons": list(tuning_seasons),
                "frozen_seasons": list(frozen_seasons),
                "target": "available games / known listed player-games",
            "availability_positive_states": [
                    "played",
                    "available_dnp_active",
                    "available_dnp_coach",
                    "available_g_league_assignment",
                ],
                "contract": (
                    "age curve plus an exposure-shrunk filtered player availability state; "
                    "previous-season NBA minutes per available game enters only as a lagged "
                    "workload transition"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return ForwardAvailabilityRun(run_dir=run_dir, run_id=run_id)


def summarize_availability_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Summarize season-level availability probabilities without GP substitution."""

    _validate_predictions(predictions)
    actual = predictions["available_share"].to_numpy(dtype=float)
    predicted = predictions["predicted_available_share"].to_numpy(dtype=float)
    exposure = predictions["known_player_games"].to_numpy(dtype=float)
    available = predictions["available_games"].to_numpy(dtype=float)
    return {
        "player_mae": float(np.abs(actual - predicted).mean()),
        "player_rmse": float(np.sqrt(np.square(actual - predicted).mean())),
        "weighted_brier": float(np.average(np.square(actual - predicted), weights=exposure)),
        "binomial_log_loss": float(
            -np.sum(
                available * np.log(np.clip(predicted, _EPSILON, 1.0 - _EPSILON))
                + (exposure - available)
                * np.log(np.clip(1.0 - predicted, _EPSILON, 1.0 - _EPSILON))
            )
            / exposure.sum()
        ),
        "mean_actual_available_share": float(actual.mean()),
        "mean_predicted_available_share": float(predicted.mean()),
        "player_season_count": float(len(predictions)),
    }


def _filtered_player_state(
    history: pd.DataFrame,
    *,
    config: ForwardAvailabilityConfig,
) -> dict[int, dict[str, float]]:
    state: dict[int, dict[str, float]] = {}
    ordered = history.sort_values(["season_start_year", "player_id"], kind="stable")
    for year in sorted(ordered["season_start_year"].unique()):
        completed = ordered.loc[ordered["season_start_year"].lt(year)]
        current = ordered.loc[ordered["season_start_year"].eq(year)]
        baseline_data = completed if not completed.empty else current
        age_model = fit_availability_baseline(
            baseline_data,
            use_age_baseline=config.use_age_baseline,
        )
        workload_mean = float(completed["log_workload"].mean()) if not completed.empty else 0.0
        workload_scale = float(completed["log_workload"].std()) if not completed.empty else 1.0
        if workload_scale <= 1e-12:
            workload_scale = 1.0
        for row in current.itertuples(index=False):
            player_id = int(row.player_id)
            prior = state.get(player_id)
            if prior is None:
                # First observed seasons are evidence, not certainty. This is
                # essential for players who missed an entire rookie season.
                age_probability = float(
                    expit(age_model.predict_logit(np.array([row.age]))[0])
                )
                posterior_probability = _shrunken_probability(
                    prior_probability=age_probability,
                    available_games=float(row.available_games),
                    known_player_games=float(row.known_player_games),
                    prior_strength=config.initial_prior_strength,
                )
            else:
                age_logit = float(age_model.predict_logit(np.array([row.age]))[0])
                prior_age_baseline = float(age_model.predict_logit(np.array([prior["age"]]))[0])
                gap = max(int(row.season_start_year) - int(prior["season_start_year"]), 1)
                deviation = float(prior["posterior_logit"] - prior_age_baseline)
                workload_z = (float(prior["log_workload"]) - workload_mean) / workload_scale
                prior_logit = age_logit + (config.persistence**gap) * (
                    deviation + config.workload_weight * workload_z
                )
                prior_probability = float(expit(prior_logit))
                posterior_probability = _shrunken_probability(
                    prior_probability=prior_probability,
                    available_games=float(row.available_games),
                    known_player_games=float(row.known_player_games),
                    prior_strength=config.prior_strength,
                )
            state[player_id] = {
                "season_start_year": float(row.season_start_year),
                "posterior_logit": float(
                    logit(np.clip(posterior_probability, _EPSILON, 1.0 - _EPSILON))
                ),
                "age": float(row.age),
                "log_workload": float(row.log_workload),
            }
    return state


def _shrunken_probability(
    *,
    prior_probability: float,
    available_games: float,
    known_player_games: float,
    prior_strength: float,
) -> float:
    """Return a beta-binomial posterior mean from a pseudo-game prior."""

    return float(
        (prior_strength * prior_probability + available_games)
        / (prior_strength + known_player_games)
    )


def _attach_ages(
    summary: pd.DataFrame,
    *,
    player_panel_path: Path | str,
    roster_dir: Path | str = DEFAULT_ROSTER_DIR,
) -> pd.DataFrame:
    panel = pd.read_parquet(player_panel_path, columns=["season", "player_id", "age"])
    panel = panel.dropna(subset=["age"]).drop_duplicates(["season", "player_id"], keep="last")
    output = summary.merge(panel, on=["season", "player_id"], how="left", validate="one_to_one")
    output["age"] = pd.to_numeric(output["age"], errors="coerce")
    output["age_source"] = np.where(output["age"].notna(), "season_panel", pd.NA)
    # Players with no target-season box-score panel row, often because they
    # never played, inherit the latest known age forward by season difference.
    for _player_id, indexes in output.groupby("player_id", sort=False).groups.items():
        player_rows = output.loc[indexes].sort_values("season_start_year", kind="stable")
        known = player_rows.dropna(subset=["age"])
        if known.empty:
            continue
        last_age = float(known.iloc[-1]["age"])
        last_year = int(known.iloc[-1]["season_start_year"])
        missing = player_rows.index[player_rows["age"].isna()]
        for index in missing:
            output.loc[index, "age"] = (
                last_age + int(output.loc[index, "season_start_year"]) - last_year
            )
            output.loc[index, "age_source"] = "player_history"
    birth_dates = _roster_birth_dates(roster_dir)
    output = output.merge(birth_dates, on="player_id", how="left", validate="many_to_one")
    missing = output["age"].isna() & output["birth_date"].notna()
    if missing.any():
        season_end = pd.to_datetime(
            output.loc[missing, "season_start_year"].astype(int).add(1).astype(str)
            + "-05-01",
            utc=True,
        )
        birth_date = pd.to_datetime(
            output.loc[missing, "birth_date"],
            format="%b %d, %Y",
            errors="coerce",
            utc=True,
        )
        output.loc[missing, "age"] = np.floor((season_end - birth_date).dt.days / 365.2425)
        output.loc[missing, "age_source"] = "roster_birth_date"
    output = output.drop(columns="birth_date")
    output["age_known"] = output["age"].notna()
    return output


def _roster_birth_dates(roster_dir: Path | str) -> pd.DataFrame:
    """Return immutable player birth dates from locally cached roster snapshots."""

    root = Path(roster_dir)
    paths = sorted(root.glob("*/part-*.parquet"))
    rows: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        required = {"player_id", "birth_date"}
        if required <= set(frame.columns):
            rows.append(frame.loc[:, ["player_id", "birth_date"]])
    if not rows:
        return pd.DataFrame(columns=["player_id", "birth_date"])
    output = pd.concat(rows, ignore_index=True)
    output["player_id"] = pd.to_numeric(output["player_id"], errors="coerce")
    output = output.dropna(subset=["player_id", "birth_date"])
    output["player_id"] = output["player_id"].astype(int)
    return output.drop_duplicates("player_id", keep="last")


def _season_year(season: str) -> int:
    return int(str(season).split("-", maxsplit=1)[0])


def _validate_summary(summary: pd.DataFrame) -> None:
    required = {"age", "available_games", "known_player_games"}
    missing = required - set(summary)
    if missing:
        raise ValueError(f"Availability summary missing columns: {sorted(missing)}")
    if summary.empty or summary["known_player_games"].le(0).any():
        raise ValueError("Availability summary requires positive known-game denominators")


def _validate_predictions(predictions: pd.DataFrame) -> None:
    required = {
        "available_share",
        "predicted_available_share",
        "available_games",
        "known_player_games",
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Availability predictions missing columns: {sorted(missing)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the forward availability-state pilot")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--player-panel-path", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run = run_forward_availability(
        curated_dir=args.curated_dir,
        player_panel_path=args.player_panel_path,
        artifacts_dir=args.artifacts_dir,
    )
    print(run.run_dir)
