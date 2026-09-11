"""Frozen exposure screens for availability-loss episode frequency."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nba_lineup_model.rotation.availability_episode_mart import (
    DEFAULT_OUTPUT_DIR as DEFAULT_EPISODE_DIR,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/rotation/actuarial_availability/exposure_screen")
DEFAULT_FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
_CANDIDATES = {
    "GP": "prior_panel_gp",
    "GS": "prior_panel_gs",
    "Minutes": "prior_panel_minutes",
    "Minutes per GP": "prior_minutes_per_gp",
    "Available rostered games": "prior_available_rostered_games",
}
_RIDGE = 1e-4


def build_forward_exposure_pairs(
    player_seasons: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Pair prior workload with strictly next-season episode starts.

    An availability-loss episode that starts before a target season is a carried episode,
    not a new target-season event. Such rows are retained and flagged for the
    audit, but excluded from the frequency screen because they have already
    entered unavailability before the prediction horizon begins.
    """

    required = {
        "season",
        "player_id",
        "player_name",
        "availability_episode_count",
        "unavailability_loss_cost",
        "available_rostered_games",
        "rostered_team_games",
        "panel_gp",
        "panel_gs",
        "panel_minutes",
    }
    missing = sorted(required - set(player_seasons))
    if missing:
        raise ValueError(f"Episode player-season table missing columns: {missing}")
    frame = player_seasons.copy()
    if "panel_age" not in frame:
        frame["panel_age"] = np.nan
    frame["season_start_year"] = frame["season"].map(_season_year)
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype(int)
    for column in required - {"season", "player_id", "player_name"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prior = frame.loc[
        :,
        [
            "season",
            "season_start_year",
            "player_id",
            "player_name",
            "availability_episode_count",
            "unavailability_loss_cost",
            "available_rostered_games",
            "panel_gp",
            "panel_gs",
            "panel_minutes",
            "panel_age",
        ],
    ].rename(
        columns={
            "season": "prior_season",
            "season_start_year": "prior_season_start_year",
            "player_name": "prior_player_name",
            "availability_episode_count": "prior_availability_episode_count",
            "unavailability_loss_cost": "prior_unavailability_loss_cost",
            "available_rostered_games": "prior_available_rostered_games",
            "panel_gp": "prior_panel_gp",
            "panel_gs": "prior_panel_gs",
            "panel_minutes": "prior_panel_minutes",
        }
    )
    target = frame.loc[
        :,
        [
            "season",
            "season_start_year",
            "player_id",
            "player_name",
            "availability_episode_count",
            "rostered_team_games",
            "panel_age",
        ],
    ].rename(
        columns={
            "season": "target_season",
            "season_start_year": "target_season_start_year",
            "player_name": "target_player_name",
            "availability_episode_count": "target_availability_episode_count",
            "rostered_team_games": "target_rostered_team_games",
            "panel_age": "target_age",
        }
    )
    pairs = prior.merge(target, on="player_id", how="inner", validate="many_to_many")
    pairs = pairs.loc[
        pairs["target_season_start_year"].eq(pairs["prior_season_start_year"] + 1)
    ].copy()
    pairs["prior_minutes_per_gp"] = (
        pairs["prior_panel_minutes"] / pairs["prior_panel_gp"].clip(lower=1.0)
    )
    pairs["has_cross_season_unavailability_carryover"] = False
    if not episodes.empty:
        required_episodes = {
            "player_id",
            "episode_start_season",
            "episode_end_season",
        }
        missing_episodes = sorted(required_episodes - set(episodes))
        if missing_episodes:
            raise ValueError(f"Episodes missing continuity columns: {missing_episodes}")
        carried: set[tuple[int, str]] = set()
        for row in episodes.loc[:, sorted(required_episodes)].itertuples(index=False):
            start = _season_year(row.episode_start_season)
            end = _season_year(row.episode_end_season)
            carried.update(
                (int(row.player_id), f"{year}-{str(year + 1)[-2:]}")
                for year in range(start + 1, end + 1)
            )
        pairs["has_cross_season_unavailability_carryover"] = [
            (int(row.player_id), str(row.target_season)) in carried
            for row in pairs.loc[:, ["player_id", "target_season"]].itertuples(index=False)
        ]
    return pairs.sort_values(
        ["target_season_start_year", "player_id"], kind="stable"
    ).reset_index(drop=True)


def run_frozen_exposure_screen(
    pairs: pd.DataFrame,
    *,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score each lagged workload candidate with expanding-window Poisson fits."""

    required = {
        "target_season",
        "target_season_start_year",
        "target_availability_episode_count",
        "target_rostered_team_games",
        "has_cross_season_unavailability_carryover",
        "prior_unavailability_loss_cost",
        "prior_availability_episode_count",
        "target_age",
        *_CANDIDATES.values(),
    }
    missing = sorted(required - set(pairs))
    if missing:
        raise ValueError(f"Forward exposure pairs missing columns: {missing}")
    eligible = pairs.loc[
        ~pairs["has_cross_season_unavailability_carryover"].astype(bool)
        & pairs["target_rostered_team_games"].gt(0)
    ].copy()
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for candidate_name, candidate_column in _CANDIDATES.items():
        candidate_predictions: list[pd.DataFrame] = []
        for target_season in frozen_seasons:
            target_year = _season_year(target_season)
            train = eligible.loc[eligible["target_season_start_year"].lt(target_year)].copy()
            test = eligible.loc[eligible["target_season"].eq(target_season)].copy()
            if train.empty or test.empty:
                raise ValueError(f"Frozen exposure screen lacks data for {target_season}")
            columns = [
                "target_age",
                "prior_unavailability_loss_cost",
                "prior_availability_episode_count",
                candidate_column,
            ]
            train_x, test_x = _standardized_design(train, test, columns)
            coefficients = _fit_poisson_glm(
                train_x,
                train["target_availability_episode_count"].to_numpy(dtype=float),
                np.log(train["target_rostered_team_games"].to_numpy(dtype=float)),
            )
            expected = _predict_poisson_glm(
                test_x,
                np.log(test["target_rostered_team_games"].to_numpy(dtype=float)),
                coefficients,
            )
            candidate_predictions.append(
                test.assign(
                    candidate=candidate_name,
                    candidate_value=test[candidate_column].to_numpy(dtype=float),
                    predicted_episode_count=expected,
                )
            )
        combined = pd.concat(candidate_predictions, ignore_index=True)
        deviance = poisson_deviance(
            combined["target_availability_episode_count"].to_numpy(dtype=float),
            combined["predicted_episode_count"].to_numpy(dtype=float),
        )
        metrics.append(
            {
                "candidate": candidate_name,
                "poisson_deviance": deviance,
                "mean_predicted_episode_count": float(combined["predicted_episode_count"].mean()),
                "mean_observed_episode_count": float(
                    combined["target_availability_episode_count"].mean()
                ),
                "concentration_gini": concentration_gini(
                    combined["target_availability_episode_count"].to_numpy(dtype=float),
                    combined["predicted_episode_count"].to_numpy(dtype=float),
                ),
                "player_seasons": int(len(combined)),
            }
        )
        predictions.append(combined)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics_frame = pd.DataFrame(metrics).sort_values("poisson_deviance", kind="stable")
    deciles = build_risk_decile_table(prediction_frame)
    return metrics_frame.reset_index(drop=True), prediction_frame, deciles


def build_risk_decile_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frozen predictions into equal-count risk deciles by candidate."""

    rows: list[dict[str, object]] = []
    for candidate, group in predictions.groupby("candidate", sort=False):
        ordered = group.sort_values(
            ["predicted_episode_count", "player_id"], kind="stable"
        ).copy()
        ordered["risk_decile"] = pd.qcut(
            ordered["predicted_episode_count"].rank(method="first"),
            q=10,
            labels=False,
        ).astype(int) + 1
        for decile, values in ordered.groupby("risk_decile", sort=True):
            observed = float(values["target_availability_episode_count"].sum())
            predicted = float(values["predicted_episode_count"].sum())
            exposure = float(values["target_rostered_team_games"].sum())
            rows.append(
                {
                    "candidate": candidate,
                    "risk_decile": int(decile),
                    "player_seasons": int(len(values)),
                    "mean_candidate_value": float(values["candidate_value"].mean()),
                    "target_rostered_team_games": exposure,
                    "observed_episode_count": observed,
                    "predicted_episode_count": predicted,
                    "observed_episodes_per_100_games": 100.0 * observed / exposure,
                    "predicted_episodes_per_100_games": 100.0 * predicted / exposure,
                    "observed_to_expected": observed / predicted if predicted > 0.0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def poisson_deviance(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Return the total Poisson deviance for count predictions."""

    y = np.asarray(observed, dtype=float)
    mu = np.clip(np.asarray(predicted, dtype=float), 1e-12, None)
    positive = y > 0.0
    terms = np.where(positive, y * np.log(np.clip(y, 1e-12, None) / mu) - (y - mu), mu)
    return float(2.0 * terms.sum())


def concentration_gini(observed: np.ndarray, scores: np.ndarray) -> float:
    """Return a normalized concentration Gini for non-negative count outcomes."""

    y = np.asarray(observed, dtype=float)
    if y.sum() <= 0.0:
        return float("nan")
    order = np.argsort(np.asarray(scores, dtype=float), kind="stable")
    cumulative = np.cumsum(y[order]) / y.sum()
    random = (np.arange(1, len(y) + 1, dtype=float) / len(y))
    area = np.trapezoid(np.concatenate(([0.0], cumulative)), np.concatenate(([0.0], random)))
    return float(1.0 - 2.0 * area)


def plot_risk_decile_lift(deciles: pd.DataFrame, output_path: Path | str) -> None:
    """Write pooled frozen observed-versus-predicted unavailable-episode lift panels."""

    candidates = list(deciles["candidate"].drop_duplicates())
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for axis, candidate in zip(axes.flat, candidates, strict=False):
        values = deciles.loc[deciles["candidate"].eq(candidate)].sort_values("risk_decile")
        axis.plot(
            values["risk_decile"],
            values["observed_episodes_per_100_games"],
            color="#1f5d9d",
            marker="o",
            label="Observed",
        )
        axis.plot(
            values["risk_decile"],
            values["predicted_episodes_per_100_games"],
            color="#d97706",
            marker="o",
            label="Predicted",
        )
        axis.set_title(candidate)
        axis.set_xlabel("Predicted frequency-risk decile")
        axis.set_xticks(range(1, 11))
        axis.grid(axis="y", alpha=0.25)
    for axis in axes[:, 0]:
        axis.set_ylabel("Episodes per 100 rostered games")
    for axis in axes.flat[len(candidates) :]:
        axis.axis("off")
    figure.legend(["Observed", "Predicted"], loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Frozen unavailable-episode frequency lift and calibration", y=0.99)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _standardized_design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_values: list[np.ndarray] = []
    test_values: list[np.ndarray] = []
    for column in columns:
        train_column = (
            pd.to_numeric(train[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        test_column = (
            pd.to_numeric(test[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        mean = float(train_column.mean())
        scale = float(train_column.std())
        if scale <= 1e-12:
            scale = 1.0
        train_values.append((train_column - mean) / scale)
        test_values.append((test_column - mean) / scale)
    return (
        np.column_stack([np.ones(len(train)), *train_values]),
        np.column_stack([np.ones(len(test)), *test_values]),
    )


def _fit_poisson_glm(design: np.ndarray, observed: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Fit a lightly ridge-stabilized Poisson log-link model with an offset."""

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        linear = np.clip(offset + design @ coefficients, -30.0, 30.0)
        expected = np.exp(linear)
        value = float((expected - observed * linear).sum())
        gradient = design.T @ (expected - observed)
        value += 0.5 * _RIDGE * float(np.dot(coefficients[1:], coefficients[1:]))
        gradient[1:] += _RIDGE * coefficients[1:]
        return value, gradient

    result = minimize(
        fun=lambda values: objective(values)[0],
        x0=np.zeros(design.shape[1], dtype=float),
        jac=lambda values: objective(values)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Poisson exposure screen failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _predict_poisson_glm(
    design: np.ndarray,
    offset: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    return np.exp(np.clip(offset + design @ coefficients, -30.0, 30.0))


def _season_year(season: str) -> int:
    return int(str(season)[:4])


def main() -> int:
    """Run and persist the frozen GP/GS/minutes exposure-frequency screen."""

    parser = argparse.ArgumentParser(
        description="Screen real basketball exposure for availability-loss frequency"
    )
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    player_seasons = pd.read_parquet(args.episode_dir / "player_seasons.parquet")
    episodes = pd.read_parquet(args.episode_dir / "episodes.parquet")
    pairs = build_forward_exposure_pairs(player_seasons, episodes)
    metrics, predictions, deciles = run_frozen_exposure_screen(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.output_dir / "forward_pairs.parquet", index=False)
    metrics.to_csv(args.output_dir / "frozen_metrics.csv", index=False)
    predictions.to_parquet(args.output_dir / "frozen_predictions.parquet", index=False)
    deciles.to_csv(args.output_dir / "risk_deciles.csv", index=False)
    plot_risk_decile_lift(deciles, args.output_dir / "risk_decile_lift.png")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
