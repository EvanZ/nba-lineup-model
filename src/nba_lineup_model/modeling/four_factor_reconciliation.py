"""Reconcile Four Factors differentials to observed game net rating."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from nba_lineup_model.modeling.four_factor_rapm import FOUR_FACTORS, read_four_factor_stints
from nba_lineup_model.modeling.prior_rapm import HISTORICAL_SEASONS


DEFAULT_ARTIFACTS_DIR = Path("artifacts/models/analysis/four_factor_reconciliation")
FEATURE_COLUMNS = ("efg_pct", "turnover_avoidance", "orb_pct", "ftr")


def build_four_factor_reconciliation(
    *,
    through_season: str = "2025-26",
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> Path:
    """Fit a pooled team-game reconciliation with an explicit residual term.

    This is a descriptive accounting model, not a player prior and not a NAIL
    candidate. Its only role is to map observed Four Factors differentials to
    observed home net rating and show the remaining unexplained component.
    """

    seasons = tuple(season for season in (*HISTORICAL_SEASONS, "2025-26") if season <= through_season)
    games = pd.concat([_season_game_frame(season) for season in seasons], ignore_index=True)
    model = LinearRegression().fit(games.loc[:, FEATURE_COLUMNS], games["home_net_rating"], sample_weight=games["pace"])
    games["four_factor_predicted_net_rating"] = model.predict(games.loc[:, FEATURE_COLUMNS])
    games["unexplained_net_rating"] = games["home_net_rating"] - games["four_factor_predicted_net_rating"]
    contributions = games.loc[:, ["season", "game_id", "home_net_rating", "four_factor_predicted_net_rating", "unexplained_net_rating"]].copy()
    for feature, coefficient in zip(FEATURE_COLUMNS, model.coef_, strict=True):
        contributions[f"{feature}_contribution"] = games[feature] * coefficient
    coefficient_frame = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "points_per_100_per_percentage_point": model.coef_ / 100.0,
            "points_per_100_per_unit": model.coef_,
        }
    )
    season_summary = (
        contributions.groupby("season", as_index=False)
        .agg(
            game_count=("game_id", "size"),
            observed_net_rating_sd=("home_net_rating", "std"),
            unexplained_net_rating_rmse=("unexplained_net_rating", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        )
        .merge(
            games.groupby("season", as_index=False).apply(
                lambda frame: _weighted_r2(frame["home_net_rating"], frame["four_factor_predicted_net_rating"], frame["pace"]),
                include_groups=False,
            ).rename(columns={None: "weighted_r2"}),
            on="season",
            validate="one_to_one",
        )
    )
    return _write_artifact(
        through_season=through_season,
        coefficients=coefficient_frame,
        game_reconciliation=contributions,
        season_summary=season_summary,
        intercept=float(model.intercept_),
        artifacts_dir=Path(artifacts_dir),
    )


def _season_game_frame(season: str) -> pd.DataFrame:
    stints = read_four_factor_stints(season)
    sums = {
        "points_home": "sum",
        "points_away": "sum",
        "home_field_goal_attempts": "sum",
        "away_field_goal_attempts": "sum",
        "home_free_throw_attempts": "sum",
        "away_free_throw_attempts": "sum",
        "home_turnovers": "sum",
        "away_turnovers": "sum",
        "home_offensive_rebounds": "sum",
        "away_offensive_rebounds": "sum",
        "home_defensive_rebounds": "sum",
        "away_defensive_rebounds": "sum",
        "home_field_goals_made": "sum",
        "away_field_goals_made": "sum",
        "home_three_point_makes": "sum",
        "away_three_point_makes": "sum",
    }
    game = stints.groupby("game_id", as_index=False).agg(sums)
    home_possessions = (
        game["home_field_goal_attempts"]
        + 0.44 * game["home_free_throw_attempts"]
        + game["home_turnovers"]
        - game["home_offensive_rebounds"]
    )
    away_possessions = (
        game["away_field_goal_attempts"]
        + 0.44 * game["away_free_throw_attempts"]
        + game["away_turnovers"]
        - game["away_offensive_rebounds"]
    )
    game["pace"] = (home_possessions + away_possessions) / 2.0
    game["efg_pct"] = (
        (game["home_field_goals_made"] + 0.5 * game["home_three_point_makes"])
        / game["home_field_goal_attempts"]
        - (game["away_field_goals_made"] + 0.5 * game["away_three_point_makes"])
        / game["away_field_goal_attempts"]
    )
    game["turnover_avoidance"] = -(
        game["home_turnovers"] / (game["home_field_goal_attempts"] + 0.44 * game["home_free_throw_attempts"] + game["home_turnovers"])
        - game["away_turnovers"] / (game["away_field_goal_attempts"] + 0.44 * game["away_free_throw_attempts"] + game["away_turnovers"])
    )
    game["orb_pct"] = (
        game["home_offensive_rebounds"] / (game["home_offensive_rebounds"] + game["away_defensive_rebounds"])
        - game["away_offensive_rebounds"] / (game["away_offensive_rebounds"] + game["home_defensive_rebounds"])
    )
    game["ftr"] = (
        game["home_free_throw_attempts"] / game["home_field_goal_attempts"]
        - game["away_free_throw_attempts"] / game["away_field_goal_attempts"]
    )
    game["home_net_rating"] = 100.0 * (game["points_home"] - game["points_away"]) / game["pace"]
    game["season"] = season
    output = game.loc[:, ["season", "game_id", "pace", "home_net_rating", *FEATURE_COLUMNS]].dropna()
    if output.empty or not output["pace"].gt(0).all():
        raise ValueError(f"Four Factor reconciliation has no valid game rows for {season}")
    return output


def _weighted_r2(observed: pd.Series, predicted: pd.Series, weights: pd.Series) -> float:
    mean = np.average(observed, weights=weights)
    residual = np.average((observed - predicted) ** 2, weights=weights)
    baseline = np.average((observed - mean) ** 2, weights=weights)
    return float(1.0 - residual / baseline) if baseline > 0 else np.nan


def _write_artifact(
    *,
    through_season: str,
    coefficients: pd.DataFrame,
    game_reconciliation: pd.DataFrame,
    season_summary: pd.DataFrame,
    intercept: float,
    artifacts_dir: Path,
) -> Path:
    now = datetime.now(UTC)
    run_id = f"four-factor-reconciliation-{through_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / through_season
    target = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        coefficients.to_parquet(temporary / "factor_point_values.parquet", index=False)
        game_reconciliation.to_parquet(temporary / "game_reconciliation.parquet", index=False)
        season_summary.to_parquet(temporary / "season_reconciliation_summary.parquet", index=False)
        (temporary / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "through_season": through_season,
                    "created_at": now.isoformat(),
                    "purpose": "descriptive_game_level_four_factor_reconciliation",
                    "intercept_points_per_100": intercept,
                    "features": list(FEATURE_COLUMNS),
                    "residual": "observed home net rating minus fitted Four Factors net rating",
                    "not_a_player_prior": True,
                },
                indent=2,
            )
            + "\n"
        )
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile Four Factors to game net rating")
    parser.add_argument("--through-season", default="2025-26")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    args = parser.parse_args()
    print(build_four_factor_reconciliation(through_season=args.through_season, artifacts_dir=args.artifacts_dir))


if __name__ == "__main__":
    main()
