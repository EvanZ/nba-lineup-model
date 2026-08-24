"""Forward-looking game-schedule controls for lineup models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACK_TO_BACK_COLUMN = "home_minus_away_back_to_back"
REQUIRED_CATALOG_COLUMNS = {
    "game_id",
    "season",
    "game_date",
    "home_team_id",
    "away_team_id",
}
COMPETITIVE_SEASON_TYPES = {"regular", "play_in", "playoffs", "nba_cup_final"}
ACTIVE_GAME_STATUSES = {"final", "live", "scheduled"}


@dataclass(frozen=True)
class BackToBackScheduleModel:
    """One completed-season, antisymmetric back-to-back adjustment."""

    pipeline: Pipeline
    source_season: str

    def predict_games(
        self,
        games: pd.DataFrame,
        schedule_features: pd.DataFrame,
    ) -> np.ndarray:
        """Return home-minus-away schedule adjustment for arbitrary game rows."""

        features = attach_back_to_back_feature(games, schedule_features)
        return np.asarray(
            self.pipeline.predict(features.loc[:, [BACK_TO_BACK_COLUMN]]), dtype=float
        )


def build_back_to_back_game_features(catalog: pd.DataFrame) -> pd.DataFrame:
    """Derive one known-before-tipoff back-to-back flag for each game side.

    Dates are evaluated within a team-season across every cataloged game, not
    merely games whose possessions happened to survive modeling validation.
    """

    missing = REQUIRED_CATALOG_COLUMNS - set(catalog)
    if missing:
        raise ValueError(f"Game catalog lacks schedule columns: {sorted(missing)}")
    games = catalog.copy()
    if "season_type" in games:
        games = games.loc[games["season_type"].isin(COMPETITIVE_SEASON_TYPES)].copy()
    if "game_status" in games:
        games = games.loc[games["game_status"].isin(ACTIVE_GAME_STATUSES)].copy()
    games = games.loc[:, sorted(REQUIRED_CATALOG_COLUMNS)].copy()
    if games.empty:
        raise ValueError("Game catalog has no competitive scheduled games")
    if games["game_id"].duplicated().any():
        raise ValueError("Game catalog must have one row per game")
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="raise").dt.normalize()
    if games[["season", "home_team_id", "away_team_id"]].isna().any().any():
        raise ValueError("Game catalog schedule fields cannot be null")

    home = games.loc[:, ["game_id", "season", "game_date", "home_team_id"]].rename(
        columns={"home_team_id": "team_id"}
    )
    away = games.loc[:, ["game_id", "season", "game_date", "away_team_id"]].rename(
        columns={"away_team_id": "team_id"}
    )
    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(
        ["season", "team_id", "game_date", "game_id"], kind="stable"
    )
    previous_date = team_games.groupby(["season", "team_id"], sort=False)["game_date"].shift()
    team_games["is_back_to_back"] = (
        (team_games["game_date"] - previous_date).dt.days.eq(1).astype(int)
    )
    flags = team_games.loc[:, ["game_id", "team_id", "is_back_to_back"]]
    home_flags = flags.rename(
        columns={"team_id": "home_team_id", "is_back_to_back": "home_back_to_back"}
    )
    away_flags = flags.rename(
        columns={"team_id": "away_team_id", "is_back_to_back": "away_back_to_back"}
    )
    output = games.loc[:, ["game_id", "season", "home_team_id", "away_team_id"]]
    output = output.merge(
        home_flags,
        on=["game_id", "home_team_id"],
        how="left",
        validate="one_to_one",
    )
    output = output.merge(
        away_flags,
        on=["game_id", "away_team_id"],
        how="left",
        validate="one_to_one",
    )
    if output[["home_back_to_back", "away_back_to_back"]].isna().any().any():
        raise ValueError("Unable to assign a back-to-back flag to every cataloged game")
    output[BACK_TO_BACK_COLUMN] = (
        output["home_back_to_back"].astype(float) - output["away_back_to_back"].astype(float)
    )
    return output.sort_values(["season", "game_id"], kind="stable").reset_index(drop=True)


def attach_back_to_back_feature(
    games: pd.DataFrame,
    schedule_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the cataloged control to stints or possession rows by game id."""

    if "game_id" not in games:
        raise ValueError("Schedule-control rows require game_id")
    required = {"game_id", BACK_TO_BACK_COLUMN}
    missing = required - set(schedule_features)
    if missing:
        raise ValueError(f"Schedule feature frame lacks: {sorted(missing)}")
    lookup = schedule_features.loc[:, ["game_id", BACK_TO_BACK_COLUMN]].copy()
    lookup["game_id"] = lookup["game_id"].astype(str)
    if lookup["game_id"].duplicated().any():
        raise ValueError("Schedule feature frame must be unique by game_id")
    output = games.copy()
    output["game_id"] = output["game_id"].astype(str)
    output = output.drop(columns=[BACK_TO_BACK_COLUMN], errors="ignore").merge(
        lookup,
        on="game_id",
        how="left",
        validate="many_to_one",
    )
    if output[BACK_TO_BACK_COLUMN].isna().any():
        missing_ids = sorted(output.loc[output[BACK_TO_BACK_COLUMN].isna(), "game_id"].unique())
        raise ValueError(
            "Schedule control is missing cataloged games: " + ", ".join(missing_ids[:10])
        )
    return output


def fit_back_to_back_schedule_model(
    stints: pd.DataFrame,
    target: np.ndarray,
    *,
    alpha: float,
    source_season: str,
) -> BackToBackScheduleModel:
    """Fit a weighted Ridge coefficient for the signed back-to-back control."""

    if alpha <= 0:
        raise ValueError("Back-to-back schedule alpha must be positive")
    if BACK_TO_BACK_COLUMN not in stints or "possessions" not in stints:
        raise ValueError("Schedule fitting requires back-to-back and possession columns")
    values = np.asarray(target, dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    features = stints.loc[:, [BACK_TO_BACK_COLUMN]].astype(float)
    if len(features) != len(values) or not np.isfinite(values).all():
        raise ValueError("Schedule model target must be finite and align with stints")
    if (
        not np.isfinite(features.to_numpy()).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Schedule model features and weights must be finite and positive")
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha, fit_intercept=False))]
    )
    pipeline.fit(features, values, ridge__sample_weight=weights)
    return BackToBackScheduleModel(pipeline=pipeline, source_season=source_season)


def schedule_model_metadata(
    model: BackToBackScheduleModel,
    stints: pd.DataFrame,
    *,
    alpha: float,
) -> dict[str, object]:
    """Return inspectable per-season control metadata in raw and standardized units."""

    scale = model.pipeline.named_steps["scale"]
    ridge = model.pipeline.named_steps["ridge"]
    standardized = float(np.asarray(ridge.coef_, dtype=float).item())
    feature_scale = float(np.asarray(scale.scale_, dtype=float).item())
    raw = standardized / feature_scale if feature_scale else 0.0
    values = stints[BACK_TO_BACK_COLUMN].to_numpy(dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    return {
        "schedule_control": "home_minus_away_back_to_back",
        "schedule_control_alpha": alpha,
        "schedule_control_standardized_weight": standardized,
        "schedule_control_raw_weight": raw,
        "schedule_control_feature_standard_deviation": feature_scale,
        "schedule_control_nonzero_stint_count": int(np.count_nonzero(values)),
        "schedule_control_nonzero_possession_share": float(
            np.average(values != 0.0, weights=weights)
        ),
    }
