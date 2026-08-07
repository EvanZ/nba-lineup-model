"""Lineup-composition features for the contextual residual prior."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_profiles import PROFILE_RATE_COLUMNS

SHOOTING_COLUMN = "three_pm_per_100"
PASSING_COLUMN = "assists_per_100"
USAGE_COLUMN = "usage_per_100"
OFFENSIVE_REBOUND_COLUMN = "offensive_rebounds_per_100"
DEFENSIVE_REBOUND_COLUMN = "defensive_rebounds_per_100"
CREDIBLE_SHOOTER_THREE_PM_PER_100 = 2.0


def lineup_context_features(
    home_lineups: Sequence[Sequence[int]],
    away_lineups: Sequence[Sequence[int]],
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Encode fixed five-player home and away lineup compositions.

    Values are home-minus-away to align with a home-net-rating residual. The
    profile map is deliberately complete; missing player profiles are a data
    contract error rather than an implicit zero-value player.
    """

    if len(home_lineups) != len(away_lineups):
        raise ValueError("Contextual home and away lineup sequences must align")
    required = {"player_id", *PROFILE_RATE_COLUMNS, "profile_imputed", "profile_replacement_weight"}
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Contextual player profiles missing columns: {sorted(missing)}")
    if profiles["player_id"].duplicated().any():
        raise ValueError("Contextual player profiles contain duplicate player IDs")
    values = profiles.set_index("player_id")
    rows: list[dict[str, float]] = []
    for home, away in zip(home_lineups, away_lineups, strict=True):
        home_values = _lineup_values(home, values)
        away_values = _lineup_values(away, values)
        rows.append(_feature_row(home_values, away_values))
    return pd.DataFrame(rows, columns=contextual_feature_columns())


def contextual_feature_columns() -> tuple[str, ...]:
    """Return the fixed feature order used by the contextual residual model."""

    return (
        *(f"home_minus_away_{column}" for column in PROFILE_RATE_COLUMNS),
        "home_minus_away_bottom_two_three_pm",
        "home_minus_away_credible_shooter_count",
        "home_minus_away_top_two_assists",
        "home_minus_away_usage_concentration",
        "home_minus_away_sqrt_offensive_rebounds",
        "home_minus_away_sqrt_defensive_rebounds",
        "home_minus_away_imputed_count",
        "home_minus_away_replacement_weight",
        "home_minus_away_shooting_usage_interaction",
        "home_minus_away_shooter_passing_interaction",
        "home_minus_away_rebounding_usage_interaction",
    )


def _lineup_values(lineup: Sequence[int], values: pd.DataFrame) -> pd.DataFrame:
    player_ids = [int(player_id) for player_id in lineup]
    if len(player_ids) != 5 or len(set(player_ids)) != 5:
        raise ValueError("Contextual lineup features require five unique players")
    missing = sorted(set(player_ids) - set(values.index.astype(int)))
    if missing:
        raise ValueError(f"Contextual player profiles missing lineup players: {missing}")
    return values.loc[player_ids]


def _feature_row(home: pd.DataFrame, away: pd.DataFrame) -> dict[str, float]:
    result = {
        f"home_minus_away_{column}": float(home[column].sum() - away[column].sum())
        for column in PROFILE_RATE_COLUMNS
    }
    home_summary = _summary(home)
    away_summary = _summary(away)
    for name, home_value in home_summary.items():
        result[f"home_minus_away_{name}"] = float(home_value - away_summary[name])
    return result


def _summary(lineup: pd.DataFrame) -> dict[str, float]:
    shooting = np.sort(lineup[SHOOTING_COLUMN].to_numpy(dtype=float))
    assists = np.sort(lineup[PASSING_COLUMN].to_numpy(dtype=float))
    usage = lineup[USAGE_COLUMN].to_numpy(dtype=float)
    total_usage = float(usage.sum())
    usage_concentration = float(np.sort(usage)[-2:].sum() / total_usage) if total_usage else 0.0
    bottom_two = float(shooting[:2].sum())
    credible_shooters = float((shooting >= CREDIBLE_SHOOTER_THREE_PM_PER_100).sum())
    top_two_assists = float(assists[-2:].sum())
    sqrt_offensive_rebounds = float(np.sqrt(lineup[OFFENSIVE_REBOUND_COLUMN].sum()))
    sqrt_defensive_rebounds = float(np.sqrt(lineup[DEFENSIVE_REBOUND_COLUMN].sum()))
    return {
        "bottom_two_three_pm": bottom_two,
        "credible_shooter_count": credible_shooters,
        "top_two_assists": top_two_assists,
        "usage_concentration": usage_concentration,
        "sqrt_offensive_rebounds": sqrt_offensive_rebounds,
        "sqrt_defensive_rebounds": sqrt_defensive_rebounds,
        "imputed_count": float(lineup["profile_imputed"].sum()),
        "replacement_weight": float(lineup["profile_replacement_weight"].sum()),
        "shooting_usage_interaction": bottom_two * usage_concentration,
        "shooter_passing_interaction": credible_shooters * top_two_assists,
        "rebounding_usage_interaction": sqrt_offensive_rebounds * usage_concentration,
    }
