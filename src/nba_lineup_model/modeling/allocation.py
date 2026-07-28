from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.stints import _assign_segments_to_stints

PossessionAllocationPolicy = Literal[
    "equal_segments",
    "starting_lineup",
    "terminal_lineup",
    "boundary_split",
    "exclude_multi_lineup",
]
POSSESSION_ALLOCATION_POLICIES: tuple[PossessionAllocationPolicy, ...] = (
    "equal_segments",
    "starting_lineup",
    "terminal_lineup",
    "boundary_split",
    "exclude_multi_lineup",
)

_STINT_COLUMNS = (
    "game_id",
    "game_date",
    "game_time_utc",
    "stint_index",
    "home_team_id",
    "away_team_id",
    "home_team_tricode",
    "away_team_tricode",
    "home_player_ids",
    "away_player_ids",
    "duration_seconds",
)


def allocation_policy_stints(
    lineup_stints: pd.DataFrame,
    possession_segments: pd.DataFrame,
    policy: PossessionAllocationPolicy,
) -> pd.DataFrame:
    """Construct comparable RAPM stints under one possession-allocation policy."""

    if policy not in POSSESSION_ALLOCATION_POLICIES:
        raise ValueError(f"Unknown possession allocation policy: {policy}")
    stints = lineup_stints.copy()
    segments = possession_segments.copy()
    segments["_stint_index"] = _assign_segments_to_stints(stints, segments)
    group_keys = ["game_id", "possession_id"]
    groups = segments.groupby(group_keys, sort=False)
    segments["_segment_count"] = groups["possession_id"].transform("size")
    segments["_segment_position"] = groups.cumcount()
    segments["_possession_points_home"] = groups["points_home"].transform("sum")
    segments["_possession_points_away"] = groups["points_away"].transform("sum")

    share = _allocation_share(segments, policy)
    segments["_possession_share"] = share
    if policy == "equal_segments":
        segments["_assigned_points_home"] = segments["points_home"].astype(float)
        segments["_assigned_points_away"] = segments["points_away"].astype(float)
    elif policy == "exclude_multi_lineup":
        included = segments["_segment_count"].eq(1)
        segments["_assigned_points_home"] = segments["points_home"].where(
            included,
            0.0,
        )
        segments["_assigned_points_away"] = segments["points_away"].where(
            included,
            0.0,
        )
    else:
        segments["_assigned_points_home"] = segments["_possession_points_home"] * share
        segments["_assigned_points_away"] = segments["_possession_points_away"] * share

    home_offense = segments["offense_team_id"].eq(segments["catalog_home_team_id"])
    away_offense = segments["offense_team_id"].eq(segments["catalog_away_team_id"])
    if not (home_offense | away_offense).all():
        raise ValueError("Possession segment offense team is not a game team")
    segments["_home_possessions"] = share.where(home_offense, 0.0)
    segments["_away_possessions"] = share.where(away_offense, 0.0)
    allocated = (
        segments.groupby(["game_id", "_stint_index"], sort=False)
        .agg(
            points_home=("_assigned_points_home", "sum"),
            points_away=("_assigned_points_away", "sum"),
            home_offensive_possessions=("_home_possessions", "sum"),
            away_offensive_possessions=("_away_possessions", "sum"),
        )
        .reset_index()
        .rename(columns={"_stint_index": "stint_index"})
    )

    base = stints.rename(
        columns={
            "catalog_home_team_id": "home_team_id",
            "catalog_away_team_id": "away_team_id",
            "catalog_home_team_tricode": "home_team_tricode",
            "catalog_away_team_tricode": "away_team_tricode",
        }
    )
    output = base.loc[:, _STINT_COLUMNS].merge(
        allocated,
        on=["game_id", "stint_index"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "points_home",
        "points_away",
        "home_offensive_possessions",
        "away_offensive_possessions",
    ):
        output[column] = output[column].fillna(0.0).astype(float)
    output["possessions"] = (
        output["home_offensive_possessions"] + output["away_offensive_possessions"]
    ) / 2.0
    output = output.loc[output["possessions"].gt(0)].copy()
    output["home_margin"] = output["points_home"] - output["points_away"]
    output["target_home_net_rating"] = 100.0 * output["home_margin"] / output["possessions"]
    output["allocation_policy"] = policy
    return output.sort_values(
        ["game_time_utc", "game_id", "stint_index"],
        kind="stable",
    ).reset_index(drop=True)


def _allocation_share(
    segments: pd.DataFrame,
    policy: PossessionAllocationPolicy,
) -> pd.Series:
    count = segments["_segment_count"]
    position = segments["_segment_position"]
    if policy == "equal_segments":
        return 1.0 / count.astype(float)
    if policy == "starting_lineup":
        return position.eq(0).astype(float)
    if policy == "terminal_lineup":
        return position.eq(count - 1).astype(float)
    if policy == "exclude_multi_lineup":
        return count.eq(1).astype(float)
    single = count.eq(1)
    boundary = position.eq(0) | position.eq(count - 1)
    return pd.Series(
        np.where(single, 1.0, np.where(boundary, 0.5, 0.0)),
        index=segments.index,
        dtype=float,
    )
