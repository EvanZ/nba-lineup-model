"""Forward-safe empirical calibration of lineup rebound capacity.

Player ORB% and DRB% are individual claims on a finite set of rebound
opportunities.  This module learns how those claims translate into a team's
realized defensive-rebound probability, rather than imposing a hand-chosen
cap on their sum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nba_lineup_model.modeling.contextual_profiles import PROFILE_REBOUND_PERCENT_COLUMNS

DEFENSIVE_REBOUND_COLUMN: Final = "defensive_rebound_pct"
OFFENSIVE_REBOUND_COLUMN: Final = "offensive_rebound_pct"
REFERENCE_GRID_SIZE: Final = 21


@dataclass(frozen=True)
class ReboundOpportunityModel:
    """Maps two five-player rebound-claim totals to realized rebound rates."""

    pipeline: Pipeline
    reference_defensive_claims: np.ndarray
    reference_offensive_claims: np.ndarray
    training_season: str
    training_opportunity_count: int

    def predict_defensive_rebound_probability(
        self,
        defensive_claims: np.ndarray | float,
        offensive_claims: np.ndarray | float,
    ) -> np.ndarray:
        """Predict the defensive unit's chance of collecting an opportunity."""

        defensive = np.asarray(defensive_claims, dtype=float)
        offensive = np.asarray(offensive_claims, dtype=float)
        defensive, offensive = np.broadcast_arrays(defensive, offensive)
        values = pd.DataFrame(
            {
                "defensive_claims": defensive.ravel(),
                "offensive_claims": offensive.ravel(),
            }
        )
        probability = self.pipeline.predict_proba(values)[:, 1]
        return probability.reshape(defensive.shape)

    def predict_unit_rebound_rates(
        self,
        offensive_claims: np.ndarray | float,
        defensive_claims: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return portable expected ORB% and DRB% against the reference field."""

        offense = np.asarray(offensive_claims, dtype=float).reshape(-1)
        defense = np.asarray(defensive_claims, dtype=float).reshape(-1)
        if len(offense) != len(defense):
            raise ValueError("Offensive and defensive claim arrays must align")
        expected_defense = self.predict_defensive_rebound_probability(
            defense[:, None], self.reference_offensive_claims[None, :]
        ).mean(axis=1)
        expected_offense = 1.0 - self.predict_defensive_rebound_probability(
            self.reference_defensive_claims[None, :], offense[:, None]
        ).mean(axis=1)
        return 100.0 * expected_offense, 100.0 * expected_defense


def fit_rebound_opportunity_model(
    season: str,
    profiles: pd.DataFrame,
    *,
    curated_dir: Path | str,
) -> ReboundOpportunityModel:
    """Fit season ``t`` rebound realization from pre-``t`` player profiles.

    The fitted object is a completed-season state.  It is only consumed by
    HPM's next-season context model and therefore never uses target-season
    outcomes in a frozen forecast.
    """

    opportunities = rebound_opportunity_frame(season, profiles, curated_dir=curated_dir)
    if len(opportunities) < 1_000:
        raise ValueError(f"{season} has too few usable rebound opportunities")
    features = opportunities.loc[:, ["defensive_claims", "offensive_claims"]]
    target = opportunities["defensive_rebound"].to_numpy(dtype=int)
    spline = SplineTransformer(n_knots=4, degree=2, extrapolation="linear")
    scale = StandardScaler()
    classifier = LogisticRegression(C=0.1, max_iter=500, solver="lbfgs")
    pipeline = Pipeline(
        [("spline", spline), ("scale", scale), ("logistic", classifier)]
    )
    pipeline.fit(features, target)
    return ReboundOpportunityModel(
        pipeline=pipeline,
        reference_defensive_claims=_reference_grid(opportunities["defensive_claims"]),
        reference_offensive_claims=_reference_grid(opportunities["offensive_claims"]),
        training_season=season,
        training_opportunity_count=len(opportunities),
    )


def rebound_opportunity_frame(
    season: str,
    profiles: pd.DataFrame,
    *,
    curated_dir: Path | str,
) -> pd.DataFrame:
    """Return one labeled non-dead-ball rebound opportunity per event.

    Rows are canonicalized from the defending unit's point of view: a
    defensive rebound has target one, while an offensive rebound has target
    zero.  The two feature values are the *prior-profile* DRB% and ORB%
    claims of the defending and offensive five-player units respectively.
    """

    required = {"player_id", *PROFILE_REBOUND_PERCENT_COLUMNS}
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Rebound calibration profiles missing columns: {sorted(missing)}")
    if profiles["player_id"].duplicated().any():
        raise ValueError("Rebound calibration profiles contain duplicate player IDs")
    root = Path(curated_dir)
    events = _read_partition_files(
        root / "events" / season / "regular",
        ["event_id", "event_type", "event_subtype", "team_id"],
    )
    events = events.loc[
        events["event_type"].astype("string").str.lower().eq("rebound")
        & events["event_subtype"].astype("string").str.lower().isin(["offensive", "defensive"])
    ].copy()
    lineups = _read_partition_files(
        root / "event_lineups" / season / "regular",
        [
            "event_id",
            "home_player_ids_before",
            "away_player_ids_before",
            "catalog_home_team_id",
            "catalog_away_team_id",
        ],
    )
    frame = events.merge(lineups, on="event_id", how="inner", validate="one_to_one")
    frame["team_id"] = pd.to_numeric(frame["team_id"], errors="coerce")
    frame = frame.loc[
        frame["team_id"].isin(
            pd.concat([frame["catalog_home_team_id"], frame["catalog_away_team_id"]])
        )
    ].copy()
    if frame.empty:
        raise ValueError(f"No aligned rebound opportunities for {season}")
    profiles_by_player = profiles.set_index("player_id")
    defensive_claims: list[float] = []
    offensive_claims: list[float] = []
    targets: list[int] = []
    for row in frame.itertuples(index=False):
        event_team_is_home = int(row.team_id) == int(row.catalog_home_team_id)
        is_defensive = str(row.event_subtype).lower() == "defensive"
        defense_is_home = event_team_is_home if is_defensive else not event_team_is_home
        defense_lineup = row.home_player_ids_before if defense_is_home else row.away_player_ids_before
        offense_lineup = row.away_player_ids_before if defense_is_home else row.home_player_ids_before
        if not _valid_lineup(defense_lineup) or not _valid_lineup(offense_lineup):
            continue
        defender_ids = [int(player_id) for player_id in defense_lineup]
        offender_ids = [int(player_id) for player_id in offense_lineup]
        if (
            not set(defender_ids).issubset(profiles_by_player.index)
            or not set(offender_ids).issubset(profiles_by_player.index)
        ):
            continue
        defensive_claims.append(
            float(profiles_by_player.loc[defender_ids, DEFENSIVE_REBOUND_COLUMN].sum())
        )
        offensive_claims.append(
            float(profiles_by_player.loc[offender_ids, OFFENSIVE_REBOUND_COLUMN].sum())
        )
        targets.append(int(is_defensive))
    output = pd.DataFrame(
        {
            "defensive_claims": defensive_claims,
            "offensive_claims": offensive_claims,
            "defensive_rebound": targets,
        }
    )
    if output.empty:
        raise ValueError(f"No rebound opportunities with complete profiles for {season}")
    return output


def _read_partition_files(path: Path, columns: list[str]) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet partitions found at {path}")
    return pd.concat([pd.read_parquet(file, columns=columns) for file in files], ignore_index=True)


def _reference_grid(values: pd.Series) -> np.ndarray:
    return np.quantile(values.to_numpy(dtype=float), np.linspace(0.05, 0.95, REFERENCE_GRID_SIZE))


def _valid_lineup(value: object) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) == 5 and len(set(value)) == 5
