"""Forward empirical model for allocation of lineup terminal usage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

USAGE_COLUMN = "usage_per_100"
TURNOVER_COLUMN = "turnovers_per_100"
USAGE_EVENT_WEIGHTS = {"2pt": 1.0, "3pt": 1.0, "freethrow": 0.44, "turnover": 1.0}


@dataclass(frozen=True)
class UsageAllocationModel:
    """Conditional-logit allocation of terminal offensive actions within a unit."""

    temperature: float
    claim_scale: float
    claim_budget: float
    training_season: str
    training_event_count: int

    def predict_shares(self, claims: np.ndarray) -> np.ndarray:
        """Return fitted five-player action shares for aligned claim rows."""

        values = np.asarray(claims, dtype=float)
        if values.ndim != 2 or values.shape[1] != 5:
            raise ValueError("Usage allocation requires an n-by-5 claim matrix")
        logits = self.temperature * values / self.claim_scale
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

    def lineup_features(self, claims: np.ndarray, turnovers: np.ndarray) -> pd.DataFrame:
        """Return continuous, cutoff-free usage-pressure context features."""

        values = np.asarray(claims, dtype=float)
        turnover_rates = np.asarray(turnovers, dtype=float)
        shares = self.predict_shares(values)
        if turnover_rates.shape != values.shape:
            raise ValueError("Usage claims and turnover rates must have the same shape")
        demand = values / values.sum(axis=1, keepdims=True)
        entropy = -np.sum(shares * np.log(shares), axis=1) / np.log(5.0)
        midpoint = 0.5 * (demand + shares)
        js = 0.5 * np.sum(demand * np.log(demand / midpoint), axis=1) + 0.5 * np.sum(
            shares * np.log(shares / midpoint), axis=1
        )
        return pd.DataFrame(
            {
                "excess_usage_demand": np.maximum(values.sum(axis=1) - self.claim_budget, 0.0),
                "allocation_entropy": entropy,
                "role_reallocation_js": js,
                "allocation_weighted_turnover_burden": np.sum(shares * turnover_rates, axis=1),
            }
        )


def fit_usage_allocation_model(
    season: str,
    profiles: pd.DataFrame,
    *,
    curated_dir: Path | str,
) -> UsageAllocationModel:
    """Fit completed-season allocation from pre-season player usage profiles."""

    frame = usage_allocation_frame(season, profiles, curated_dir=curated_dir)
    claims = frame.loc[:, _claim_columns()].to_numpy(dtype=float)
    choice = frame["chosen_position"].to_numpy(dtype=int)
    weights = frame["event_weight"].to_numpy(dtype=float)
    scale = float(np.std(claims))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Usage claims have no variation for {season}")

    def objective(temperature: float) -> float:
        logits = temperature * claims / scale
        chosen = logits[np.arange(len(logits)), choice]
        return float(np.sum(weights * (logsumexp(logits, axis=1) - chosen)))

    selected = minimize_scalar(objective, bounds=(0.0, 10.0), method="bounded")
    if not selected.success:
        raise RuntimeError(f"Usage allocation fit failed for {season}: {selected.message}")
    return UsageAllocationModel(
        temperature=float(selected.x),
        claim_scale=scale,
        claim_budget=float(np.average(claims.sum(axis=1), weights=weights)),
        training_season=season,
        training_event_count=len(frame),
    )


def usage_allocation_frame(
    season: str,
    profiles: pd.DataFrame,
    *,
    curated_dir: Path | str,
) -> pd.DataFrame:
    """Return historical terminal actions with their pre-season five-man claims."""

    required = {"player_id", USAGE_COLUMN}
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Usage allocation profiles missing columns: {sorted(missing)}")
    root = Path(curated_dir)
    events = _read_partition_files(
        root / "events" / season / "regular",
        ["event_id", "event_type", "team_id", "player_id"],
    )
    events["event_type"] = events["event_type"].astype("string").str.lower()
    events = events.loc[events["event_type"].isin(USAGE_EVENT_WEIGHTS)].copy()
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
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    profiles_by_player = profiles.set_index("player_id")
    rows: list[dict[str, float | int]] = []
    for row in frame.itertuples(index=False):
        if pd.isna(row.team_id) or pd.isna(row.player_id):
            continue
        team_is_home = int(row.team_id) == int(row.catalog_home_team_id)
        if int(row.team_id) not in {int(row.catalog_home_team_id), int(row.catalog_away_team_id)}:
            continue
        lineup = row.home_player_ids_before if team_is_home else row.away_player_ids_before
        if not _valid_lineup(lineup):
            continue
        player_ids = [int(player_id) for player_id in lineup]
        actor = int(row.player_id)
        if actor not in player_ids or not set(player_ids).issubset(profiles_by_player.index):
            continue
        claims = profiles_by_player.loc[player_ids, USAGE_COLUMN].to_numpy(dtype=float)
        output: dict[str, float | int] = {
            "chosen_position": player_ids.index(actor),
            "event_weight": USAGE_EVENT_WEIGHTS[str(row.event_type)],
        }
        output.update({column: float(value) for column, value in zip(_claim_columns(), claims, strict=True)})
        rows.append(output)
    result = pd.DataFrame(rows)
    if len(result) < 1_000:
        raise ValueError(f"{season} has too few usable terminal actions")
    return result


def _claim_columns() -> list[str]:
    return [f"usage_claim_{index}" for index in range(5)]


def _read_partition_files(path: Path, columns: list[str]) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet partitions found at {path}")
    return pd.concat([pd.read_parquet(file, columns=columns) for file in files], ignore_index=True)


def _valid_lineup(value: object) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) == 5 and len(set(value)) == 5
