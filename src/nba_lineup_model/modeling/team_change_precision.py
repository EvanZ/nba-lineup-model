"""Forward-only player-prior precision candidates for clean offseason team moves."""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_CHANGE_VARIANCE_RATIO_GRID = (0.0, 0.05, 0.1, 0.2, 0.4)


def build_team_change_precision_candidates(
    *,
    season: str,
    panel: pd.DataFrame,
    player_ids: tuple[int, ...],
    player_priors: pd.DataFrame,
    exposure_history: list[pd.DataFrame],
) -> tuple[dict[str, dict[int, float]], dict[str, object]]:
    """Create clean-offseason-move precision vectors available before a season.

    A player is treated as a mover only when they were on exactly one team in
    each of two consecutive completed-season panel rows and those primary teams
    differ.  Players acquired or traded during a season are excluded.  The
    variance-ratio grid is selected jointly with player lambda on chronological
    folds of the already source-adjusted target.
    """

    if not player_ids:
        raise ValueError("Team-change precision requires a non-empty player vocabulary")
    if not TEAM_CHANGE_VARIANCE_RATIO_GRID or TEAM_CHANGE_VARIANCE_RATIO_GRID[0] != 0.0:
        raise RuntimeError("The team-change grid must include the uniform-precision baseline")
    player_set = {int(player_id) for player_id in player_ids}
    profile = _team_change_profile(
        season=season,
        panel=panel,
        player_ids=player_set,
        player_priors=player_priors,
        exposure_history=exposure_history,
    )
    candidates: dict[str, dict[int, float]] = {}
    for variance_ratio in TEAM_CHANGE_VARIANCE_RATIO_GRID:
        values = dict.fromkeys(player_ids, 1.0)
        movers = profile.loc[profile["clean_offseason_team_change"]]
        values.update(
            dict(
                zip(
                    movers["player_id"].astype(int),
                    1.0
                    / (
                        1.0
                        + variance_ratio * movers["prior_exposure_scale"].astype(float)
                    ),
                    strict=True,
                )
            )
        )
        candidates[_candidate_key(variance_ratio)] = values
    mover_scale = profile.loc[profile["clean_offseason_team_change"], "prior_exposure_scale"]
    return candidates, {
        "team_change_precision_enabled": True,
        "team_change_precision_candidate_grid": list(TEAM_CHANGE_VARIANCE_RATIO_GRID),
        "team_change_precision_candidate_keys": list(candidates),
        "team_change_precision_contract": (
            "Clean offseason movers receive relative prior precision "
            "1 / (1 + q * normalized_log_prior_possessions); non-movers remain 1. "
            "q is jointly selected with player lambda on chronological source-season folds."
        ),
        "team_change_precision_assignment": (
            "consecutive completed single-primary-team panel rows only; excludes "
            "multi-team seasons, rookies, gaps, and in-season moves"
        ),
        "team_change_precision_clean_mover_count": int(
            profile["clean_offseason_team_change"].sum()
        ),
        "team_change_precision_eligible_returner_count": int(
            profile["has_prior_exposure"].sum()
        ),
        "team_change_precision_mover_exposure_scale_median": (
            float(mover_scale.median()) if not mover_scale.empty else 0.0
        ),
    }


def _team_change_profile(
    *,
    season: str,
    panel: pd.DataFrame,
    player_ids: set[int],
    player_priors: pd.DataFrame,
    exposure_history: list[pd.DataFrame],
) -> pd.DataFrame:
    """Return the auditable clean-move classification for active players."""

    required_panel = {"season", "player_id", "primary_team_tricode", "team_count"}
    missing_panel = required_panel - set(panel)
    if missing_panel:
        raise ValueError(f"Player panel missing team-change columns: {sorted(missing_panel)}")
    required_prior = {"player_id", "lagged_rapm_prior"}
    missing_prior = required_prior - set(player_priors)
    if missing_prior:
        raise ValueError(f"Player priors missing columns: {sorted(missing_prior)}")
    current = panel.loc[
        panel["season"].eq(season),
        ["player_id", "primary_team_tricode", "team_count"],
    ].rename(
        columns={
            "primary_team_tricode": "current_primary_team",
            "team_count": "current_team_count",
        }
    )
    previous_season = _previous_season(season)
    previous = panel.loc[
        panel["season"].eq(previous_season),
        ["player_id", "primary_team_tricode", "team_count"],
    ].rename(
        columns={
            "primary_team_tricode": "previous_primary_team",
            "team_count": "previous_team_count",
        }
    )
    output = pd.DataFrame({"player_id": sorted(player_ids)}).merge(
        current, on="player_id", how="left", validate="one_to_one"
    ).merge(previous, on="player_id", how="left", validate="one_to_one")
    prior_ids = set(player_priors["player_id"].astype(int))
    exposure = exposure_history[-1] if exposure_history else pd.DataFrame()
    if exposure.empty:
        exposure_values = pd.DataFrame(
            columns=["player_id", "prior_on_court_possessions"]
        )
    else:
        if not {"player_id", "on_court_possessions"}.issubset(exposure):
            raise ValueError("Prior exposure history is missing on_court_possessions")
        exposure_values = exposure.loc[:, ["player_id", "on_court_possessions"]].rename(
            columns={"on_court_possessions": "prior_on_court_possessions"}
        )
    output = output.merge(
        exposure_values, on="player_id", how="left", validate="one_to_one"
    )
    output["prior_on_court_possessions"] = (
        pd.to_numeric(output["prior_on_court_possessions"], errors="coerce")
        .astype("float64")
        .fillna(0.0)
    )
    output["has_prior_exposure"] = (
        output["player_id"].isin(prior_ids)
        & output["prior_on_court_possessions"].gt(0)
    )
    output["clean_offseason_team_change"] = (
        output["has_prior_exposure"]
        & output["current_team_count"].eq(1)
        & output["previous_team_count"].eq(1)
        & output["current_primary_team"].notna()
        & output["previous_primary_team"].notna()
        & output["current_primary_team"].ne(output["previous_primary_team"])
    )
    logged_exposure = np.log1p(
        output.loc[output["has_prior_exposure"], "prior_on_court_possessions"]
    )
    scale_denominator = float(logged_exposure.median()) if not logged_exposure.empty else 1.0
    if not np.isfinite(scale_denominator) or scale_denominator <= 0:
        scale_denominator = 1.0
    output["prior_exposure_scale"] = np.log1p(
        output["prior_on_court_possessions"].astype(float)
    ) / scale_denominator
    return output


def _candidate_key(variance_ratio: float) -> str:
    return f"move_variance_ratio={variance_ratio:g}"


def _previous_season(season: str) -> str:
    year = int(season[:4]) - 1
    return f"{year}-{str(year + 1)[-2:]}"
