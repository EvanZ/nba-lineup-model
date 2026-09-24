"""Strictly forward, component-native offense/defense Four Factors state.

This runner deliberately does not use total NAIL, box-score-profile context,
or a completed target-season outcome to form any player prior.  Each of the
eight factor-side player states has its own recursive aging/cold-start branch.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.four_factor_offense_defense import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR as DEFAULT_OD_ARTIFACTS_DIR,
    DEFAULT_GAME_CATALOG_PATH,
    FOUR_FACTORS,
    OffenseDefenseFactorFit,
    fit_completed_offense_defense_factor_season,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import (
    _combine_priors,
)
from nba_lineup_model.modeling.gap_returner_prior import (
    build_centered_value_conditioned_aging_gap_returner_priors,
)
from nba_lineup_model.modeling.prior_rapm import (
    HISTORICAL_SEASONS,
    PRIOR_MEAN_COLUMN,
    ForwardLaggedRapmSeason,
)
from nba_lineup_model.modeling.replacement_level import player_exposure_shares
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.season.schema import validate_season


DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models/analysis/forward_four_factor_offense_defense")
DEFAULT_SIDE_REGULARIZATION_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)
RUN_PREFIX = "forward-four-factor-od"
CHECKPOINT_FILENAME = ".checkpoint-v2.joblib"


@dataclass(frozen=True)
class ForwardFourFactorODRun:
    """Persisted factor-side recursive state through one completed season."""

    run_dir: Path
    run_id: str
    through_season: str


def train_forward_four_factor_offense_defense(
    *,
    through_season: str = "2025-26",
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    checkpoint_path: Path | str | None = None,
    max_seasons: int | None = None,
    side_regularization_grid: tuple[float, ...] = DEFAULT_SIDE_REGULARIZATION_GRID,
) -> ForwardFourFactorODRun | None:
    """Train the eight strictly-forward Four Factors player states.

    A factor-side prior for season ``t`` can use only completed factor-side
    estimates through ``t-1``.  The completed seasonal fit is then retained as
    the state for a later season; it is never scored as a forecast of itself.
    """

    last_season = validate_season(through_season)
    if not side_regularization_grid or any(value < 0 for value in side_regularization_grid):
        raise ValueError("Factor-side regularization grid must be non-negative")
    panel = pd.read_parquet(player_season_panel_path)
    _validate_panel(panel, last_season)
    seasons = tuple(season for season in HISTORICAL_SEASONS if season <= last_season)
    if last_season not in seasons:
        seasons = (*seasons, last_season)
    checkpoint = Path(checkpoint_path) if checkpoint_path else (
        Path(artifacts_dir) / last_season / CHECKPOINT_FILENAME
    )
    state = _load_checkpoint(checkpoint, seasons)
    completed: dict[str, dict[str, list[ForwardLaggedRapmSeason]]] = state["completed"]
    exposure_history: dict[str, dict[str, list[pd.DataFrame]]] = state["exposure_history"]
    replacement_tokens: dict[str, dict[str, list[dict[str, object]]]] = state[
        "replacement_tokens"
    ]
    prior_records: list[pd.DataFrame] = state["prior_records"]
    effect_records: list[pd.DataFrame] = state["effect_records"]
    control_records: list[pd.DataFrame] = state["control_records"]
    cv_records: list[pd.DataFrame] = state["cv_records"]
    regularization_pairs = tuple(product(side_regularization_grid, repeat=2))
    # One combined effect frame is checkpointed per completed season.
    start_count = len(effect_records)

    for season in seasons[start_count:]:
        print(f"Fitting forward Four Factors O/D: {season}", flush=True)
        prior_by_factor, prior_frame = _build_prior_frame(
            season=season,
            panel=panel,
            completed=completed,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
        )
        fits, effects = fit_completed_offense_defense_factor_season(
            season,
            analytical_dir=analytical_dir,
            game_catalog_path=game_catalog_path,
            prior_by_factor=prior_by_factor,
            regularization_pairs=regularization_pairs,
        )
        fit_map = {fit.factor.name: fit for fit in fits}
        _append_completed_state(
            season=season,
            fits=fit_map,
            panel=panel,
            generic_exposure=player_exposure_shares(
                read_rapm_stints(season, analytical_dir=analytical_dir)
            ),
            completed=completed,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
        )
        prior_records.append(prior_frame)
        effect_records.append(effects)
        control_records.append(_controls_frame(season, fits))
        cv_records.append(
            pd.concat(
                [fit.cv_results.assign(season=season, factor=fit.factor.name) for fit in fits],
                ignore_index=True,
            )
        )
        _write_checkpoint(
            checkpoint,
            seasons=seasons,
            completed=completed,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
            prior_records=prior_records,
            effect_records=effect_records,
            control_records=control_records,
            cv_records=cv_records,
        )
        if max_seasons is not None and len(effect_records) >= start_count + max_seasons:
            return None

    run = _write_run(
        through_season=last_season,
        priors=pd.concat(prior_records, ignore_index=True),
        effects=pd.concat(effect_records, ignore_index=True),
        controls=pd.concat(control_records, ignore_index=True),
        cv_results=pd.concat(cv_records, ignore_index=True),
        artifacts_dir=Path(artifacts_dir),
        side_regularization_grid=side_regularization_grid,
    )
    checkpoint.unlink(missing_ok=True)
    return run


def _build_prior_frame(
    *,
    season: str,
    panel: pd.DataFrame,
    completed: dict[str, dict[str, list[ForwardLaggedRapmSeason]]],
    exposure_history: dict[str, dict[str, list[pd.DataFrame]]],
    replacement_tokens: dict[str, dict[str, list[dict[str, object]]]],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    by_factor: dict[str, pd.DataFrame] = {}
    records: list[pd.DataFrame] = []
    for factor in FOUR_FACTORS:
        side_priors: dict[str, pd.DataFrame] = {}
        for side in ("offense", "defense"):
            priors, _metadata = build_centered_value_conditioned_aging_gap_returner_priors(
                season=season,
                panel=panel,
                completed_results=completed[factor][side],
                exposure_history=exposure_history[factor][side],
                replacement_tokens=replacement_tokens[factor][side],
            )
            side_priors[side] = priors.rename(columns={PRIOR_MEAN_COLUMN: f"{side}_prior"})
            records.append(side_priors[side].assign(season=season, factor=factor, side=side))
        combined = side_priors["offense"].merge(
            side_priors["defense"], on="player_id", how="outer", validate="one_to_one"
        )
        combined[["offense_prior", "defense_prior"]] = combined[
            ["offense_prior", "defense_prior"]
        ].fillna(0.0)
        by_factor[factor] = combined
    record = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    return by_factor, record


def _append_completed_state(
    *,
    season: str,
    fits: dict[str, OffenseDefenseFactorFit],
    panel: pd.DataFrame,
    generic_exposure: pd.DataFrame,
    completed: dict[str, dict[str, list[ForwardLaggedRapmSeason]]],
    exposure_history: dict[str, dict[str, list[pd.DataFrame]]],
    replacement_tokens: dict[str, dict[str, list[dict[str, object]]]],
) -> None:
    for factor, fit in fits.items():
        for side, effect, prior, opportunity, selected_lambda in (
            (
                "offense",
                fit.offense_effects,
                fit.offense_priors,
                fit.offense_opportunities,
                fit.selected_offense_lambda,
            ),
            (
                "defense",
                fit.defense_effects,
                fit.defense_priors,
                fit.defense_opportunities,
                fit.selected_defense_lambda,
            ),
        ):
            estimates = pd.DataFrame(
                {
                    "season": season,
                    "player_id": fit.player_ids,
                    "rapm": effect,
                    "prior_rapm": prior,
                    "rapm_adjustment_from_prior": effect - prior,
                    "prior_available": True,
                    "selected_lambda": selected_lambda,
                }
            )
            exposure = _component_exposure_history(
                season=season,
                player_ids=fit.player_ids,
                component_opportunity=opportunity,
                panel=panel,
                generic_exposure=generic_exposure,
            )
            completed[factor][side].append(
                ForwardLaggedRapmSeason(
                    season=season,
                    selected_lambda=selected_lambda,
                    cv_results=fit.cv_results,
                    player_estimates=estimates,
                    player_priors=pd.DataFrame(),
                )
            )
            exposure_history[factor][side].append(exposure)
            replacement_tokens[factor][side].append(
                {
                    "season": season,
                    "replacement_token_rapm": _replacement_effect(estimates, exposure),
                }
            )


def _replacement_effect(estimates: pd.DataFrame, exposure: pd.DataFrame) -> float:
    """Component-specific low-exposure state used only for rookie gating."""

    frame = estimates.loc[:, ["player_id", "rapm"]].merge(
        exposure, on="player_id", validate="one_to_one"
    )
    cutoff = frame["on_court_possessions"].quantile(0.2)
    low = frame.loc[frame["on_court_possessions"].le(cutoff)]
    return float(np.average(low["rapm"], weights=low["on_court_possessions"]))


def _component_exposure_history(
    *,
    season: str,
    player_ids: tuple[int, ...],
    component_opportunity: np.ndarray,
    panel: pd.DataFrame,
    generic_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Attach component reliability and generic rookie-gate exposure inputs."""

    bio_columns = (
        "season",
        "season_start_year",
        "player_id",
        "player_name",
        "is_rookie",
        "draft_year",
        "draft_round",
        "draft_number",
        "is_undrafted",
        "age",
        "height_inches",
        "weight_pounds",
    )
    bios = panel.loc[panel["season"].eq(season), bio_columns]
    component = pd.DataFrame(
        {"player_id": player_ids, "on_court_possessions": component_opportunity}
    )
    generic = generic_exposure.loc[
        :, ["player_id", "team_opportunity_possessions", "exposure_share"]
    ]
    output = (
        component.merge(bios, on="player_id", how="left", validate="one_to_one")
        .merge(generic, on="player_id", how="left", validate="one_to_one")
    )
    if output[["exposure_share", "on_court_possessions"]].isna().any().any() or not output[
        "on_court_possessions"
    ].gt(0).all():
        raise ValueError("Factor-side exposure history could not be fully joined")
    # Some early lineup-only IDs do not have a player-panel biography. They
    # retain their completed factor state but cannot enter a rookie/aging row.
    output["is_rookie"] = output["is_rookie"].astype("boolean").fillna(False).astype(bool)
    return output


def _controls_frame(season: str, fits: list[OffenseDefenseFactorFit]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "factor": fit.factor.name,
                "selected_offense_lambda": fit.selected_offense_lambda,
                "selected_defense_lambda": fit.selected_defense_lambda,
                **{f"{key}_percentage_points": 100.0 * value for key, value in fit.controls.items()},
            }
            for fit in fits
        ]
    )


def _initial_state() -> dict[str, object]:
    return {
        "completed": {factor: {side: [] for side in ("offense", "defense")} for factor in FOUR_FACTORS},
        "exposure_history": {factor: {side: [] for side in ("offense", "defense")} for factor in FOUR_FACTORS},
        "replacement_tokens": {factor: {side: [] for side in ("offense", "defense")} for factor in FOUR_FACTORS},
        "prior_records": [],
        "effect_records": [],
        "control_records": [],
        "cv_records": [],
    }


def _load_checkpoint(checkpoint: Path, seasons: tuple[str, ...]) -> dict[str, object]:
    if not checkpoint.exists():
        return _initial_state()
    state = joblib.load(checkpoint)
    if not isinstance(state, dict) or tuple(state.get("seasons", ())) != seasons:
        raise ValueError("Forward Four Factors O/D checkpoint is incompatible")
    expected = set(_initial_state())
    if expected - set(state):
        raise ValueError("Forward Four Factors O/D checkpoint is incomplete")
    return state


def _write_checkpoint(
    checkpoint: Path,
    *,
    seasons: tuple[str, ...],
    **state: object,
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    joblib.dump({"seasons": seasons, **state}, temporary)
    temporary.replace(checkpoint)


def _write_run(
    *,
    through_season: str,
    priors: pd.DataFrame,
    effects: pd.DataFrame,
    controls: pd.DataFrame,
    cv_results: pd.DataFrame,
    artifacts_dir: Path,
    side_regularization_grid: tuple[float, ...],
) -> ForwardFourFactorODRun:
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}-{through_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / through_season
    target = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        priors.to_parquet(temporary / "season_factor_side_priors.parquet", index=False)
        effects.to_parquet(temporary / "season_factor_side_effects.parquet", index=False)
        controls.to_parquet(temporary / "season_factor_side_controls.parquet", index=False)
        cv_results.to_parquet(temporary / "season_factor_side_lambda_cv.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "through_season": through_season,
            "created_at": now.isoformat(),
            "model": "strictly_forward_component_native_four_factors_od",
            "player_prior_contract": (
                "factor-side recursive value-conditioned aging, exposure-gated cold starts, "
                "and last-observed annual aging gap bridge"
            ),
            "state_dimensions": [f"{factor}_{side}" for factor in FOUR_FACTORS for side in ("offense", "defense")],
            "regularization_selection": "within-season chronological CV over offense/defense penalty pairs",
            "side_regularization_grid": list(side_regularization_grid),
            "excludes": ["total_nail", "nail_additive_profiles", "nail_nonadditive_context"],
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return ForwardFourFactorODRun(target, run_id, through_season)


def _validate_panel(panel: pd.DataFrame, through_season: str) -> None:
    required = {
        "season",
        "player_id",
        "player_name",
        "age",
        "nba_experience_years",
        "is_rookie",
        "draft_year",
        "draft_number",
        "height_inches",
        "weight_pounds",
        "is_undrafted",
        "rapm_seconds",
        "rapm_exposure_eligible",
    }
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Four Factors O/D player panel lacks columns: {sorted(missing)}")
    if not panel["season"].eq(through_season).any():
        raise ValueError(f"Four Factors O/D panel has no {through_season} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forward component-native Four Factors O/D")
    parser.add_argument("--through-season", default="2025-26")
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--game-catalog-path", default=str(DEFAULT_GAME_CATALOG_PATH))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--max-seasons", type=int)
    args = parser.parse_args()
    run = train_forward_four_factor_offense_defense(
        through_season=args.through_season,
        player_season_panel_path=args.player_season_panel_path,
        analytical_dir=args.analytical_dir,
        game_catalog_path=args.game_catalog_path,
        artifacts_dir=args.artifacts_dir,
        checkpoint_path=args.checkpoint_path,
        max_seasons=args.max_seasons,
    )
    print("Forward Four Factors O/D checkpoint saved" if run is None else run.run_dir)


if __name__ == "__main__":
    main()
