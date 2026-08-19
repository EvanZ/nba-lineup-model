"""Materialize observed-five lineup rankings for the NBA GESTALT web API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from nba_lineup_model.modeling.contextual_profiles import (
    ProfilePaddingContract,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.matchup_contextual import MatchupContextualModel
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.web_api.inference import (
    DEFAULT_ARTIFACTS_DIR,
    DISPLAY_SEASON,
    MODEL_ARTIFACT,
    LineupEvaluator,
    _compiled_linear_x3_coefficients,
    _is_compiled_linear_x3,
    _player_catalog,
    _player_rating_center,
    build_observed_lineup_rankings,
    build_published_player_ratings,
    lineup_rankings_path,
    published_player_ratings_path,
)

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")


def main() -> None:
    """Build one immutable observed-lineup table for the published model run."""

    parser = argparse.ArgumentParser(description="Build NBA GESTALT observed lineup rankings")
    parser.add_argument("--season", default=DISPLAY_SEASON)
    parser.add_argument("--all-seasons", action="store_true")
    args = parser.parse_args()
    if args.all_seasons:
        _build_all_seasons()
    else:
        _build_one_season(args.season)


def _build_one_season(season: str) -> None:
    """Materialize one selected completed-fit season."""

    evaluator = LineupEvaluator.from_latest_artifact(season=DISPLAY_SEASON)
    state = _artifact_state(
        evaluator.run_id,
        padding_contract=evaluator.profile_padding_contract,
    )
    _write_published_player_ratings(run_id=evaluator.run_id, **state)
    _write_season(season, run_id=evaluator.run_id, **state)


def _build_all_seasons() -> None:
    """Materialize every season with a completed contextual model."""

    evaluator = LineupEvaluator.from_latest_artifact(season=DISPLAY_SEASON)
    state = _artifact_state(
        evaluator.run_id,
        padding_contract=evaluator.profile_padding_contract,
    )
    _write_published_player_ratings(run_id=evaluator.run_id, **state)
    seasons = sorted(state["models"])
    for index, season in enumerate(seasons, start=1):
        print(
            f"Building observed lineup rankings for {season} ({index}/{len(seasons)})",
            flush=True,
        )
        _write_season(season, run_id=evaluator.run_id, **state)


def _artifact_state(
    run_id: str,
    *,
    padding_contract: ProfilePaddingContract,
) -> dict[str, object]:
    """Load the completed artifact state shared by every historical season."""

    root = DEFAULT_ARTIFACTS_DIR / MODEL_ARTIFACT / DISPLAY_SEASON / run_id
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("model") != MODEL_ARTIFACT:
        raise ValueError("The published NAIL-RAPM artifact has an unexpected model identity")
    return {
        "panel": pd.read_parquet(DEFAULT_PANEL_PATH),
        "coefficients": pd.read_parquet(root / "historical_player_coefficients.parquet"),
        "ratings": pd.read_parquet(root / "player_season_ratings.parquet"),
        "models": joblib.load(root / "season_context_models.joblib"),
        "padding_contract": padding_contract,
    }


def _write_published_player_ratings(
    *,
    run_id: str,
    panel: pd.DataFrame,
    ratings: pd.DataFrame,
    models: dict[str, MatchupContextualModel],
    padding_contract: ProfilePaddingContract,
    **_: object,
) -> None:
    """Materialize compiled player ratings used by ranking and biography views."""

    output = published_player_ratings_path(MODEL_ARTIFACT, run_id)
    if output.is_file():
        print(f"Using existing published player ratings: {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    build_published_player_ratings(
        ratings,
        panel=panel,
        models=models,
        padding_contract=padding_contract,
    ).to_parquet(output, index=False)
    print(f"Materialized published player ratings: {output}", flush=True)


def _write_season(
    season: str,
    *,
    run_id: str,
    panel: pd.DataFrame,
    coefficients: pd.DataFrame,
    models: dict[str, MatchupContextualModel],
    padding_contract: ProfilePaddingContract,
    **_: object,
) -> None:
    """Build one completed-fit observed-lineup table from shared artifact state."""

    output = lineup_rankings_path(MODEL_ARTIFACT, run_id, season)
    if output.is_file():
        print(f"Using existing {season}: {output}", flush=True)
        return
    context_model = models.get(season)
    if context_model is None:
        raise ValueError(f"No completed context model is available for {season}")
    stints = read_rapm_stints(season)
    player_ids = set().union(*stints["home_player_ids"], *stints["away_player_ids"])
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(DISPLAY_SEASON)],
        through_season=DISPLAY_SEASON,
    )
    profiles = build_contextual_player_profiles(
        panel,
        target_season=season,
        target_player_ids=player_ids,
        exposure_cohort=exposure_cohort,
        padding_contract=padding_contract,
    )
    season_coefficients = coefficients.loc[
        coefficients["season"].astype(str).eq(season), ["player_id", "rapm"]
    ].copy()
    if season_coefficients.empty:
        raise ValueError(f"No completed player coefficients are available for {season}")
    if _is_compiled_linear_x3(context_model):
        uncentered_coefficients = _compiled_linear_x3_coefficients(
            season_coefficients,
            profiles,
            context_model,
        )
        uncentered_players = _player_catalog(
            uncentered_coefficients,
            profiles,
            panel_path=DEFAULT_PANEL_PATH,
            season=season,
        )
        season_coefficients = _compiled_linear_x3_coefficients(
            season_coefficients,
            profiles,
            context_model,
            center=_player_rating_center(uncentered_coefficients, uncentered_players),
        )
    players = _player_catalog(
        season_coefficients,
        profiles,
        panel_path=DEFAULT_PANEL_PATH,
        season=season,
    )
    rankings = build_observed_lineup_rankings(
        season=season,
        profiles=profiles,
        players=players,
        coefficients=season_coefficients,
        context_model=context_model,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_parquet(output, index=False)
    print(f"Materialized {season}: {output} ({len(rankings)} units)", flush=True)


if __name__ == "__main__":
    main()
