"""Exact preseason-NAIL extension of L20 preseason minute-share allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_profiles import build_contextual_player_profiles
from nba_lineup_model.rotation.l0_opening_minute_share_persistence import (
    read_preseason_roster_candidates,
)
from nba_lineup_model.rotation.l1_minute_share_persistence import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    read_regular_game_minutes,
)
from nba_lineup_model.rotation.l20_nail_minute_share import (
    DEFAULT_NAIL_WEIGHT_GRID,
    DEFAULT_SEASONS,
    L20NailMinuteShareRun,
    RatingAwareSeasonInputs,
    run_l20_nail_minute_share,
)
from nba_lineup_model.rotation.l20_preseason_minute_share_persistence import (
    DEFAULT_EPSILON_GRID,
    DEFAULT_TEAM_GAMES,
    previous_season,
)
from nba_lineup_model.web_api.inference import (
    MODEL_ARTIFACT,
    _compiled_linear_x3_coefficients,
    _player_rating_center,
    _published_profile_padding_contract,
    exposure_cohort_path,
)

MODEL_NAME = "l20_preseason_nail_forecast_minute_share"
MODEL_VERSION = "v0.2"
DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_MODEL_RUN_DIR = Path(
    "artifacts/models/forward_nail_rapm_v1212_residualized_lambda/2025-26/"
    "forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25"
)


def build_preseason_nail_forecast_ratings(
    season: str,
    opening_candidates: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
) -> pd.Series:
    """Reconstruct the web app's preseason NAIL forecast for an opening roster.

    The target's forward player prior was materialized before the target season.
    The frozen previous-season additive profile is then applied using only
    prior-season player statistics. No target-season minutes, box score, or
    RAPM update enters this calculation.
    """

    run_path = Path(run_dir)
    metadata = json.loads((run_path / "metadata.json").read_text())
    candidate_ids = opening_candidates["player_id"].astype(int).drop_duplicates().tolist()
    prior_frame = pd.read_parquet(run_path / "season_player_priors.parquet")
    base = prior_frame.loc[
        prior_frame["season"].astype(str).eq(season), ["player_id", "prior_rapm"]
    ].copy()
    if base.empty:
        raise ValueError(f"NAIL run has no frozen prior state for {season}")
    base["player_id"] = base["player_id"].astype(int)
    base = pd.DataFrame({"player_id": candidate_ids}).merge(
        base, on="player_id", how="left", validate="one_to_one"
    )
    base["prior_rapm"] = base["prior_rapm"].fillna(_replacement_rating(run_path, season))
    base = base.rename(columns={"prior_rapm": "rapm"})

    padding_contract = _published_profile_padding_contract(metadata)
    use_last_observed_profile = (
        metadata.get("profile_padding_contract", {}).get("gap_returner_profile_method")
        == "last_observed_padded_profile"
    )
    exposure_cohort = pd.read_parquet(exposure_cohort_path(MODEL_ARTIFACT, run_path.name))
    profiles = build_contextual_player_profiles(
        panel,
        target_season=season,
        target_player_ids=candidate_ids,
        exposure_cohort=exposure_cohort,
        padding_contract=padding_contract,
        use_last_observed_profile=use_last_observed_profile,
    )
    models = joblib.load(run_path / "season_context_models.joblib")
    source_season = previous_season(season)
    context_model = models.get(source_season)
    if context_model is None:
        raise ValueError(f"NAIL run has no frozen context state for {source_season}")
    uncentered = _compiled_linear_x3_coefficients(base, profiles, context_model)
    source_exposure = exposure_cohort.loc[
        exposure_cohort["season"].astype(str).eq(source_season),
        ["player_id", "on_court_possessions"],
    ].rename(columns={"on_court_possessions": "possessions"})
    center = _player_rating_center(uncentered, source_exposure)
    forecast = _compiled_linear_x3_coefficients(base, profiles, context_model, center=center)
    if forecast["player_id"].duplicated().any() or len(forecast) != len(candidate_ids):
        raise ValueError(f"Invalid preseason NAIL forecast coverage for {season}")
    return forecast.set_index("player_id")["rapm"].astype(float)


def build_l20_preseason_nail_forecast_inputs(
    season: str,
    *,
    panel: pd.DataFrame,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
) -> RatingAwareSeasonInputs:
    """Load all opening inputs plus the exact preseason NAIL rating forecast."""

    candidates = read_preseason_roster_candidates(season, curated_dir=curated_dir)
    return RatingAwareSeasonInputs(
        season=season,
        prior_game_minutes=read_regular_game_minutes(
            previous_season(season), curated_dir=curated_dir
        ),
        target_game_minutes=read_regular_game_minutes(season, curated_dir=curated_dir),
        opening_candidates=candidates,
        completed_nail_ratings=build_preseason_nail_forecast_ratings(
            season, candidates, panel=panel, run_dir=run_dir
        ),
    )


def run_l20_preseason_nail_forecast_minute_share(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
    epsilon_grid: tuple[float, ...] = DEFAULT_EPSILON_GRID,
    nail_weight_grid: tuple[float, ...] = DEFAULT_NAIL_WEIGHT_GRID,
    team_games: int = DEFAULT_TEAM_GAMES,
) -> L20NailMinuteShareRun:
    """Run expanding frozen tests using exact target-season preseason NAIL."""

    panel = pd.read_parquet(panel_path)
    return run_l20_nail_minute_share(
        seasons=seasons,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
        epsilon_grid=epsilon_grid,
        nail_weight_grid=nail_weight_grid,
        team_games=team_games,
        input_loader=lambda season: build_l20_preseason_nail_forecast_inputs(
            season, panel=panel, curated_dir=curated_dir, run_dir=run_dir
        ),
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        contract=(
            "For target season t, use NAIL's exact t preseason forecast: a forward "
            "prior plus the t-1 frozen additive profile. Parameters for each holdout "
            "season use only earlier completed target seasons."
        ),
    )


def _replacement_rating(run_dir: Path, season: str) -> float:
    """Return the target season's stored no-information cold-start prior."""

    metadata = pd.read_parquet(run_dir / "season_model_metadata.parquet")
    row = metadata.loc[metadata["season"].astype(str).eq(season)]
    if row.empty:
        raise ValueError(f"NAIL run has no model metadata for {season}")
    cold_start = json.loads(str(row.iloc[0]["cold_start"]))
    replacement = cold_start.get("replacement_rapm")
    if replacement is None or not np.isfinite(float(replacement)):
        raise ValueError(f"NAIL run has no replacement rating for {season}")
    return float(replacement)


def main() -> None:
    """Run the exact-preseason-NAIL L20 minute-share replay."""

    parser = argparse.ArgumentParser(description="Evaluate exact preseason NAIL minutes")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--team-games", type=int, default=DEFAULT_TEAM_GAMES)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_MODEL_RUN_DIR)
    args = parser.parse_args()
    run = run_l20_preseason_nail_forecast_minute_share(
        seasons=tuple(args.seasons),
        panel_path=args.panel_path,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        run_dir=args.run_dir,
        team_games=args.team_games,
    )
    print(run.run_dir)
