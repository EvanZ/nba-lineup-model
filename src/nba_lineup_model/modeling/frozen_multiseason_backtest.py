"""Replay completed recursive states as strict multi-season frozen forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from nba_lineup_model.evaluation.metrics import mean_absolute_error, rmse
from nba_lineup_model.modeling.contextual_features import lineup_side_context_features
from nba_lineup_model.modeling.contextual_prior import (
    _contextual_stint_predictions,
    _score_possessions,
)
from nba_lineup_model.modeling.contextual_profiles import build_contextual_player_profiles
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    _previous_season,
)
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    _load_recursive_model_mapping,
)
from nba_lineup_model.modeling.frozen_game_outcomes import score_full_game_outcomes
from nba_lineup_model.modeling.frozen_prior_evaluation import (
    PythagoreanWinModel,
    _game_prediction_frame,
    _historical_team_seasons,
    _read_playoff_possessions,
    _read_regular_possessions,
    _recover_home_intercept,
    _team_net_rating_metrics,
    _team_win_evaluation,
    fit_pythagorean_win_model,
    score_possession_cohort,
)
from nba_lineup_model.modeling.matchup_contextual import MatchupContextualModel
from nba_lineup_model.modeling.progress import format_progress_bar
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.schedule_controls import (
    BackToBackScheduleModel,
    build_back_to_back_game_features,
)
from nba_lineup_model.modeling.shot_portfolio import add_shot_portfolio_profiles
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.season.schema import validate_season

DEFAULT_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_DOCS_PATH: Path | None = None
REPORT_NAME = "frozen_multiseason_backtest"


@dataclass(frozen=True)
class BacktestModel:
    """One recursive artifact family with a public-facing comparison label."""

    model: str
    label: str
    profile_transformer: Callable[[pd.DataFrame, str], pd.DataFrame] | None = None
    profile_builder: Callable[..., pd.DataFrame] | None = None
    uses_context: bool = True
    uses_schedule_control: bool = False
    run_target_season: str | None = None


BACKTEST_MODELS = (
    BacktestModel(
        "forward_centered_value_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm",
        "Value-Conditioned Aging HPM",
    ),
    BacktestModel("forward_hpm_v2_depth_aware_shooting", "HPM v2 shooting composition"),
    BacktestModel(
        "forward_hpm_v21_empirical_rebound_capacity",
        "HPM v2.1 empirical rebound capacity",
    ),
    BacktestModel(
        "forward_hpm_v22_usage_allocation",
        "HPM v2.2 usage allocation",
    ),
    BacktestModel(
        "forward_hpm_v23_shot_portfolio",
        "HPM v2.3 shot portfolio",
        profile_transformer=lambda profiles, target: add_shot_portfolio_profiles(
            profiles,
            target_season=target,
        ),
    ),
    BacktestModel("forward_hpm_x1_orb_claim_total", "HPM x1 ORB claim context"),
    BacktestModel("forward_hpm_x2_orb_per_100_total", "HPM x2 raw OREB/100 context"),
    BacktestModel(
        "forward_hpm_x3_v1_orb_claim_replacement",
        "HPM x3 ORB claim rebound replacement",
    ),
    BacktestModel(
        "forward_hpm_x3_linear_ridge_context",
        "Compiled-additive linear HPM x3",
    ),
    BacktestModel(
        "forward_hpm_x3_linear_ridge_without_uncertainty",
        "Compiled-linear HPM x3 without uncertainty",
    ),
    BacktestModel(
        "forward_hpm_x3_linear_quadratic_side_context",
        "Compiled-additive HPM x3 plus quadratic side context",
    ),
    BacktestModel(
        "forward_additive_profile_linear_shape_context_rapm",
        "Additive prior plus linear shape context",
    ),
    BacktestModel("forward_hpm_x4_orb_claim_blocks_only", "HPM x4 ORB claims plus blocks"),
)


@dataclass(frozen=True)
class FrozenMultiseasonBacktestRun:
    run_dir: Path
    run_id: str


def run_frozen_multiseason_backtest(
    *,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    models: Sequence[BacktestModel] = BACKTEST_MODELS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    game_catalog_path: Path | str = Path("data/catalog/games.parquet"),
    docs_path: Path | str | None = DEFAULT_DOCS_PATH,
    output_artifacts_dir: Path | str | None = None,
    score_possessions: bool = True,
) -> FrozenMultiseasonBacktestRun:
    """Score multiple completed seasons without refitting player or context state.

    ``output_artifacts_dir`` permits a focused candidate replay to retain its
    own immutable report without changing the shared all-model leaderboard
    pointer under ``artifacts_dir``.
    """

    target_seasons = _validated_seasons(seasons)
    root = Path(artifacts_dir)
    output_root = Path(output_artifacts_dir) if output_artifacts_dir is not None else root
    panel = pd.read_parquet(player_season_panel_path)
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(target_seasons[-1])],
        through_season=target_seasons[-1],
        analytical_dir=analytical_dir,
    )
    source_rows: list[dict[str, object]] = []
    evaluations: list[dict[str, object]] = []
    states: list[_RecursiveState] = []
    schedule_features = (
        build_back_to_back_game_features(pd.read_parquet(game_catalog_path))
        if any(candidate.uses_schedule_control for candidate in models)
        else None
    )
    for candidate in models:
        run_target = candidate.run_target_season or target_seasons[-1]
        run_dir = _latest_recursive_run(root / candidate.model / run_target)
        state = _load_state(run_dir, candidate=candidate, target_seasons=target_seasons)
        states.append(state)
        source_rows.append(
            {
                "model": candidate.model,
                "label": candidate.label,
                "source_run_id": state.run_id,
                "source_run_dir": str(run_dir),
                "source_manifest_sha256": _sha256_file(run_dir / "manifest.json"),
            }
        )

    analytical_root = Path(analytical_dir)
    curated_root = Path(curated_dir)
    for target_index, target in enumerate(target_seasons, start=1):
        print(f"Starting frozen replay for {target}", flush=True)
        source = _previous_season(target)
        target_stints = read_rapm_stints(target, analytical_dir=analytical_root)
        source_stints = read_rapm_stints(source, analytical_dir=analytical_root)
        source_possessions, _ = _read_regular_possessions(
            source,
            analytical_dir=analytical_root,
            curated_dir=curated_root,
        )
        pythagorean = fit_pythagorean_win_model(
            _historical_team_seasons(
                analytical_dir=analytical_root,
                through_season=source,
            )
        )
        profile_cache: dict[tuple[int, int, bool], pd.DataFrame] = {}
        context_feature_cache: dict[tuple[object, ...], tuple[pd.DataFrame, pd.DataFrame]] = {}
        for state in states:
            candidate = state.candidate
            profile_key = (
                id(candidate.profile_builder),
                id(candidate.profile_transformer),
                candidate.uses_context,
            )
            if profile_key not in profile_cache:
                profile_cache[profile_key] = (
                    _transformed_target_profiles(
                        target,
                        candidate=candidate,
                        panel=panel,
                        exposure_cohort=exposure_cohort,
                        analytical_dir=analytical_root,
                        stints=target_stints,
                    )
                    if candidate.uses_context
                    else pd.DataFrame()
                )
            context_correction = None
            context_model = state.context_models.get(source)
            if context_model is not None:
                context_feature_key = (
                    *profile_key,
                    context_model.feature_set,
                    id(getattr(context_model, "rebound_model", None)),
                    id(getattr(context_model, "usage_model", None)),
                )
                if context_feature_key not in context_feature_cache:
                    home_features = lineup_side_context_features(
                        target_stints["home_player_ids"].tolist(),
                        profile_cache[profile_key],
                        feature_set=context_model.feature_set,
                        rebound_model=getattr(context_model, "rebound_model", None),
                        usage_model=getattr(context_model, "usage_model", None),
                    )
                    away_features = lineup_side_context_features(
                        target_stints["away_player_ids"].tolist(),
                        profile_cache[profile_key],
                        feature_set=context_model.feature_set,
                        rebound_model=getattr(context_model, "rebound_model", None),
                        usage_model=getattr(context_model, "usage_model", None),
                    )
                    context_feature_cache[context_feature_key] = (
                        home_features,
                        away_features,
                    )
                home_features, away_features = context_feature_cache[context_feature_key]
                context_correction = context_model.predict_side_pairs(
                    home_features,
                    away_features,
                )
            schedule_model = state.schedule_models.get(source)
            if candidate.uses_schedule_control and schedule_model is None:
                raise ValueError(f"{candidate.label} lacks a schedule state for {source}")
            evaluation = _replay_regular_target_season(
                target,
                state=state,
                panel=panel,
                exposure_cohort=exposure_cohort,
                analytical_dir=analytical_root,
                curated_dir=curated_root,
                profiles=profile_cache[profile_key],
                score_possessions=score_possessions,
                stints=target_stints,
                source_stints=source_stints,
                source_possessions=source_possessions,
                pythagorean=pythagorean,
                context_correction=context_correction,
                schedule_model=schedule_model,
                schedule_features=schedule_features,
            )
            evaluations.append({"candidate": candidate, "target": target, **evaluation})
        print(
            format_progress_bar(
                target_index,
                len(target_seasons),
                label=f"Completed frozen replay for {target}",
            ),
            flush=True,
        )
    outputs = _collect_outputs(evaluations)
    run = _write_run(
        target_seasons=target_seasons,
        outputs=outputs,
        sources=pd.DataFrame(source_rows),
        artifacts_dir=output_root,
    )
    if docs_path is not None:
        _update_docs(Path(docs_path), target_seasons=target_seasons, outputs=outputs, run=run)
    return run


def _transformed_target_profiles(
    target: str,
    *,
    candidate: BacktestModel,
    panel: pd.DataFrame,
    exposure_cohort: pd.DataFrame,
    analytical_dir: Path,
    stints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    profiles = _target_contextual_profiles(
        target,
        panel=panel,
        exposure_cohort=exposure_cohort,
        analytical_dir=analytical_dir,
        stints=stints,
        profile_builder=candidate.profile_builder,
    )
    if candidate.profile_transformer is not None:
        profiles = candidate.profile_transformer(profiles, target)
    return profiles


@dataclass(frozen=True)
class _RecursiveState:
    candidate: BacktestModel
    run_dir: Path
    run_id: str
    priors: pd.DataFrame
    coefficients: pd.DataFrame
    context_models: dict[str, MatchupContextualModel]
    schedule_models: dict[str, BackToBackScheduleModel] = field(default_factory=dict)


def _load_state(
    run_dir: Path,
    *,
    candidate: BacktestModel,
    target_seasons: tuple[str, ...],
) -> _RecursiveState:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata.get("model") != candidate.model:
        raise ValueError(f"Unexpected model in {run_dir}")
    if str(metadata.get("target_season")) < target_seasons[-1]:
        raise ValueError(f"{candidate.label} artifact does not reach {target_seasons[-1]}")
    priors = pd.read_parquet(run_dir / "season_player_priors.parquet")
    coefficients = pd.read_parquet(run_dir / "historical_player_coefficients.parquet")
    models = _load_recursive_model_mapping(run_dir, "season_context_models.joblib")
    schedule_models = _load_recursive_model_mapping(
        run_dir,
        "season_schedule_models.joblib",
    )
    required_priors = {"season", "player_id", "prior_rapm"}
    required_coefficients = {"season", "player_id", "rapm"}
    if required_priors - set(priors) or required_coefficients - set(coefficients):
        raise ValueError(f"{candidate.label} lacks replayable player state")
    if priors.duplicated(["season", "player_id"]).any():
        raise ValueError(f"{candidate.label} priors are not unique by player-season")
    for target in target_seasons:
        source = _previous_season(target)
        if priors.loc[priors["season"].eq(target)].empty:
            raise ValueError(f"{candidate.label} lacks prior vector for {target}")
        if coefficients.loc[coefficients["season"].eq(source)].empty:
            raise ValueError(f"{candidate.label} lacks player coefficients for {source}")
        if candidate.uses_context and source not in models:
            raise ValueError(f"{candidate.label} lacks context state for {source}")
        if candidate.uses_schedule_control and source not in schedule_models:
            raise ValueError(f"{candidate.label} lacks schedule state for {source}")
    return _RecursiveState(
        candidate=candidate,
        run_dir=run_dir,
        run_id=str(metadata["run_id"]),
        priors=priors,
        coefficients=coefficients,
        context_models=models,
        schedule_models=schedule_models,
    )


def _replay_regular_target_season(
    target: str,
    *,
    state: _RecursiveState,
    panel: pd.DataFrame,
    exposure_cohort: pd.DataFrame,
    analytical_dir: Path,
    curated_dir: Path,
    profiles: pd.DataFrame | None = None,
    score_possessions: bool = True,
    stints: pd.DataFrame | None = None,
    source_stints: pd.DataFrame | None = None,
    source_possessions: pd.DataFrame | None = None,
    pythagorean: PythagoreanWinModel | None = None,
    context_correction: np.ndarray | None = None,
    schedule_model: BackToBackScheduleModel | None = None,
    schedule_features: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    """Replay regular season and, when available, playoffs from one frozen state."""

    source = _previous_season(target)
    target_stints = (
        stints
        if stints is not None
        else read_rapm_stints(target, analytical_dir=analytical_dir)
    )
    if profiles is None and state.candidate.uses_context:
        profiles = _target_contextual_profiles(
            target,
            panel=panel,
            exposure_cohort=exposure_cohort,
            analytical_dir=analytical_dir,
            stints=target_stints,
        )
    model = state.context_models.get(source)
    context_predictor = model.predict_lineups if model is not None else _zero_context_predictor
    schedule_predictor = (
        _schedule_predictor(schedule_model, schedule_features)
        if schedule_model is not None and schedule_features is not None
        else None
    )
    stint_context_predictor = (
        _fixed_context_predictor(context_correction)
        if context_correction is not None
        else context_predictor
    )
    prior_frame = state.priors.loc[
        state.priors["season"].eq(target), ["player_id", "prior_rapm"]
    ].rename(columns={"prior_rapm": "prior_rapm_mean"})
    source_coefficients = state.coefficients.loc[
        state.coefficients["season"].eq(source), ["player_id", "rapm"]
    ]
    source_stint_frame = (
        source_stints
        if source_stints is not None
        else read_rapm_stints(source, analytical_dir=analytical_dir)
    )
    source_home_intercept = _recover_home_intercept(
        source_stint_frame,
        source_coefficients,
    )
    source_possession_frame = source_possessions
    if source_possession_frame is None:
        source_possession_frame, _ = _read_regular_possessions(
            source,
            analytical_dir=analytical_dir,
            curated_dir=curated_dir,
        )
    source_mean = float(source_possession_frame["target_offense_margin"].mean())
    predictions: list[pd.DataFrame] = []
    cohort_metrics: list[pd.DataFrame] = []
    playoff_available = score_possessions and _playoff_partition_exists(target, curated_dir)
    if score_possessions:
        regular_possessions, _ = _read_regular_possessions(
            target,
            analytical_dir=analytical_dir,
            curated_dir=curated_dir,
        )
        regular_predictions = _score_possessions(
            regular_possessions,
            cohort="regular_season",
            profiles=profiles,
            context_predictor=context_predictor,
            schedule_predictor=schedule_predictor,
            priors=prior_frame,
            source_mean=source_mean,
            source_home_intercept=source_home_intercept,
        )
        predictions.append(regular_predictions)
        cohort_metrics.append(
            score_possession_cohort(
                regular_predictions,
                source_mean=source_mean,
                model=state.candidate.model,
            )
        )
        if playoff_available:
            playoff_possessions, _ = _read_playoff_possessions(target, curated_dir)
            playoff_predictions = _score_possessions(
                playoff_possessions,
                cohort="playoffs",
                profiles=profiles,
                context_predictor=context_predictor,
                schedule_predictor=schedule_predictor,
                priors=prior_frame,
                source_mean=source_mean,
                source_home_intercept=source_home_intercept,
            )
            predictions.append(playoff_predictions)
            cohort_metrics.append(
                score_possession_cohort(
                    playoff_predictions,
                    source_mean=source_mean,
                    model=state.candidate.model,
                )
            )
    regular_games, team_net_ratings = _contextual_stint_predictions(
        target_stints,
        profiles=profiles,
        context_predictor=stint_context_predictor,
        priors=prior_frame,
        source_home_intercept=source_home_intercept,
        schedule_predictor=schedule_predictor,
    )
    win_model = pythagorean
    if win_model is None:
        win_model = fit_pythagorean_win_model(
            _historical_team_seasons(analytical_dir=analytical_dir, through_season=source)
        )
    team_wins, team_win_metrics = _team_win_evaluation(
        regular_games,
        team_net_ratings,
        win_model,
        model=state.candidate.model,
    )
    return {
        "source_state": {
            "target_season": target,
            "source_season": source,
            "source_offense_margin_mean": source_mean,
            "source_home_intercept_net_rating": source_home_intercept,
            "target_season_refit": False,
            "target_regular_outcomes_used_for_fit": False,
            "target_playoff_outcomes_used_for_fit": False,
            "oracle_information": "realized target-season regular-season lineups and exposure only",
            "playoffs_evaluated": playoff_available,
            "playoffs_exclusion_reason": (
                None
                if playoff_available
                else "possession scoring disabled"
                if not score_possessions
                else "historical playoff possession partition unavailable"
            ),
            "source_run_id": state.run_id,
            "replay_mode": "persisted_recursive_state_no_refit",
            "replay_context_model_season": source if model is not None else None,
            "replay_schedule_model_season": source if schedule_model is not None else None,
            "replay_player_prior_season": target,
        },
        "cohort_metrics": (
            pd.concat(cohort_metrics, ignore_index=True) if cohort_metrics else pd.DataFrame()
        ),
        "possession_predictions": (
            pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        ),
        "game_predictions": (
            pd.concat(
                [_game_prediction_frame(frame) for frame in predictions], ignore_index=True
            )
            if predictions
            else pd.DataFrame()
        ),
        "regular_game_predictions": regular_games,
        "team_net_rating_predictions": team_net_ratings,
        "team_net_rating_metrics": _team_net_rating_metrics(
            team_net_ratings, model=state.candidate.model
        ),
        "team_win_predictions": team_wins,
        "team_win_metrics": team_win_metrics,
    }


def _zero_context_predictor(
    home_lineups: object,
    away_lineups: object,
    profiles: object,
) -> np.ndarray:
    """Return the explicit zero lineup correction for player-prior-only models."""

    del away_lineups, profiles
    return np.zeros(len(home_lineups), dtype=float)  # type: ignore[arg-type]


def _schedule_predictor(
    model: BackToBackScheduleModel,
    schedule_features: pd.DataFrame,
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Bind one frozen schedule state to the canonical game schedule."""

    def predict(rows: pd.DataFrame) -> np.ndarray:
        return model.predict_games(rows, schedule_features)

    return predict


def _fixed_context_predictor(correction: np.ndarray) -> Callable[..., np.ndarray]:
    values = np.asarray(correction, dtype=float)

    def predict(
        home_lineups: object,
        away_lineups: object,
        profiles: object,
    ) -> np.ndarray:
        del away_lineups, profiles
        if len(home_lineups) != len(values):  # type: ignore[arg-type]
            raise ValueError("Cached context correction has the wrong row count")
        return values

    return predict


def _target_contextual_profiles(
    target: str,
    *,
    panel: pd.DataFrame,
    exposure_cohort: pd.DataFrame,
    analytical_dir: Path,
    stints: pd.DataFrame | None = None,
    profile_builder: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build the common pre-target trait table once for every model family."""

    target_stints = stints
    if target_stints is None:
        target_stints = read_rapm_stints(target, analytical_dir=analytical_dir)
    participants = set().union(*target_stints["home_player_ids"], *target_stints["away_player_ids"])
    builder = profile_builder or build_contextual_player_profiles
    return builder(
        panel,
        target_season=target,
        target_player_ids=participants,
        analytical_dir=str(analytical_dir),
        exposure_cohort=exposure_cohort,
    )


def _playoff_partition_exists(season: str, curated_dir: Path) -> bool:
    return (curated_dir / "possession_segments" / season / "playoffs" / "_manifest.json").is_file()


def _collect_outputs(evaluations: Sequence[dict[str, object]]) -> dict[str, pd.DataFrame]:
    tables: dict[str, list[pd.DataFrame]] = {
        "cohort_metrics": [],
        "possession_predictions": [],
        "game_predictions": [],
        "regular_game_predictions": [],
        "team_net_rating_predictions": [],
        "team_net_rating_metrics": [],
        "team_win_predictions": [],
        "team_win_metrics": [],
        "source_states": [],
    }
    for item in evaluations:
        candidate = item["candidate"]
        target = str(item["target"])
        if not isinstance(candidate, BacktestModel):
            raise TypeError("Backtest evaluation lacks a candidate")
        for key in tuple(tables):
            value = item["source_state"] if key == "source_states" else item[key]
            if isinstance(value, dict):
                frame = pd.DataFrame([{**value}])
            elif isinstance(value, pd.DataFrame):
                frame = value.copy()
            else:
                raise TypeError(f"Backtest evaluation {key} has an invalid type")
            frame = frame.drop(columns=["season", "model", "label"], errors="ignore")
            frame.insert(0, "season", target)
            frame.insert(1, "model", candidate.model)
            frame.insert(2, "label", candidate.label)
            tables[key].append(frame)
    output = {key: pd.concat(frames, ignore_index=True) for key, frames in tables.items()}
    output["aggregate_metrics"] = _aggregate_metrics(output)
    return output


def _aggregate_metrics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, possessions in tables["possession_predictions"].groupby("model", sort=False):
        label = possessions["label"].iat[0]
        for cohort, frame in possessions.groupby("cohort", sort=False):
            actual = frame["target_offense_margin"].to_numpy(dtype=float)
            predicted = frame["prediction_offense_margin"].to_numpy(dtype=float)
            # The stored per-season source mean is not a target statistic. Recover it from
            # prediction minus player/context terms is needlessly fragile; aggregate skill
            # from the already materialized per-season MSE instead.
            season_metrics = tables["cohort_metrics"].loc[
                (tables["cohort_metrics"]["model"].eq(model))
                & (tables["cohort_metrics"]["cohort"].eq(cohort))
            ]
            n = season_metrics["possession_count"].to_numpy(dtype=float)
            mean_mse = np.average(
                np.square(season_metrics["frozen_mean_reference_possession_rmse"]), weights=n
            )
            game_frame = tables["game_predictions"].loc[
                (tables["game_predictions"]["model"].eq(model))
                & (tables["game_predictions"]["cohort"].eq(cohort))
            ]
            game_errors = game_frame["margin_error"].to_numpy(dtype=float)
            mean_game_mse = np.average(
                np.square(season_metrics["frozen_mean_reference_game_margin_rmse"]),
                weights=season_metrics["game_count"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "model": model,
                    "label": label,
                    "scope": f"pooled_{cohort}",
                    "season_count": int(frame["season"].nunique()),
                    "game_count": int(game_frame["game_id"].size),
                    "possession_count": len(frame),
                    "possession_rmse": rmse(actual, predicted),
                    "possession_mae": mean_absolute_error(actual, predicted),
                    "eligible_game_margin_rmse": float(np.sqrt(np.mean(np.square(game_errors)))),
                    "possession_skill_vs_frozen_mean": float(
                        1.0 - np.mean(np.square(actual - predicted)) / mean_mse
                    ),
                    "eligible_game_skill_vs_frozen_mean": float(
                        1.0 - np.mean(np.square(game_errors)) / mean_game_mse
                    ),
                }
            )
        regular_games = (
            tables["regular_game_predictions"]
            .loc[tables["regular_game_predictions"]["model"].eq(model)]
            .copy()
        )
        regular_games["game_id"] = (
            regular_games["season"] + ":" + regular_games["game_id"].astype(str)
        )
        full = score_full_game_outcomes(regular_games)
        teams = tables["team_net_rating_predictions"].loc[
            tables["team_net_rating_predictions"]["model"].eq(model)
        ]
        actual_net = teams["actual_net_rating"].to_numpy(dtype=float)
        predicted_net = teams["predicted_net_rating"].to_numpy(dtype=float)
        wins = tables["team_win_predictions"].loc[tables["team_win_predictions"]["model"].eq(model)]
        rows.append(
            {
                "model": model,
                "label": label,
                "scope": "pooled_regular_full_games_and_teams",
                "season_count": int(regular_games["season"].nunique()),
                "game_count": len(regular_games),
                "full_game_margin_rmse": full["full_game_margin_rmse"],
                "full_game_margin_mae": full["full_game_margin_mae"],
                "game_winner_accuracy": full["game_winner_accuracy"],
                "team_net_rating_rmse": rmse(actual_net, predicted_net),
                "team_net_rating_mae": mean_absolute_error(actual_net, predicted_net),
                "team_net_rating_pearson": float(pearsonr(actual_net, predicted_net).statistic),
                "team_net_rating_spearman": float(spearmanr(actual_net, predicted_net).statistic),
                "pythagorean_win_rmse": rmse(wins["wins"], wins["pythagorean_wins"]),
                "pythagorean_win_mae": mean_absolute_error(wins["wins"], wins["pythagorean_wins"]),
                "pythagorean_win_spearman": float(
                    spearmanr(wins["wins"], wins["pythagorean_wins"]).statistic
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_run(
    *,
    target_seasons: tuple[str, ...],
    outputs: dict[str, pd.DataFrame],
    sources: pd.DataFrame,
    artifacts_dir: Path,
) -> FrozenMultiseasonBacktestRun:
    now = datetime.now(UTC)
    run_id = (
        f"{REPORT_NAME}-{target_seasons[0]}-to-{target_seasons[-1]}-"
        f"{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    root = artifacts_dir / REPORT_NAME / f"{target_seasons[0]}_to_{target_seasons[-1]}"
    run_dir = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for name, frame in outputs.items():
            frame.to_parquet(temporary / f"{name}.parquet", index=False)
        sources.to_parquet(temporary / "sources.parquet", index=False)
        metadata = {
            "run_id": run_id,
            "created_at": now.isoformat(),
            "target_seasons": list(target_seasons),
            "information_boundary": (
                "persisted target-season player priors plus immediately prior completed "
                "context state; no replay-season player or context refit"
            ),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        records = [
            {"filename": path.name, "byte_count": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(run_dir)
        return FrozenMultiseasonBacktestRun(run_dir, run_id)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _update_docs(
    path: Path,
    *,
    target_seasons: tuple[str, ...],
    outputs: dict[str, pd.DataFrame],
    run: FrozenMultiseasonBacktestRun,
) -> None:
    start = "<!-- frozen-multiseason-results:start -->"
    end = "<!-- frozen-multiseason-results:end -->"
    aggregate = outputs["aggregate_metrics"]
    regular = aggregate.loc[aggregate["scope"].eq("pooled_regular_season")].copy()
    full = aggregate.loc[aggregate["scope"].eq("pooled_regular_full_games_and_teams")].copy()
    season_metrics = (
        outputs["cohort_metrics"]
        .loc[outputs["cohort_metrics"]["cohort"].eq("regular_season")]
        .copy()
    )
    lines = [
        start,
        "## Results",
        "",
        f"Artifact: `{run.run_dir}`.",
        "",
        "### Pooled Regular Season",
        "",
        (
            "| Model | Possession RMSE | Eligible game RMSE | Full-game RMSE | "
            "Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    full_by_model = full.set_index("model")
    best_regular = {
        "possession_rmse": regular["possession_rmse"].min(),
        "eligible_game_margin_rmse": regular["eligible_game_margin_rmse"].min(),
    }
    best_full = {
        "full_game_margin_rmse": full["full_game_margin_rmse"].min(),
        "game_winner_accuracy": full["game_winner_accuracy"].max(),
        "team_net_rating_rmse": full["team_net_rating_rmse"].min(),
        "pythagorean_win_rmse": full["pythagorean_win_rmse"].min(),
    }
    for row in regular.itertuples(index=False):
        game = full_by_model.loc[row.model]
        possession = _format_metric(
            row.possession_rmse, best_regular["possession_rmse"], precision=6
        )
        eligible_game = _format_metric(
            row.eligible_game_margin_rmse,
            best_regular["eligible_game_margin_rmse"],
            precision=4,
        )
        full_game = _format_metric(
            game.full_game_margin_rmse,
            best_full["full_game_margin_rmse"],
            precision=4,
        )
        winner_accuracy = _format_metric(
            game.game_winner_accuracy,
            best_full["game_winner_accuracy"],
            precision=2,
            percent=True,
        )
        team_net_rating = _format_metric(
            game.team_net_rating_rmse,
            best_full["team_net_rating_rmse"],
            precision=4,
        )
        pythagorean = _format_metric(
            game.pythagorean_win_rmse,
            best_full["pythagorean_win_rmse"],
            precision=4,
        )
        lines.append(
            f"| {row.label} | {possession} | {eligible_game} | {full_game} | "
            f"{winner_accuracy} | {team_net_rating} | {pythagorean} |"
        )
    lines.extend(
        [
            "",
            "### Per-Season Regular Results",
            "",
            "| Season | Model | Possession RMSE | Eligible game RMSE |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    ordered_season_metrics = season_metrics.sort_values(["season", "label"], kind="stable")
    for row in ordered_season_metrics.itertuples(index=False):
        lines.append(
            f"| {row.season} | {row.label} | {row.possession_rmse:.6f} | "
            f"{row.eligible_possession_game_margin_rmse:.4f} |"
        )
    playoff_metrics = (
        outputs["cohort_metrics"].loc[outputs["cohort_metrics"]["cohort"].eq("playoffs")].copy()
    )
    if not playoff_metrics.empty:
        lines.extend(
            [
                "",
                "### Frozen Playoff Check",
                "",
                (
                    "Each playoff cohort uses the same pre-playoff player priors and prior-season "
                    "context state as its matching regular-season forecast. Playoff outcomes are "
                    "evaluation-only and never enter the fitted state."
                ),
                "",
                "| Season | Model | Possession RMSE | Eligible game RMSE |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for row in playoff_metrics.sort_values(["season", "label"], kind="stable").itertuples(
            index=False
        ):
            lines.append(
                f"| {row.season} | {row.label} | {row.possession_rmse:.6f} | "
                f"{row.eligible_possession_game_margin_rmse:.4f} |"
            )
    lines.extend([end, ""])
    content = path.read_text()
    before, marker, after = content.partition(start)
    if not marker:
        raise ValueError(f"Backtest results marker missing from {path}")
    _, end_marker, tail = after.partition(end)
    if not end_marker:
        raise ValueError(f"Backtest results end marker missing from {path}")
    path.write_text(before + "\n".join(lines) + tail)


def _format_metric(
    value: float,
    best: float,
    *,
    precision: int,
    percent: bool = False,
) -> str:
    formatted = f"{value:.{precision}%}" if percent else f"{value:.{precision}f}"
    return f"**{formatted}**" if np.isclose(value, best) else formatted


def _latest_recursive_run(root: Path) -> Path:
    latest = root / "latest.json"
    if not latest.is_file():
        raise ValueError(f"Recursive artifact has no latest pointer: {root}")
    run_id = json.loads(latest.read_text()).get("run_id")
    run_dir = root / str(run_id)
    if not (run_dir / "manifest.json").is_file():
        raise ValueError(f"Recursive artifact latest pointer is invalid: {root}")
    return run_dir


def _validated_seasons(seasons: Sequence[str]) -> tuple[str, ...]:
    validated = tuple(validate_season(str(season)) for season in seasons)
    if len(validated) < 2 or len(set(validated)) != len(validated):
        raise ValueError("Frozen multiseason backtest requires at least two distinct seasons")
    if tuple(sorted(validated, key=lambda value: int(value[:4]))) != validated:
        raise ValueError("Frozen multiseason target seasons must be chronological")
    return validated


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay recursive HPM states across frozen seasons"
    )
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    args = parser.parse_args()
    run = run_frozen_multiseason_backtest(seasons=args.seasons)
    print(f"Frozen multiseason backtest: run={run.run_dir}")


if __name__ == "__main__":
    main()
