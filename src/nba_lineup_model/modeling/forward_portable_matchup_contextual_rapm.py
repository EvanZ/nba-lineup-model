"""Recursive RAPM with portable unit context plus matchup residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import lineup_side_context_features
from nba_lineup_model.modeling.contextual_prior import _evaluate_target, _lineup_effects
from nba_lineup_model.modeling.contextual_profiles import build_contextual_player_profiles
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONTEXT_ALPHA,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    DEFAULT_TARGET_SEASON,
    _previous_season,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import (
    _cold_start_priors,
    _combine_priors,
    _fit_replacement_token,
    _latest_run,
    _returning_priors,
)
from nba_lineup_model.modeling.frozen_prior_evaluation import _recover_home_intercept
from nba_lineup_model.modeling.matchup_contextual import (
    MatchupContextualModel,
    fit_matchup_contextual_model,
    model_metadata,
)
from nba_lineup_model.modeling.prior_rapm import (
    HISTORICAL_SEASONS,
    PRIOR_MEAN_COLUMN,
    ForwardLaggedRapmSeason,
    fit_forward_lagged_rapm_season,
)
from nba_lineup_model.modeling.replacement_level import (
    player_exposure_shares,
    prepare_player_exposure_cohort,
)
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.season.schema import validate_season

MODEL_NAME = "forward_portable_matchup_contextual_rapm"
RUN_PREFIX = "forward-portable-matchup-contextual-rapm"


@dataclass(frozen=True)
class ForwardPortableMatchupContextualRapmRun:
    """One immutable recursive portable-plus-matchup context artifact."""

    run_dir: Path
    run_id: str


def train_forward_portable_matchup_contextual_rapm(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    context_curvature_alpha: float = 0.0,
    context_temporal_alpha: float = 0.0,
    model_name: str = MODEL_NAME,
    run_prefix: str = RUN_PREFIX,
    player_prior_builder: Callable[..., tuple[pd.DataFrame, dict[str, object]]] | None = None,
    player_prior_description: str = "forward exposure-gated RAPM plus portable-matchup context",
    context_fit: Callable[..., MatchupContextualModel] = fit_matchup_contextual_model,
    context_metadata: Callable[[MatchupContextualModel], dict[str, object]] = model_metadata,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Roll additive RAPM and reference-identified contextual state forward."""

    target = validate_season(through_season)
    if context_alpha <= 0 or context_curvature_alpha < 0 or context_temporal_alpha < 0:
        raise ValueError("Contextual penalties must be non-negative, with alpha positive")
    panel = pd.read_parquet(player_season_panel_path)
    artifact_root = Path(artifacts_dir)
    reference_root = _latest_run(artifact_root / "forward_exposure_gated_rapm" / target)
    reference_coefficients = pd.read_parquet(
        reference_root / "historical_player_coefficients.parquet"
    )
    lambda_schedule = reference_coefficients.groupby("season", as_index=True)[
        "selected_lambda"
    ].agg(lambda values: float(values.iloc[0]))
    seasons = tuple(season for season in HISTORICAL_SEASONS if season <= target)
    if target not in seasons:
        seasons = (*seasons, target)
    source = _previous_season(target)
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(target)],
        through_season=target,
        analytical_dir=analytical_dir,
    )
    results: list[ForwardLaggedRapmSeason] = []
    priors_by_season: list[pd.DataFrame] = []
    exposure_history: list[pd.DataFrame] = []
    replacement_tokens: list[dict[str, object]] = []
    prior_metadata: list[dict[str, object]] = []
    contextual_models: dict[str, MatchupContextualModel] = {}
    contextual_metadata: list[dict[str, object]] = []
    target_priors: pd.DataFrame | None = None
    target_profiles: pd.DataFrame | None = None
    prior_builder = player_prior_builder or _exposure_gated_player_priors

    for season in seasons:
        print(f"Fitting portable-plus-matchup context state for {season}", flush=True)
        raw_stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        participants = set().union(*raw_stints["home_player_ids"], *raw_stints["away_player_ids"])
        profiles = (
            build_contextual_player_profiles(
                panel,
                target_season=season,
                target_player_ids=participants,
                analytical_dir=str(analytical_dir),
                exposure_cohort=exposure_cohort,
            )
            if season != seasons[0]
            else None
        )
        previous_model = contextual_models.get(_previous_season(season))
        offset = (
            _context_offset(raw_stints, previous_model, profiles)
            if previous_model is not None and profiles is not None
            else np.zeros(len(raw_stints), dtype=float)
        )
        adjusted_stints = raw_stints.copy()
        adjusted_stints["target_home_net_rating"] = (
            raw_stints["target_home_net_rating"].to_numpy(dtype=float) - offset
        )
        priors, prior_row = prior_builder(
            season=season,
            panel=panel,
            completed_results=results,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
        )
        prior_metadata.append(prior_row)
        prior_rows = priors.rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm"}).copy()
        prior_rows["season"] = season
        prior_rows["context_offset_source_season"] = (
            _previous_season(season) if previous_model is not None else pd.NA
        )
        priors_by_season.append(prior_rows)
        season_lambda = float(lambda_schedule.loc[season])
        if not np.isfinite(season_lambda) or season_lambda < 0:
            raise ValueError(f"Published RAPM lambda is invalid for {season}")
        fitted = fit_forward_lagged_rapm_season(
            season,
            adjusted_stints,
            priors,
            lambda_grid=(season_lambda,),
        )
        results.append(fitted)
        exposure = player_exposure_shares(raw_stints)
        exposure_history.append(
            panel.loc[panel["season"].eq(season)].merge(
                exposure, on="player_id", how="inner", validate="one_to_one"
            )
        )
        replacement_tokens.append(
            _fit_replacement_token(season, adjusted_stints, exposure, fitted, panel)
        )
        if profiles is not None:
            model, row = _fit_matchup_contextual_season(
                raw_stints,
                fitted,
                profiles,
                alpha=context_alpha,
                curvature_alpha=context_curvature_alpha,
                temporal_alpha=context_temporal_alpha if previous_model is not None else 0.0,
                previous_model=previous_model,
                context_fit=context_fit,
                context_metadata=context_metadata,
            )
            contextual_models[season] = model
            contextual_metadata.append(row)
        if season == target:
            target_priors = priors.rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm_mean"})
            target_profiles = profiles

    if target_priors is None or target_profiles is None:
        raise ValueError("Portable-matchup contextual RAPM did not create the target state")
    forecast_model = contextual_models.get(source)
    if forecast_model is None:
        raise ValueError("Portable-matchup contextual RAPM has no source context state")
    historical_coefficients = pd.concat(
        [result.player_estimates for result in results], ignore_index=True
    )
    state_priors = pd.concat(priors_by_season, ignore_index=True)
    evaluation = _evaluate_target(
        target,
        model=forecast_model,  # type: ignore[arg-type]
        profiles=target_profiles,
        priors=state_priors,
        coefficients=historical_coefficients,
        analytical_dir=Path(analytical_dir),
        curated_dir=Path(curated_dir),
        evaluation_model=model_name,
        context_predictor=forecast_model.predict_lineups,
    )
    evaluation["source_state"] = {
        **evaluation["source_state"],  # type: ignore[arg-type]
        "player_prior_method": player_prior_description,
        "context_contract": "C_(t-1)(home, away) = h(home) - h(away) + q(home, away)",
        "reference_context_source_season": source,
    }
    return _write_run(
        target=target,
        results=results,
        priors=state_priors,
        contextual_models=contextual_models,
        contextual_metadata=pd.DataFrame(contextual_metadata),
        prior_metadata=pd.DataFrame(prior_metadata),
        target_priors=target_priors,
        target_profiles=target_profiles,
        forecast_reference=forecast_model.reference_features.assign(
            reference_weight=forecast_model.reference_weights
        ),
        evaluation=evaluation,
        context_alpha=context_alpha,
        context_curvature_alpha=context_curvature_alpha,
        context_temporal_alpha=context_temporal_alpha,
        model_name=model_name,
        run_prefix=run_prefix,
        artifacts_dir=artifact_root,
    )


def _context_offset(
    stints: pd.DataFrame,
    model: MatchupContextualModel,
    profiles: pd.DataFrame,
) -> np.ndarray:
    return model.predict_lineups(
        stints["home_player_ids"].tolist(),
        stints["away_player_ids"].tolist(),
        profiles,
    )


def _exposure_gated_player_priors(
    *,
    season: str,
    panel: pd.DataFrame,
    completed_results: list[ForwardLaggedRapmSeason],
    exposure_history: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Preserve the established lagged-returner and cold-start prior state."""

    cold, cold_metadata = _cold_start_priors(
        season=season,
        panel=panel,
        completed_results=completed_results,
        exposure_history=exposure_history,
        replacement_tokens=replacement_tokens,
    )
    return _combine_priors(_returning_priors(completed_results), cold), {
        "season": season,
        "player_prior_method": "lagged_rapm_plus_exposure_gated_cold_start",
        "cold_start": cold_metadata,
    }


def _fit_matchup_contextual_season(
    stints: pd.DataFrame,
    fitted: ForwardLaggedRapmSeason,
    profiles: pd.DataFrame,
    *,
    alpha: float,
    curvature_alpha: float = 0.0,
    temporal_alpha: float = 0.0,
    previous_model: MatchupContextualModel | None = None,
    context_fit: Callable[..., MatchupContextualModel] = fit_matchup_contextual_model,
    context_metadata: Callable[[MatchupContextualModel], dict[str, object]] = model_metadata,
) -> tuple[MatchupContextualModel, dict[str, object]]:
    coefficients = fitted.player_estimates.loc[:, ["player_id", "rapm"]]
    values = dict(zip(coefficients["player_id"].astype(int), coefficients["rapm"], strict=True))
    effects, unknown = _lineup_effects(stints, values)
    intercept = _recover_home_intercept(stints, coefficients)
    home = lineup_side_context_features(stints["home_player_ids"].tolist(), profiles)
    away = lineup_side_context_features(stints["away_player_ids"].tolist(), profiles)
    target = stints["target_home_net_rating"].to_numpy(dtype=float) - effects - intercept
    print(f"  Fitting antisymmetric spline on {len(stints):,} stints", flush=True)
    model = context_fit(
        home,
        away,
        target,
        stints["possessions"].to_numpy(dtype=float),
        alpha=alpha,
        curvature_alpha=curvature_alpha,
        temporal_alpha=temporal_alpha,
        previous_model=previous_model,
    )
    print(
        f"  Stored {len(model.reference_features):,} reference units for {fitted.season}",
        flush=True,
    )
    return model, {
        "season": fitted.season,
        "context_alpha": alpha,
        "context_curvature_alpha": curvature_alpha,
        "context_temporal_alpha": temporal_alpha,
        "context_temporal_source_season": (
            _previous_season(fitted.season) if previous_model is not None else pd.NA
        ),
        "context_training_stint_count": len(stints),
        "context_unknown_player_exposures": int(unknown.sum()),
        "context_home_intercept": intercept,
        **context_metadata(model),
    }


def _write_run(
    *,
    target: str,
    results: list[ForwardLaggedRapmSeason],
    priors: pd.DataFrame,
    contextual_models: dict[str, MatchupContextualModel],
    contextual_metadata: pd.DataFrame,
    prior_metadata: pd.DataFrame,
    target_priors: pd.DataFrame,
    target_profiles: pd.DataFrame,
    forecast_reference: pd.DataFrame,
    evaluation: dict[str, pd.DataFrame | dict[str, object]],
    context_alpha: float,
    context_curvature_alpha: float,
    context_temporal_alpha: float,
    model_name: str,
    run_prefix: str,
    artifacts_dir: Path,
) -> ForwardPortableMatchupContextualRapmRun:
    now = datetime.now(UTC)
    run_id = f"{run_prefix}-{target}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / model_name / target
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        tables: dict[str, pd.DataFrame] = {
            "historical_player_coefficients.parquet": pd.concat(
                [result.player_estimates for result in results], ignore_index=True
            ),
            "season_player_priors.parquet": priors,
            "season_context_metadata.parquet": contextual_metadata,
            "season_player_prior_metadata.parquet": prior_metadata,
            "frozen_2025_26_player_priors.parquet": target_priors,
            "target_player_profiles.parquet": target_profiles,
            "forecast_reference_units.parquet": forecast_reference,
            "cohort_metrics.parquet": evaluation["cohort_metrics"],  # type: ignore[dict-item]
            "possession_predictions.parquet": evaluation["possession_predictions"],  # type: ignore[dict-item]
            "game_predictions.parquet": evaluation["game_predictions"],  # type: ignore[dict-item]
            "regular_game_predictions.parquet": evaluation["regular_game_predictions"],  # type: ignore[dict-item]
            "team_net_rating_predictions.parquet": evaluation["team_net_rating_predictions"],  # type: ignore[dict-item]
            "team_net_rating_metrics.parquet": evaluation["team_net_rating_metrics"],  # type: ignore[dict-item]
            "team_win_predictions.parquet": evaluation["team_win_predictions"],  # type: ignore[dict-item]
            "team_win_metrics.parquet": evaluation["team_win_metrics"],  # type: ignore[dict-item]
        }
        for filename, frame in tables.items():
            frame.to_parquet(temporary / filename, index=False)
        joblib.dump(contextual_models, temporary / "season_context_models.joblib")
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": model_name,
            "target_season": target,
            "context_alpha": context_alpha,
            "context_curvature_alpha": context_curvature_alpha,
            "context_temporal_alpha": context_temporal_alpha,
            "contextual_offset_contract": "C_(t-1)(home, away) is subtracted before season t RAPM",
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint(
                (Path(__file__), Path(__file__).with_name("matchup_contextual.py"))
            ),
            "source_state": evaluation["source_state"],
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {"filename": path.name, "byte_count": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return ForwardPortableMatchupContextualRapmRun(output, run_id)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Train the published portable-plus-matchup contextual RAPM exemplar."""

    parser = argparse.ArgumentParser(
        description="Train forward RAPM with portable lineup context and matchup residuals"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    args = parser.parse_args()
    run = train_forward_portable_matchup_contextual_rapm(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
    )
    print(f"Forward portable-matchup contextual RAPM: run={run.run_dir}")


if __name__ == "__main__":
    main()
