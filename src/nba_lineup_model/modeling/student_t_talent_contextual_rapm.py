"""Forward contextual RAPM with a Student-t prior on player adjustments."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_prior import _evaluate_target
from nba_lineup_model.modeling.contextual_profiles import build_contextual_player_profiles
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONTEXT_ALPHA,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    _context_offset,
    _fit_contextual_season,
    _previous_season,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import (
    _cold_start_priors,
    _combine_priors,
    _fit_replacement_token,
    _returning_priors,
)
from nba_lineup_model.modeling.prior_rapm import (
    HISTORICAL_SEASONS,
    PRIOR_MEAN_COLUMN,
    ForwardLaggedRapmSeason,
)
from nba_lineup_model.modeling.replacement_level import (
    player_exposure_shares,
    prepare_player_exposure_cohort,
)
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.modeling.student_t_forward_rapm import _baseline_lambda_schedule
from nba_lineup_model.modeling.student_t_talent_forward_rapm import (
    DEFAULT_DEGREES_OF_FREEDOM,
    DEFAULT_PRIOR_SCALE,
)
from nba_lineup_model.modeling.student_t_talent_forward_rapm import (
    _fit_season as _fit_student_t_talent_season,
)
from nba_lineup_model.season.schema import validate_season

DEFAULT_TARGET_SEASON = "2025-26"
MODEL_NAME = "student_t_talent_forward_contextual_rapm"
RUN_PREFIX = "student-t-talent-contextual-rapm"


@dataclass(frozen=True)
class StudentTTalentContextualRapmRun:
    """Immutable output for the combined forward model."""

    run_dir: Path
    run_id: str
    target_season: str


def train_student_t_talent_contextual_rapm(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    degrees_of_freedom: float = DEFAULT_DEGREES_OF_FREEDOM,
    prior_scale: float = DEFAULT_PRIOR_SCALE,
    max_iterations: int = 120,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    checkpoint_path: Path | str | None = None,
    max_seasons: int | None = None,
) -> StudentTTalentContextualRapmRun | None:
    r"""Fit Student-t player state and contextual offsets one season at a time.

    The order is fixed. A completed \(g_{t-1}\) is subtracted before fitting
    the Student-t player state for season \(t\); the completed player state
    then supplies residuals used to fit \(g_t\). Cold starts, lambda schedule,
    and profiles use the existing forward exposure-gated contracts.
    """

    target = validate_season(through_season)
    if context_alpha <= 0 or degrees_of_freedom <= 0 or prior_scale <= 0:
        raise ValueError("Context alpha and Student-t prior settings must be positive")
    if max_iterations < 1:
        raise ValueError("Student-t maximum iterations must be positive")
    panel = pd.read_parquet(player_season_panel_path)
    seasons = tuple(season for season in HISTORICAL_SEASONS if season <= target)
    if target not in seasons:
        seasons = (*seasons, target)
    if not seasons or seasons[-1] != target:
        raise ValueError(f"Unsupported through-season: {target}")

    artifact_root = Path(artifacts_dir)
    lambda_schedule = _baseline_lambda_schedule(artifact_root, target)
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else artifact_root / "student_t_talent_contextual_rapm" / target / ".checkpoint.joblib"
    )
    state = _load_checkpoint(checkpoint, seasons)
    results: list[ForwardLaggedRapmSeason] = state["results"]
    priors_by_season: list[pd.DataFrame] = state["priors_by_season"]
    exposure_history: list[pd.DataFrame] = state["exposure_history"]
    replacement_tokens: list[dict[str, object]] = state["replacement_tokens"]
    cold_metadata: list[dict[str, object]] = state["cold_metadata"]
    contextual_models: dict[str, object] = state["contextual_models"]
    contextual_metadata: list[dict[str, object]] = state["contextual_metadata"]
    target_priors: pd.DataFrame | None = state["target_priors"]
    target_profiles: pd.DataFrame | None = state["target_profiles"]
    start_count = len(results)
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(target)],
        through_season=target,
        analytical_dir=analytical_dir,
    )

    for season in seasons[len(results) :]:
        print(f"Fitting Student-t contextual state for {season}", flush=True)
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
        cold, metadata = _cold_start_priors(
            season=season,
            panel=panel,
            completed_results=results,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
        )
        priors = _combine_priors(_returning_priors(results), cold)
        prior_rows = priors.rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm"}).copy()
        prior_rows["season"] = season
        prior_rows["context_offset_source_season"] = (
            _previous_season(season) if previous_model is not None else pd.NA
        )
        priors_by_season.append(prior_rows)
        if season == target:
            target_priors = prior_rows.copy()
            target_profiles = profiles

        regularization = float(lambda_schedule[season])
        fitted, diagnostics = _fit_student_t_talent_season(
            season,
            adjusted_stints,
            priors,
            regularization,
            degrees_of_freedom,
            prior_scale,
            max_iterations,
        )
        results.append(fitted)
        metadata.update(diagnostics)
        cold_metadata.append(metadata)
        exposure = player_exposure_shares(raw_stints)
        exposure_history.append(
            panel.loc[panel["season"].eq(season)].merge(
                exposure, on="player_id", how="inner", validate="one_to_one"
            )
        )
        replacement_tokens.append(
            _fit_replacement_token(season, raw_stints, exposure, fitted, panel)
        )
        if profiles is not None:
            context_model, context_row = _fit_contextual_season(
                raw_stints,
                fitted,
                profiles,
                context_alpha,
            )
            contextual_models[season] = context_model
            contextual_metadata.append(context_row)

        _write_checkpoint(
            checkpoint,
            seasons=seasons,
            results=results,
            priors_by_season=priors_by_season,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
            cold_metadata=cold_metadata,
            contextual_models=contextual_models,
            contextual_metadata=contextual_metadata,
            target_priors=target_priors,
            target_profiles=target_profiles,
        )
        if max_seasons is not None and len(results) >= start_count + max_seasons:
            return None

    if target_priors is None or target_profiles is None:
        raise ValueError("Student-t contextual run did not create the target frozen state")
    source = _previous_season(target)
    forecast_model = contextual_models.get(source)
    if forecast_model is None:
        raise ValueError("Student-t contextual run has no completed source-season context state")
    coefficients = pd.concat([result.player_estimates for result in results], ignore_index=True)
    priors = pd.concat(priors_by_season, ignore_index=True)
    evaluation = _evaluate_target(
        target,
        model=forecast_model,
        profiles=target_profiles,
        priors=priors,
        coefficients=coefficients,
        analytical_dir=Path(analytical_dir),
        curated_dir=Path(curated_dir),
        evaluation_model=MODEL_NAME,
    )
    evaluation["source_state"].update(
        {
            "player_prior_method": (
                "forward exposure-gated RAPM with a Student-t talent prior and contextual "
                "spline residual"
            ),
            "coefficient_prior_distribution": "Student-t",
            "student_t_degrees_of_freedom": degrees_of_freedom,
            "student_t_prior_scale": prior_scale,
        }
    )
    run = _write_run(
        target=target,
        results=results,
        priors=priors,
        replacement_tokens=replacement_tokens,
        cold_metadata=cold_metadata,
        contextual_models=contextual_models,
        contextual_metadata=pd.DataFrame(contextual_metadata),
        target_priors=target_priors,
        target_profiles=target_profiles,
        evaluation=evaluation,
        context_alpha=context_alpha,
        degrees_of_freedom=degrees_of_freedom,
        prior_scale=prior_scale,
        artifacts_dir=artifact_root,
    )
    checkpoint.unlink(missing_ok=True)
    return run


def _load_checkpoint(path: Path, seasons: tuple[str, ...]) -> dict[str, object]:
    if not path.is_file():
        return {
            "results": [],
            "priors_by_season": [],
            "exposure_history": [],
            "replacement_tokens": [],
            "cold_metadata": [],
            "contextual_models": {},
            "contextual_metadata": [],
            "target_priors": None,
            "target_profiles": None,
        }
    state = joblib.load(path)
    required = {
        "seasons",
        "results",
        "priors_by_season",
        "exposure_history",
        "replacement_tokens",
        "cold_metadata",
        "contextual_models",
        "contextual_metadata",
        "target_priors",
        "target_profiles",
    }
    if not isinstance(state, dict) or required - set(state) or tuple(state["seasons"]) != seasons:
        raise ValueError("Student-t contextual checkpoint is incompatible")
    lengths = [
        len(state["results"]),
        len(state["priors_by_season"]),
        len(state["exposure_history"]),
        len(state["replacement_tokens"]),
        len(state["cold_metadata"]),
    ]
    if len(set(lengths)) != 1 or lengths[0] > len(seasons):
        raise ValueError("Student-t contextual checkpoint is incomplete")
    return state


def _write_checkpoint(
    path: Path,
    *,
    seasons: tuple[str, ...],
    results: list[ForwardLaggedRapmSeason],
    priors_by_season: list[pd.DataFrame],
    exposure_history: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
    cold_metadata: list[dict[str, object]],
    contextual_models: dict[str, object],
    contextual_metadata: list[dict[str, object]],
    target_priors: pd.DataFrame | None,
    target_profiles: pd.DataFrame | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(
        {
            "seasons": seasons,
            "results": results,
            "priors_by_season": priors_by_season,
            "exposure_history": exposure_history,
            "replacement_tokens": replacement_tokens,
            "cold_metadata": cold_metadata,
            "contextual_models": contextual_models,
            "contextual_metadata": contextual_metadata,
            "target_priors": target_priors,
            "target_profiles": target_profiles,
        },
        temporary,
    )
    temporary.replace(path)


def _write_run(
    *,
    target: str,
    results: list[ForwardLaggedRapmSeason],
    priors: pd.DataFrame,
    replacement_tokens: list[dict[str, object]],
    cold_metadata: list[dict[str, object]],
    contextual_models: dict[str, object],
    contextual_metadata: pd.DataFrame,
    target_priors: pd.DataFrame,
    target_profiles: pd.DataFrame,
    evaluation: dict[str, pd.DataFrame | dict[str, object]],
    context_alpha: float,
    degrees_of_freedom: float,
    prior_scale: float,
    artifacts_dir: Path,
) -> StudentTTalentContextualRapmRun:
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}-{target}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "student_t_talent_contextual_rapm" / target
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
            "season_replacement_tokens.parquet": pd.DataFrame(replacement_tokens),
            "season_cold_start_metadata.parquet": pd.DataFrame(cold_metadata),
            "season_context_metadata.parquet": contextual_metadata,
            "frozen_2025_26_player_priors.parquet": target_priors,
            "target_player_profiles.parquet": target_profiles,
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
            "model": MODEL_NAME,
            "target_season": target,
            "context_alpha": context_alpha,
            "stint_error_distribution": "Gaussian",
            "coefficient_prior_distribution": "Student-t",
            "student_t_degrees_of_freedom": degrees_of_freedom,
            "student_t_prior_scale": prior_scale,
            "lambda_policy": "fixed to completed Gaussian forward exposure-gated run",
            "contextual_offset_contract": "g_(t-1) is subtracted before fitting RAPM in season t",
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint(
                (
                    Path(__file__),
                    Path(__file__).with_name("forward_contextual_rapm.py"),
                    Path(__file__).with_name("student_t.py"),
                )
            ),
            "source_state": evaluation["source_state"],
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {
                "filename": path.name,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return StudentTTalentContextualRapmRun(output, run_id, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train forward contextual RAPM with a Student-t talent prior"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    parser.add_argument("--degrees-of-freedom", type=float, default=DEFAULT_DEGREES_OF_FREEDOM)
    parser.add_argument("--prior-scale", type=float, default=DEFAULT_PRIOR_SCALE)
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--max-seasons", type=int)
    args = parser.parse_args()
    run = train_student_t_talent_contextual_rapm(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
        degrees_of_freedom=args.degrees_of_freedom,
        prior_scale=args.prior_scale,
        max_iterations=args.max_iterations,
        checkpoint_path=args.checkpoint_path,
        max_seasons=args.max_seasons,
    )
    print(
        "Student-t contextual RAPM checkpoint saved"
        if run is None
        else f"Student-t contextual RAPM: run={run.run_dir}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
