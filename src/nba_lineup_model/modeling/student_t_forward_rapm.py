"""Recursive forward exposure-gated RAPM with Student-t stint errors."""

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

from nba_lineup_model.modeling.forward_exposure_gated_rapm import (
    _cold_start_priors,
    _combine_priors,
    _fit_replacement_token,
    _returning_priors,
)
from nba_lineup_model.modeling.frozen_prior_evaluation import run_frozen_lagged_evaluation
from nba_lineup_model.modeling.prior_rapm import (
    HISTORICAL_SEASONS,
    PRIOR_MEAN_COLUMN,
    ForwardLaggedRapmSeason,
    _prior_vector,
)
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.modeling.student_t import fit_student_t_prior_centered_ridge
from nba_lineup_model.models.baselines import (
    entity_vocabulary,
    signed_entity_matrix,
    vocabulary_mapping,
)

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_DEGREES_OF_FREEDOM = 5.0


@dataclass(frozen=True)
class StudentTForwardRapmRun:
    run_dir: Path
    run_id: str
    through_season: str


def train_student_t_forward_rapm(
    *,
    through_season: str = "2025-26",
    degrees_of_freedom: float = DEFAULT_DEGREES_OF_FREEDOM,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    max_iterations: int = 20,
    checkpoint_path: Path | str | None = None,
    max_seasons: int | None = None,
) -> StudentTForwardRapmRun | None:
    """Fit strict forward Student-t RAPM using the Gaussian run's lambda schedule."""

    if degrees_of_freedom <= 2:
        raise ValueError("Student-t degrees of freedom must exceed two")
    panel = pd.read_parquet(player_season_panel_path)
    seasons = tuple(season for season in HISTORICAL_SEASONS if season <= through_season)
    if through_season not in seasons:
        seasons = (*seasons, through_season)
    if not seasons or seasons[-1] != through_season:
        raise ValueError(f"Unsupported through-season: {through_season}")
    roots = Path(artifacts_dir)
    lambda_schedule = _baseline_lambda_schedule(roots, through_season)
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path
        else (roots / "student_t_forward_rapm" / through_season / ".checkpoint.joblib")
    )
    state = _load_checkpoint(checkpoint, seasons)
    results: list[ForwardLaggedRapmSeason] = state["results"]
    season_priors: list[pd.DataFrame] = state["season_priors"]
    replacement_tokens: list[dict[str, object]] = state["replacement_tokens"]
    exposure_history: list[pd.DataFrame] = state["exposure_history"]
    cold_metadata: list[dict[str, object]] = state["cold_metadata"]
    frozen_priors: pd.DataFrame | None = state["frozen_priors"]
    start_count = len(results)
    for season in seasons[len(results) :]:
        print(f"Fitting Student-t {season}", flush=True)
        stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        cold, metadata = _cold_start_priors(
            season=season,
            panel=panel,
            completed_results=results,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
        )
        priors = _combine_priors(_returning_priors(results), cold)
        if season == "2025-26":
            frozen_priors = priors.copy()
        result, diagnostics = _fit_season(
            season,
            stints,
            priors,
            lambda_schedule[season],
            degrees_of_freedom,
            max_iterations,
        )
        results.append(result)
        metadata.update(diagnostics)
        cold_metadata.append(metadata)
        from nba_lineup_model.modeling.replacement_level import player_exposure_shares

        exposure = player_exposure_shares(stints)
        exposure_history.append(
            panel.loc[panel["season"].eq(season)].merge(
                exposure, on="player_id", how="inner", validate="one_to_one"
            )
        )
        replacement_tokens.append(_fit_replacement_token(season, stints, exposure, result, panel))
        season_priors.append(priors.assign(season=season))
        _write_checkpoint(
            checkpoint,
            seasons=seasons,
            results=results,
            season_priors=season_priors,
            replacement_tokens=replacement_tokens,
            exposure_history=exposure_history,
            cold_metadata=cold_metadata,
            frozen_priors=frozen_priors,
        )
        if max_seasons is not None and len(results) >= start_count + max_seasons:
            return None
    if frozen_priors is None:
        raise ValueError("Student-t forward run requires a frozen 2025-26 prior")
    evaluation = _evaluate_frozen_priors(
        frozen_priors, roots, Path(analytical_dir), Path(curated_dir)
    )
    run = _write_run(
        through_season=through_season,
        degrees_of_freedom=degrees_of_freedom,
        results=results,
        priors=season_priors,
        replacement_tokens=replacement_tokens,
        cold_metadata=cold_metadata,
        frozen_priors=frozen_priors,
        evaluation=evaluation,
        panel=panel,
        artifacts_dir=roots,
    )
    checkpoint.unlink(missing_ok=True)
    return run


def _fit_season(
    season: str,
    stints: pd.DataFrame,
    priors: pd.DataFrame,
    regularization: float,
    degrees_of_freedom: float,
    max_iterations: int,
) -> tuple[ForwardLaggedRapmSeason, dict[str, object]]:
    player_ids = entity_vocabulary(stints, "home_player_ids", "away_player_ids", multiple=True)
    matrix = signed_entity_matrix(
        stints, "home_player_ids", "away_player_ids", vocabulary_mapping(player_ids), multiple=True
    )
    prior, prior_frame = _prior_vector(player_ids, priors)
    fit = fit_student_t_prior_centered_ridge(
        matrix,
        stints["target_home_net_rating"].to_numpy(dtype=float),
        stints["possessions"].to_numpy(dtype=float),
        prior,
        regularization=regularization,
        degrees_of_freedom=degrees_of_freedom,
        max_iterations=max_iterations,
    )
    estimates = pd.DataFrame(
        {
            "season": season,
            "player_id": player_ids,
            "rapm": fit.model.coef_,
            "prior_rapm": prior,
            "rapm_adjustment_from_prior": fit.model.adjustment_,
        }
    ).merge(prior_frame.loc[:, ["player_id", "prior_available"]], on="player_id")
    estimates["selected_lambda"] = regularization
    return (
        ForwardLaggedRapmSeason(
            season,
            regularization,
            pd.DataFrame(),
            estimates.sort_values("player_id").reset_index(drop=True),
            prior_frame,
        ),
        {
            "student_t_scale": fit.scale,
            "student_t_iterations": fit.iterations,
            "student_t_converged": fit.converged,
        },
    )


def _baseline_lambda_schedule(artifacts_dir: Path, through_season: str) -> dict[str, float]:
    root = artifacts_dir / "forward_exposure_gated_rapm" / through_season
    run = root / json.loads((root / "latest.json").read_text())["run_id"]
    coefficients = pd.read_parquet(run / "historical_player_coefficients.parquet")
    schedule = coefficients.groupby("season", sort=False)["selected_lambda"].first().to_dict()
    required = {season for season in HISTORICAL_SEASONS if season <= through_season}
    required.add(through_season)
    if required - set(schedule):
        raise ValueError("Gaussian forward artifact lacks the required lambda schedule")
    return {str(season): float(value) for season, value in schedule.items()}


def _evaluate_frozen_priors(
    priors: pd.DataFrame,
    artifacts_dir: Path,
    analytical_dir: Path,
    curated_dir: Path,
    *,
    player_prior_method: str = "strictly forward Student-t RAPM state",
    evaluation_model: str = "frozen_student_t_forward_exposure_gated_rapm",
) -> object:
    source_root = artifacts_dir / "prior_rapm" / "2025-26"
    source_run = source_root / json.loads((source_root / "latest.json").read_text())["run_id"]
    return run_frozen_lagged_evaluation(
        season="2025-26",
        prior_run_dir=source_run,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        player_priors_override=priors.rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm_mean"}),
        source_state_overrides={
            "player_prior_method": player_prior_method,
            "cold_start_prior": "exposure-gated draft and replacement prior",
        },
        evaluation_model=evaluation_model,
    )


def _load_checkpoint(path: Path, seasons: tuple[str, ...]) -> dict[str, object]:
    if not path.is_file():
        return {
            "results": [],
            "season_priors": [],
            "replacement_tokens": [],
            "exposure_history": [],
            "cold_metadata": [],
            "frozen_priors": None,
        }
    state = joblib.load(path)
    required = {
        "seasons",
        "results",
        "season_priors",
        "replacement_tokens",
        "exposure_history",
        "cold_metadata",
        "frozen_priors",
    }
    if not isinstance(state, dict) or required - set(state) or tuple(state["seasons"]) != seasons:
        raise ValueError("Student-t forward checkpoint is incompatible")
    lengths = [
        len(state["results"]),
        len(state["season_priors"]),
        len(state["replacement_tokens"]),
        len(state["exposure_history"]),
        len(state["cold_metadata"]),
    ]
    if len(set(lengths)) != 1 or lengths[0] > len(seasons):
        raise ValueError("Student-t forward checkpoint is incomplete")
    return state


def _write_checkpoint(
    path: Path,
    *,
    seasons: tuple[str, ...],
    results: list[ForwardLaggedRapmSeason],
    season_priors: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
    exposure_history: list[pd.DataFrame],
    cold_metadata: list[dict[str, object]],
    frozen_priors: pd.DataFrame | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(
        {
            "seasons": seasons,
            "results": results,
            "season_priors": season_priors,
            "replacement_tokens": replacement_tokens,
            "exposure_history": exposure_history,
            "cold_metadata": cold_metadata,
            "frozen_priors": frozen_priors,
        },
        temporary,
    )
    temporary.replace(path)


def _write_run(
    *,
    through_season: str,
    degrees_of_freedom: float,
    results: list[ForwardLaggedRapmSeason],
    priors: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
    cold_metadata: list[dict[str, object]],
    frozen_priors: pd.DataFrame,
    evaluation: object,
    panel: pd.DataFrame,
    artifacts_dir: Path,
) -> StudentTForwardRapmRun:
    now = datetime.now(UTC)
    run_id = f"student-t-forward-rapm-{through_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "student_t_forward_rapm" / through_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        final = results[-1].player_estimates.loc[
            :, ["player_id", "rapm", "prior_rapm", "rapm_adjustment_from_prior"]
        ]
        season_panel = panel.loc[
            panel["season"].eq(through_season),
            ["player_id", "player_name", "listed_position", "rapm_possessions"],
        ]
        rankings = final.merge(
            season_panel, on="player_id", how="left", validate="one_to_one"
        ).sort_values(
            ["rapm", "rapm_possessions", "player_id"],
            ascending=[False, False, True],
            kind="stable",
        )
        rankings["rank"] = np.arange(1, len(rankings) + 1)
        tables = {
            "historical_player_coefficients.parquet": pd.concat(
                [result.player_estimates for result in results], ignore_index=True
            ),
            "season_player_priors.parquet": pd.concat(priors, ignore_index=True),
            "season_replacement_tokens.parquet": pd.DataFrame(replacement_tokens),
            "season_cold_start_metadata.parquet": pd.DataFrame(cold_metadata),
            "frozen_2025_26_player_priors.parquet": frozen_priors,
            "next_season_returning_rankings.parquet": rankings,
            "next_season_top_100_returning_rankings.parquet": rankings.head(100),
            "cohort_metrics.parquet": evaluation.cohort_metrics,
            "possession_predictions.parquet": evaluation.possession_predictions,
            "game_predictions.parquet": evaluation.game_predictions,
            "regular_game_predictions.parquet": evaluation.regular_game_predictions,
            "team_net_rating_predictions.parquet": evaluation.team_net_rating_predictions,
            "team_net_rating_metrics.parquet": evaluation.team_net_rating_metrics,
            "team_win_predictions.parquet": evaluation.team_win_predictions,
            "team_win_metrics.parquet": evaluation.team_win_metrics,
        }
        for name, table in tables.items():
            table.to_parquet(temporary / name, index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": "student_t_forward_exposure_gated_rapm",
            "through_season": through_season,
            "degrees_of_freedom": degrees_of_freedom,
            "lambda_policy": "fixed to completed Gaussian forward run",
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint(
                (Path(__file__), Path(__file__).with_name("student_t.py"))
            ),
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
        return StudentTForwardRapmRun(output, run_id, through_season)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Student-t forward exposure-gated RAPM")
    parser.add_argument("--through-season", default="2025-26")
    parser.add_argument("--degrees-of-freedom", type=float, default=DEFAULT_DEGREES_OF_FREEDOM)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--curated-dir", default=str(DEFAULT_CURATED_DIR))
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--max-seasons", type=int)
    args = parser.parse_args()
    run = train_student_t_forward_rapm(
        through_season=args.through_season,
        degrees_of_freedom=args.degrees_of_freedom,
        max_iterations=args.max_iterations,
        artifacts_dir=args.artifacts_dir,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        player_season_panel_path=args.player_season_panel_path,
        checkpoint_path=args.checkpoint_path,
        max_seasons=args.max_seasons,
    )
    print(
        "Student-t forward RAPM checkpoint saved"
        if run is None
        else f"Student-t forward RAPM: run={run.run_dir}"
    )


if __name__ == "__main__":
    main()
