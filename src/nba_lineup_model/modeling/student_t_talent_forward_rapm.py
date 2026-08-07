"""Recursive forward RAPM with a Student-t talent prior and Gaussian stint errors."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.forward_exposure_gated_rapm import (
    _cold_start_priors,
    _combine_priors,
    _fit_replacement_token,
    _returning_priors,
)
from nba_lineup_model.modeling.prior_rapm import (
    HISTORICAL_SEASONS,
    ForwardLaggedRapmSeason,
    _prior_vector,
)
from nba_lineup_model.modeling.replacement_level import player_exposure_shares
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.modeling.student_t import fit_student_t_coefficient_prior_ridge
from nba_lineup_model.modeling.student_t_forward_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    _baseline_lambda_schedule,
    _evaluate_frozen_priors,
    _load_checkpoint,
    _sha256_file,
    _write_checkpoint,
)
from nba_lineup_model.models.baselines import (
    entity_vocabulary,
    signed_entity_matrix,
    vocabulary_mapping,
)

DEFAULT_DEGREES_OF_FREEDOM = 3.0
DEFAULT_PRIOR_SCALE = 3.0
MODEL_NAME = "student_t_talent_forward_exposure_gated_rapm"
DEFAULT_RANKINGS_PAGE = Path("docs/models/2026-27-student-t-talent-rankings.md")


@dataclass(frozen=True)
class StudentTTalentForwardRapmRun:
    run_dir: Path
    run_id: str
    through_season: str


def train_student_t_talent_forward_rapm(
    *,
    through_season: str = "2025-26",
    degrees_of_freedom: float = DEFAULT_DEGREES_OF_FREEDOM,
    prior_scale: float = DEFAULT_PRIOR_SCALE,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    max_iterations: int = 120,
    checkpoint_path: Path | str | None = None,
    max_seasons: int | None = None,
) -> StudentTTalentForwardRapmRun | None:
    """Fit strict forward Student-t talent RAPM using the Gaussian lambda schedule."""

    if degrees_of_freedom <= 0 or prior_scale <= 0:
        raise ValueError("Student-t talent-prior settings must be positive")
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
        else (roots / "student_t_talent_forward_rapm" / through_season / ".checkpoint.joblib")
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
        print(f"Fitting Student-t talent prior {season}", flush=True)
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
            prior_scale,
            max_iterations,
        )
        results.append(result)
        metadata.update(diagnostics)
        cold_metadata.append(metadata)
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
        raise ValueError("Student-t talent forward run requires a frozen 2025-26 prior")
    evaluation = _evaluate_frozen_priors(
        frozen_priors,
        roots,
        Path(analytical_dir),
        Path(curated_dir),
        player_prior_method="strictly forward Student-t talent-prior RAPM state",
        evaluation_model="frozen_student_t_talent_forward_exposure_gated_rapm",
    )
    run = _write_run(
        through_season=through_season,
        degrees_of_freedom=degrees_of_freedom,
        prior_scale=prior_scale,
        results=results,
        priors=season_priors,
        replacement_tokens=replacement_tokens,
        cold_metadata=cold_metadata,
        frozen_priors=frozen_priors,
        evaluation=evaluation,
        panel=panel,
        artifacts_dir=roots,
    )
    render_student_t_talent_rankings_page(run.run_dir)
    checkpoint.unlink(missing_ok=True)
    return run


def _fit_season(
    season: str,
    stints: pd.DataFrame,
    priors: pd.DataFrame,
    regularization: float,
    degrees_of_freedom: float,
    prior_scale: float,
    max_iterations: int,
) -> tuple[ForwardLaggedRapmSeason, dict[str, object]]:
    player_ids = entity_vocabulary(stints, "home_player_ids", "away_player_ids", multiple=True)
    matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        vocabulary_mapping(player_ids),
        multiple=True,
    )
    prior, prior_frame = _prior_vector(player_ids, priors)
    fit = fit_student_t_coefficient_prior_ridge(
        matrix,
        stints["target_home_net_rating"].to_numpy(dtype=float),
        stints["possessions"].to_numpy(dtype=float),
        prior,
        regularization=regularization,
        degrees_of_freedom=degrees_of_freedom,
        prior_scale=prior_scale,
        max_iterations=max_iterations,
    )
    if not fit.converged:
        raise RuntimeError(
            f"Student-t talent-prior IRLS did not converge for {season} within "
            f"{max_iterations} iterations"
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
            "student_t_talent_iterations": fit.iterations,
            "student_t_talent_converged": fit.converged,
        },
    )


def _write_run(
    *,
    through_season: str,
    degrees_of_freedom: float,
    prior_scale: float,
    results: list[ForwardLaggedRapmSeason],
    priors: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
    cold_metadata: list[dict[str, object]],
    frozen_priors: pd.DataFrame,
    evaluation: object,
    panel: pd.DataFrame,
    artifacts_dir: Path,
) -> StudentTTalentForwardRapmRun:
    now = datetime.now(UTC)
    run_id = (
        f"student-t-talent-forward-rapm-{through_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    root = artifacts_dir / "student_t_talent_forward_rapm" / through_season
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
            "model": MODEL_NAME,
            "through_season": through_season,
            "stint_error_distribution": "Gaussian",
            "coefficient_prior_distribution": "Student-t",
            "degrees_of_freedom": degrees_of_freedom,
            "prior_scale": prior_scale,
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
        return StudentTTalentForwardRapmRun(output, run_id, through_season)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def render_student_t_talent_rankings_page(
    run_dir: Path | str,
    *,
    page_path: Path | str = DEFAULT_RANKINGS_PAGE,
) -> Path:
    """Render the sortable 2026-27 top-100 table from an immutable run."""

    root = Path(run_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Student-t talent rankings require a matching model artifact")
    rankings = pd.read_parquet(root / "next_season_top_100_returning_rankings.parquet")
    required = {
        "rank",
        "player_name",
        "listed_position",
        "rapm",
        "prior_rapm",
        "rapm_adjustment_from_prior",
        "rapm_possessions",
    }
    if required - set(rankings):
        raise ValueError("Student-t talent ranking artifact has an invalid schema")
    page = Path(page_path)
    page.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 2026-27 Student-t Talent-Prior Rankings",
        "",
        "These are the top 100 returning-player preseason values from the completed "
        "2025-26 Gaussian-error RAPM fit with a Student-t talent prior. They are "
        "2026-27 priors, not retrospective 2025-26 evaluations.",
        "",
        "The table is sortable by every column. `2025-26 adjustment` is the "
        "completed-season movement from the prior that entered 2025-26. Low-exposure "
        "players remain in the ranking; interpret their values alongside possessions.",
        "",
        "Source model: [Student-t Talent-Prior RAPM](student-t-talent-forward-rapm.md).",
        f"Immutable artifact: `{root}`.",
        "",
        "## Top 100 Returning Players",
        "",
        "| Rank | 2026-27 RAPM prior | Player | Pos. | 2025-26 preseason prior | "
        "2025-26 adjustment | 2025-26 possessions |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rankings.itertuples(index=False):
        position = "" if pd.isna(row.listed_position) else str(row.listed_position)
        lines.append(
            f"| {int(row.rank)} | {float(row.rapm):+.2f} | {row.player_name} | "
            f"{position} | {float(row.prior_rapm):+.2f} | "
            f"{float(row.rapm_adjustment_from_prior):+.2f} | "
            f"{float(row.rapm_possessions):,.0f} |"
        )
    page.write_text("\n".join(lines) + "\n")
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Student-t talent-prior forward RAPM")
    parser.add_argument("--through-season", default="2025-26")
    parser.add_argument("--degrees-of-freedom", type=float, default=DEFAULT_DEGREES_OF_FREEDOM)
    parser.add_argument("--prior-scale", type=float, default=DEFAULT_PRIOR_SCALE)
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--curated-dir", default=str(DEFAULT_CURATED_DIR))
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--max-seasons", type=int)
    args = parser.parse_args()
    run = train_student_t_talent_forward_rapm(
        through_season=args.through_season,
        degrees_of_freedom=args.degrees_of_freedom,
        prior_scale=args.prior_scale,
        max_iterations=args.max_iterations,
        artifacts_dir=args.artifacts_dir,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        player_season_panel_path=args.player_season_panel_path,
        checkpoint_path=args.checkpoint_path,
        max_seasons=args.max_seasons,
    )
    print(
        "Student-t talent forward RAPM checkpoint saved"
        if run is None
        else f"Student-t talent forward RAPM: run={run.run_dir}"
    )


if __name__ == "__main__":
    main()
