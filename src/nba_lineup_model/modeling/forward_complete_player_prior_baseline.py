"""Recovered-coverage, no-context player-prior RAPM baseline.

This is the complete HIPSTER PM player-prior system with context and box-score
residual branches removed. It reruns all seasonal precursor models forward:
replacement tokens, rookie draft rates, exposure gates, aging transitions,
value conditioning, and possession-weighted centering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.modeling.forward_aging_player_prior import (
    build_centered_value_conditioned_aging_exposure_gated_priors,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _fit_replacement_token
from nba_lineup_model.modeling.forward_rapm_memory_baselines import (
    DEFAULT_TARGET_SEASONS,
    RapmMemoryBaseline,
    _collect_evaluations,
    _evaluate_target_season,
    _playoff_partition_exists,
    _validate_target_seasons,
)
from nba_lineup_model.modeling.prior_rapm import (
    HISTORICAL_SEASONS,
    PRIOR_MEAN_COLUMN,
    ForwardLaggedRapmSeason,
    fit_forward_lagged_rapm_season,
)
from nba_lineup_model.modeling.replacement_level import player_exposure_shares
from nba_lineup_model.modeling.train import DEFAULT_LAMBDA_GRID
from nba_lineup_model.season.schema import validate_season

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_DOCS_PATH = Path("docs/models/complete-player-prior-baseline.md")
MODEL_NAME = "forward_complete_player_prior_rapm"
RUN_PREFIX = "forward-complete-player-prior-rapm"
LABEL = "Complete player-prior RAPM (no context, no box score)"


@dataclass(frozen=True)
class CompletePlayerPriorBaselineRun:
    """Location and immutable id for a completed player-prior control run."""

    run_dir: Path
    run_id: str


def train_complete_player_prior_baseline(
    *,
    target_seasons: tuple[str, ...] = DEFAULT_TARGET_SEASONS,
    through_season: str | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    docs_path: Path | str | None = DEFAULT_DOCS_PATH,
) -> CompletePlayerPriorBaselineRun:
    """Rebuild and evaluate the full forward player-prior state without context."""

    targets = _validate_target_seasons(target_seasons)
    final_season = validate_season(through_season or targets[-1])
    if int(final_season[:4]) < int(targets[-1][:4]):
        raise ValueError("through_season must reach the final evaluation season")
    seasons = _seasons_through(final_season)
    panel = pd.read_parquet(player_season_panel_path)
    analytical_root = Path(analytical_dir)
    curated_root = Path(curated_dir)
    from nba_lineup_model.modeling.stints import read_rapm_stints

    stints_by_season = {
        season: read_rapm_stints(season, analytical_dir=analytical_root) for season in seasons
    }
    candidate = RapmMemoryBaseline(MODEL_NAME, LABEL, 1)
    results: list[ForwardLaggedRapmSeason] = []
    exposure_history: list[pd.DataFrame] = []
    replacement_tokens: list[dict[str, object]] = []
    evaluations: list[dict[str, object]] = []
    prior_frames: list[pd.DataFrame] = []
    prior_metadata: list[dict[str, object]] = []
    coefficient_frames: list[pd.DataFrame] = []
    cv_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    aging_curve_frames: list[pd.DataFrame] = []
    season_rows: list[dict[str, object]] = []

    for season in seasons:
        print(f"Fitting complete player-prior state for {season}", flush=True)
        priors, metadata = build_centered_value_conditioned_aging_exposure_gated_priors(
            season=season,
            panel=panel,
            completed_results=results,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
        )
        metadata = dict(metadata)
        aging_curve = metadata.pop("_aging_curve_grid", None)
        metadata.pop("_aging_model", None)
        if isinstance(aging_curve, pd.DataFrame):
            aging_curve_frames.append(aging_curve.assign(season=season))
        if season in targets:
            if not results:
                raise ValueError("Target forecast requires a completed source state")
            evaluations.append(
                _evaluate_target_season(
                    candidate,
                    target_season=season,
                    player_priors=priors,
                    source_result=results[-1],
                    source_stints=stints_by_season[results[-1].season],
                    analytical_dir=analytical_root,
                    curated_dir=curated_root,
                )
            )
        fitted = fit_forward_lagged_rapm_season(
            season,
            stints_by_season[season],
            priors,
            lambda_grid=lambda_grid,
        )
        results.append(fitted)
        exposure = player_exposure_shares(stints_by_season[season])
        exposure_with_bios = panel.loc[panel["season"].eq(season)].merge(
            exposure, on="player_id", how="inner", validate="one_to_one"
        )
        exposure_history.append(exposure_with_bios)
        token = _fit_replacement_token(
            season,
            stints_by_season[season],
            exposure,
            fitted,
            panel,
        )
        replacement_tokens.append(token)
        prior_frames.append(
            priors.rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm"}).assign(
                season=season,
                model=MODEL_NAME,
                label=LABEL,
            )
        )
        prior_metadata.append({"season": season, **metadata})
        coefficient_frames.append(
            fitted.player_estimates.assign(model=MODEL_NAME, label=LABEL)
        )
        cv_frames.append(fitted.cv_results.assign(season=season, model=MODEL_NAME, label=LABEL))
        exposure_frames.append(exposure_with_bios.assign(model=MODEL_NAME))
        season_rows.append(
            {
                "season": season,
                "model": MODEL_NAME,
                "label": LABEL,
                "selected_lambda": fitted.selected_lambda,
                "prior_player_count": len(priors),
                "fitted_player_count": len(fitted.player_estimates),
                "stint_count": len(stints_by_season[season]),
                "possessions": float(stints_by_season[season]["possessions"].sum()),
                "target_forecast_materialized_before_fit": season in targets,
                "replacement_token_rapm": token["replacement_token_rapm"],
            }
        )

    outputs = _collect_evaluations(evaluations)
    outputs.update(
        {
            "historical_player_coefficients": pd.concat(coefficient_frames, ignore_index=True),
            "season_player_priors": pd.concat(prior_frames, ignore_index=True),
            "season_player_prior_metadata": pd.DataFrame(prior_metadata),
            "season_replacement_tokens": pd.DataFrame(replacement_tokens),
            "season_exposure_history": pd.concat(exposure_frames, ignore_index=True),
            "season_cv_results": pd.concat(cv_frames, ignore_index=True),
            "season_model_metadata": pd.DataFrame(season_rows),
            "season_aging_curve_grid": (
                pd.concat(aging_curve_frames, ignore_index=True)
                if aging_curve_frames
                else pd.DataFrame()
            ),
        }
    )
    run = _write_run(
        outputs=outputs,
        targets=targets,
        final_season=final_season,
        lambda_grid=lambda_grid,
        artifacts_dir=Path(artifacts_dir),
    )
    if docs_path is not None:
        _update_docs(
            Path(docs_path),
            outputs=outputs,
            run=run,
            curated_dir=curated_root,
        )
    return run


def _seasons_through(final_season: str) -> tuple[str, ...]:
    seasons = tuple(
        season
        for season in (*HISTORICAL_SEASONS, "2025-26")
        if int(season[:4]) <= int(final_season[:4])
    )
    if not seasons or seasons[-1] != final_season:
        raise ValueError(f"No continuous RAPM stint history through {final_season}")
    return seasons


def _write_run(
    *,
    outputs: dict[str, pd.DataFrame],
    targets: tuple[str, ...],
    final_season: str,
    lambda_grid: tuple[float, ...],
    artifacts_dir: Path,
) -> CompletePlayerPriorBaselineRun:
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}-{targets[0]}-to-{targets[-1]}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / MODEL_NAME / f"{targets[0]}_to_{targets[-1]}"
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for name, frame in outputs.items():
            frame.to_parquet(temporary / f"{name}.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "label": LABEL,
            "created_at": now.isoformat(),
            "target_seasons": list(targets),
            "through_season": final_season,
            "lambda_grid": list(lambda_grid),
            "context_enabled": False,
            "box_score_prior_enabled": False,
            "historical_playoff_training": False,
            "player_prior_components": [
                "lagged annual RAPM",
                "value-conditioned aging transition",
                "draft and physical profile features",
                "exposure-gated rookie cold start",
                "replacement-token mixture",
                "prior-season possession-weighted centering",
            ],
            "information_boundary": (
                "every target forecast is written before target outcomes; all prior components "
                "are refit from earlier completed recovered regular seasons only"
            ),
            "historical_playoff_evaluation": "2025-26 only; earlier target partitions unavailable",
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
        temporary.replace(output)
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return CompletePlayerPriorBaselineRun(run_dir=output, run_id=run_id)


def _update_docs(
    path: Path,
    *,
    outputs: dict[str, pd.DataFrame],
    run: CompletePlayerPriorBaselineRun,
    curated_dir: Path,
) -> None:
    start = "<!-- complete-player-prior-results:start -->"
    end = "<!-- complete-player-prior-results:end -->"
    aggregate = outputs["aggregate_metrics"]
    regular = aggregate.loc[aggregate["scope"].eq("pooled_regular_season")].iloc[0]
    full = aggregate.loc[
        aggregate["scope"].eq("pooled_regular_full_games_and_teams")
    ].iloc[0]
    lines = [
        start,
        "## Recovered-Coverage Results",
        "",
        f"Artifact: `{run.run_dir}`.",
        "",
        (
            "| Possession RMSE | Eligible game RMSE | Full-game RMSE | Winner accuracy | "
            "Team NetRtg RMSE | Pythagorean-win RMSE |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {regular.possession_rmse:.6f} | {regular.eligible_game_margin_rmse:.4f} | "
            f"{full.full_game_margin_rmse:.4f} | {full.game_winner_accuracy:.2%} | "
            f"{full.team_net_rating_rmse:.4f} | {full.pythagorean_win_rmse:.4f} |"
        ),
        "",
        "### Per-Season Regular Results",
        "",
        "| Season | Possession RMSE | Eligible game RMSE |",
        "| --- | ---: | ---: |",
    ]
    seasons = outputs["cohort_metrics"].loc[
        outputs["cohort_metrics"]["cohort"].eq("regular_season")
    ].sort_values("season", kind="stable")
    for row in seasons.itertuples(index=False):
        lines.append(
            f"| {row.season} | {row.possession_rmse:.6f} | "
            f"{row.eligible_possession_game_margin_rmse:.4f} |"
        )
    playoff_metrics = outputs["cohort_metrics"].loc[
        outputs["cohort_metrics"]["cohort"].eq("playoffs")
    ]
    if not playoff_metrics.empty:
        lines.extend(
            [
                "",
                "### Frozen Playoff Check",
                "",
                "| Season | Possession RMSE | Eligible game RMSE |",
                "| --- | ---: | ---: |",
            ]
        )
        for playoff in playoff_metrics.sort_values("season", kind="stable").itertuples(index=False):
            lines.append(
                f"| {playoff.season} | {playoff.possession_rmse:.6f} | "
                f"{playoff.eligible_possession_game_margin_rmse:.4f} |"
            )
    lines.extend([end, ""])
    content = path.read_text()
    before, marker, after = content.partition(start)
    if not marker:
        raise ValueError(f"Player-prior result marker missing from {path}")
    _, end_marker, tail = after.partition(end)
    if not end_marker:
        raise ValueError(f"Player-prior result end marker missing from {path}")
    path.write_text(before + "\n".join(lines) + tail)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a complete no-context, no-box-score player-prior RAPM baseline"
    )
    parser.add_argument("--targets", nargs=3, default=DEFAULT_TARGET_SEASONS)
    parser.add_argument("--through-season", default=None)
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--curated-dir", default=str(DEFAULT_CURATED_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--no-docs", action="store_true")
    return parser


def main() -> None:
    """CLI entry point for the complete no-context player-prior baseline."""

    args = _build_parser().parse_args()
    run = train_complete_player_prior_baseline(
        target_seasons=tuple(args.targets),
        through_season=args.through_season,
        player_season_panel_path=args.player_season_panel_path,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        docs_path=None if args.no_docs else DEFAULT_DOCS_PATH,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run.run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(f"Complete player-prior baseline: run={run.run_dir}{tracking_text}")


if __name__ == "__main__":  # pragma: no cover
    main()
