"""Strict rolling one- and three-season RAPM-prior baseline backtests.

Each completed season is fit as prior-centered ridge RAPM.  The only model
input beyond the lineup design matrix is a player RAPM prior made from prior
completed seasons.  No age, box score, draft, exposure gate, or context
features are used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.frozen_multiseason_backtest import _aggregate_metrics
from nba_lineup_model.modeling.frozen_prior_evaluation import (
    _game_prediction_frame,
    _historical_team_seasons,
    _read_playoff_possessions,
    _read_regular_possessions,
    _recover_home_intercept,
    _regular_stint_predictions,
    _team_net_rating_metrics,
    _team_win_evaluation,
    fit_pythagorean_win_model,
    score_frozen_possessions,
    score_possession_cohort,
)
from nba_lineup_model.modeling.prior_rapm import (
    HISTORICAL_SEASONS,
    PRIOR_MEAN_COLUMN,
    ForwardLaggedRapmSeason,
    fit_forward_lagged_rapm_season,
)
from nba_lineup_model.modeling.replacement_level import player_exposure_shares
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.modeling.train import DEFAULT_LAMBDA_GRID
from nba_lineup_model.season.schema import validate_season

DEFAULT_TARGET_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_DOCS_PATH = Path("docs/models/forward-rapm-memory-baselines.md")
MODEL_FAMILY = "forward_rapm_memory_prior_baselines"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RapmMemoryBaseline:
    """A rolling RAPM control defined only by how many seasons it remembers."""

    model: str
    label: str
    memory_seasons: int


BASELINES = (
    RapmMemoryBaseline(
        "forward_one_year_rapm_prior",
        "Forward 1-year RAPM-prior baseline",
        1,
    ),
    RapmMemoryBaseline(
        "forward_three_year_rapm_prior",
        "Forward 3-year RAPM-prior baseline",
        3,
    ),
)


@dataclass(frozen=True)
class ForwardRapmMemoryBaselineRun:
    """Location and immutable id for a completed rolling-control run."""

    run_dir: Path
    run_id: str


def build_rapm_memory_prior(
    history: Sequence[ForwardLaggedRapmSeason],
    exposures_by_season: dict[str, pd.DataFrame],
    *,
    memory_seasons: int,
) -> pd.DataFrame:
    """Return a possession-weighted RAPM prior from completed annual states.

    A player is averaged only over seasons in which the player appeared.  A
    player absent from every remembered season is intentionally absent from the
    table and receives the canonical zero cold-start value at scoring time.
    """

    if memory_seasons not in {1, 3}:
        raise ValueError("RAPM memory baseline supports one or three seasons")
    selected = tuple(history[-memory_seasons:])
    if not selected:
        return pd.DataFrame(
            columns=[
                "player_id",
                PRIOR_MEAN_COLUMN,
                "prior_available",
                "prior_observed_season_count",
                "prior_weight_possessions",
                "prior_source_seasons",
            ]
        )

    frames: list[pd.DataFrame] = []
    for result in selected:
        exposures = exposures_by_season.get(result.season)
        if exposures is None:
            raise ValueError(f"Missing player exposure weights for {result.season}")
        frame = result.player_estimates.loc[:, ["player_id", "rapm"]].merge(
            exposures.loc[:, ["player_id", "on_court_possessions"]],
            on="player_id",
            how="inner",
            validate="one_to_one",
        )
        if len(frame) != len(result.player_estimates):
            raise ValueError(f"Exposure weights do not cover annual RAPM state for {result.season}")
        if not frame["on_court_possessions"].gt(0).all():
            raise ValueError(f"Non-positive annual exposure weight in {result.season}")
        frames.append(frame.assign(source_season=result.season))

    stacked = pd.concat(frames, ignore_index=True)
    weighted = stacked.assign(
        weighted_rapm=stacked["rapm"] * stacked["on_court_possessions"]
    ).groupby("player_id", as_index=False, sort=True).agg(
        weighted_rapm=("weighted_rapm", "sum"),
        prior_weight_possessions=("on_court_possessions", "sum"),
        prior_observed_season_count=("source_season", "nunique"),
        prior_source_seasons=("source_season", lambda values: ",".join(values)),
    )
    weighted[PRIOR_MEAN_COLUMN] = (
        weighted["weighted_rapm"] / weighted["prior_weight_possessions"]
    )
    weighted["prior_available"] = True
    return weighted.loc[
        :,
        [
            "player_id",
            PRIOR_MEAN_COLUMN,
            "prior_available",
            "prior_observed_season_count",
            "prior_weight_possessions",
            "prior_source_seasons",
        ],
    ].sort_values("player_id", kind="stable").reset_index(drop=True)


def train_forward_rapm_memory_baselines(
    *,
    target_seasons: Sequence[str] = DEFAULT_TARGET_SEASONS,
    through_season: str | None = None,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    docs_path: Path | str | None = DEFAULT_DOCS_PATH,
) -> ForwardRapmMemoryBaselineRun:
    """Fit rolling RAPM-only chains and score fixed pre-season forecasts.

    The target-season forecast is materialized before that season's annual
    RAPM fit.  The fitted target state is used only by a later season, which
    enforces the chronological information boundary.
    """

    targets = _validate_target_seasons(target_seasons)
    final_season = validate_season(through_season or targets[-1])
    if int(final_season[:4]) < int(targets[-1][:4]):
        raise ValueError("through_season must reach the final evaluation season")
    seasons = _training_seasons_through(final_season)
    analytical_root = Path(analytical_dir)
    curated_root = Path(curated_dir)
    stints_by_season = {
        season: read_rapm_stints(season, analytical_dir=analytical_root) for season in seasons
    }
    exposures_by_season = {
        season: player_exposure_shares(stints) for season, stints in stints_by_season.items()
    }

    evaluations: list[dict[str, object]] = []
    coefficient_frames: list[pd.DataFrame] = []
    prior_frames: list[pd.DataFrame] = []
    cv_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    season_metadata_rows: list[dict[str, object]] = []
    for baseline in BASELINES:
        LOGGER.info(
            "Starting %s (%d-season RAPM memory) through %s",
            baseline.label,
            baseline.memory_seasons,
            final_season,
        )
        history: list[ForwardLaggedRapmSeason] = []
        for season in seasons:
            prior = build_rapm_memory_prior(
                history,
                exposures_by_season,
                memory_seasons=baseline.memory_seasons,
            )
            if season in targets:
                evaluations.append(
                    _evaluate_target_season(
                        baseline,
                        target_season=season,
                        player_priors=prior,
                        source_result=history[-1],
                        source_stints=stints_by_season[history[-1].season],
                        analytical_dir=analytical_root,
                        curated_dir=curated_root,
                    )
                )
            result = fit_forward_lagged_rapm_season(
                season,
                stints_by_season[season],
                prior,
                lambda_grid=lambda_grid,
            )
            LOGGER.info(
                "%s: fitted %s; lambda=%.6g; prior_players=%d; fitted_players=%d",
                baseline.label,
                season,
                result.selected_lambda,
                len(prior),
                len(result.player_estimates),
            )
            history.append(result)
            coefficient_frames.append(
                result.player_estimates.assign(
                    model=baseline.model,
                    label=baseline.label,
                    memory_seasons=baseline.memory_seasons,
                )
            )
            if not prior.empty:
                prior_frames.append(
                    prior.assign(
                        season=season,
                        model=baseline.model,
                        label=baseline.label,
                        memory_seasons=baseline.memory_seasons,
                    )
                )
            cv_frames.append(
                result.cv_results.assign(
                    season=season,
                    model=baseline.model,
                    label=baseline.label,
                    memory_seasons=baseline.memory_seasons,
                    selected_lambda=result.selected_lambda,
                )
            )
            exposure_frames.append(
                exposures_by_season[season].assign(
                    season=season,
                    model=baseline.model,
                    memory_seasons=baseline.memory_seasons,
                )
            )
            season_metadata_rows.append(
                {
                    "model": baseline.model,
                    "label": baseline.label,
                    "memory_seasons": baseline.memory_seasons,
                    "season": season,
                    "selected_lambda": result.selected_lambda,
                    "prior_player_count": len(prior),
                    "fitted_player_count": len(result.player_estimates),
                    "season_stint_count": len(stints_by_season[season]),
                    "season_possessions": float(stints_by_season[season]["possessions"].sum()),
                    "target_season_forecast_materialized_before_fit": season in targets,
                }
            )

    outputs = _collect_evaluations(evaluations)
    outputs["historical_player_coefficients"] = pd.concat(
        coefficient_frames, ignore_index=True
    )
    outputs["season_player_priors"] = pd.concat(prior_frames, ignore_index=True)
    outputs["season_cv_results"] = pd.concat(cv_frames, ignore_index=True)
    outputs["season_exposure_weights"] = pd.concat(exposure_frames, ignore_index=True)
    outputs["season_model_metadata"] = pd.DataFrame(season_metadata_rows)
    run = _write_run(
        outputs=outputs,
        targets=targets,
        final_season=final_season,
        lambda_grid=lambda_grid,
        artifacts_dir=Path(artifacts_dir),
    )
    if docs_path is not None:
        _update_docs(Path(docs_path), outputs=outputs, run=run)
    return run


def _evaluate_target_season(
    baseline: RapmMemoryBaseline,
    *,
    target_season: str,
    player_priors: pd.DataFrame,
    source_result: ForwardLaggedRapmSeason,
    source_stints: pd.DataFrame,
    analytical_dir: Path,
    curated_dir: Path,
) -> dict[str, object]:
    """Score a target season before fitting its annual update."""

    source_season = source_result.season
    scoring_priors = player_priors.rename(
        columns={PRIOR_MEAN_COLUMN: "prior_rapm_mean"}
    )
    source_coefficients = source_result.player_estimates.loc[:, ["player_id", "rapm"]]
    source_home_intercept = _recover_home_intercept(source_stints, source_coefficients)
    source_possessions, _ = _read_regular_possessions(
        source_season, analytical_dir=analytical_dir, curated_dir=curated_dir
    )
    source_mean = float(source_possessions["target_offense_margin"].mean())
    regular_possessions, _ = _read_regular_possessions(
        target_season, analytical_dir=analytical_dir, curated_dir=curated_dir
    )
    regular_predictions = score_frozen_possessions(
        regular_possessions,
        scoring_priors,
        source_mean=source_mean,
        source_home_intercept=source_home_intercept,
        cohort="regular_season",
    )
    regular_stints = read_rapm_stints(target_season, analytical_dir=analytical_dir)
    regular_games, team_net_ratings = _regular_stint_predictions(
        regular_stints,
        scoring_priors,
        source_home_intercept=source_home_intercept,
    )
    pythagorean = fit_pythagorean_win_model(
        _historical_team_seasons(analytical_dir=analytical_dir, through_season=source_season)
    )
    team_wins, team_win_metrics = _team_win_evaluation(
        regular_games,
        team_net_ratings,
        pythagorean,
        model=baseline.model,
    )
    predictions = [regular_predictions]
    cohort_metrics = [
        score_possession_cohort(
            regular_predictions, source_mean=source_mean, model=baseline.model
        )
    ]
    playoff_available = _playoff_partition_exists(target_season, curated_dir)
    if playoff_available:
        playoff_possessions, _ = _read_playoff_possessions(target_season, curated_dir)
        playoff_predictions = score_frozen_possessions(
            playoff_possessions,
            scoring_priors,
            source_mean=source_mean,
            source_home_intercept=source_home_intercept,
            cohort="playoffs",
        )
        predictions.append(playoff_predictions)
        cohort_metrics.append(
            score_possession_cohort(
                playoff_predictions, source_mean=source_mean, model=baseline.model
            )
        )
    return {
        "baseline": baseline,
        "target_season": target_season,
        "source_state": {
            "target_season": target_season,
            "source_season": source_season,
            "source_offense_margin_mean": source_mean,
            "source_home_intercept_net_rating": source_home_intercept,
            "source_selected_lambda": source_result.selected_lambda,
            "target_season_refit": False,
            "target_regular_outcomes_used_for_forecast": False,
            "target_playoff_outcomes_used_for_forecast": False,
            "target_regular_outcomes_used_for_next_state": True,
            "prior_memory_seasons": baseline.memory_seasons,
            "oracle_information": "realized target-season regular-season lineups and exposure only",
            "playoffs_evaluated": playoff_available,
            "playoffs_exclusion_reason": (
                None if playoff_available else "historical playoff possession partition unavailable"
            ),
        },
        "frozen_player_priors": player_priors,
        "cohort_metrics": pd.concat(cohort_metrics, ignore_index=True),
        "possession_predictions": pd.concat(predictions, ignore_index=True),
        "game_predictions": pd.concat(
            [_game_prediction_frame(frame) for frame in predictions], ignore_index=True
        ),
        "regular_game_predictions": regular_games,
        "team_net_rating_predictions": team_net_ratings,
        "team_net_rating_metrics": _team_net_rating_metrics(
            team_net_ratings, model=baseline.model
        ),
        "team_win_predictions": team_wins,
        "team_win_metrics": team_win_metrics,
        "pythagorean_calibration_team_seasons": _historical_team_seasons(
            analytical_dir=analytical_dir, through_season=source_season
        ),
    }


def _collect_evaluations(evaluations: Sequence[dict[str, object]]) -> dict[str, pd.DataFrame]:
    keys = (
        "source_states",
        "frozen_player_priors",
        "cohort_metrics",
        "possession_predictions",
        "game_predictions",
        "regular_game_predictions",
        "team_net_rating_predictions",
        "team_net_rating_metrics",
        "team_win_predictions",
        "team_win_metrics",
        "pythagorean_calibration_team_seasons",
    )
    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in keys}
    for evaluation in evaluations:
        baseline = evaluation["baseline"]
        target = str(evaluation["target_season"])
        if not isinstance(baseline, RapmMemoryBaseline):
            raise TypeError("RAPM memory evaluation lacks baseline metadata")
        for key in keys:
            source_key = "source_state" if key == "source_states" else key
            value = evaluation[source_key]
            frame = pd.DataFrame([value]) if isinstance(value, dict) else value.copy()
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"RAPM memory evaluation has invalid {key} output")
            frame = frame.drop(columns=["season", "model", "label"], errors="ignore")
            frame.insert(0, "season", target)
            frame.insert(1, "model", baseline.model)
            frame.insert(2, "label", baseline.label)
            frames[key].append(frame)
    output = {key: pd.concat(value, ignore_index=True) for key, value in frames.items()}
    output["aggregate_metrics"] = _aggregate_metrics(output)
    return output


def _write_run(
    *,
    outputs: dict[str, pd.DataFrame],
    targets: tuple[str, ...],
    final_season: str,
    lambda_grid: tuple[float, ...],
    artifacts_dir: Path,
) -> ForwardRapmMemoryBaselineRun:
    now = datetime.now(UTC)
    run_id = (
        f"forward-rapm-memory-baselines-{targets[0]}-to-{targets[-1]}-"
        f"{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    root = artifacts_dir / MODEL_FAMILY / f"{targets[0]}_to_{targets[-1]}"
    run_dir = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for name, frame in outputs.items():
            frame.to_parquet(temporary / f"{name}.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": now.isoformat(),
            "model_family": MODEL_FAMILY,
            "models": [baseline.model for baseline in BASELINES],
            "target_seasons": list(targets),
            "through_season": final_season,
            "season_type": "regular",
            "lambda_grid": list(lambda_grid),
            "prior_definitions": {
                "forward_one_year_rapm_prior": "immediately preceding completed annual RAPM",
                "forward_three_year_rapm_prior": (
                    "possession-weighted player-specific mean of the preceding up to three "
                    "completed annual RAPM estimates"
                ),
            },
            "cold_start_prior": 0.0,
            "excluded_features": ["age", "box_score", "draft", "exposure_gate", "context"],
            "information_boundary": (
                "each target forecast uses only player estimates from completed seasons before "
                "the target; target outcomes update only later season priors"
            ),
            "oracle_information": "realized target-season regular-season lineup allocation",
            "historical_playoff_training": False,
            "historical_playoff_evaluation": "2025-26 only; earlier target partitions unavailable",
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
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
            json.dumps({**metadata, "artifacts": records}, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(run_dir)
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return ForwardRapmMemoryBaselineRun(run_dir=run_dir, run_id=run_id)


def _update_docs(
    path: Path,
    *,
    outputs: dict[str, pd.DataFrame],
    run: ForwardRapmMemoryBaselineRun,
) -> None:
    start = "<!-- forward-rapm-memory-results:start -->"
    end = "<!-- forward-rapm-memory-results:end -->"
    aggregate = outputs["aggregate_metrics"]
    regular = aggregate.loc[aggregate["scope"].eq("pooled_regular_season")].copy()
    full = aggregate.loc[
        aggregate["scope"].eq("pooled_regular_full_games_and_teams")
    ].copy()
    full_by_model = full.set_index("model")
    best = {
        "possession_rmse": regular["possession_rmse"].min(),
        "eligible_game_margin_rmse": regular["eligible_game_margin_rmse"].min(),
        "full_game_margin_rmse": full["full_game_margin_rmse"].min(),
        "game_winner_accuracy": full["game_winner_accuracy"].max(),
        "team_net_rating_rmse": full["team_net_rating_rmse"].min(),
        "pythagorean_win_rmse": full["pythagorean_win_rmse"].min(),
    }
    lines = [
        start,
        "## Results",
        "",
        f"Artifact: `{run.run_dir}`.",
        "",
        (
            "The regular-season pool combines 2023-24, 2024-25, and 2025-26. "
            "Lower is better except winner accuracy."
        ),
        "",
        (
            "| Model | Possession RMSE | Eligible game RMSE | Full-game RMSE | "
            "Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order_column = "memory_seasons" if "memory_seasons" in regular else "label"
    for row in regular.sort_values(order_column).itertuples(index=False):
        companion = full_by_model.loc[row.model]
        lines.append(
            (
                "| {label} | {possession} | {eligible} | {full_game} | "
                "{winner} | {team} | {wins} |"
            ).format(
                label=row.label,
                possession=_format_metric(row.possession_rmse, best["possession_rmse"], 6),
                eligible=_format_metric(
                    row.eligible_game_margin_rmse, best["eligible_game_margin_rmse"], 4
                ),
                full_game=_format_metric(
                    companion.full_game_margin_rmse, best["full_game_margin_rmse"], 4
                ),
                winner=_format_metric(
                    companion.game_winner_accuracy, best["game_winner_accuracy"], 2, percent=True
                ),
                team=_format_metric(
                    companion.team_net_rating_rmse, best["team_net_rating_rmse"], 4
                ),
                wins=_format_metric(
                    companion.pythagorean_win_rmse, best["pythagorean_win_rmse"], 4
                ),
            )
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
    seasons = outputs["cohort_metrics"].loc[
        outputs["cohort_metrics"]["cohort"].eq("regular_season")
    ].sort_values(["season", "label"], kind="stable")
    for row in seasons.itertuples(index=False):
        lines.append(
            f"| {row.season} | {row.label} | {row.possession_rmse:.6f} | "
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
                (
                    "Each playoff cohort is scored from the matching frozen pre-season state; "
                    "playoff outcomes never enter the fit."),
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
        raise ValueError(f"Forward RAPM baseline result marker missing from {path}")
    _, end_marker, tail = after.partition(end)
    if not end_marker:
        raise ValueError(f"Forward RAPM baseline result end marker missing from {path}")
    path.write_text(before + "\n".join(lines) + tail)


def _training_seasons_through(final_season: str) -> tuple[str, ...]:
    seasons = tuple(
        season
        for season in (*HISTORICAL_SEASONS, "2025-26")
        if int(season[:4]) <= int(final_season[:4])
    )
    if not seasons or seasons[-1] != final_season:
        raise ValueError(f"No continuous regular-season RAPM history through {final_season}")
    return seasons


def _validate_target_seasons(seasons: Sequence[str]) -> tuple[str, ...]:
    validated = tuple(validate_season(str(season)) for season in seasons)
    if len(validated) != 3 or len(set(validated)) != len(validated):
        raise ValueError("Forward RAPM baseline requires exactly three distinct target seasons")
    if tuple(sorted(validated, key=lambda season: int(season[:4]))) != validated:
        raise ValueError("Target seasons must be chronological")
    return validated


def _playoff_partition_exists(season: str, curated_dir: Path) -> bool:
    return (curated_dir / "possession_segments" / season / "playoffs" / "_manifest.json").is_file()


def _format_metric(value: float, best: float, precision: int, *, percent: bool = False) -> str:
    text = f"{value:.{precision}%}" if percent else f"{value:.{precision}f}"
    return f"**{text}**" if np.isclose(value, best) else text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate rolling one- and three-year RAPM-prior baselines"
    )
    parser.add_argument("--targets", nargs=3, default=DEFAULT_TARGET_SEASONS)
    parser.add_argument("--through-season", default=None)
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--curated-dir", default=str(DEFAULT_CURATED_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--no-docs", action="store_true")
    return parser


def main() -> None:
    """CLI entry point for the strict rolling RAPM-only controls."""

    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = train_forward_rapm_memory_baselines(
        target_seasons=tuple(args.targets),
        through_season=args.through_season,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        docs_path=None if args.no_docs else DEFAULT_DOCS_PATH,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run.run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(f"Forward RAPM memory baselines: run={run.run_dir}{tracking_text}")


if __name__ == "__main__":  # pragma: no cover
    main()
