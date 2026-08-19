"""Fixed-raw context Ridge grid for NAIL-RAPM v1.1."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_nail_context_regularization import (
    model_name_for_raw_context_alpha,
    train_nail_fixed_context_regularization,
)
from nba_lineup_model.modeling.forward_nail_profile_padding import PUBLISHED_MODEL_NAME
from nba_lineup_model.modeling.frozen_model_tournament import _paired_metrics
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)
from nba_lineup_model.modeling.nail_context_regularization_study import (
    DEFAULT_VALIDATION_SEASONS,
    RECENT_SEASON_COUNT,
)

MODEL_NAME = "nail_fixed_context_regularization_study"
RAW_CONTEXT_ALPHAS = (1_000.0, 5_000.0, 10_000.0, 20_000.0)
TRAINING_ALPHAS = (1_000.0, 5_000.0, 20_000.0)
ROLLING_OUTPUT_ROOT = Path("artifacts/models/nail_fixed_context_regularization_rolling")
FROZEN_OUTPUT_ROOT = Path("artifacts/models/nail_fixed_context_regularization_frozen")
BOOTSTRAP_OUTPUT_ROOT = Path(
    "artifacts/models/nail_fixed_context_regularization_bootstrap/2023-24_to_2025-26"
)
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20_260_819


@dataclass(frozen=True)
class FixedContextSelectionRun:
    run_dir: Path
    run_id: str
    selected_raw_alpha: float


def candidate_model_name(raw_alpha: float) -> str:
    if raw_alpha == 10_000.0:
        return PUBLISHED_MODEL_NAME
    return model_name_for_raw_context_alpha(raw_alpha)


def candidate_label(raw_alpha: float) -> str:
    return f"NAIL-RAPM v1.1 fixed raw alpha={raw_alpha:,.0f}"


def train_fixed_context_candidates(
    *,
    raw_alphas: tuple[float, ...] = TRAINING_ALPHAS,
    through_season: str = "2025-26",
    evaluate_target: bool = False,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> tuple[Path, ...]:
    """Train independently named fixed-alpha recursive candidates."""

    outputs: list[Path] = []
    for raw_alpha in raw_alphas:
        if raw_alpha == 10_000.0:
            raise ValueError("The published alpha=10,000 artifact already exists")
        run = train_nail_fixed_context_regularization(
            context_alpha=raw_alpha,
            through_season=through_season,
            evaluate_target=evaluate_target,
            artifacts_dir=artifacts_dir,
        )
        outputs.append(run.run_dir)
        print(f"Fixed raw context alpha={raw_alpha:,.0f}: run={run.run_dir}", flush=True)
    return tuple(outputs)


def run_fixed_context_selection(
    *,
    validation_seasons: tuple[str, ...] = DEFAULT_VALIDATION_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> FixedContextSelectionRun:
    """Select a fixed raw alpha using only seasons ending by 2022-23."""

    root = Path(artifacts_dir)
    replay = run_frozen_multiseason_backtest(
        seasons=validation_seasons,
        models=_backtest_models(),
        artifacts_dir=root,
        output_artifacts_dir=ROLLING_OUTPUT_ROOT,
        docs_path=None,
        score_possessions=False,
    )
    games = pd.read_parquet(replay.run_dir / "regular_game_predictions.parquet")
    season_metrics = _season_metrics(games)
    selection = _selection_summary(season_metrics)
    selected = selection.loc[selection["selected_by_one_se_rule"]].iloc[0]
    return _write_selection_run(
        season_metrics=season_metrics,
        selection=selection,
        validation_seasons=validation_seasons,
        replay_run_dir=replay.run_dir,
        selected_raw_alpha=float(selected["raw_context_alpha"]),
        artifacts_dir=root,
    )


def run_fixed_context_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    """Replay every fixed-alpha candidate on the three frozen seasons."""

    return run_frozen_multiseason_backtest(
        seasons=seasons,
        models=_backtest_models(),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=FROZEN_OUTPUT_ROOT,
        docs_path=None,
    )


def run_fixed_context_bootstrap(
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Bootstrap each fixed-alpha candidate against published alpha=10,000."""

    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    frozen_root = FROZEN_OUTPUT_ROOT / "frozen_multiseason_backtest" / "2023-24_to_2025-26"
    source_run = _latest_directory(frozen_root)
    games = pd.read_parquet(source_run / "regular_game_predictions.parquet")
    possessions = pd.read_parquet(source_run / "possession_predictions.parquet")

    def source(model: str) -> dict[str, pd.DataFrame]:
        return {
            "games": games.loc[games["model"].eq(model)].copy(),
            "possessions": possessions.loc[
                possessions["model"].eq(model)
                & possessions["cohort"].eq("regular_season")
            ].copy(),
        }

    incumbent = source(PUBLISHED_MODEL_NAME)
    rows: list[pd.DataFrame] = []
    for raw_alpha in RAW_CONTEXT_ALPHAS:
        if raw_alpha == 10_000.0:
            continue
        challenger = candidate_model_name(raw_alpha)
        metrics = _paired_metrics(
            incumbent,
            source(challenger),
            draws=draws,
            seed=seed + int(raw_alpha),
        )
        metrics.insert(0, "raw_context_alpha", raw_alpha)
        metrics.insert(1, "incumbent_model", PUBLISHED_MODEL_NAME)
        metrics.insert(2, "challenger_model", challenger)
        rows.append(metrics)
    output = pd.concat(rows, ignore_index=True)
    run_id = (
        f"nail-fixed-context-bootstrap-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-"
        f"{uuid4().hex[:8]}"
    )
    run_dir = BOOTSTRAP_OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output.to_parquet(run_dir / "paired_bootstrap_metrics.parquet", index=False)
    metadata = {
        "run_id": run_id,
        "source_run_dir": str(source_run),
        "raw_context_alphas": list(RAW_CONTEXT_ALPHAS),
        "draws": draws,
        "seed": seed,
        "resampling_unit": "games stratified by season",
        "created_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    BOOTSTRAP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (BOOTSTRAP_OUTPUT_ROOT / "latest.json").write_text(
        json.dumps({"run_id": run_id}, indent=2) + "\n"
    )
    return run_dir


def _backtest_models() -> tuple[BacktestModel, ...]:
    builder = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
    )
    return tuple(
        BacktestModel(
            candidate_model_name(raw_alpha),
            candidate_label(raw_alpha),
            profile_builder=builder,
            run_target_season="2025-26",
        )
        for raw_alpha in RAW_CONTEXT_ALPHAS
    )


def _season_metrics(games: pd.DataFrame) -> pd.DataFrame:
    alpha_by_model = {
        candidate_model_name(raw_alpha): raw_alpha for raw_alpha in RAW_CONTEXT_ALPHAS
    }
    rows: list[dict[str, object]] = []
    for (model, label, season), frame in games.groupby(
        ["model", "label", "season"], sort=False
    ):
        error = frame["margin_error"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "label": label,
                "season": season,
                "raw_context_alpha": alpha_by_model[str(model)],
                "game_count": len(frame),
                "full_game_margin_mse": float(np.mean(np.square(error))),
                "full_game_margin_rmse": float(np.sqrt(np.mean(np.square(error)))),
                "full_game_margin_mae": float(np.mean(np.abs(error))),
                "game_winner_accuracy": float(
                    np.mean(frame["actual_home_win"] == frame["predicted_home_win"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["raw_context_alpha", "season"], kind="stable"
    )


def _selection_summary(season_metrics: pd.DataFrame) -> pd.DataFrame:
    recent = set(sorted(season_metrics["season"].unique())[-RECENT_SEASON_COUNT:])
    rows: list[dict[str, object]] = []
    for (model, label, raw_alpha), frame in season_metrics.groupby(
        ["model", "label", "raw_context_alpha"], sort=False
    ):
        recent_frame = frame.loc[frame["season"].isin(recent)]
        rows.append(
            {
                "model": model,
                "label": label,
                "raw_context_alpha": float(raw_alpha),
                "season_count": len(frame),
                "mean_season_full_game_mse": float(frame["full_game_margin_mse"].mean()),
                "equal_season_full_game_rmse": float(
                    np.sqrt(frame["full_game_margin_mse"].mean())
                ),
                "recent_equal_season_full_game_rmse": float(
                    np.sqrt(recent_frame["full_game_margin_mse"].mean())
                ),
                "mean_winner_accuracy": float(frame["game_winner_accuracy"].mean()),
            }
        )
    output = pd.DataFrame(rows)
    best = output.sort_values("mean_season_full_game_mse", kind="stable").iloc[0]
    best_model = str(best["model"])
    paired = season_metrics.pivot(
        index="season", columns="model", values="full_game_margin_mse"
    )
    best_mse = paired[best_model]
    paired_stats: dict[str, tuple[float, float]] = {}
    for model in paired.columns:
        delta = (paired[model] - best_mse).dropna().to_numpy(dtype=float)
        standard_error = (
            float(np.std(delta, ddof=1) / np.sqrt(len(delta)))
            if len(delta) > 1
            else 0.0
        )
        paired_stats[str(model)] = (float(np.mean(delta)), standard_error)
    output["paired_mse_delta"] = output["model"].map(
        lambda model: paired_stats[str(model)][0]
    )
    output["paired_mse_delta_standard_error"] = output["model"].map(
        lambda model: paired_stats[str(model)][1]
    )
    output["exact_mse_minimum"] = output["model"].eq(best_model)
    output["within_one_standard_error"] = output["paired_mse_delta"].le(
        output["paired_mse_delta_standard_error"]
    )
    eligible = output.loc[output["within_one_standard_error"]].sort_values(
        ["raw_context_alpha", "mean_season_full_game_mse"],
        ascending=[False, True],
        kind="stable",
    )
    selected = eligible.iloc[0]
    output["selected_by_one_se_rule"] = output["model"].eq(selected["model"])
    output["full_history_rank"] = output["mean_season_full_game_mse"].rank(
        method="min"
    ).astype(int)
    output["recent_history_rank"] = output[
        "recent_equal_season_full_game_rmse"
    ].rank(method="min").astype(int)
    return output.sort_values("raw_context_alpha", kind="stable")


def _write_selection_run(
    *,
    season_metrics: pd.DataFrame,
    selection: pd.DataFrame,
    validation_seasons: tuple[str, ...],
    replay_run_dir: Path,
    selected_raw_alpha: float,
    artifacts_dir: Path,
) -> FixedContextSelectionRun:
    now = datetime.now(UTC)
    run_id = (
        f"nail-fixed-context-{validation_seasons[0]}-to-{validation_seasons[-1]}-"
        f"{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    root = (
        artifacts_dir
        / MODEL_NAME
        / f"{validation_seasons[0]}_to_{validation_seasons[-1]}"
    )
    run_dir = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        season_metrics.to_parquet(temporary / "season_metrics.parquet", index=False)
        selection.to_parquet(temporary / "selection_summary.parquet", index=False)
        metadata = {
            "run_id": run_id,
            "model": MODEL_NAME,
            "validation_seasons": list(validation_seasons),
            "frozen_evaluation_seasons": list(DEFAULT_SEASONS),
            "selection_objective": "equal-season mean full-game margin squared error",
            "selection_rule": (
                "largest raw alpha whose paired season MSE delta from the minimum "
                "is within one standard error"
            ),
            "selected_raw_alpha": selected_raw_alpha,
            "replay_run_dir": str(replay_run_dir),
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(run_dir)
        (root / "latest.json").write_text(
            json.dumps({"run_id": run_id}, indent=2) + "\n"
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return FixedContextSelectionRun(run_dir, run_id, selected_raw_alpha)


def _latest_directory(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No fixed-context frozen runs under {root}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fixed raw NAIL context alphas")
    parser.add_argument("action", choices=("train", "select", "evaluate", "bootstrap"))
    parser.add_argument("--raw-alpha", type=float, action="append")
    parser.add_argument("--through-season", default="2025-26")
    parser.add_argument("--evaluate-target", action="store_true")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.action == "train":
        train_fixed_context_candidates(
            raw_alphas=tuple(args.raw_alpha or TRAINING_ALPHAS),
            through_season=args.through_season,
            evaluate_target=args.evaluate_target,
        )
    elif args.action == "select":
        run = run_fixed_context_selection()
        print(
            f"Fixed context selection: alpha={run.selected_raw_alpha:,.0f} "
            f"run={run.run_dir}"
        )
    elif args.action == "evaluate":
        run = run_fixed_context_frozen_backtest()
        print(f"Fixed context frozen evaluation: run={run.run_dir}")
    else:
        path = run_fixed_context_bootstrap(draws=args.draws, seed=args.seed)
        print(f"Fixed context bootstrap: run={path}")


if __name__ == "__main__":
    main()
