"""Leakage-safe rolling selection of NAIL context regularization."""

from __future__ import annotations

import argparse
import hashlib
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
    BASE_CONTEXT_LAMBDA,
    DEFAULT_CONTEXT_LAMBDAS,
    model_name_for_context_lambda,
)
from nba_lineup_model.modeling.forward_nail_profile_padding import PUBLISHED_MODEL_NAME
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    BacktestModel,
    run_frozen_multiseason_backtest,
)

MODEL_NAME = "nail_context_regularization_study"
DEFAULT_VALIDATION_SEASONS = tuple(
    f"{year}-{str(year + 1)[-2:]}" for year in range(2000, 2023)
)
RECENT_SEASON_COUNT = 10


@dataclass(frozen=True)
class NailContextRegularizationStudyRun:
    run_dir: Path
    run_id: str
    selected_context_lambda: float


def run_nail_context_regularization_study(
    *,
    context_lambdas: tuple[float, ...] = DEFAULT_CONTEXT_LAMBDAS,
    validation_seasons: tuple[str, ...] = DEFAULT_VALIDATION_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    reuse_latest: bool = False,
) -> NailContextRegularizationStudyRun:
    """Replay pre-frozen seasons and apply a one-standard-error selection rule."""

    if not context_lambdas or any(value <= 0 for value in context_lambdas):
        raise ValueError("Context regularization study requires positive lambdas")
    root = Path(artifacts_dir)
    profile_builder = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
    )
    prior_metrics, prior_replays = _latest_compatible_metrics(
        root,
        validation_seasons=validation_seasons,
    ) if reuse_latest else (pd.DataFrame(), ())
    existing_models = set(prior_metrics.get("model", pd.Series(dtype=str)).astype(str))
    candidates: list[BacktestModel] = []
    if PUBLISHED_MODEL_NAME not in existing_models:
        candidates.append(
            BacktestModel(
                PUBLISHED_MODEL_NAME,
                "NAIL v1.1 fixed raw alpha=10,000",
                profile_builder=profile_builder,
                run_target_season="2025-26",
            )
        )
    candidates.extend(
        BacktestModel(
            model_name_for_context_lambda(context_lambda),
            f"NAIL v1.1 normalized lambda_C={context_lambda:.8g}",
            profile_builder=profile_builder,
            run_target_season=validation_seasons[-1],
        )
        for context_lambda in context_lambdas
        if model_name_for_context_lambda(context_lambda) not in existing_models
    )
    replay_dirs = list(prior_replays)
    new_metrics = pd.DataFrame()
    if candidates:
        replay = run_frozen_multiseason_backtest(
            seasons=validation_seasons,
            models=tuple(candidates),
            artifacts_dir=root,
            output_artifacts_dir=root / MODEL_NAME / "rolling_replay",
            score_possessions=False,
        )
        replay_dirs.append(str(replay.run_dir))
        games = pd.read_parquet(replay.run_dir / "regular_game_predictions.parquet")
        new_metrics = _season_metrics(games, context_lambdas=context_lambdas)
    season_metrics = (
        pd.concat([prior_metrics, new_metrics], ignore_index=True)
        .drop_duplicates(["model", "season"], keep="last")
        .sort_values(["context_lambda", "regularization_contract", "season"], kind="stable")
    )
    selection = _selection_summary(season_metrics)
    selected = selection.loc[selection["selected_by_one_se_rule"]].iloc[0]
    return _write_run(
        season_metrics=season_metrics,
        selection=selection,
        validation_seasons=validation_seasons,
        replay_run_dirs=tuple(replay_dirs),
        selected_context_lambda=float(selected["context_lambda"]),
        artifacts_dir=root,
    )


def _latest_compatible_metrics(
    artifacts_dir: Path,
    *,
    validation_seasons: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    root = artifacts_dir / MODEL_NAME / f"{validation_seasons[0]}_to_{validation_seasons[-1]}"
    latest_path = root / "latest.json"
    if not latest_path.is_file():
        return pd.DataFrame(), ()
    latest = json.loads(latest_path.read_text())
    run_dir = root / str(latest["run_id"])
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if tuple(metadata.get("validation_seasons", ())) != validation_seasons:
        return pd.DataFrame(), ()
    replay_dirs = metadata.get("replay_run_dirs")
    if replay_dirs is None:
        replay_dirs = [metadata.get("replay_run_dir")]
    return (
        pd.read_parquet(run_dir / "season_metrics.parquet"),
        tuple(str(path) for path in replay_dirs if path),
    )


def _season_metrics(
    games: pd.DataFrame,
    *,
    context_lambdas: tuple[float, ...],
) -> pd.DataFrame:
    lambda_by_model = {
        model_name_for_context_lambda(value): value for value in context_lambdas
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
                "regularization_contract": (
                    "fixed_raw_alpha" if model == PUBLISHED_MODEL_NAME else "mean_weighted_loss"
                ),
                "context_lambda": lambda_by_model.get(str(model), BASE_CONTEXT_LAMBDA),
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
        ["context_lambda", "regularization_contract", "season"], kind="stable"
    )


def _selection_summary(season_metrics: pd.DataFrame) -> pd.DataFrame:
    recent = set(sorted(season_metrics["season"].unique())[-RECENT_SEASON_COUNT:])
    rows: list[dict[str, object]] = []
    for (model, label, contract, context_lambda), frame in season_metrics.groupby(
        ["model", "label", "regularization_contract", "context_lambda"], sort=False
    ):
        mse = frame["full_game_margin_mse"].to_numpy(dtype=float)
        recent_frame = frame.loc[frame["season"].isin(recent)]
        rows.append(
            {
                "model": model,
                "label": label,
                "regularization_contract": contract,
                "context_lambda": float(context_lambda),
                "season_count": len(frame),
                "mean_season_full_game_mse": float(np.mean(mse)),
                "equal_season_full_game_rmse": float(np.sqrt(np.mean(mse))),
                "season_mse_standard_error": float(np.std(mse, ddof=1) / np.sqrt(len(mse))),
                "recent_season_count": len(recent_frame),
                "recent_equal_season_full_game_rmse": float(
                    np.sqrt(recent_frame["full_game_margin_mse"].mean())
                ),
                "mean_full_game_mae": float(frame["full_game_margin_mae"].mean()),
                "mean_winner_accuracy": float(frame["game_winner_accuracy"].mean()),
            }
        )
    output = pd.DataFrame(rows)
    selectable = output.loc[output["regularization_contract"].eq("mean_weighted_loss")]
    if selectable.empty:
        raise ValueError("Selection requires at least one normalized context candidate")
    best = selectable.sort_values("mean_season_full_game_mse", kind="stable").iloc[0]
    best_model = str(best["model"])
    paired = season_metrics.loc[
        season_metrics["regularization_contract"].eq("mean_weighted_loss"),
        ["model", "season", "full_game_margin_mse"],
    ].pivot(index="season", columns="model", values="full_game_margin_mse")
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
        lambda model: paired_stats.get(str(model), (np.nan, np.nan))[0]
    )
    output["paired_mse_delta_standard_error"] = output["model"].map(
        lambda model: paired_stats.get(str(model), (np.nan, np.nan))[1]
    )
    output["one_se_mse_threshold"] = (
        float(best["mean_season_full_game_mse"])
        + output["paired_mse_delta_standard_error"]
    )
    output["exact_mse_minimum"] = output["model"].eq(best["model"])
    output["within_one_standard_error"] = (
        output["regularization_contract"].eq("mean_weighted_loss")
        & output["paired_mse_delta"].le(
            output["paired_mse_delta_standard_error"]
        )
    )
    eligible = output.loc[output["within_one_standard_error"]].sort_values(
        ["context_lambda", "mean_season_full_game_mse"],
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
    return output.sort_values(
        ["full_history_rank", "context_lambda"], ascending=[True, False], kind="stable"
    )


def _write_run(
    *,
    season_metrics: pd.DataFrame,
    selection: pd.DataFrame,
    validation_seasons: tuple[str, ...],
    replay_run_dirs: tuple[str, ...],
    selected_context_lambda: float,
    artifacts_dir: Path,
) -> NailContextRegularizationStudyRun:
    now = datetime.now(UTC)
    run_id = (
        f"nail-context-regularization-{validation_seasons[0]}-to-{validation_seasons[-1]}-"
        f"{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    root = artifacts_dir / MODEL_NAME / f"{validation_seasons[0]}_to_{validation_seasons[-1]}"
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        season_metrics.to_parquet(temporary / "season_metrics.parquet", index=False)
        selection.to_parquet(temporary / "selection_summary.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "validation_seasons": list(validation_seasons),
            "frozen_evaluation_seasons": ["2023-24", "2024-25", "2025-26"],
            "selection_objective": "equal-season mean full-game margin squared error",
            "selection_rule": (
                "largest normalized lambda whose paired season MSE delta from the "
                "minimum is within one standard error"
            ),
            "selected_context_lambda": selected_context_lambda,
            "replay_run_dirs": list(replay_run_dirs),
            "created_at": now.isoformat(),
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
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return NailContextRegularizationStudyRun(
            output,
            run_id,
            selected_context_lambda,
        )
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
    parser = argparse.ArgumentParser(
        description="Select NAIL v1.1 context regularization before frozen seasons"
    )
    parser.add_argument("--context-lambda", type=float, action="append")
    parser.add_argument(
        "--reuse-latest",
        action="store_true",
        help="Reuse compatible persisted season metrics and replay only new candidates",
    )
    args = parser.parse_args()
    values = tuple(args.context_lambda or DEFAULT_CONTEXT_LAMBDAS)
    run = run_nail_context_regularization_study(
        context_lambdas=values,
        reuse_latest=args.reuse_latest,
    )
    print(
        f"NAIL context regularization study: lambda={run.selected_context_lambda:.8g} "
        f"run={run.run_dir}"
    )


if __name__ == "__main__":
    main()
