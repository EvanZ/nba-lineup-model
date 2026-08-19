"""Frozen evaluation and bootstrap for selected NAIL context regularization."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_nail_context_regularization import (
    model_name_for_context_lambda,
    train_nail_normalized_context_regularization,
)
from nba_lineup_model.modeling.forward_nail_profile_padding import PUBLISHED_MODEL_NAME
from nba_lineup_model.modeling.frozen_model_tournament import _paired_metrics
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)
from nba_lineup_model.modeling.nail_context_regularization_study import MODEL_NAME as STUDY_MODEL

OUTPUT_ARTIFACTS_DIR = Path("artifacts/models/nail_context_regularization_frozen_backtest")
BOOTSTRAP_OUTPUT_ROOT = Path(
    "artifacts/models/nail_context_regularization_bootstrap/2023-24_to_2025-26"
)
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20_260_818


def selected_context_lambda(artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR) -> float:
    root = Path(artifacts_dir) / STUDY_MODEL / "2000-01_to_2022-23"
    latest = json.loads((root / "latest.json").read_text())
    metadata = json.loads((root / str(latest["run_id"]) / "metadata.json").read_text())
    return float(metadata["selected_context_lambda"])


def train_selected_nail_context_regularization(
    *,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> Path:
    context_lambda = selected_context_lambda(artifacts_dir)
    run = train_nail_normalized_context_regularization(
        context_lambda=context_lambda,
        through_season="2025-26",
        artifacts_dir=artifacts_dir,
    )
    return run.run_dir


def run_selected_context_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_artifacts_dir: Path | str = OUTPUT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    context_lambda = selected_context_lambda(artifacts_dir)
    builder = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
    )
    return run_frozen_multiseason_backtest(
        seasons=seasons,
        models=(
            BacktestModel(
                PUBLISHED_MODEL_NAME,
                "NAIL-RAPM v1.1 fixed raw alpha=10,000",
                profile_builder=builder,
            ),
            BacktestModel(
                model_name_for_context_lambda(context_lambda),
                f"NAIL normalized context lambda_C={context_lambda:.8g}",
                profile_builder=builder,
            ),
        ),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=output_artifacts_dir,
        docs_path=None,
    )


def run_selected_context_bootstrap(
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    backtest_root: Path | str = OUTPUT_ARTIFACTS_DIR,
    output_root: Path | str = BOOTSTRAP_OUTPUT_ROOT,
) -> Path:
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    frozen_root = Path(backtest_root) / "frozen_multiseason_backtest" / "2023-24_to_2025-26"
    source_run = _latest_directory(frozen_root)
    games = pd.read_parquet(source_run / "regular_game_predictions.parquet")
    possessions = pd.read_parquet(source_run / "possession_predictions.parquet")
    context_lambda = selected_context_lambda()
    challenger = model_name_for_context_lambda(context_lambda)

    def source(model: str) -> dict[str, pd.DataFrame]:
        return {
            "games": games.loc[games["model"].eq(model)].copy(),
            "possessions": possessions.loc[
                possessions["model"].eq(model)
                & possessions["cohort"].eq("regular_season")
            ].copy(),
        }

    metrics = _paired_metrics(
        source(PUBLISHED_MODEL_NAME),
        source(challenger),
        draws=draws,
        seed=seed,
    )
    metrics.insert(0, "incumbent_model", PUBLISHED_MODEL_NAME)
    metrics.insert(1, "challenger_model", challenger)
    root = Path(output_root)
    run_id = (
        f"nail-context-regularization-bootstrap-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-"
        f"{uuid4().hex[:8]}"
    )
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_parquet(run_dir / "paired_bootstrap_metrics.parquet", index=False)
    metadata = {
        "run_id": run_id,
        "source_run_dir": str(source_run),
        "context_lambda": context_lambda,
        "draws": draws,
        "seed": seed,
        "resampling_unit": "games stratified by season",
        "created_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return run_dir


def _latest_directory(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No frozen context-regularization runs under {root}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or bootstrap selected NAIL context regularization"
    )
    parser.add_argument("action", choices=("train", "evaluate", "bootstrap"))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.action == "train":
        path = train_selected_nail_context_regularization()
    elif args.action == "evaluate":
        path = run_selected_context_frozen_backtest().run_dir
    else:
        path = run_selected_context_bootstrap(draws=args.draws, seed=args.seed)
    print(f"NAIL context regularization {args.action}: run={path}")


if __name__ == "__main__":
    main()
