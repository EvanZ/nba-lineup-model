"""Pre-frozen development selection for NAIL v1.2.1.3 block penalties."""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stdout
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
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import (
    MODEL_NAME as INCUMBENT_MODEL,
)
from nba_lineup_model.modeling.forward_nail_v1213_block_penalty import (
    NO_NONADDITIVE_MODEL_NAME,
    NONADDITIVE_RATIOS,
    model_name_for_ratio,
)
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)


DEVELOPMENT_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23")
DEVELOPMENT_RUN_TARGET = DEVELOPMENT_SEASONS[-1]
INCUMBENT_RUN_TARGET = "2025-26"
OUTPUT_ARTIFACTS_DIR = Path("artifacts/models/nail_v1213_block_penalty_development")
LOCKED_OUTPUT_ARTIFACTS_DIR = Path(
    "artifacts/models/nail_v1213_block_penalty_locked_backtest"
)
LOCKED_INCUMBENT_OUTPUT_ARTIFACTS_DIR = Path(
    "artifacts/models/nail_v1213_block_penalty_locked_incumbent_backtest"
)


def _candidates() -> tuple[BacktestModel, ...]:
    profiles = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    )
    output = [
        BacktestModel(
            INCUMBENT_MODEL,
            "NAIL-RAPM v1.2.1.3 shared alpha (r=1.00)",
            profile_builder=profiles,
            uses_schedule_control=True,
            # This completed recursive run contains every pre-target state needed
            # for development replay; the evaluator selects the appropriate one.
            run_target_season=INCUMBENT_RUN_TARGET,
        )
    ]
    output.extend(
        BacktestModel(
            model_name_for_ratio(ratio),
            f"NAIL-RAPM v1.2.1.3 non-additive penalty ratio r={ratio:.2f}",
            profile_builder=profiles,
            uses_schedule_control=True,
            run_target_season=DEVELOPMENT_RUN_TARGET,
        )
        for ratio in NONADDITIVE_RATIOS
        if ratio != 1.0
    )
    output.append(
        BacktestModel(
            NO_NONADDITIVE_MODEL_NAME,
            "NAIL-RAPM v1.2.1.3 additive-only context (no non-additive terms)",
            profile_builder=profiles,
            uses_schedule_control=True,
            run_target_season=DEVELOPMENT_RUN_TARGET,
        )
    )
    return tuple(output)


def run_development_backtest(
    *, artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR
) -> FrozenMultiseasonBacktestRun:
    """Replay all pre-registered ratios on only the development seasons."""

    return run_frozen_multiseason_backtest(
        seasons=DEVELOPMENT_SEASONS,
        models=_candidates(),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=OUTPUT_ARTIFACTS_DIR,
        docs_path=None,
    )


def run_locked_backtest(
    *, artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR
) -> FrozenMultiseasonBacktestRun:
    """Replay the selected r=16 candidate on the untouched frozen seasons."""

    profiles = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    )


def run_locked_incumbent_backtest(
    *, artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR
) -> FrozenMultiseasonBacktestRun:
    """Score production under the same corrected evaluator as the selected candidate."""

    profiles = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    )
    incumbent = BacktestModel(
        INCUMBENT_MODEL,
        "NAIL-RAPM v1.2.1.2 production (corrected evaluator)",
        profile_builder=profiles,
        uses_schedule_control=True,
        run_target_season=DEFAULT_SEASONS[-1],
    )
    return run_frozen_multiseason_backtest(
        seasons=DEFAULT_SEASONS,
        models=(incumbent,),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=LOCKED_INCUMBENT_OUTPUT_ARTIFACTS_DIR,
        docs_path=None,
    )
    candidate = BacktestModel(
        model_name_for_ratio(16.0),
        "NAIL-RAPM v1.2.1.3 non-additive penalty ratio r=16.00",
        profile_builder=profiles,
        uses_schedule_control=True,
        run_target_season=DEFAULT_SEASONS[-1],
    )
    return run_frozen_multiseason_backtest(
        seasons=DEFAULT_SEASONS,
        models=(candidate,),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=LOCKED_OUTPUT_ARTIFACTS_DIR,
        docs_path=None,
    )


def summarize_development(run_dir: Path) -> pd.DataFrame:
    """Choose a ratio by equal-season full-game MSE and one-standard-error rule."""

    games = pd.read_parquet(run_dir / "regular_game_predictions.parquet")
    summary = (
        games.assign(squared_error=lambda frame: np.square(frame["margin_error"]))
        .groupby(["model", "label", "season"], as_index=False)["squared_error"]
        .mean()
        .groupby(["model", "label"], as_index=False)["squared_error"]
        .mean()
        .rename(columns={"squared_error": "equal_season_full_game_mse"})
    )
    ratio_by_model: dict[str, float | None] = {INCUMBENT_MODEL: 1.0}
    ratio_by_model.update(
        {model_name_for_ratio(ratio): ratio for ratio in NONADDITIVE_RATIOS if ratio != 1.0}
    )
    ratio_by_model[NO_NONADDITIVE_MODEL_NAME] = None
    summary["nonadditive_penalty_ratio"] = summary["model"].map(ratio_by_model)
    summary["nonadditive_terms_included"] = summary["model"].ne(NO_NONADDITIVE_MODEL_NAME)
    best = summary.loc[summary["equal_season_full_game_mse"].idxmin()]
    best_model = str(best["model"])
    season_mse = (
        games.assign(squared_error=lambda frame: np.square(frame["margin_error"]))
        .groupby(["model", "season"], as_index=False)["squared_error"]
        .mean()
        .pivot(index="season", columns="model", values="squared_error")
    )
    best_values = season_mse[best_model]
    deltas = {
        model: (season_mse[model] - best_values).dropna().to_numpy(dtype=float)
        for model in summary["model"]
    }
    summary["paired_mse_delta_from_minimum"] = summary["model"].map(
        lambda model: float(deltas[str(model)].mean())
    )
    summary["paired_mse_delta_standard_error"] = summary["model"].map(
        lambda model: float(
            np.std(deltas[str(model)], ddof=1) / np.sqrt(len(deltas[str(model)]))
        )
    )
    summary["within_one_standard_error"] = summary.apply(
        lambda row: row["paired_mse_delta_from_minimum"]
        <= row["paired_mse_delta_standard_error"],
        axis=1,
    )
    eligible = summary.loc[summary["within_one_standard_error"]].copy()
    eligible["regularization_order"] = eligible["nonadditive_penalty_ratio"].fillna(float("inf"))
    eligible = eligible.sort_values(
        ["regularization_order", "equal_season_full_game_mse"],
        ascending=[False, True],
        kind="stable",
    )
    selected_model = str(eligible.iloc[0]["model"])
    summary["selected_by_one_standard_error_rule"] = summary["model"].eq(selected_model)
    return summary.sort_values(
        ["nonadditive_terms_included", "nonadditive_penalty_ratio"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def write_selection(run: FrozenMultiseasonBacktestRun) -> Path:
    """Persist the development decision independently of the candidate artifacts."""

    summary = summarize_development(run.run_dir)
    selected = summary.loc[summary["selected_by_one_standard_error_rule"]].iloc[0]
    now = datetime.now(UTC)
    run_id = f"block-penalty-development-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = OUTPUT_ARTIFACTS_DIR / "selection" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary.to_parquet(run_dir / "selection_summary.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "development_seasons": list(DEVELOPMENT_SEASONS),
                "development_replay_run_dir": str(run.run_dir),
                "objective": "equal-season full-game margin MSE",
                "selection_rule": (
                    "largest non-additive penalty ratio within one paired-season "
                    "standard error of the development minimum"
                ),
                "selected_nonadditive_penalty_ratio": (
                    float(selected["nonadditive_penalty_ratio"])
                    if pd.notna(selected["nonadditive_penalty_ratio"])
                    else None
                ),
                "selected_nonadditive_terms_included": bool(
                    selected["nonadditive_terms_included"]
                ),
                "created_at": now.isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NAIL v1.2.1.3 block-penalty development selection"
    )
    parser.add_argument("--log-path")
    parser.add_argument("--locked-backtest", action="store_true")
    parser.add_argument("--locked-incumbent", action="store_true")
    args = parser.parse_args()
    if args.locked_backtest and args.locked_incumbent:
        parser.error("Choose one locked replay mode")

    def execute() -> None:
        if args.locked_backtest:
            run = run_locked_backtest()
            print(f"Locked block-penalty backtest: run={run.run_dir}", flush=True)
            return
        if args.locked_incumbent:
            run = run_locked_incumbent_backtest()
            print(f"Locked incumbent backtest: run={run.run_dir}", flush=True)
            return
        run = run_development_backtest()
        selection = write_selection(run)
        print(f"Block-penalty development selection: run={selection}", flush=True)

    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", buffering=1) as handle, redirect_stdout(handle):
            execute()
        return
    execute()


if __name__ == "__main__":
    main()
