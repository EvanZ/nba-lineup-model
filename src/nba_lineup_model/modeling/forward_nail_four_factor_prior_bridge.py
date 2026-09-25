"""Production-contract NAIL with a strictly forward Four Factors prior bridge."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd

from nba_lineup_model.modeling.four_factor_prior_bridge import (
    DEFAULT_BRIDGE_ALPHA_GRID,
    DEFAULT_EFFECTS_PATH,
    apply_four_factor_prior_bridge,
    build_four_factor_prior_bridge,
    load_factor_side_effects,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_CONTEXT_ALPHA, DEFAULT_TARGET_SEASON
from nba_lineup_model.modeling.forward_nail_v13_additive_profiles import _Tee
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import train_nail_v1212_back_to_back
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import RESIDUALIZED_LAMBDA_GRID
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    ForwardPortableMatchupContextualRapmRun,
)
from nba_lineup_model.modeling.gap_returner_prior import (
    GAP_RETURNER_METHOD,
    build_centered_value_conditioned_aging_gap_returner_priors,
)


MODEL_NAME = "forward_nail_four_factor_prior_bridge"
RUN_PREFIX = "forward-nail-four-factor-prior-bridge"
ZERO_MODEL_NAME = "forward_nail_four_factor_prior_bridge_zero"
ZERO_RUN_PREFIX = "forward-nail-four-factor-prior-bridge-zero"


def train_nail_four_factor_prior_bridge(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    bridge_weight: float = 1.0,
    bridge_alpha_grid: tuple[float, ...] = DEFAULT_BRIDGE_ALPHA_GRID,
    factor_effects_path: Path | str = DEFAULT_EFFECTS_PATH,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    schedule_alpha: float | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    model_name: str | None = None,
    run_prefix: str | None = None,
) -> ForwardPortableMatchupContextualRapmRun:
    """Train the frozen candidate without changing the production NAIL fit."""

    effects = load_factor_side_effects(factor_effects_path)
    builder = partial(
        _four_factor_bridge_priors,
        effects=effects,
        bridge_weight=bridge_weight,
        bridge_alpha_grid=bridge_alpha_grid,
    )
    return train_nail_v1212_back_to_back(
        through_season=through_season,
        context_alpha=context_alpha,
        schedule_alpha=schedule_alpha,
        player_lambda_mode="residualized_cv",
        residualized_lambda_grid=RESIDUALIZED_LAMBDA_GRID,
        player_prior_builder=builder,
        player_prior_description=(
            "production NAIL-RAPM v1.2.1.3 prior plus a strictly forward, "
            "prior-season Four Factors O/D residual bridge; no total NAIL input, "
            "no target-season Four Factors input, and "
            f"{GAP_RETURNER_METHOD} base prior"
        ),
        profile_contract_metadata_updates={
            "four_factor_prior_bridge_contract": {
                "feature_source": "prior-season completed Four Factors O/D player effects",
                "target": "completed NAIL player effect minus its pre-season NAIL prior",
                "features": "eight factor-side player states",
                "estimator": "no-intercept Ridge on RMS-scaled feature states",
                "alpha_selection": "three expanding chronological completed-season folds",
                "cold_start_behavior": "zero adjustment without a completed source factor state",
                "bridge_weight": bridge_weight,
                "alpha_grid": list(bridge_alpha_grid),
            }
        },
        model_name=model_name or (ZERO_MODEL_NAME if bridge_weight == 0.0 else MODEL_NAME),
        run_prefix=run_prefix or (ZERO_RUN_PREFIX if bridge_weight == 0.0 else RUN_PREFIX),
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def _four_factor_bridge_priors(
    *,
    season: str,
    panel: pd.DataFrame,
    completed_results: list[object],
    exposure_history: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
    effects: pd.DataFrame,
    bridge_weight: float,
    bridge_alpha_grid: tuple[float, ...],
) -> tuple[pd.DataFrame, dict[str, object]]:
    base_priors, metadata = build_centered_value_conditioned_aging_gap_returner_priors(
        season=season,
        panel=panel,
        completed_results=completed_results,  # type: ignore[arg-type]
        exposure_history=exposure_history,
        replacement_tokens=replacement_tokens,
    )
    bridge = (
        build_four_factor_prior_bridge(
            completed_results=completed_results,  # type: ignore[arg-type]
            exposure_history=exposure_history,
            effects=effects,
            alpha_grid=bridge_alpha_grid,
        )
        if bridge_weight > 0.0
        else None
    )
    priors, bridge_metadata = apply_four_factor_prior_bridge(
        priors=base_priors,
        target_season=season,
        effects=effects,
        bridge=bridge,
        weight=bridge_weight,
    )
    metadata.update(bridge_metadata)
    return priors, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NAIL with a forward Four Factors prior bridge")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--bridge-weight", type=float, default=1.0)
    parser.add_argument("--log-path")
    args = parser.parse_args()
    kwargs = {"through_season": args.through_season, "bridge_weight": args.bridge_weight}
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_nail_four_factor_prior_bridge(**kwargs)
                print(f"NAIL Four Factors prior bridge: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_nail_four_factor_prior_bridge(**kwargs)
    print(f"NAIL Four Factors prior bridge: run={run.run_dir}")


if __name__ == "__main__":
    main()
