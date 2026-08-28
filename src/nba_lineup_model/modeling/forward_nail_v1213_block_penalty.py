"""Additive-versus-non-additive penalty candidates for NAIL-RAPM v1.2.1.3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONTEXT_ALPHA,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    DEFAULT_TARGET_SEASON,
)
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import (
    train_nail_v1212_back_to_back,
)
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import (
    RESIDUALIZED_LAMBDA_GRID,
)
from nba_lineup_model.modeling.forward_nail_v13_additive_profiles import _Tee
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    ForwardPortableMatchupContextualRapmRun,
)
from nba_lineup_model.modeling.matchup_contextual import (
    fit_block_penalized_linear_ridge_matchup_contextual_model,
)


ADDITIVE_ALPHA = DEFAULT_CONTEXT_ALPHA
NONADDITIVE_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NONADDITIVE_FEATURES = ("top_two_assists", "usage_concentration")
FIXED_B2B_ALPHA = DEFAULT_CONTEXT_ALPHA
NO_NONADDITIVE_MODEL_NAME = "forward_nail_rapm_v1213_additive_only_context"
NO_NONADDITIVE_RUN_PREFIX = "forward-nail-rapm-v1213-additive-only-context"


def ratio_slug(ratio: float) -> str:
    if ratio <= 0:
        raise ValueError("nonadditive_ratio must be positive")
    return f"{ratio:.8g}".replace(".", "p").replace("-", "m")


def model_name_for_ratio(ratio: float) -> str:
    return f"forward_nail_rapm_v1213_block_penalty_r{ratio_slug(ratio)}"


def run_prefix_for_ratio(ratio: float) -> str:
    return f"forward-nail-rapm-v1213-block-penalty-r{ratio_slug(ratio)}"


def train_nail_v1213_block_penalty(
    *,
    nonadditive_ratio: float,
    through_season: str = DEFAULT_TARGET_SEASON,
    resume_from: Path | str | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Fit a candidate that changes only the non-additive Ridge block penalty."""

    nonadditive_alpha = ADDITIVE_ALPHA * nonadditive_ratio
    return train_nail_v1212_back_to_back(
        through_season=through_season,
        evaluate_target=False,
        context_alpha=ADDITIVE_ALPHA,
        schedule_alpha=FIXED_B2B_ALPHA,
        player_lambda_mode="residualized_cv",
        residualized_lambda_grid=RESIDUALIZED_LAMBDA_GRID,
        model_name=model_name_for_ratio(nonadditive_ratio),
        run_prefix=run_prefix_for_ratio(nonadditive_ratio),
        resume_from=resume_from,
        context_fit=fit_block_penalized_linear_ridge_matchup_contextual_model,
        context_fit_kwargs={
            "additive_alpha": ADDITIVE_ALPHA,
            "nonadditive_alpha": nonadditive_alpha,
            "additive_features": LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
            "nonadditive_features": NONADDITIVE_FEATURES,
        },
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def train_nail_v1213_additive_only_context(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Fit the structural control with no non-additive context coefficients."""

    return train_nail_v1212_back_to_back(
        through_season=through_season,
        evaluate_target=False,
        context_alpha=ADDITIVE_ALPHA,
        schedule_alpha=FIXED_B2B_ALPHA,
        player_lambda_mode="residualized_cv",
        residualized_lambda_grid=RESIDUALIZED_LAMBDA_GRID,
        model_name=NO_NONADDITIVE_MODEL_NAME,
        run_prefix=NO_NONADDITIVE_RUN_PREFIX,
        context_fit=fit_block_penalized_linear_ridge_matchup_contextual_model,
        context_fit_kwargs={
            "additive_alpha": ADDITIVE_ALPHA,
            "nonadditive_alpha": None,
            "additive_features": LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
            "nonadditive_features": NONADDITIVE_FEATURES,
        },
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NAIL-RAPM v1.2.1.3 additive/non-additive penalty candidates"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--nonadditive-ratio", type=float, action="append")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--no-nonadditive", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.all and (args.nonadditive_ratio or args.no_nonadditive):
        parser.error("Use --all alone, or explicit candidate arguments")
    ratios = NONADDITIVE_RATIOS if args.all else tuple(args.nonadditive_ratio or ())
    if not ratios and not args.no_nonadditive:
        parser.error("Provide --all, --no-nonadditive, or a non-additive ratio")

    def run_all() -> None:
        for ratio in ratios:
            if ratio == 1.0:
                print(
                    "Skipping r=1.00: the locked v1.2.1.3 incumbent is exactly "
                    "the shared-alpha control.",
                    flush=True,
                )
                continue
            run = train_nail_v1213_block_penalty(
                nonadditive_ratio=ratio,
                through_season=args.through_season,
                resume_from=args.resume_from,
            )
            print(
                f"Block-penalty ratio {ratio:.2f}: run={run.run_dir}", flush=True
            )
        if args.no_nonadditive:
            run = train_nail_v1213_additive_only_context(
                through_season=args.through_season,
            )
            print(f"Additive-only context: run={run.run_dir}", flush=True)

    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            sys.stderr = _Tee(original_stderr, handle)  # type: ignore[assignment]
            try:
                run_all()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        return
    run_all()


if __name__ == "__main__":
    main()
