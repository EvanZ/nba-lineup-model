"""Frozen FCM-only test of incumbent preseason team strength for rookies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_CURATED_DIR,
    PROMOTED_AVAILABILITY_CONFIG,
    build_availability_season_summary,
    predict_availability_season,
)
from nba_lineup_model.rotation.forward_conditional_minutes import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_COLD_START_ALPHA_GRID,
    DEFAULT_PLAYER_PANEL_PATH,
    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
    PROMOTED_CONDITIONAL_MINUTES_CONFIG,
    ColdStartConditionalMinutesConfig,
    attach_cold_start_biographies,
    predict_conditional_minutes_season,
    summarize_cold_start_minutes_metrics,
    summarize_conditional_minutes_metrics,
    tune_cold_start_conditional_minutes,
)
from nba_lineup_model.rotation.l0_opening_minute_share_persistence import (
    read_preseason_roster_candidates,
)
from nba_lineup_model.rotation.l20_preseason_nail_forecast_minute_share import (
    DEFAULT_MODEL_RUN_DIR,
    build_preseason_nail_forecast_ratings,
)

MODEL_NAME = "forward_conditional_team_strength"
MODEL_VERSION = "v0.1"
DEFAULT_TUNING_SEASONS = ("2020-21", "2021-22", "2022-23")
DEFAULT_FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
TEAM_STRENGTH_COLUMN = "incumbent_team_nail"


@dataclass(frozen=True)
class ForwardConditionalTeamStrengthRun:
    """Immutable output location for one FCM-only frozen candidate."""

    run_dir: Path
    run_id: str


def build_incumbent_team_strength_summary(
    summary: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    seasons: tuple[str, ...],
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach target-preseason incumbent NAIL strength to every player-season.

    The feature is a raw-minute-weighted average of exact target-preseason
    NAIL forecasts for opening-roster players who are both non-rookies and
    have a prior conditional-minutes state. Rookies receive their team's value
    but are excluded from its construction, preventing feedback into the
    cold-start prediction.
    """

    output = summary.copy()
    output[TEAM_STRENGTH_COLUMN] = np.nan
    first_year = int(pd.to_numeric(output["season_start_year"], errors="raise").min())
    diagnostic_rows: list[dict[str, object]] = []
    for index, season in enumerate(seasons, start=1):
        target = output.loc[output["season"].astype(str).eq(season)].copy()
        if target.empty:
            raise ValueError(f"Team-strength input lacks target season {season}")
        if int(target["season_start_year"].iloc[0]) <= first_year:
            continue
        opening = read_preseason_roster_candidates(season, curated_dir=curated_dir).loc[
            :, ["team_id", "team", "player_id", "player_name"]
        ].copy()
        opening["player_id"] = pd.to_numeric(opening["player_id"], errors="raise").astype(int)
        ratings = build_preseason_nail_forecast_ratings(
            season, opening, panel=panel, run_dir=run_dir
        ).rename("preseason_nail")
        conditional, _age_model = predict_conditional_minutes_season(
            output,
            target_season=season,
            config=PROMOTED_CONDITIONAL_MINUTES_CONFIG,
            cold_start_config=None,
        )
        availability, _availability_age, _metadata = predict_availability_season(
            output,
            target_season=season,
            config=PROMOTED_AVAILABILITY_CONFIG,
        )
        incumbent = (
            opening.merge(
                conditional.loc[
                    :, [
                        "player_id",
                        "has_prior_minutes_state",
                        "is_rookie",
                        "predicted_minutes_per_available_game",
                    ]
                ],
                on="player_id",
                how="left",
                validate="one_to_one",
            )
            .merge(
                availability.loc[:, ["player_id", "predicted_available_share"]],
                on="player_id",
                how="left",
                validate="one_to_one",
            )
        )
        incumbent["preseason_nail"] = incumbent["player_id"].map(ratings).fillna(0.0)
        incumbent["raw_incumbent_weight"] = (
            pd.to_numeric(incumbent["predicted_available_share"], errors="coerce").fillna(0.0)
            * pd.to_numeric(
                incumbent["predicted_minutes_per_available_game"], errors="coerce"
            ).fillna(0.0)
        )
        eligible = incumbent.loc[
            incumbent["has_prior_minutes_state"].fillna(False).astype(bool)
            & ~incumbent["is_rookie"].fillna(False).astype(bool)
            & incumbent["raw_incumbent_weight"].gt(0.0)
        ].copy()
        eligible["weighted_nail"] = eligible["raw_incumbent_weight"] * eligible["preseason_nail"]
        team_strength = eligible.groupby(["team_id", "team"], as_index=False).agg(
            incumbent_weight=("raw_incumbent_weight", "sum"),
            weighted_nail=("weighted_nail", "sum"),
            incumbent_player_count=("player_id", "nunique"),
        )
        if team_strength.empty:
            raise ValueError(f"No eligible incumbent strength weights for {season}")
        team_strength[TEAM_STRENGTH_COLUMN] = (
            team_strength["weighted_nail"] / team_strength["incumbent_weight"]
        )
        league_fallback = float(
            np.average(
                team_strength[TEAM_STRENGTH_COLUMN],
                weights=team_strength["incumbent_weight"],
            )
        )
        player_team_strength = opening.merge(
            team_strength.loc[:, ["team_id", TEAM_STRENGTH_COLUMN]],
            on="team_id",
            how="left",
            validate="many_to_one",
        )
        player_strength = player_team_strength.set_index("player_id")[TEAM_STRENGTH_COLUMN]
        target_strength = (
            target["player_id"].astype(int).map(player_strength).fillna(league_fallback)
        )
        output.loc[target.index, TEAM_STRENGTH_COLUMN] = target_strength.to_numpy(dtype=float)
        diagnostic_rows.extend(
            team_strength.assign(
                season=season,
                league_strength_fallback=league_fallback,
                build_index=index,
            ).to_dict("records")
        )
    missing_mask = (
        pd.to_numeric(output["season_start_year"], errors="raise").gt(first_year)
        & output[TEAM_STRENGTH_COLUMN].isna()
    )
    if missing_mask.any():
        missing = output.loc[missing_mask, "season"].unique().tolist()
        raise ValueError(f"Team-strength feature is missing seasons: {missing}")
    return output, pd.DataFrame(diagnostic_rows)


def evaluate_forward_conditional_team_strength(
    *,
    augmented_summary: pd.DataFrame,
    base_summary: pd.DataFrame,
    season: str,
    candidate_config: ColdStartConditionalMinutesConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Compare conditional MPG only; availability and roster squash are excluded."""

    candidate, _candidate_age = predict_conditional_minutes_season(
        augmented_summary,
        target_season=season,
        config=PROMOTED_CONDITIONAL_MINUTES_CONFIG,
        cold_start_config=candidate_config,
    )
    control, _control_age = predict_conditional_minutes_season(
        base_summary,
        target_season=season,
        config=PROMOTED_CONDITIONAL_MINUTES_CONFIG,
        cold_start_config=PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
    )
    metrics = {
        "season": season,
        **{
            f"candidate_{name}": value
            for name, value in summarize_conditional_minutes_metrics(candidate).items()
        },
        **{
            f"control_{name}": value
            for name, value in summarize_conditional_minutes_metrics(control).items()
        },
        **{
            f"candidate_{name}": value
            for name, value in summarize_cold_start_minutes_metrics(candidate).items()
        },
        **{
            f"control_{name}": value
            for name, value in summarize_cold_start_minutes_metrics(control).items()
        },
    }
    for name in (
        "conditional_mae",
        "conditional_rmse",
        "available_game_weighted_mae",
        "available_game_weighted_rmse",
        "cold_start_conditional_mae",
        "cold_start_conditional_rmse",
        "cold_start_available_game_weighted_mae",
        "cold_start_available_game_weighted_rmse",
    ):
        metrics[f"{name}_delta"] = metrics[f"candidate_{name}"] - metrics[f"control_{name}"]
    return candidate, control, metrics


def run_forward_conditional_team_strength(
    *,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    nail_run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
) -> ForwardConditionalTeamStrengthRun:
    """Tune incumbent strength on source seasons, then score frozen FCM targets."""

    all_targets = tuple(dict.fromkeys((*tuning_seasons, *frozen_seasons)))
    final_year = max(int(season[:4]) for season in all_targets)
    seasons = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, final_year + 1))
    print(f"Team-strength FCM: summarizing {len(seasons)} seasons", flush=True)
    base_summary = attach_cold_start_biographies(
        build_availability_season_summary(
            seasons, curated_dir=curated_dir, player_panel_path=player_panel_path
        ),
        player_panel_path=player_panel_path,
    )
    panel = pd.read_parquet(player_panel_path)
    print("Team-strength FCM: building incumbent preseason NAIL features", flush=True)
    augmented_summary, diagnostics = build_incumbent_team_strength_summary(
        base_summary,
        panel=panel,
        seasons=seasons,
        curated_dir=curated_dir,
        run_dir=nail_run_dir,
    )
    print("Team-strength FCM: tuning rookie ridge", flush=True)
    candidate_config, tuning_grid = tune_cold_start_conditional_minutes(
        augmented_summary,
        state_config=PROMOTED_CONDITIONAL_MINUTES_CONFIG,
        target_seasons=tuning_seasons,
        alpha_grid=DEFAULT_COLD_START_ALPHA_GRID,
        draft_pick_half_life_grid=(None,),
        include_incumbent_team_strength=True,
    )
    print(f"Team-strength FCM: selected {asdict(candidate_config)}", flush=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"forward-conditional-team-strength-{timestamp}-{uuid4().hex[:7]}"
    output_dir = Path(artifacts_dir) / MODEL_NAME / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    base_summary.to_parquet(output_dir / "base_player_season_summary.parquet", index=False)
    augmented_summary.to_parquet(
        output_dir / "team_strength_player_season_summary.parquet", index=False
    )
    diagnostics.to_parquet(output_dir / "incumbent_team_strength.parquet", index=False)
    tuning_grid.to_parquet(output_dir / "cold_start_tuning_grid.parquet", index=False)

    metrics_rows: list[dict[str, float]] = []
    candidate_rows: list[pd.DataFrame] = []
    control_rows: list[pd.DataFrame] = []
    for index, season in enumerate(frozen_seasons, start=1):
        print(f"Team-strength FCM: frozen {index}/{len(frozen_seasons)} {season}", flush=True)
        candidate, control, metrics = evaluate_forward_conditional_team_strength(
            augmented_summary=augmented_summary,
            base_summary=base_summary,
            season=season,
            candidate_config=candidate_config,
        )
        candidate_rows.append(candidate)
        control_rows.append(control)
        metrics_rows.append(metrics)
    candidate_predictions = pd.concat(candidate_rows, ignore_index=True)
    control_predictions = pd.concat(control_rows, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    candidate_predictions.to_parquet(
        output_dir / "frozen_candidate_predictions.parquet", index=False
    )
    control_predictions.to_parquet(output_dir / "frozen_control_predictions.parquet", index=False)
    metrics.to_parquet(output_dir / "frozen_fcm_comparison.parquet", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "tuning_seasons": list(tuning_seasons),
                "frozen_seasons": list(frozen_seasons),
                "candidate_config": asdict(candidate_config),
                "control_config": asdict(PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG),
                "evaluation": "conditional MPG only; no availability, squash, allocation, or wins",
                "feature_contract": (
                    "Raw-minute-weighted exact preseason NAIL strength from opening-roster "
                    "non-rookies with prior minutes state; rookies receive but do not create "
                    "the feature."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return ForwardConditionalTeamStrengthRun(run_dir=output_dir, run_id=run_id)


def main() -> None:
    """Run the incumbent-strength FCM-only replay."""

    parser = argparse.ArgumentParser(description="Evaluate incumbent team strength in FCM")
    parser.add_argument("--tuning-seasons", nargs="+", default=list(DEFAULT_TUNING_SEASONS))
    parser.add_argument("--frozen-seasons", nargs="+", default=list(DEFAULT_FROZEN_SEASONS))
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--player-panel-path", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--nail-run-dir", type=Path, default=DEFAULT_MODEL_RUN_DIR)
    args = parser.parse_args()
    run = run_forward_conditional_team_strength(
        tuning_seasons=tuple(args.tuning_seasons),
        frozen_seasons=tuple(args.frozen_seasons),
        curated_dir=args.curated_dir,
        player_panel_path=args.player_panel_path,
        artifacts_dir=args.artifacts_dir,
        nail_run_dir=args.nail_run_dir,
    )
    print(run.run_dir)
