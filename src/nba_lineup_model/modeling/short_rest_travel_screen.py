"""Fast frozen residual screen for the schedule travel candidate."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import lineup_side_context_features
from nba_lineup_model.modeling.contextual_prior import _lineup_effects
from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    _previous_season,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import MODEL_NAME
from nba_lineup_model.modeling.frozen_feature_screen import (
    DEFAULT_SEASONS,
    FeatureCandidate,
    _render_residual_screen,
    summarize_feature_bins,
    weighted_correlation,
)
from nba_lineup_model.modeling.frozen_prior_evaluation import _recover_home_intercept
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.modeling.travel_mart import build_short_rest_travel_game_features

FEATURE_NAME = "short_rest_travel"
FEATURE_LABEL = "Travel distance within 48 scheduled-tipoff hours"
DEFAULT_TRAVEL_MART_DIR = Path("data/analytical/team_game_travel")
DEFAULT_GAME_CATALOG_PATH = Path("data/catalog/games.parquet")
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/short_rest_travel_screen")
DEFAULT_CHART_PATH = Path(
    "docs/assets/images/schedule-controls/short-rest-travel-residual-screen.svg"
)
DEFAULT_MODEL_ARTIFACT_SEASON = "2025-26"


def build_short_rest_travel_screen(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
    travel_mart_dir: Path | str = DEFAULT_TRAVEL_MART_DIR,
    model_artifact_season: str = DEFAULT_MODEL_ARTIFACT_SEASON,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
) -> Path:
    """Screen short-rest travel against full frozen production residuals."""

    target_seasons = tuple(sorted({str(season) for season in seasons}))
    if not target_seasons:
        raise ValueError("At least one target season is required")
    artifacts = Path(artifacts_dir)
    analytical = Path(analytical_dir)
    panel = pd.read_parquet(player_season_panel_path)
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(target_seasons[-1])],
        through_season=target_seasons[-1],
        analytical_dir=analytical,
    )
    model_root = _latest_run(artifacts / MODEL_NAME / model_artifact_season)
    metadata = json.loads((model_root / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Travel screen requires the promoted NAIL-RAPM v1.2.1.3 artifact")
    priors = pd.read_parquet(model_root / "season_player_priors.parquet")
    coefficients = pd.read_parquet(model_root / "historical_player_coefficients.parquet")
    context_models = joblib.load(model_root / "season_context_models.joblib")
    schedule_models = joblib.load(model_root / "season_schedule_models.joblib")
    schedule_features = build_back_to_back_game_features(pd.read_parquet(game_catalog_path))
    travel_rows = pd.read_parquet(Path(travel_mart_dir) / "team_game_travel.parquet")
    travel_features = build_short_rest_travel_game_features(travel_rows)
    profile_builder = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    )

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    bins: list[pd.DataFrame] = []
    for target in target_seasons:
        print(f"Screening {FEATURE_NAME}: {target}", flush=True)
        source = _previous_season(target)
        stints = read_rapm_stints(target, analytical_dir=analytical)
        target_priors = priors.loc[priors["season"].eq(target), ["player_id", "prior_rapm"]]
        source_coefficients = coefficients.loc[
            coefficients["season"].eq(source), ["player_id", "rapm"]
        ]
        context_model = context_models.get(source)
        schedule_model = schedule_models.get(source)
        if target_priors.empty or source_coefficients.empty or context_model is None:
            raise ValueError(f"Production state is incomplete for {target}")
        if schedule_model is None:
            raise ValueError(f"Production B2B schedule state is incomplete for {source}")
        participants = set().union(*stints["home_player_ids"], *stints["away_player_ids"])
        profiles = profile_builder(
            panel,
            target_season=target,
            target_player_ids=participants,
            analytical_dir=str(analytical),
            curated_dir=str(curated_dir),
            exposure_cohort=exposure_cohort,
        )
        prior_map = dict(
            zip(target_priors["player_id"].astype(int), target_priors["prior_rapm"], strict=True)
        )
        player_edge, unknown = _lineup_effects(stints, prior_map)
        home_context = lineup_side_context_features(
            stints["home_player_ids"].tolist(), profiles, feature_set=context_model.feature_set
        )
        away_context = lineup_side_context_features(
            stints["away_player_ids"].tolist(), profiles, feature_set=context_model.feature_set
        )
        source_stints = read_rapm_stints(source, analytical_dir=analytical)
        source_intercept = _recover_home_intercept(source_stints, source_coefficients)
        predicted = (
            player_edge
            + source_intercept
            + context_model.predict_side_pairs(home_context, away_context)
            + schedule_model.predict_games(stints, schedule_features)
        )
        frame = stints.loc[
            :, ["season", "game_id", "stint_index", "possessions", "target_home_net_rating"]
        ].copy()
        frame = frame.merge(travel_features, on="game_id", how="left", validate="many_to_one")
        if frame["has_complete_short_rest_travel"].isna().any():
            raise ValueError(f"Travel mart is missing target-season games for {target}")
        frame["source_season"] = source
        frame["frozen_prediction_net_rating"] = predicted
        frame["frozen_residual_net_rating"] = (
            frame["target_home_net_rating"].to_numpy(dtype=float) - predicted
        )
        frame["unknown_player_exposures"] = unknown
        eligible = frame.loc[frame["has_complete_short_rest_travel"]].copy()
        eligible["feature_edge"] = eligible[
            "home_minus_away_short_rest_travel_thousand_miles"
        ].astype(float)
        summary, summary_bins = _summarize(target, source, eligible)
        summaries.append(summary)
        bins.append(summary_bins)
        frames.append(frame)
        print(
            f"Completed {target}: eligible_stints={len(eligible):,} "
            f"eligible_possessions={int(eligible['possessions'].sum()):,}",
            flush=True,
        )

    screen = pd.concat(frames, ignore_index=True)
    eligible_screen = screen.loc[screen["has_complete_short_rest_travel"]].copy()
    eligible_screen["feature_edge"] = eligible_screen[
        "home_minus_away_short_rest_travel_thousand_miles"
    ].astype(float)
    pooled_summary, pooled_bins = _summarize("pooled", "multiple", eligible_screen)
    summaries.append(pooled_summary)
    bins.append(pooled_bins)
    bin_frame = pd.concat(bins, ignore_index=True)
    resolved_chart = Path(chart_path)
    resolved_chart.parent.mkdir(parents=True, exist_ok=True)
    _render_residual_screen(
        bin_frame,
        FeatureCandidate(
            name=FEATURE_NAME,
            label=FEATURE_LABEL,
            description=(
                "Signed home-minus-away travel distance gated at 48 scheduled-tipoff hours."
            ),
        ),
        target_seasons,
        resolved_chart,
    )
    run_id = f"short-rest-travel-screen-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    screen.to_parquet(run_dir / "stint_residuals.parquet", index=False)
    bin_frame.to_parquet(run_dir / "residual_bins.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(run_dir / "season_summary.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "feature": FEATURE_NAME,
                "label": FEATURE_LABEL,
                "model": MODEL_NAME,
                "model_run_id": metadata["run_id"],
                "target_seasons": list(target_seasons),
                "baseline_contract": (
                    "Full frozen production prediction, including source-season home-court "
                    "intercept, B2B schedule control, player edge, and lineup context."
                ),
                "missing_travel_contract": (
                    "Games where either team lacks a prior competitive game in the same season "
                    "are retained in stint_residuals but excluded from the screen."
                ),
                "travel_mart_path": str(travel_mart_dir),
                "chart_path": str(resolved_chart),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (Path(output_root) / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return run_dir


def _summarize(
    season: str,
    source_season: str,
    frame: pd.DataFrame,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    weights = frame["possessions"].to_numpy(dtype=float)
    feature = frame["feature_edge"].to_numpy(dtype=float)
    residual = frame["frozen_residual_net_rating"].to_numpy(dtype=float)
    feature_mean = float(np.average(feature, weights=weights))
    feature_scale = float(np.sqrt(np.average(np.square(feature - feature_mean), weights=weights)))
    standardized = (feature - feature_mean) / feature_scale
    design = np.column_stack((np.ones(len(frame)), standardized))
    coefficients, *_ = np.linalg.lstsq(
        design * np.sqrt(weights)[:, None], residual * np.sqrt(weights), rcond=None
    )
    bins = summarize_feature_bins(frame)
    bins.insert(0, "season", season)
    bins.insert(1, "source_season", source_season)
    bins.insert(2, "feature", FEATURE_NAME)
    summary = {
        "season": season,
        "source_season": source_season,
        "stint_count": len(frame),
        "possession_count": float(weights.sum()),
        "feature_standard_deviation_thousand_miles": feature_scale,
        "weighted_correlation": weighted_correlation(feature, residual, weights),
        "standardized_residual_weight": float(coefficients[1]),
        "raw_residual_weight_per_thousand_miles": float(coefficients[1] / feature_scale),
    }
    bins = pd.concat(
        [
            bins,
            pd.DataFrame(
                [
                    {
                        "season": season,
                        "source_season": source_season,
                        "feature": FEATURE_NAME,
                        "bin_kind": "season_summary",
                        "bin": np.nan,
                        "stint_count": len(frame),
                        "possession_count": float(weights.sum()),
                        "feature_mean": feature_mean,
                        "residual_mean": float(np.average(residual, weights=weights)),
                        "residual_ci_low": np.nan,
                        "residual_ci_high": np.nan,
                        "effective_stint_count": np.nan,
                        "weighted_correlation": summary["weighted_correlation"],
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    return summary, bins


def main() -> None:
    """Run the frozen short-rest travel residual screen."""

    parser = argparse.ArgumentParser(description="Screen a short-rest travel schedule control")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--travel-mart-dir", default=str(DEFAULT_TRAVEL_MART_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    args = parser.parse_args()
    run = build_short_rest_travel_screen(
        seasons=tuple(args.seasons),
        travel_mart_dir=args.travel_mart_dir,
        output_root=args.output_root,
        chart_path=args.chart_path,
    )
    print(f"Short-rest travel screen: run={run}")


if __name__ == "__main__":
    main()
