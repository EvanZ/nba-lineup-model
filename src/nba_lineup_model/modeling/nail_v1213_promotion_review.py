"""Materialize ranking and coefficient-history review for NAIL v1.2.1.3."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
)
from nba_lineup_model.modeling.contextual_profiles import (
    build_contextual_player_profiles,
    prepare_player_exposure_cohort,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import MODEL_NAME
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    _load_recursive_model_mapping,
)
from nba_lineup_model.modeling.nail_v121_pruned_nonadditive_weight_audit import (
    RETAINED_FEATURES,
)
from nba_lineup_model.modeling.nail_v13_additive_weight_audit import (
    render_additive_weight_trajectories,
    standardized_additive_weights,
    summarize_additive_weights,
)
from nba_lineup_model.web_api.inference import (
    _compiled_linear_x3_coefficients,
    _fold_imputed_profile_into_prior,
    _player_rating_center,
    _published_profile_padding_contract,
)


DEFAULT_ARTIFACT_SEASON = "2025-26"
DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/nail_v1213_promotion_review")
DEFAULT_CONTROL_CHART_PATH = Path("docs/assets/images/nail-v1213/control-trajectories.svg")
DEFAULT_ADDITIVE_CHART_PATH = Path(
    "docs/assets/images/nail-v1213/additive-profile-weight-trajectories.svg"
)
DEFAULT_NONADDITIVE_CHART_PATH = Path(
    "docs/assets/images/nail-v1213/nonadditive-weight-trajectories.svg"
)


@dataclass(frozen=True)
class NailV1213PromotionReview:
    """Persisted review artifact derived solely from a completed candidate fit."""

    run_dir: Path
    source_run_dir: Path
    control_chart_path: Path
    additive_chart_path: Path
    nonadditive_chart_path: Path


def _latest_run(root: Path) -> Path:
    latest_path = root / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"No latest artifact pointer: {latest_path}")
    run_id = str(json.loads(latest_path.read_text())["run_id"])
    return root / run_id


def _completed_top_25(source: Path, panel: pd.DataFrame) -> pd.DataFrame:
    metadata = json.loads((source / "metadata.json").read_text())
    target = str(metadata["target_season"])
    raw = pd.read_parquet(source / "player_season_ratings.parquet")
    season_ratings = raw.loc[raw["season"].eq(target)].copy()
    if season_ratings.empty:
        raise ValueError(f"Candidate artifact has no completed ratings for {target}")
    models = _load_recursive_model_mapping(source, "season_context_models.joblib")
    model = models.get(target)
    if model is None:
        raise ValueError(f"Candidate artifact has no context model for {target}")
    padding_contract = _published_profile_padding_contract(metadata)
    profile_ids = season_ratings["player_id"].astype(int).tolist()
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(target)],
        through_season=target,
    )
    profiles = build_contextual_player_profiles(
        panel,
        target_season=target,
        target_player_ids=profile_ids,
        exposure_cohort=exposure_cohort,
        padding_contract=padding_contract,
        use_last_observed_profile=(
            metadata.get("profile_padding_contract", {}).get("gap_returner_profile_method")
            == "last_observed_padded_profile"
        ),
    )
    base = season_ratings.loc[:, ["player_id", "rapm"]].copy()
    uncentered = _compiled_linear_x3_coefficients(base, profiles, model)
    exposure = panel.loc[
        panel["season"].astype(str).eq(target), ["player_id", "rapm_possessions"]
    ].copy()
    if exposure["player_id"].duplicated().any():
        exposure = (
            exposure.sort_values("rapm_possessions", ascending=False, kind="stable")
            .drop_duplicates("player_id", keep="first")
        )
    center = _player_rating_center(
        uncentered,
        exposure.rename(columns={"rapm_possessions": "possessions"}),
    )
    compiled = _compiled_linear_x3_coefficients(base, profiles, model, center=center)
    compiled_by_player = compiled.set_index("player_id")["rapm"]
    output = season_ratings.copy()
    output["additive_profile_adjustment"] = (
        output["player_id"].map(compiled_by_player).astype(float) - output["rapm"].astype(float)
    )
    output["rapm"] = output["player_id"].map(compiled_by_player).astype(float)
    output = _fold_imputed_profile_into_prior(output, profiles)
    details = panel.loc[
        panel["season"].astype(str).eq(target),
        ["player_id", "primary_team_tricode", "listed_position", "rapm_possessions"],
    ].copy()
    output = output.merge(details, on="player_id", how="left", validate="one_to_one")
    output = output.sort_values(["rapm", "player_name", "player_id"], ascending=[False, True, True])
    output.insert(0, "rank", np.arange(1, len(output) + 1))
    columns = [
        "rank",
        "player_id",
        "player_name",
        "primary_team_tricode",
        "listed_position",
        "rapm",
        "prior_rapm",
        "rapm_adjustment_from_prior",
        "additive_profile_adjustment",
        "rapm_possessions",
    ]
    return output.loc[:, columns].head(25).reset_index(drop=True)


def _control_history(source: Path) -> pd.DataFrame:
    context = pd.read_parquet(source / "season_context_metadata.parquet")
    schedule = pd.read_parquet(source / "season_schedule_control_metadata.parquet")
    required_context = {"season", "context_home_intercept"}
    required_schedule = {"season", "schedule_control_raw_weight"}
    if required_context - set(context) or required_schedule - set(schedule):
        raise ValueError("Candidate artifact lacks HCA or B2B coefficient history")
    output = context.loc[:, ["season", "context_home_intercept"]].merge(
        schedule.loc[:, ["season", "schedule_control_raw_weight"]],
        on="season",
        how="inner",
        validate="one_to_one",
    )
    output["season_start_year"] = output["season"].str[:4].astype(int)
    return output.sort_values("season_start_year", kind="stable").reset_index(drop=True)


def _render_control_history(history: pd.DataFrame, path: Path, *, title_prefix: str) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True, layout="constrained")
    series = (
        ("context_home_intercept", "Home-court advantage", "#2f6ea8"),
        ("schedule_control_raw_weight", "Home-minus-away back-to-back effect", "#e8502f"),
    )
    for axis, (column, title, color) in zip(axes, series, strict=True):
        axis.axhline(0.0, color="#767d76", linewidth=0.8, linestyle="--", zorder=0)
        axis.plot(
            history["season_start_year"],
            history[column],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
        )
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("Home net-rating points", fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8)
    axes[-1].set_xlabel("Completed source season start year", fontsize=9)
    figure.suptitle(
        f"{title_prefix} external-control coefficients by completed source season",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.99,
        0.975,
        "Blue: season-specific HCA intercept | Orange: home B2B minus away B2B",
        ha="right",
        va="top",
        fontsize=8.5,
        color="#5f6860",
    )
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def build_nail_v1213_promotion_review(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    control_chart_path: Path | str = DEFAULT_CONTROL_CHART_PATH,
    additive_chart_path: Path | str = DEFAULT_ADDITIVE_CHART_PATH,
    nonadditive_chart_path: Path | str = DEFAULT_NONADDITIVE_CHART_PATH,
    expected_model_name: str | None = MODEL_NAME,
    chart_title_prefix: str = "NAIL-RAPM v1.2.1.3",
) -> NailV1213PromotionReview:
    """Write inspectable rankings and all fitted HCA, B2B, and context histories."""

    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    source_model = str(metadata.get("model"))
    if expected_model_name is not None and source_model != expected_model_name:
        raise ValueError(
            f"Promotion review expected {expected_model_name!r}, found {source_model!r}"
        )
    panel = pd.read_parquet(panel_path)
    top_25 = _completed_top_25(source, panel)
    controls = _control_history(source)
    models = _load_recursive_model_mapping(source, "season_context_models.joblib")
    additive = standardized_additive_weights(
        models,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
        features=LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
        accent_features=frozenset(),
    )
    nonadditive = standardized_additive_weights(
        models,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
        features=RETAINED_FEATURES,
        accent_features=frozenset(RETAINED_FEATURES),
    )
    control_chart = Path(control_chart_path)
    additive_chart = Path(additive_chart_path)
    nonadditive_chart = Path(nonadditive_chart_path)
    for path in (control_chart, additive_chart, nonadditive_chart):
        path.parent.mkdir(parents=True, exist_ok=True)
    _render_control_history(controls, control_chart, title_prefix=chart_title_prefix)
    render_additive_weight_trajectories(
        additive,
        summarize_additive_weights(additive),
        additive_chart,
        title=f"{chart_title_prefix} additive profile weights by completed source season",
        features=LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
        accent_features=frozenset(),
        legend="Blue: player-attributable additive profile term",
    )
    render_additive_weight_trajectories(
        nonadditive,
        summarize_additive_weights(nonadditive),
        nonadditive_chart,
        title=f"{chart_title_prefix} retained non-additive weights by completed source season",
        features=RETAINED_FEATURES,
        accent_features=frozenset(RETAINED_FEATURES),
        legend="Orange: retained non-additive lineup term",
    )
    root = Path(output_root)
    run_id = f"nail-v1213-promotion-review-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    top_25.to_parquet(run_dir / "completed_2025_26_top_25.parquet", index=False)
    controls.to_parquet(run_dir / "hca_b2b_time_series.parquet", index=False)
    additive.to_parquet(run_dir / "additive_context_weight_time_series.parquet", index=False)
    nonadditive.to_parquet(run_dir / "nonadditive_context_weight_time_series.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source),
                "source_model": source_model,
                "rating_definition": (
                    "Completed NAIL rating = fitted player RAPM plus the centered, "
                    "exactly compilable additive profile adjustment."
                ),
                "hca_definition": "Season-specific home-net-rating intercept.",
                "b2b_definition": "Raw home-minus-away back-to-back coefficient.",
                "context_weight_definition": (
                    "Ridge coefficient after StandardScaler of the home-minus-away "
                    "five-man aggregate feature."
                ),
                "control_chart_path": str(control_chart),
                "additive_chart_path": str(additive_chart),
                "nonadditive_chart_path": str(nonadditive_chart),
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return NailV1213PromotionReview(
        run_dir=run_dir,
        source_run_dir=source,
        control_chart_path=control_chart,
        additive_chart_path=additive_chart,
        nonadditive_chart_path=nonadditive_chart,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v1.2.1.3 promotion review")
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--control-chart-path", default=str(DEFAULT_CONTROL_CHART_PATH))
    parser.add_argument("--additive-chart-path", default=str(DEFAULT_ADDITIVE_CHART_PATH))
    parser.add_argument("--nonadditive-chart-path", default=str(DEFAULT_NONADDITIVE_CHART_PATH))
    parser.add_argument("--expected-model-name", default=MODEL_NAME)
    parser.add_argument("--chart-title-prefix", default="NAIL-RAPM v1.2.1.3")
    args = parser.parse_args()
    run = build_nail_v1213_promotion_review(
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        control_chart_path=args.control_chart_path,
        additive_chart_path=args.additive_chart_path,
        nonadditive_chart_path=args.nonadditive_chart_path,
        expected_model_name=args.expected_model_name,
        chart_title_prefix=args.chart_title_prefix,
    )
    print(
        f"NAIL v1.2.1.3 promotion review: run={run.run_dir}; "
        f"controls={run.control_chart_path}; additive={run.additive_chart_path}; "
        f"nonadditive={run.nonadditive_chart_path}"
    )


if __name__ == "__main__":
    main()
