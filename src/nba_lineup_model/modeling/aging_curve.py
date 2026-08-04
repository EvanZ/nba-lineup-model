from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from nba_lineup_model.modeling.aging import (
    AGING_FEATURE_COLUMNS,
    fit_aging_pipeline,
    prepare_aging_transitions,
    validate_aging_model_run,
)
from nba_lineup_model.modeling.player_history import validate_player_season_panel
from nba_lineup_model.season.schema import validate_season


@dataclass(frozen=True)
class AgingCurveCaseStudy:
    """Published aging-curve report and the immutable aging model it describes."""

    report_id: str
    report_dir: Path
    source_aging_run_id: str
    source_aging_manifest_sha256: str
    reference_age: int
    bootstrap_samples: int
    bootstrap_seed: int
    training_player_season_count: int
    training_target_seasons: tuple[str, ...]
    curve: pd.DataFrame
    draft_adjusted_curves: pd.DataFrame
    forward_trajectory_curve: pd.DataFrame
    physical_profile_curves: pd.DataFrame


def extract_partial_age_curve(
    model: Pipeline,
    training: pd.DataFrame,
    *,
    reference_age: int,
    ages: np.ndarray | None = None,
    draft_cohort: str = "population",
    advance_experience: bool = False,
    physical_profile: tuple[float, float] | None = None,
    reference_override: pd.Series | None = None,
) -> pd.DataFrame:
    """Extract the model's shared age spline, centered at one reference age."""

    if training.empty:
        raise ValueError("Aging curve extraction requires nonempty training transitions")
    if reference_age < int(np.floor(training["target_age"].min())) or reference_age > int(
        np.ceil(training["target_age"].max())
    ):
        raise ValueError("Reference age must fall within observed training ages")

    age_values = (
        np.arange(
            int(np.floor(training["target_age"].min())),
            int(np.ceil(training["target_age"].max())) + 1,
            dtype=float,
        )
        if ages is None
        else np.asarray(ages, dtype=float)
    )
    if age_values.ndim != 1 or len(age_values) == 0:
        raise ValueError("Aging curve ages must be a nonempty one-dimensional array")

    reference = _reference_features(training, reference_override=reference_override)
    features = pd.DataFrame(
        {column: np.repeat(reference[column], len(age_values)) for column in AGING_FEATURE_COLUMNS}
    )
    features["target_age"] = age_values
    if advance_experience:
        features["target_nba_experience_years"] = (
            reference["target_nba_experience_years"] + age_values - float(reference_age)
        ).clip(min=0.0)
    # The population-reference curve averages the cohort-specific age slopes.
    for cohort in ("early_entry", "late_entry", "undrafted"):
        features[f"age_by_{cohort}"] = (
            (age_values - float(reference_age)) * reference[f"share_{cohort}"]
        )
    if draft_cohort != "population":
        _apply_draft_cohort(features, age_values, reference_age, draft_cohort)
    if physical_profile is not None:
        _apply_physical_profile(features, age_values, reference_age, physical_profile)
    if reference_override is not None:
        _apply_reference_profile_interactions(features, age_values, reference_age, reference)
    predictions = np.asarray(model.predict(features.loc[:, AGING_FEATURE_COLUMNS]), dtype=float)
    reference_features = features.iloc[[0]].copy()
    reference_features.loc[:, "target_age"] = float(reference_age)
    reference_features.loc[
        :,
        ["age_by_early_entry", "age_by_late_entry", "age_by_undrafted"],
    ] = 0.0
    if advance_experience:
        reference_features.loc[:, "target_nba_experience_years"] = reference[
            "target_nba_experience_years"
        ]
    reference_prediction = float(model.predict(reference_features.loc[:, AGING_FEATURE_COLUMNS])[0])
    curve = pd.DataFrame(
        {
            "age": age_values.astype(int),
            "partial_age_effect": predictions - reference_prediction,
        }
    )
    curve.loc[curve["age"].eq(reference_age), "partial_age_effect"] = 0.0
    curve["annual_change"] = curve["partial_age_effect"].shift(-1) - curve["partial_age_effect"]
    return curve


def _apply_draft_cohort(
    features: pd.DataFrame,
    ages: np.ndarray,
    reference_age: int,
    cohort: str,
) -> None:
    if cohort not in {"early_entry", "late_entry", "undrafted"}:
        raise ValueError(f"Unknown draft cohort: {cohort}")
    features.loc[:, ["age_by_early_entry", "age_by_late_entry", "age_by_undrafted"]] = 0.0
    features["is_undrafted"] = float(cohort == "undrafted")
    features["has_draft_age_estimate"] = float(cohort != "undrafted")
    features["has_draft_pick"] = float(cohort != "undrafted")
    features["draft_record_unknown"] = 0.0
    draft_ages = {"early_entry": 20.0, "late_entry": 24.0, "undrafted": np.nan}
    features["draft_age_estimate"] = draft_ages[cohort]
    features["draft_pick"] = np.nan if cohort == "undrafted" else features["draft_pick"]
    features[f"age_by_{cohort}"] = ages - float(reference_age)


def _apply_physical_profile(
    features: pd.DataFrame,
    ages: np.ndarray,
    reference_age: int,
    profile: tuple[float, float],
) -> None:
    height, weight = profile
    bmi = 703.0 * weight / height**2
    features["height_inches"] = height
    features["weight_pounds"] = weight
    features["body_mass_index"] = bmi
    features["age_by_height"] = (ages - float(reference_age)) * height
    features["age_by_body_mass_index"] = (ages - float(reference_age)) * bmi


def _apply_reference_profile_interactions(
    features: pd.DataFrame,
    ages: np.ndarray,
    reference_age: int,
    reference: dict[str, float],
) -> None:
    centered = ages - float(reference_age)
    draft_age = reference["draft_age_estimate"]
    features.loc[:, ["age_by_early_entry", "age_by_late_entry", "age_by_undrafted"]] = 0.0
    if reference["is_undrafted"] >= 0.5:
        features["age_by_undrafted"] = centered
    elif np.isfinite(draft_age) and draft_age <= 20.5:
        features["age_by_early_entry"] = centered
    elif np.isfinite(draft_age) and draft_age > 22.5:
        features["age_by_late_entry"] = centered
    features["age_by_height"] = centered * reference["height_inches"]
    features["age_by_body_mass_index"] = centered * reference["body_mass_index"]


def season_block_bootstrap_age_curve(
    training: pd.DataFrame,
    *,
    regularization: float,
    age_spline_knots: int,
    age_spline_degree: int,
    reference_age: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    ages: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Refit fixed-specification curves after resampling target seasons."""

    if bootstrap_samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    seasons = tuple(sorted(training["target_season"].astype(str).unique()))
    if len(seasons) < 2:
        raise ValueError("Season-block bootstrap requires at least two target seasons")
    groups = {
        season: training.loc[training["target_season"].astype(str).eq(season)].copy()
        for season in seasons
    }
    generator = np.random.default_rng(bootstrap_seed)
    curve_draws = np.empty((bootstrap_samples, len(ages)), dtype=float)
    annual_draws = np.empty((bootstrap_samples, len(ages)), dtype=float)
    for draw_index in range(bootstrap_samples):
        selected_seasons = generator.choice(seasons, size=len(seasons), replace=True)
        sampled = pd.concat([groups[str(season)] for season in selected_seasons], ignore_index=True)
        model = fit_aging_pipeline(
            sampled,
            regularization=regularization,
            age_spline_knots=age_spline_knots,
            age_spline_degree=age_spline_degree,
        )
        curve = extract_partial_age_curve(
            model,
            sampled,
            reference_age=reference_age,
            ages=ages,
        )
        curve_draws[draw_index] = curve["partial_age_effect"].to_numpy(dtype=float)
        annual_draws[draw_index] = curve["annual_change"].fillna(0.0).to_numpy(dtype=float)
    return curve_draws, annual_draws


def build_aging_curve_case_study(
    season: str,
    *,
    aging_run_id: str | None = None,
    artifacts_dir: Path | str = Path("artifacts/models"),
    panel_dir: Path | str = Path("data/analytical/player_season_panel"),
    reports_dir: Path | str = Path("artifacts/reports"),
    docs_asset_dir: Path | str = Path("docs/assets/images/aging"),
    reference_age: int = 29,
    bootstrap_samples: int = 250,
    bootstrap_seed: int = 20260801,
) -> AgingCurveCaseStudy:
    """Publish an immutable, source-validated case study for one aging model."""

    season = validate_season(season)
    source_dir = _resolve_aging_run(Path(artifacts_dir), season, aging_run_id)
    aging_manifest = validate_aging_model_run(source_dir)
    panel_root = Path(panel_dir)
    validate_player_season_panel(panel_root)
    panel_manifest_path = panel_root / "_manifest.json"
    if _sha256_file(panel_manifest_path) != aging_manifest.source_panel_manifest_sha256:
        raise ValueError("Aging curve panel manifest does not match the source aging run")

    parameters = json.loads((source_dir / "model_parameters.json").read_text())
    training_seasons = tuple(str(value) for value in parameters["training_target_seasons"])
    transitions = prepare_aging_transitions(pd.read_parquet(panel_root / "transitions.parquet"))
    training = transitions.loc[transitions["target_season"].isin(training_seasons)].copy()
    if len(training) != aging_manifest.training_player_season_count:
        raise ValueError("Aging curve training transition count does not match source run")
    model = joblib.load(source_dir / "model.joblib")
    if not isinstance(model, Pipeline):
        raise ValueError("Aging model artifact is not a scikit-learn Pipeline")

    curve = extract_partial_age_curve(model, training, reference_age=reference_age)
    forward_trajectory_curve = extract_partial_age_curve(
        model,
        training,
        reference_age=reference_age,
        advance_experience=True,
    )
    draft_adjusted_curves = pd.concat(
        [
            extract_partial_age_curve(
                model,
                training,
                reference_age=reference_age,
                draft_cohort=cohort,
            ).assign(draft_cohort=cohort)
            for cohort in ("early_entry", "late_entry", "undrafted")
        ],
        ignore_index=True,
    )
    physical_profiles = {
        "small_guard": (75.0, 190.0),
        "wing": (79.0, 220.0),
        "center": (83.0, 255.0),
    }
    physical_profile_curves = pd.concat(
        [
            extract_partial_age_curve(
                model,
                training,
                reference_age=reference_age,
                advance_experience=True,
                physical_profile=profile,
            ).assign(physical_profile=name)
            for name, profile in physical_profiles.items()
        ],
        ignore_index=True,
    )
    stars = {
        406: "Shaquille O'Neal",
        1495: "Tim Duncan",
        977: "Kobe Bryant",
        2544: "LeBron James",
        201939: "Stephen Curry",
        201142: "Kevin Durant",
        203999: "Nikola Jokić",
    }
    observed_stars = pd.read_parquet(panel_root / "player_seasons.parquet").loc[
        lambda frame: frame["player_id"].isin(stars),
        ["player_id", "player_name", "season", "age", "rapm"],
    ].sort_values(["player_name", "season"], kind="stable")
    ages = curve["age"].to_numpy(dtype=float)
    curve_draws, annual_draws = season_block_bootstrap_age_curve(
        training,
        regularization=float(parameters["selected_regularization"]),
        age_spline_knots=int(parameters["age_spline_knots"]),
        age_spline_degree=int(parameters["age_spline_degree"]),
        reference_age=reference_age,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        ages=ages,
    )
    curve["partial_age_effect_p05"] = np.quantile(curve_draws, 0.05, axis=0)
    curve["partial_age_effect_p95"] = np.quantile(curve_draws, 0.95, axis=0)
    curve["annual_change_p05"] = np.quantile(annual_draws, 0.05, axis=0)
    curve["annual_change_p95"] = np.quantile(annual_draws, 0.95, axis=0)
    curve.loc[curve.index[-1], ["annual_change_p05", "annual_change_p95"]] = np.nan
    curve = _merge_age_support(curve, training)

    now = datetime.now(UTC)
    report_id = f"aging-curve-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    season_dir = Path(reports_dir) / "aging_curve" / season
    report_dir = season_dir / report_id
    temporary = season_dir / f".{report_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        curve.to_parquet(temporary / "curve.parquet", index=False)
        draft_adjusted_curves.to_parquet(temporary / "draft_adjusted_curves.parquet", index=False)
        physical_profile_curves.to_parquet(
            temporary / "physical_profile_curves.parquet",
            index=False,
        )
        observed_stars.to_parquet(temporary / "star_observed_rapm.parquet", index=False)
        forward_trajectory_curve.to_parquet(
            temporary / "forward_trajectory_curve.parquet",
            index=False,
        )
        _write_curve_chart(curve, reference_age, temporary / "aging-curve.svg")
        _write_forward_trajectory_chart(
            forward_trajectory_curve,
            reference_age,
            temporary / "forward-aging-trajectory.svg",
        )
        _write_draft_adjusted_chart(
            draft_adjusted_curves,
            reference_age,
            temporary / "draft-adjusted-aging-curves.svg",
        )
        _write_physical_profile_chart(
            physical_profile_curves,
            reference_age,
            temporary / "physical-profile-aging-curves.svg",
        )
        _write_star_observed_chart(observed_stars, temporary / "star-observed-rapm.svg")
        _write_support_chart(curve, temporary / "age-support.svg")
        report = AgingCurveCaseStudy(
            report_id=report_id,
            report_dir=report_dir,
            source_aging_run_id=aging_manifest.run_id,
            source_aging_manifest_sha256=_sha256_file(source_dir / "manifest.json"),
            reference_age=reference_age,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            training_player_season_count=len(training),
            training_target_seasons=training_seasons,
            curve=curve,
            draft_adjusted_curves=draft_adjusted_curves,
            forward_trajectory_curve=forward_trajectory_curve,
            physical_profile_curves=physical_profile_curves,
        )
        (temporary / "report.md").write_text(_report_markdown(report))
        manifest = {
            "schema_version": 1,
            "report_id": report_id,
            "created_at": now.isoformat(),
            "season": season,
            "source_aging_run_id": aging_manifest.run_id,
            "source_aging_manifest_sha256": report.source_aging_manifest_sha256,
            "source_panel_manifest_sha256": aging_manifest.source_panel_manifest_sha256,
            "training_target_seasons": list(training_seasons),
            "training_player_season_count": len(training),
            "reference_age": reference_age,
            "bootstrap": {
                "method": "target-season block resampling; fixed selected specification",
                "samples": bootstrap_samples,
                "seed": bootstrap_seed,
                "interval": "5th to 95th percentile",
            },
            "generator_code_sha256": _sha256_file(Path(__file__)),
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        _write_artifact_index(temporary)
        temporary.replace(report_dir)
        asset_dir = Path(docs_asset_dir) / season
        asset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report_dir / "aging-curve.svg", asset_dir / "aging-curve.svg")
        shutil.copyfile(
            report_dir / "draft-adjusted-aging-curves.svg",
            asset_dir / "draft-adjusted-aging-curves.svg",
        )
        shutil.copyfile(
            report_dir / "forward-aging-trajectory.svg",
            asset_dir / "forward-aging-trajectory.svg",
        )
        shutil.copyfile(
            report_dir / "physical-profile-aging-curves.svg",
            asset_dir / "physical-profile-aging-curves.svg",
        )
        shutil.copyfile(
            report_dir / "star-observed-rapm.svg",
            asset_dir / "star-observed-rapm.svg",
        )
        observed_stars.to_json(
            asset_dir / "star-observed-rapm.json",
            orient="records",
            indent=2,
        )
        shutil.copyfile(report_dir / "age-support.svg", asset_dir / "age-support.svg")
        latest_path = season_dir / "latest.json"
        latest_path.write_text(json.dumps({"run_id": report_id}, indent=2) + "\n")
        return report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _reference_features(
    training: pd.DataFrame,
    *,
    reference_override: pd.Series | None = None,
) -> dict[str, float]:
    weights = training["target_rapm_possessions"].to_numpy(dtype=float)
    excluded = {
        "target_age",
        "age_by_early_entry",
        "age_by_late_entry",
        "age_by_undrafted",
    }
    reference = {
        column: float(
            np.average(training[column].fillna(training[column].median()), weights=weights)
        )
        for column in AGING_FEATURE_COLUMNS
        if column not in excluded
    }
    reference["target_age"] = 29.0
    reference["has_prior_season"] = 1.0
    reference["is_rookie"] = 0.0
    reference["share_early_entry"] = float(
        np.average(training["draft_age_estimate"].le(20.5), weights=weights)
    )
    reference["share_late_entry"] = float(
        np.average(training["draft_age_estimate"].gt(22.5), weights=weights)
    )
    reference["share_undrafted"] = float(np.average(training["is_undrafted"], weights=weights))
    for column in ("age_by_early_entry", "age_by_late_entry", "age_by_undrafted"):
        reference[column] = 0.0
    if reference_override is not None:
        for column in AGING_FEATURE_COLUMNS:
            value = reference_override.get(column)
            if value is not None and pd.notna(value):
                reference[column] = float(value)
    return reference


def _merge_age_support(curve: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    support = (
        training.groupby("target_age", as_index=False)
        .agg(
            player_seasons=("player_id", "size"),
            distinct_players=("player_id", "nunique"),
            target_rapm_possessions=("target_rapm_possessions", "sum"),
        )
        .rename(columns={"target_age": "age"})
    )
    support["age"] = support["age"].astype(int)
    output = curve.merge(support, on="age", how="left")
    for column in ("player_seasons", "distinct_players", "target_rapm_possessions"):
        output[column] = output[column].fillna(0).astype(int)
    return output


def _write_curve_chart(curve: pd.DataFrame, reference_age: int, output_path: Path) -> None:
    plt = _pyplot()
    figure, (curve_axis, change_axis) = plt.subplots(
        nrows=2,
        figsize=(9.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1]},
    )
    ages = curve["age"].to_numpy(dtype=float)
    curve_axis.fill_between(
        ages,
        curve["partial_age_effect_p05"],
        curve["partial_age_effect_p95"],
        color="#77a9d5",
        alpha=0.28,
        label="Season-block bootstrap 90% interval",
    )
    curve_axis.plot(ages, curve["partial_age_effect"], color="#1e628f", linewidth=2.5)
    curve_axis.axhline(0, color="#697786", linewidth=1, linestyle="--")
    curve_axis.axvline(reference_age, color="#d05f3c", linewidth=1.2, linestyle="--")
    curve_axis.set_ylabel("Partial RAPM effect\n(points / 100 possessions)")
    curve_axis.set_title("Conditional age effect in the forward aging model", loc="left", pad=14)
    curve_axis.legend(loc="upper left", frameon=False)

    change = curve.iloc[:-1]
    change_axis.fill_between(
        change["age"],
        change["annual_change_p05"],
        change["annual_change_p95"],
        color="#d8a06b",
        alpha=0.3,
    )
    change_axis.plot(change["age"], change["annual_change"], color="#a24c25", linewidth=2.3)
    change_axis.axhline(0, color="#697786", linewidth=1, linestyle="--")
    change_axis.set_xlabel("Target-season age")
    change_axis.set_ylabel("Estimated change\nto next age")
    _style_axes(curve_axis)
    _style_axes(change_axis)
    figure.tight_layout()
    _save_svg(figure, output_path)
    plt.close(figure)


def _write_support_chart(curve: pd.DataFrame, output_path: Path) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9.5, 3.8))
    axis.bar(curve["age"], curve["player_seasons"], width=0.78, color="#4f8c78")
    axis.set_xlabel("Target-season age")
    axis.set_ylabel("Training player-seasons")
    axis.set_title("Observed age support for the fitted curve", loc="left", pad=14)
    _style_axes(axis)
    figure.tight_layout()
    _save_svg(figure, output_path)
    plt.close(figure)


def _write_draft_adjusted_chart(
    curves: pd.DataFrame,
    reference_age: int,
    output_path: Path,
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    styles = {
        "early_entry": ("#1e628f", "Early-entry drafted (estimated age 20)"),
        "late_entry": ("#a24c25", "Late-entry drafted (estimated age 24)"),
        "undrafted": ("#4f8c78", "Undrafted"),
    }
    for cohort, (color, label) in styles.items():
        curve = curves.loc[curves["draft_cohort"].eq(cohort)]
        axis.plot(
            curve["age"],
            curve["partial_age_effect"],
            color=color,
            linewidth=2.4,
            label=label,
        )
    axis.axhline(0, color="#697786", linewidth=1, linestyle="--")
    axis.axvline(reference_age, color="#d05f3c", linewidth=1.2, linestyle="--")
    axis.set_xlabel("Target-season age")
    axis.set_ylabel("Conditional RAPM effect\n(points / 100 possessions)")
    axis.set_title("Draft-adjusted aging curves", loc="left", pad=14)
    axis.legend(frameon=False, loc="lower left")
    _style_axes(axis)
    figure.tight_layout()
    _save_svg(figure, output_path)
    plt.close(figure)


def _write_forward_trajectory_chart(
    curve: pd.DataFrame,
    reference_age: int,
    output_path: Path,
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    axis.plot(curve["age"], curve["partial_age_effect"], color="#614f9b", linewidth=2.5)
    axis.axhline(0, color="#697786", linewidth=1, linestyle="--")
    axis.axvline(reference_age, color="#d05f3c", linewidth=1.2, linestyle="--")
    axis.set_xlabel("Target-season age")
    axis.set_ylabel("Conditional RAPM effect\n(points / 100 possessions)")
    axis.set_title(
        "Forward aging trajectory (age and experience advance together)",
        loc="left",
        pad=14,
    )
    _style_axes(axis)
    figure.tight_layout()
    _save_svg(figure, output_path)
    plt.close(figure)


def _write_physical_profile_chart(
    curves: pd.DataFrame,
    reference_age: int,
    output_path: Path,
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    styles = {
        "small_guard": ("#1e628f", "Small guard (6'3\", 190 lb)"),
        "wing": ("#a24c25", "Wing (6'7\", 220 lb)"),
        "center": ("#4f8c78", "Center (6'11\", 255 lb)"),
    }
    for profile, (color, label) in styles.items():
        curve = curves.loc[curves["physical_profile"].eq(profile)]
        axis.plot(
            curve["age"],
            curve["partial_age_effect"],
            color=color,
            linewidth=2.4,
            label=label,
        )
    axis.axhline(0, color="#697786", linewidth=1, linestyle="--")
    axis.axvline(reference_age, color="#d05f3c", linewidth=1.2, linestyle="--")
    axis.set_xlabel("Target-season age")
    axis.set_ylabel("Conditional RAPM effect\n(points / 100 possessions)")
    axis.set_title("Forward height and weight profile aging curves", loc="left", pad=14)
    axis.legend(frameon=False, loc="lower left")
    _style_axes(axis)
    figure.tight_layout()
    _save_svg(figure, output_path)
    plt.close(figure)


def _write_star_observed_chart(
    observed: pd.DataFrame,
    output_path: Path,
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(10, 6.2))
    for player_name, points in observed.groupby("player_name", sort=False):
        line = axis.plot(
            points["age"],
            points["rapm"],
            linewidth=2,
            label=player_name,
        )[0]
        axis.scatter(points["age"], points["rapm"], color=line.get_color(), s=18)
    axis.axhline(0, color="#697786", linewidth=1, linestyle="--")
    axis.set_xlabel("Age")
    axis.set_ylabel("One-season RAPM\n(points / 100 possessions)")
    axis.set_title("Observed one-season RAPM by age", loc="left", pad=14)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    _style_axes(axis)
    figure.tight_layout()
    _save_svg(figure, output_path)
    plt.close(figure)


def _report_markdown(report: AgingCurveCaseStudy) -> str:
    lines = [
        "# Aging Curve Case Study",
        "",
        f"Source aging run: `{report.source_aging_run_id}`.",
        "",
        "The fixed-experience background curve is centered at "
        f"age {report.reference_age}. Bands are 5th to 95th percentiles from "
        f"{report.bootstrap_samples} target-season block bootstrap refits with the selected "
        "model specification fixed.",
        "",
        "`draft_adjusted_curves.parquet` and `draft-adjusted-aging-curves.svg` "
        "compare early-entry, late-entry, and undrafted counterfactual profiles.",
        "",
        "`forward_trajectory_curve.parquet` and `forward-aging-trajectory.svg` "
        "advance age and NBA experience together from the same reference profile.",
        "",
        "| Age | Partial effect | 90% interval | Next-age change | Player-seasons |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.curve.itertuples(index=False):
        change = "-" if pd.isna(row.annual_change) else f"{row.annual_change:+.2f}"
        lines.append(
            f"| {row.age} | {row.partial_age_effect:+.2f} | "
            f"[{row.partial_age_effect_p05:+.2f}, {row.partial_age_effect_p95:+.2f}] | "
            f"{change} | {row.player_seasons:,} |"
        )
    return "\n".join(lines) + "\n"


def _resolve_aging_run(
    artifacts_dir: Path,
    season: str,
    aging_run_id: str | None,
) -> Path:
    season_dir = artifacts_dir / "aging" / season
    if aging_run_id is None:
        aging_run_id = json.loads((season_dir / "latest.json").read_text())["run_id"]
    run_dir = season_dir / aging_run_id
    if not run_dir.is_dir():
        raise ValueError(f"Aging model run does not exist: {run_dir}")
    return run_dir


def _write_artifact_index(directory: Path) -> None:
    artifacts = [
        {
            "filename": path.name,
            "byte_count": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "artifacts.json"
    ]
    (directory / "artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n")


def _pyplot():
    cache_root = Path(tempfile.gettempdir()) / "nba-lineup-model-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.hashsalt"] = "nba-lineup-model"
    matplotlib.rcParams["svg.fonttype"] = "none"
    from matplotlib import pyplot as plt

    return plt


def _style_axes(axis) -> None:
    axis.figure.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")
    axis.grid(axis="y", color="#d5dee7", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#9aa8b6")
    axis.tick_params(colors="#172231", labelsize=9)
    axis.xaxis.label.set_color("#172231")
    axis.yaxis.label.set_color("#172231")
    axis.title.set_color("#183b63")


def _save_svg(figure, output_path: Path) -> None:
    figure.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "nba-lineup-model", "Date": None},
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a source-validated age-curve case study from an aging model run."
    )
    parser.add_argument("season", help="Aging model holdout season in YYYY-YY format")
    parser.add_argument("--aging-run-id")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--panel-dir", default="data/analytical/player_season_panel")
    parser.add_argument("--reports-dir", default="artifacts/reports")
    parser.add_argument("--docs-asset-dir", default="docs/assets/images/aging")
    parser.add_argument("--reference-age", type=int, default=29)
    parser.add_argument("--bootstrap-samples", type=int, default=250)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_aging_curve_case_study(
        args.season,
        aging_run_id=args.aging_run_id,
        artifacts_dir=args.artifacts_dir,
        panel_dir=args.panel_dir,
        reports_dir=args.reports_dir,
        docs_asset_dir=args.docs_asset_dir,
        reference_age=args.reference_age,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        f"Aging curve {report.report_id}: source={report.source_aging_run_id}, "
        f"ages={report.curve['age'].min()}-{report.curve['age'].max()}, "
        f"report={report.report_dir}"
    )


if __name__ == "__main__":
    main()
