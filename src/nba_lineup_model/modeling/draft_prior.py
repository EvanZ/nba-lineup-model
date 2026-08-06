"""Interpretable draft-informed cold-start RAPM prior and diagnostic report."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_lineup_model.evaluation.metrics import mean_squared_error
from nba_lineup_model.modeling.player_history import (
    player_history_code_fingerprint,
    validate_player_season_panel,
)
from nba_lineup_model.season.schema import validate_season

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_DOCS_ASSET_DIR = Path("docs/assets/images/draft-prior")
DEFAULT_REGULARIZATION_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_BOOTSTRAP_SAMPLES = 250
DEFAULT_BOOTSTRAP_SEED = 20260805
DEFAULT_STABILITY_MINIMUM_SEASONS = 12
FEATURE_COLUMNS = (
    "draft_pick_linear",
    "draft_pick_quadratic",
    "undrafted",
    "later_round",
    "draft_unknown",
    "draft_age",
    "draft_pick_draft_age_interaction",
    "height_inches",
    "body_mass_index",
)
RAW_PICK_BINS = (
    (1, 3, "1-3"),
    (4, 10, "4-10"),
    (11, 20, "11-20"),
    (21, 30, "21-30"),
    (31, 45, "31-45"),
    (46, 60, "46-60"),
)


@dataclass(frozen=True)
class DraftPriorStudy:
    """Immutable draft-prior report outputs."""

    run_dir: Path
    run_id: str
    selected_regularization: float
    training_player_count: int
    rookie_ranking_count: int


def build_draft_prior_study(
    *,
    season: str = "2025-26",
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    docs_asset_dir: Path | str = DEFAULT_DOCS_ASSET_DIR,
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> DraftPriorStudy:
    """Fit a pre-target draft prior and publish its curve and rookie rankings."""

    target_season = validate_season(season)
    if bootstrap_samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    _validate_regularization_grid(regularization_grid)
    panel_path = Path(player_season_panel_path)
    validate_player_season_panel(panel_path.parent)
    panel = pd.read_parquet(panel_path)
    training, target_rookies = prepare_draft_prior_data(panel, target_season=target_season)
    selected_regularization, cv_results = select_regularization(
        training,
        regularization_grid=regularization_grid,
    )
    regularization_stability = rolling_regularization_stability(
        training,
        regularization_grid=regularization_grid,
    )
    model = fit_draft_prior_model(training, regularization=selected_regularization)
    raw_curve = empirical_draft_curve(training)
    adjusted_curve = adjusted_draft_curve(model, training)
    bootstrap = bootstrap_adjusted_draft_curve(
        training,
        regularization=selected_regularization,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    adjusted_curve = adjusted_curve.merge(bootstrap, on="draft_pick", validate="one_to_one")
    rookie_rankings = rank_target_rookies(model, target_rookies, bootstrap, training)
    return _write_study(
        target_season=target_season,
        panel_path=panel_path,
        training=training,
        target_rookies=target_rookies,
        model=model,
        selected_regularization=selected_regularization,
        cv_results=cv_results,
        regularization_stability=regularization_stability,
        raw_curve=raw_curve,
        adjusted_curve=adjusted_curve,
        rookie_rankings=rookie_rankings,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        artifacts_dir=Path(artifacts_dir),
        docs_asset_dir=Path(docs_asset_dir),
    )


def prepare_draft_prior_data(
    panel: pd.DataFrame,
    *,
    target_season: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select first-NBA-season training labels and target preseason profiles."""

    required = {
        "season",
        "player_id",
        "player_name",
        "is_rookie",
        "rapm",
        "rapm_possessions",
        "draft_round",
        "draft_number",
        "is_undrafted",
        "age",
        "height_inches",
        "weight_pounds",
        "listed_position",
    }
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Player-season panel missing draft-prior columns: {sorted(missing)}")
    target_year = int(target_season[:4])
    rookies = panel.loc[panel["is_rookie"].astype(bool)].copy()
    rookies["season"] = rookies["season"].astype(str)
    rookies["season_start_year"] = rookies["season"].str[:4].astype(int)
    training_raw = rookies.loc[rookies["season_start_year"].lt(target_year)].copy()
    target_raw = rookies.loc[rookies["season"].eq(target_season)].copy()
    if training_raw.empty or target_raw.empty:
        raise ValueError("Draft prior requires historical and target rookie cohorts")
    training = _prepare_features(training_raw)
    historical_reference = _feature_reference(_feature_base(training_raw))
    target = _prepare_features(target_raw, reference=historical_reference)
    if not training["rapm_possessions"].gt(0).all():
        raise ValueError("Draft-prior labels require positive RAPM possessions")
    target = target.drop(columns=["rapm", "rapm_possessions"], errors="ignore")
    return (
        training.sort_values(["season_start_year", "player_id"], kind="stable").reset_index(
            drop=True
        ),
        target.sort_values("player_id", kind="stable").reset_index(drop=True),
    )


def select_regularization(
    training: pd.DataFrame,
    *,
    regularization_grid: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    """Select ridge strength using expanding season-level validation folds."""

    seasons = tuple(sorted(training["season"].unique(), key=lambda value: int(value[:4])))
    if len(seasons) < 8:
        raise ValueError("Draft prior requires at least eight historical rookie seasons")
    validation_seasons = seasons[-6:]
    rows: list[dict[str, float | int | str]] = []
    for validation_season in validation_seasons:
        train = training.loc[training["season"].lt(validation_season)]
        validation = training.loc[training["season"].eq(validation_season)]
        for regularization in regularization_grid:
            model = fit_draft_prior_model(train, regularization=regularization)
            prediction = model.predict(validation.loc[:, FEATURE_COLUMNS])
            weights = validation["rapm_possessions"].to_numpy(dtype=float)
            mse = mean_squared_error(
                validation["rapm"].to_numpy(dtype=float), prediction, weights
            )
            rows.append(
                {
                    "validation_season": validation_season,
                    "regularization": regularization,
                    "training_player_count": len(train),
                    "validation_player_count": len(validation),
                    "validation_possessions": float(weights.sum()),
                    "squared_error_sum": float(mse * weights.sum()),
                    "weighted_mse": mse,
                }
            )
    results = pd.DataFrame(rows)
    summary = results.groupby("regularization", as_index=False).agg(
        squared_error_sum=("squared_error_sum", "sum"),
        validation_possessions=("validation_possessions", "sum"),
    )
    summary["weighted_mse"] = summary["squared_error_sum"] / summary["validation_possessions"]
    selected = summary.sort_values(["weighted_mse", "regularization"], kind="stable").iloc[0]
    return float(selected["regularization"]), results.sort_values(
        ["regularization", "validation_season"], kind="stable"
    ).reset_index(drop=True)


def fit_draft_prior_model(training: pd.DataFrame, *, regularization: float) -> Pipeline:
    """Fit the fixed interpretable ridge specification used by the report."""

    if regularization < 0:
        raise ValueError("Draft-prior regularization must be non-negative")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "ridge",
                Ridge(
                    alpha=regularization * len(training),
                    solver="lsqr",
                    tol=1e-8,
                ),
            ),
        ]
    )
    weights = training["rapm_possessions"].to_numpy(dtype=float)
    model.fit(
        training.loc[:, FEATURE_COLUMNS],
        training["rapm"].to_numpy(dtype=float),
        ridge__sample_weight=weights / np.mean(weights),
    )
    return model


def rolling_regularization_stability(
    training: pd.DataFrame,
    *,
    regularization_grid: tuple[float, ...],
    minimum_seasons: int = DEFAULT_STABILITY_MINIMUM_SEASONS,
) -> pd.DataFrame:
    """Record expanding-cutoff regularization choices without future labels."""

    seasons = tuple(sorted(training["season"].unique(), key=lambda value: int(value[:4])))
    if minimum_seasons < 8 or len(seasons) < minimum_seasons:
        raise ValueError("Draft-prior stability requires at least eight historical seasons")
    rows: list[dict[str, float | int | str]] = []
    for training_last_season in seasons[minimum_seasons - 1 :]:
        snapshot = training.loc[training["season"].le(training_last_season)]
        selected, folds = select_regularization(
            snapshot,
            regularization_grid=regularization_grid,
        )
        summary = folds.groupby("regularization", as_index=False).agg(
            squared_error_sum=("squared_error_sum", "sum"),
            validation_possessions=("validation_possessions", "sum"),
        )
        summary["weighted_validation_rmse"] = np.sqrt(
            summary["squared_error_sum"] / summary["validation_possessions"]
        )
        selected_row = summary.loc[summary["regularization"].eq(selected)].iloc[0]
        rows.append(
            {
                "training_last_season": training_last_season,
                "training_season_count": len(snapshot["season"].unique()),
                "validation_first_season": str(folds["validation_season"].min()),
                "validation_last_season": str(folds["validation_season"].max()),
                "selected_regularization": selected,
                "selected_weighted_validation_rmse": float(
                    selected_row["weighted_validation_rmse"]
                ),
            }
        )
    return pd.DataFrame(rows)


def empirical_draft_curve(training: pd.DataFrame) -> pd.DataFrame:
    """Return possession-weighted observed first-season RAPM by draft tier."""

    rows: list[dict[str, float | int | str]] = []
    for lower, upper, label in RAW_PICK_BINS:
        cohort = training.loc[
            training["draft_status"].eq("drafted_1_60")
            & training["draft_number"].between(lower, upper)
        ].copy()
        rows.append(_empirical_row(label, cohort, draft_pick=(lower + upper) / 2.0))
    for label, column in (
        ("Undrafted", "undrafted"),
        ("Later round / pick > 60", "later_round"),
        ("Draft record unknown", "draft_unknown"),
    ):
        cohort = training.loc[training[column].eq(1.0)].copy()
        rows.append(_empirical_row(label, cohort, draft_pick=np.nan))
    return pd.DataFrame(rows)


def adjusted_draft_curve(model: Pipeline, training: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the fitted draft effect for picks 1--60 at a reference profile."""

    reference = _reference_profile(training)
    picks = np.arange(1, 61, dtype=float)
    curve = pd.DataFrame(
        {column: np.repeat(reference[column], len(picks)) for column in FEATURE_COLUMNS}
    )
    curve["draft_pick_linear"] = (picks - 30.5) / 29.5
    curve["draft_pick_quadratic"] = curve["draft_pick_linear"].pow(2)
    curve["draft_pick_draft_age_interaction"] = 0.0
    curve[["undrafted", "later_round", "draft_unknown"]] = 0.0
    curve["draft_pick"] = picks.astype(int)
    curve["draft_prior"] = model.predict(curve.loc[:, FEATURE_COLUMNS])
    return curve.loc[:, ["draft_pick", "draft_prior"]]


def bootstrap_adjusted_draft_curve(
    training: pd.DataFrame,
    *,
    regularization: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Season-block bootstrap intervals for the adjusted draft curve."""

    seasons = tuple(sorted(training["season"].unique(), key=lambda value: int(value[:4])))
    groups = {season: training.loc[training["season"].eq(season)] for season in seasons}
    generator = np.random.default_rng(bootstrap_seed)
    draws = np.empty((bootstrap_samples, 60), dtype=float)
    for index in range(bootstrap_samples):
        sampled_seasons = generator.choice(seasons, size=len(seasons), replace=True)
        sampled = pd.concat([groups[str(season)] for season in sampled_seasons], ignore_index=True)
        draws[index] = adjusted_draft_curve(
            fit_draft_prior_model(sampled, regularization=regularization), sampled
        )["draft_prior"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "draft_pick": np.arange(1, 61, dtype=int),
            "bootstrap_lower": np.quantile(draws, 0.05, axis=0),
            "bootstrap_upper": np.quantile(draws, 0.95, axis=0),
        }
    )


def rank_target_rookies(
    model: Pipeline,
    target_rookies: pd.DataFrame,
    bootstrap_curve: pd.DataFrame,
    training: pd.DataFrame,
) -> pd.DataFrame:
    """Score target first-NBA-season players from preseason draft profiles only."""

    columns = tuple(
        dict.fromkeys(
            (
                "player_id",
                "player_name",
                "listed_position",
                "draft_round",
                "draft_number",
                "is_undrafted",
                "age",
                "height_inches",
                "weight_pounds",
                "draft_status",
                *FEATURE_COLUMNS,
            )
        )
    )
    output = target_rookies.loc[:, columns].copy()
    output["draft_prior"] = model.predict(output.loc[:, FEATURE_COLUMNS])
    interval = bootstrap_curve.set_index("draft_pick")
    drafted = output["draft_status"].eq("drafted_1_60")
    pick = pd.to_numeric(output["draft_number"], errors="coerce").astype("Int64")
    output["prior_lower"] = pick.map(interval["bootstrap_lower"])
    output["prior_upper"] = pick.map(interval["bootstrap_upper"])
    fallback_std = _weighted_residual_std(model, training)
    output.loc[~drafted, "prior_lower"] = output.loc[~drafted, "draft_prior"] - fallback_std
    output.loc[~drafted, "prior_upper"] = output.loc[~drafted, "draft_prior"] + fallback_std
    output = output.sort_values(
        ["draft_prior", "player_id"], ascending=[False, True], kind="stable"
    )
    output["rank"] = np.arange(1, len(output) + 1)
    return output.loc[
        :,
        [
            "rank",
            "player_id",
            "player_name",
            "listed_position",
            "draft_status",
            "draft_round",
            "draft_number",
            "is_undrafted",
            "age",
            "height_inches",
            "weight_pounds",
            "draft_prior",
            "prior_lower",
            "prior_upper",
        ],
    ].reset_index(drop=True)


def _feature_base(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in (
        "draft_year",
        "draft_round",
        "draft_number",
        "age",
        "height_inches",
        "weight_pounds",
    ):
        if column not in output:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["is_undrafted"] = output["is_undrafted"].fillna(False).astype(bool)
    output["draft_status"] = np.select(
        [
            output["is_undrafted"],
            output["draft_number"].between(1, 60),
            output["draft_number"].gt(60),
        ],
        ["undrafted", "drafted_1_60", "later_round"],
        default="draft_unknown",
    )
    output["draft_pick"] = output["draft_number"].where(
        output["draft_status"].eq("drafted_1_60")
    )
    output["draft_age"] = output["age"] - (
        output["season_start_year"] - output["draft_year"]
    )
    output["draft_age"] = output["draft_age"].where(output["draft_age"].between(17.0, 30.0))
    output["body_mass_index"] = 703.0 * output["weight_pounds"] / output["height_inches"].pow(2)
    return output


def _feature_reference(frame: pd.DataFrame) -> dict[str, float]:
    reference = {
        "draft_pick": float(frame["draft_pick"].median()),
        "draft_age": float(frame["draft_age"].median()),
        "height_inches": float(frame["height_inches"].median()),
        "body_mass_index": float(frame["body_mass_index"].median()),
    }
    if not all(np.isfinite(value) for value in reference.values()):
        raise ValueError("Draft-prior training data cannot establish feature imputation values")
    return reference


def _prepare_features(
    frame: pd.DataFrame,
    *,
    reference: dict[str, float] | None = None,
) -> pd.DataFrame:
    output = _feature_base(frame)
    feature_reference = reference or _feature_reference(output)
    for column, value in feature_reference.items():
        output[column] = output[column].fillna(value)
    output["draft_pick_linear"] = (output["draft_pick"] - 30.5) / 29.5
    output["draft_pick_quadratic"] = output["draft_pick_linear"].pow(2)
    output["draft_pick_draft_age_interaction"] = np.where(
        output["draft_status"].eq("drafted_1_60"),
        output["draft_pick_linear"] * (output["draft_age"] - feature_reference["draft_age"]),
        0.0,
    )
    output["undrafted"] = output["draft_status"].eq("undrafted").astype(float)
    output["later_round"] = output["draft_status"].eq("later_round").astype(float)
    output["draft_unknown"] = output["draft_status"].eq("draft_unknown").astype(float)
    return output


def _empirical_row(
    label: str,
    cohort: pd.DataFrame,
    *,
    draft_pick: float,
) -> dict[str, float | int | str]:
    if cohort.empty:
        return {
            "draft_tier": label,
            "draft_pick": draft_pick,
            "player_count": 0,
            "possessions": 0.0,
            "observed_rapm": np.nan,
        }
    weights = cohort["rapm_possessions"].to_numpy(dtype=float)
    return {
        "draft_tier": label,
        "draft_pick": draft_pick,
        "player_count": len(cohort),
        "possessions": float(weights.sum()),
        "observed_rapm": float(np.average(cohort["rapm"], weights=weights)),
    }


def _reference_profile(training: pd.DataFrame) -> dict[str, float]:
    drafted = training.loc[training["draft_status"].eq("drafted_1_60")]
    source = drafted if not drafted.empty else training
    return {column: float(source[column].median()) for column in FEATURE_COLUMNS}


def _weighted_residual_std(model: Pipeline, training: pd.DataFrame) -> float:
    residual = training["rapm"].to_numpy(dtype=float) - model.predict(
        training.loc[:, FEATURE_COLUMNS]
    )
    weights = training["rapm_possessions"].to_numpy(dtype=float)
    return float(np.sqrt(np.average(residual**2, weights=weights)))


def _write_study(
    *,
    target_season: str,
    panel_path: Path,
    training: pd.DataFrame,
    target_rookies: pd.DataFrame,
    model: Pipeline,
    selected_regularization: float,
    cv_results: pd.DataFrame,
    regularization_stability: pd.DataFrame,
    raw_curve: pd.DataFrame,
    adjusted_curve: pd.DataFrame,
    rookie_rankings: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
    artifacts_dir: Path,
    docs_asset_dir: Path,
) -> DraftPriorStudy:
    now = datetime.now(UTC)
    run_id = f"draft-prior-{target_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "draft_prior" / target_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        tables = {
            "training_first_nba_season_players.parquet": training,
            "target_first_nba_season_players.parquet": target_rookies,
            "cross_validation.parquet": cv_results,
            "regularization_stability.parquet": regularization_stability,
            "empirical_draft_curve.parquet": raw_curve,
            "adjusted_draft_curve.parquet": adjusted_curve,
            "rookie_rankings.parquet": rookie_rankings,
        }
        for filename, table in tables.items():
            table.to_parquet(temporary / filename, index=False)
        joblib.dump(model, temporary / "model.joblib")
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": "draft_informed_first_nba_season_rapm_prior",
            "season": target_season,
            "target_season": target_season,
            "training_last_season": max(
                training["season"].unique(), key=lambda value: int(value[:4])
            ),
            "training_seasons": sorted(
                training["season"].unique(), key=lambda value: int(value[:4])
            ),
            "training_player_count": len(training),
            "target_rookie_count": len(target_rookies),
            "selected_regularization": selected_regularization,
            "regularization_validation_season_count": 6,
            "stability_minimum_training_seasons": DEFAULT_STABILITY_MINIMUM_SEASONS,
            "regularization_convention": "regularization * training_row_count",
            "features": list(FEATURE_COLUMNS),
            "target": "same-season regular-only canonical RAPM",
            "target_weight": "same-season reconstructed on-court possessions",
            "temporal_boundary": "all model inputs end before target season",
            "target_outcomes_used_for_fit": False,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "source_panel_path": str(panel_path),
            "source_panel_manifest_sha256": _sha256_file(panel_path.parent / "_manifest.json"),
            "code_version": player_history_code_fingerprint((Path(__file__),)),
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        _render_curve(raw_curve, adjusted_curve, temporary / "draft-prior-curve.svg")
        artifacts = [
            {
                "filename": path.name,
                "row_count": len(tables[path.name]) if path.name in tables else None,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": artifacts}, indent=2) + "\n"
        )
        temporary.replace(output)
        validate_draft_prior_study(output)
        docs_asset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "draft-prior-curve.svg", docs_asset_dir / "draft-prior-curve.svg")
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return DraftPriorStudy(
            run_dir=output,
            run_id=run_id,
            selected_regularization=selected_regularization,
            training_player_count=len(training),
            rookie_ranking_count=len(rookie_rankings),
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_draft_prior_study(run_dir: Path | str) -> dict[str, object]:
    """Validate immutable draft-prior study files and the pre-target contract."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("target_outcomes_used_for_fit") is not False:
        raise ValueError("Draft-prior study uses target outcomes")
    required = {
        "training_first_nba_season_players.parquet",
        "target_first_nba_season_players.parquet",
        "cross_validation.parquet",
        "regularization_stability.parquet",
        "empirical_draft_curve.parquet",
        "adjusted_draft_curve.parquet",
        "rookie_rankings.parquet",
        "model.joblib",
        "draft-prior-curve.svg",
        "metadata.json",
    }
    records = {record["filename"]: record for record in manifest["artifacts"]}
    if not required <= set(records):
        raise ValueError("Draft-prior study is missing required files")
    for filename, record in records.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["byte_count"]:
            raise ValueError(f"Draft-prior study changed: {filename}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Draft-prior study hash changed: {filename}")
        if record["row_count"] is not None and len(pd.read_parquet(path)) != record["row_count"]:
            raise ValueError(f"Draft-prior study row count changed: {filename}")
    target_year = int(str(manifest["target_season"])[:4])
    training = pd.read_parquet(root / "training_first_nba_season_players.parquet")
    target_profiles = pd.read_parquet(root / "target_first_nba_season_players.parquet")
    if training["season"].astype(str).str[:4].astype(int).ge(target_year).any():
        raise ValueError("Draft-prior study contains target-season training outcomes")
    if {"rapm", "rapm_possessions"} & set(target_profiles):
        raise ValueError("Draft-prior target profile artifact contains target outcomes")
    return manifest


def _render_curve(raw_curve: pd.DataFrame, adjusted_curve: pd.DataFrame, path: Path) -> None:
    figure, (raw_axis, adjusted_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        constrained_layout=True,
    )
    numeric = raw_curve.loc[raw_curve["draft_pick"].notna()]
    raw_axis.plot(numeric["draft_pick"], numeric["observed_rapm"], marker="o", color="#1e628f")
    for row in numeric.itertuples(index=False):
        raw_axis.annotate(
            str(row.draft_tier),
            (row.draft_pick, row.observed_rapm),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    raw_axis.axhline(0.0, color="#596879", linewidth=0.8)
    raw_axis.set(
        title="Observed first-season RAPM",
        xlabel="Draft-pick tier midpoint",
        ylabel="Possession-weighted RAPM",
    )
    adjusted_axis.fill_between(
        adjusted_curve["draft_pick"],
        adjusted_curve["bootstrap_lower"],
        adjusted_curve["bootstrap_upper"],
        color="#e66a25",
        alpha=0.2,
        label="90% season-block bootstrap interval",
    )
    adjusted_axis.plot(
        adjusted_curve["draft_pick"],
        adjusted_curve["draft_prior"],
        color="#e66a25",
        linewidth=2.2,
        label="Adjusted draft prior",
    )
    adjusted_axis.axhline(0.0, color="#596879", linewidth=0.8)
    adjusted_axis.set(
        title="Adjusted draft-pick prior",
        xlabel="Draft pick",
        ylabel="Predicted first-season RAPM",
    )
    adjusted_axis.legend(frameon=False, fontsize=8)
    for axis in (raw_axis, adjusted_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _validate_regularization_grid(values: tuple[float, ...]) -> None:
    if len(values) < 2 or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("Draft-prior regularization grid must be unique and non-negative")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a draft-informed cold-start RAPM report")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--docs-asset-dir", default=str(DEFAULT_DOCS_ASSET_DIR))
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    study = build_draft_prior_study(
        season=args.season,
        player_season_panel_path=args.player_season_panel_path,
        artifacts_dir=args.artifacts_dir,
        docs_asset_dir=args.docs_asset_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(study.run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(f"Draft-prior study: run={study.run_dir}{tracking_text}")


if __name__ == "__main__":
    main()
