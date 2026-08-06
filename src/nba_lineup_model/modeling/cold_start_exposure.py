"""Preseason exposure gate for first-NBA-season replacement-token assignment."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_lineup_model.modeling.player_history import validate_player_season_panel
from nba_lineup_model.modeling.replacement_level import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_PANEL_PATH,
    prepare_player_exposure_cohort,
)
from nba_lineup_model.modeling.stints import modeling_code_fingerprint
from nba_lineup_model.season.schema import validate_season

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_DOCS_ASSET_DIR = Path("docs/assets/images/cold-start-exposure")
DEFAULT_REPLACEMENT_SHARE_CUTOFF = 0.05
DEFAULT_C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
FEATURE_COLUMNS = (
    "draft_pick_linear",
    "draft_pick_quadratic",
    "undrafted",
    "later_round",
    "draft_unknown",
    "draft_age",
    "height_inches",
    "body_mass_index",
    "draft_pick_draft_age_interaction",
)
TARGET_OUTCOME_COLUMNS = {
    "rapm",
    "rapm_possessions",
    "on_court_possessions",
    "team_opportunity_possessions",
    "exposure_share",
    "team_exposure_share",
    "exposure_band",
    "is_replacement_candidate",
}


@dataclass(frozen=True)
class ColdStartExposureStudy:
    """Immutable outputs for the first-year low-exposure classification study."""

    run_dir: Path
    run_id: str
    selected_c: float
    training_player_count: int
    target_player_count: int


def build_cold_start_exposure_study(
    *,
    target_season: str = "2025-26",
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    docs_asset_dir: Path | str = DEFAULT_DOCS_ASSET_DIR,
    replacement_share_cutoff: float = DEFAULT_REPLACEMENT_SHARE_CUTOFF,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
) -> ColdStartExposureStudy:
    """Fit a pre-target first-year exposure gate and publish its diagnostics."""

    season = validate_season(target_season)
    if not 0.0 < replacement_share_cutoff < 1.0:
        raise ValueError("Replacement share cutoff must lie strictly between zero and one")
    _validate_c_grid(c_grid)
    panel_path = Path(player_season_panel_path)
    validate_player_season_panel(panel_path.parent)
    panel = pd.read_parquet(panel_path)
    exposure_cohort = prepare_player_exposure_cohort(
        panel,
        through_season=season,
        analytical_dir=analytical_dir,
    )
    cohort = enrich_exposure_cohort(exposure_cohort, panel)
    training, target_profiles = prepare_cold_start_exposure_data(
        cohort,
        target_season=season,
        replacement_share_cutoff=replacement_share_cutoff,
    )
    selected_c, cross_validation = select_regularization(training, c_grid=c_grid)
    oof_predictions = cross_validated_predictions(
        training,
        selected_c=selected_c,
        validation_seasons=_validation_seasons(training),
    )
    calibration = calibration_deciles(oof_predictions)
    model = fit_exposure_model(training, c=selected_c)
    target_predictions = rank_target_profiles(model, target_profiles)
    draft_curve = adjusted_draft_exposure_curve(model, training)
    metrics = evaluate_predictions(oof_predictions)
    return _write_study(
        target_season=season,
        panel_path=panel_path,
        training=training,
        target_profiles=target_profiles,
        model=model,
        selected_c=selected_c,
        cross_validation=cross_validation,
        oof_predictions=oof_predictions,
        calibration=calibration,
        target_predictions=target_predictions,
        draft_curve=draft_curve,
        metrics=metrics,
        replacement_share_cutoff=replacement_share_cutoff,
        artifacts_dir=Path(artifacts_dir),
        docs_asset_dir=Path(docs_asset_dir),
    )


def enrich_exposure_cohort(exposure_cohort: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Join predefined player bio fields onto the outcome-derived exposure cohort."""

    bio_columns = [
        "season",
        "player_id",
        "draft_year",
        "draft_round",
        "draft_number",
        "is_undrafted",
        "age",
        "height_inches",
        "weight_pounds",
    ]
    missing = set(bio_columns) - set(panel)
    if missing:
        raise ValueError(f"Player-season panel missing cold-start bio columns: {sorted(missing)}")
    bios = panel.loc[:, bio_columns].copy()
    if bios.duplicated(["season", "player_id"]).any():
        raise ValueError("Player-season panel must be unique by season and player")
    enriched = exposure_cohort.merge(
        bios,
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    if enriched[["age", "height_inches", "weight_pounds"]].isna().all(axis=None):
        raise ValueError("Cold-start exposure cohort could not join player bio data")
    return enriched


def prepare_cold_start_exposure_data(
    cohort: pd.DataFrame,
    *,
    target_season: str,
    replacement_share_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split first-NBA-season outcome rows from target preseason-only profiles."""

    required = {
        "season",
        "season_start_year",
        "player_id",
        "player_name",
        "is_rookie",
        "exposure_share",
        "draft_year",
        "draft_round",
        "draft_number",
        "is_undrafted",
        "age",
        "height_inches",
        "weight_pounds",
        "listed_position",
    }
    missing = required - set(cohort)
    if missing:
        raise ValueError(f"Exposure cohort missing cold-start columns: {sorted(missing)}")
    target_year = int(validate_season(target_season)[:4])
    rookies = cohort.loc[cohort["is_rookie"].astype(bool)].copy()
    if rookies.empty:
        raise ValueError("Cold-start exposure study requires first-NBA-season players")
    historical_raw = rookies.loc[rookies["season_start_year"].lt(target_year)].copy()
    target_raw = rookies.loc[rookies["season_start_year"].eq(target_year)].copy()
    if historical_raw.empty or target_raw.empty:
        raise ValueError("Cold-start exposure study requires historical and target cohorts")
    training = _prepare_features(historical_raw)
    training["is_replacement_candidate"] = training["exposure_share"].lt(
        replacement_share_cutoff
    )
    historical_reference = _feature_reference(_feature_base(historical_raw))
    target = _prepare_features(target_raw, reference=historical_reference)
    if training["is_replacement_candidate"].nunique() != 2:
        raise ValueError("Historical cold-start cohort must contain both exposure outcomes")
    target_profiles = target.drop(columns=TARGET_OUTCOME_COLUMNS, errors="ignore")
    return (
        training.sort_values(["season_start_year", "player_id"], kind="stable").reset_index(
            drop=True
        ),
        target_profiles.sort_values("player_id", kind="stable").reset_index(drop=True),
    )


def fit_exposure_model(training: pd.DataFrame, *, c: float) -> Pipeline:
    """Fit the fixed regularized logistic gate for first-year players only."""

    if c <= 0:
        raise ValueError("Logistic regularization C must be positive")
    _validate_training(training)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=c,
                    solver="lbfgs",
                    max_iter=10_000,
                    random_state=20260807,
                ),
            ),
        ]
    )
    model.fit(
        training.loc[:, FEATURE_COLUMNS],
        training["is_replacement_candidate"].astype(int),
    )
    return model


def select_regularization(
    training: pd.DataFrame,
    *,
    c_grid: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    """Choose C using six chronologically expanding validation cohorts."""

    _validate_c_grid(c_grid)
    folds = _validation_seasons(training)
    rows: list[dict[str, float | int | str]] = []
    for season in folds:
        fit = training.loc[training["season"].lt(season)]
        validation = training.loc[training["season"].eq(season)]
        for c in c_grid:
            prediction = fit_exposure_model(fit, c=c).predict_proba(
                validation.loc[:, FEATURE_COLUMNS]
            )[:, 1]
            labels = validation["is_replacement_candidate"].astype(int).to_numpy()
            rows.append(
                {
                    "validation_season": season,
                    "c": c,
                    "training_player_count": len(fit),
                    "validation_player_count": len(validation),
                    "candidate_rate": float(labels.mean()),
                    "log_loss": float(log_loss(labels, prediction, labels=[0, 1])),
                    "brier_score": float(brier_score_loss(labels, prediction)),
                    "roc_auc": _roc_auc(labels, prediction),
                }
            )
    results = pd.DataFrame(rows)
    summary = results.groupby("c", as_index=False).agg(
        mean_log_loss=("log_loss", "mean"),
        mean_brier_score=("brier_score", "mean"),
    )
    selected = summary.sort_values(["mean_log_loss", "c"], kind="stable").iloc[0]
    return float(selected["c"]), results.sort_values(["c", "validation_season"]).reset_index(
        drop=True
    )


def cross_validated_predictions(
    training: pd.DataFrame,
    *,
    selected_c: float,
    validation_seasons: tuple[str, ...],
) -> pd.DataFrame:
    """Return one strictly-forward prediction for every selected validation player."""

    rows: list[pd.DataFrame] = []
    for season in validation_seasons:
        fit = training.loc[training["season"].lt(season)]
        validation = training.loc[training["season"].eq(season)].copy()
        validation["predicted_replacement_probability"] = fit_exposure_model(
            fit, c=selected_c
        ).predict_proba(validation.loc[:, FEATURE_COLUMNS])[:, 1]
        validation["actual_replacement_candidate"] = validation[
            "is_replacement_candidate"
        ].astype(int)
        rows.append(validation)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["season_start_year", "player_id"], kind="stable"
    ).reset_index(drop=True)


def calibration_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize out-of-fold calibration in equal-count probability bins."""

    if predictions.empty:
        raise ValueError("Calibration requires cross-validated predictions")
    frame = predictions.copy()
    rank = frame["predicted_replacement_probability"].rank(method="first")
    frame["calibration_decile"] = pd.qcut(rank, q=min(10, len(frame)), labels=False) + 1
    return (
        frame.groupby("calibration_decile", as_index=False, sort=True)
        .agg(
            player_count=("player_id", "size"),
            mean_predicted_probability=("predicted_replacement_probability", "mean"),
            actual_candidate_rate=("actual_replacement_candidate", "mean"),
        )
        .reset_index(drop=True)
    )


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare out-of-fold draft-profile predictions to a constant-rate baseline."""

    labels = predictions["actual_replacement_candidate"].to_numpy(dtype=int)
    probabilities = predictions["predicted_replacement_probability"].to_numpy(dtype=float)
    constant = np.repeat(float(labels.mean()), len(labels))
    rows = []
    for model, values in (("draft_profile_logistic", probabilities), ("constant_rate", constant)):
        rows.append(
            {
                "model": model,
                "player_count": len(labels),
                "log_loss": float(log_loss(labels, values, labels=[0, 1])),
                "brier_score": float(brier_score_loss(labels, values)),
                "roc_auc": _roc_auc(labels, values),
            }
        )
    return pd.DataFrame(rows)


def rank_target_profiles(model: Pipeline, target_profiles: pd.DataFrame) -> pd.DataFrame:
    """Score target first-year players without accessing target exposure outcomes."""

    output = target_profiles.copy()
    output["predicted_replacement_probability"] = model.predict_proba(
        output.loc[:, FEATURE_COLUMNS]
    )[:, 1]
    output["predicted_rotation_probability"] = 1.0 - output["predicted_replacement_probability"]
    output["rank"] = (
        output["predicted_rotation_probability"].rank(method="first", ascending=False).astype(int)
    )
    columns = [
        "rank",
        "player_id",
        "player_name",
        "listed_position",
        "draft_status",
        "draft_number",
        "draft_age",
        "height_inches",
        "weight_pounds",
        "predicted_replacement_probability",
        "predicted_rotation_probability",
    ]
    return output.loc[:, columns].sort_values("rank", kind="stable").reset_index(drop=True)


def adjusted_draft_exposure_curve(model: Pipeline, training: pd.DataFrame) -> pd.DataFrame:
    """Predict low-exposure probability by pick at a pre-target reference profile."""

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
    curve["predicted_replacement_probability"] = model.predict_proba(
        curve.loc[:, FEATURE_COLUMNS]
    )[:, 1]
    return curve.loc[:, ["draft_pick", "predicted_replacement_probability"]]


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
    output["draft_pick"] = output["draft_number"].where(output["draft_status"].eq("drafted_1_60"))
    output["draft_age"] = output["age"] - (output["season_start_year"] - output["draft_year"])
    output["draft_age"] = output["draft_age"].where(output["draft_age"].between(17.0, 30.0))
    output["body_mass_index"] = 703.0 * output["weight_pounds"] / output["height_inches"].pow(2)
    return output


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


def _feature_reference(frame: pd.DataFrame) -> dict[str, float]:
    reference = {
        "draft_pick": float(frame["draft_pick"].median()),
        "draft_age": float(frame["draft_age"].median()),
        "height_inches": float(frame["height_inches"].median()),
        "body_mass_index": float(frame["body_mass_index"].median()),
    }
    if not all(np.isfinite(value) for value in reference.values()):
        raise ValueError("Cold-start exposure training cannot establish feature imputation values")
    return reference


def _reference_profile(training: pd.DataFrame) -> dict[str, float]:
    drafted = training.loc[training["draft_status"].eq("drafted_1_60")]
    source = drafted if not drafted.empty else training
    return {column: float(source[column].median()) for column in FEATURE_COLUMNS}


def _validation_seasons(training: pd.DataFrame) -> tuple[str, ...]:
    seasons = tuple(sorted(training["season"].unique(), key=lambda value: int(value[:4])))
    if len(seasons) < 8:
        raise ValueError("Cold-start exposure study requires at least eight historical seasons")
    return seasons[-6:]


def _validate_training(training: pd.DataFrame) -> None:
    missing = (set(FEATURE_COLUMNS) | {"is_replacement_candidate", "season"}) - set(training)
    if missing:
        raise ValueError(f"Exposure training frame missing columns: {sorted(missing)}")
    if training["is_replacement_candidate"].nunique() != 2:
        raise ValueError("Exposure model requires both candidate classes")
    if not np.isfinite(training.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Exposure model features must be finite")


def _validate_c_grid(values: tuple[float, ...]) -> None:
    if len(values) < 2 or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("Regularization C grid must be unique and positive")


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else np.nan


def _write_study(
    *,
    target_season: str,
    panel_path: Path,
    training: pd.DataFrame,
    target_profiles: pd.DataFrame,
    model: Pipeline,
    selected_c: float,
    cross_validation: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    target_predictions: pd.DataFrame,
    draft_curve: pd.DataFrame,
    metrics: pd.DataFrame,
    replacement_share_cutoff: float,
    artifacts_dir: Path,
    docs_asset_dir: Path,
) -> ColdStartExposureStudy:
    now = datetime.now(UTC)
    run_id = f"cold-start-exposure-{target_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "cold_start_exposure" / target_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        tables = {
            "training_first_nba_season_exposure.parquet": training,
            "target_first_nba_season_profiles.parquet": target_profiles,
            "cross_validation.parquet": cross_validation,
            "cross_validated_predictions.parquet": oof_predictions,
            "calibration_deciles.parquet": calibration,
            "target_exposure_predictions.parquet": target_predictions,
            "adjusted_draft_exposure_curve.parquet": draft_curve,
            "metrics.parquet": metrics,
        }
        for filename, table in tables.items():
            table.to_parquet(temporary / filename, index=False)
        joblib.dump(model, temporary / "model.joblib")
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": "first_nba_season_low_exposure_gate",
            "season": target_season,
            "target_season": target_season,
            "training_last_season": max(training["season"], key=lambda value: int(value[:4])),
            "training_seasons": sorted(
                training["season"].unique(), key=lambda value: int(value[:4])
            ),
            "training_player_count": len(training),
            "target_player_count": len(target_profiles),
            "cohort": "first-NBA-season players only",
            "target": (
                "realized regular-season team-possession share "
                f"< {replacement_share_cutoff:.0%}"
            ),
            "replacement_share_cutoff": replacement_share_cutoff,
            "features": list(FEATURE_COLUMNS),
            "selected_c": selected_c,
            "regularization_validation_season_count": 6,
            "temporal_boundary": "all target model inputs end before target season",
            "target_outcomes_used_for_fit": False,
            "use_for_returning_players": False,
            "scope": (
                "diagnostic preseason gate for first-NBA-season players; returning players "
                "retain their lagged RAPM prior"
            ),
            "source_panel_path": str(panel_path),
            "source_panel_manifest_sha256": _sha256_file(panel_path.parent / "_manifest.json"),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        _render_study_chart(draft_curve, calibration, temporary / "cold-start-exposure.svg")
        records = [
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
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        validate_cold_start_exposure_study(output)
        docs_asset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "cold-start-exposure.svg", docs_asset_dir / "cold-start-exposure.svg")
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return ColdStartExposureStudy(
            run_dir=output,
            run_id=run_id,
            selected_c=selected_c,
            training_player_count=len(training),
            target_player_count=len(target_profiles),
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_cold_start_exposure_study(run_dir: Path | str) -> dict[str, object]:
    """Validate artifact hashes and the cold-start-only temporal contract."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("target_outcomes_used_for_fit") is not False:
        raise ValueError("Cold-start exposure study uses target outcomes")
    if manifest.get("use_for_returning_players") is not False:
        raise ValueError("Cold-start exposure gate must be first-year-only")
    required = {
        "training_first_nba_season_exposure.parquet",
        "target_first_nba_season_profiles.parquet",
        "cross_validation.parquet",
        "cross_validated_predictions.parquet",
        "calibration_deciles.parquet",
        "target_exposure_predictions.parquet",
        "adjusted_draft_exposure_curve.parquet",
        "metrics.parquet",
        "model.joblib",
        "cold-start-exposure.svg",
        "metadata.json",
    }
    records = {record["filename"]: record for record in manifest["artifacts"]}
    if not required <= set(records):
        raise ValueError("Cold-start exposure study is missing required files")
    for filename, record in records.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["byte_count"]:
            raise ValueError(f"Cold-start exposure study changed: {filename}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Cold-start exposure study hash changed: {filename}")
        if record["row_count"] is not None and len(pd.read_parquet(path)) != record["row_count"]:
            raise ValueError(f"Cold-start exposure study row count changed: {filename}")
    target_year = int(str(manifest["target_season"])[:4])
    training = pd.read_parquet(root / "training_first_nba_season_exposure.parquet")
    target_profiles = pd.read_parquet(root / "target_first_nba_season_profiles.parquet")
    if training["season_start_year"].ge(target_year).any():
        raise ValueError("Cold-start exposure training contains target-season outcomes")
    if TARGET_OUTCOME_COLUMNS & set(target_profiles):
        raise ValueError("Cold-start exposure target profiles contain target outcomes")
    return manifest


def _render_study_chart(
    draft_curve: pd.DataFrame,
    calibration: pd.DataFrame,
    path: Path,
) -> None:
    figure, (curve_axis, calibration_axis) = plt.subplots(
        1, 2, figsize=(12, 4.8), constrained_layout=True
    )
    curve_axis.plot(
        draft_curve["draft_pick"],
        draft_curve["predicted_replacement_probability"],
        color="#1e628f",
        linewidth=2.2,
    )
    curve_axis.set(
        title="Predicted low-exposure probability by draft pick",
        xlabel="Draft pick at reference draft age and body profile",
        ylabel="P(under 5% team-possession share)",
        ylim=(0.0, 1.0),
    )
    calibration_axis.plot(
        [0.0, 1.0], [0.0, 1.0], color="#596879", linewidth=0.8, linestyle="--"
    )
    calibration_axis.scatter(
        calibration["mean_predicted_probability"],
        calibration["actual_candidate_rate"],
        color="#e66a25",
        s=36,
    )
    calibration_axis.set(
        title="Six-fold forward calibration",
        xlabel="Mean predicted probability",
        ylabel="Observed low-exposure rate",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    for axis in (curve_axis, calibration_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a first-NBA-season exposure gate study")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--docs-asset-dir", default=str(DEFAULT_DOCS_ASSET_DIR))
    parser.add_argument(
        "--replacement-share-cutoff", type=float, default=DEFAULT_REPLACEMENT_SHARE_CUTOFF
    )
    args = parser.parse_args()
    study = build_cold_start_exposure_study(
        target_season=args.season,
        player_season_panel_path=args.player_season_panel_path,
        analytical_dir=args.analytical_dir,
        artifacts_dir=args.artifacts_dir,
        docs_asset_dir=args.docs_asset_dir,
        replacement_share_cutoff=args.replacement_share_cutoff,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(study.run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(f"Cold-start exposure study: run={study.run_dir}{tracking_text}")


if __name__ == "__main__":
    main()
