"""Frozen O/D RAPM with first-year exposure-gated cold-start priors."""

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
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_lineup_model.evaluation.metrics import mean_squared_error
from nba_lineup_model.modeling.cold_start_exposure import (
    FEATURE_COLUMNS,
    _feature_base,
    _feature_reference,
    _prepare_features,
    validate_cold_start_exposure_study,
)
from nba_lineup_model.modeling.offense_defense_rapm import (
    _previous_season,
    build_side_design,
    evaluate_frozen_offense_defense_rapm,
    offense_defense_code_fingerprint,
    validate_frozen_offense_defense_run,
)
from nba_lineup_model.modeling.replacement_level import player_exposure_shares
from nba_lineup_model.modeling.replacement_token import (
    REPLACEMENT_TOKEN_ID,
    tokenize_replacement_lineups,
)
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.models.baselines import PriorCenteredRidgeLineupModel
from nba_lineup_model.season.schema import validate_season

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_RATE_REGULARIZATION_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_REPLACEMENT_SHARE_CUTOFF = 0.05
MODEL_NAME = "frozen_exposure_gated_offense_defense_cold_start_prior"


@dataclass(frozen=True)
class ExposureGatedOffenseDefenseStudy:
    """Immutable O/D cold-start prior and frozen target-season evaluation."""

    run_dir: Path
    run_id: str
    offense_replacement_rapm: float
    defense_replacement_rapm: float


def build_exposure_gated_offense_defense_study(
    *,
    season: str = "2025-26",
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    rate_regularization_grid: tuple[float, ...] = DEFAULT_RATE_REGULARIZATION_GRID,
) -> ExposureGatedOffenseDefenseStudy:
    """Fit all O/D cold-start components through the prior season and score target play."""

    target_season = validate_season(season)
    source_season = _previous_season(target_season)
    artifact_root = Path(artifacts_dir)
    od_source_root = _latest_run(artifact_root / "frozen_offense_defense_rapm" / target_season)
    validate_frozen_offense_defense_run(od_source_root)
    source_state = json.loads((od_source_root / "source_state.json").read_text())
    if source_state.get("source_season") != source_season:
        raise ValueError("Frozen O/D source does not end with the prior season")
    exposure_root = _latest_run(artifact_root / "cold_start_exposure" / target_season)
    exposure_metadata = validate_cold_start_exposure_study(exposure_root)
    if exposure_metadata.get("training_last_season") != source_season:
        raise ValueError("Exposure gate does not end with the prior season")

    panel = pd.read_parquet(player_season_panel_path)
    historical_coefficients = pd.read_parquet(
        od_source_root / "historical_player_coefficients.parquet"
    )
    rate_training, target_profiles = prepare_od_draft_rate_data(
        panel,
        historical_coefficients,
        target_season=target_season,
    )
    selected_rates, rate_cv = select_od_rate_regularization(
        rate_training,
        regularization_grid=rate_regularization_grid,
    )
    offense_model = fit_od_rate_model(
        rate_training, target_column="offense_rapm", regularization=selected_rates["offense"]
    )
    defense_model = fit_od_rate_model(
        rate_training, target_column="defense_rapm", regularization=selected_rates["defense"]
    )
    target_rates = score_od_draft_rates(offense_model, defense_model, target_profiles)
    replacement_seasons = fit_od_replacement_tokens(
        panel,
        historical_coefficients,
        through_season=source_season,
        analytical_dir=analytical_dir,
    )
    replacement = aggregate_od_replacement_tokens(replacement_seasons)
    exposure_predictions = pd.read_parquet(exposure_root / "target_exposure_predictions.parquet")
    rookie_priors = blend_od_cold_start_priors(
        target_rates,
        exposure_predictions,
        replacement_offense_rapm=float(replacement["offense_replacement_rapm"]),
        replacement_defense_rapm=float(replacement["defense_replacement_rapm"]),
    )
    frozen_coefficients = combine_frozen_od_coefficients(
        pd.read_parquet(od_source_root / "frozen_player_priors.parquet"), rookie_priors
    )
    evaluation = evaluate_frozen_offense_defense_rapm(
        season=target_season,
        coefficients=frozen_coefficients,
        league_offensive_rating=float(source_state["league_offensive_rating"]),
        home_offense_shift=float(source_state["home_offense_shift"]),
        analytical_dir=Path(analytical_dir),
        curated_dir=Path(curated_dir),
    )
    return _write_study(
        target_season=target_season,
        source_season=source_season,
        source_root=od_source_root,
        exposure_root=exposure_root,
        target_rates=target_rates,
        rookie_priors=rookie_priors,
        frozen_coefficients=frozen_coefficients,
        replacement_seasons=replacement_seasons,
        replacement=replacement,
        rate_cv=rate_cv,
        selected_rates=selected_rates,
        offense_model=offense_model,
        defense_model=defense_model,
        evaluation=evaluation,
        artifacts_dir=artifact_root,
    )


def prepare_od_draft_rate_data(
    panel: pd.DataFrame,
    historical_coefficients: pd.DataFrame,
    *,
    target_season: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join historical first-year O/D labels to the existing preseason profiles."""

    required_panel = {
        "season",
        "season_start_year",
        "player_id",
        "is_rookie",
        "rapm_possessions",
        "draft_year",
        "draft_round",
        "draft_number",
        "is_undrafted",
        "age",
        "height_inches",
        "weight_pounds",
    }
    required_coefficients = {"season", "player_id", "offense_rapm", "defense_rapm"}
    missing = required_panel - set(panel)
    missing_coefficients = required_coefficients - set(historical_coefficients)
    if missing or missing_coefficients:
        raise ValueError(
            "O/D draft-rate inputs missing columns: "
            f"panel={sorted(missing)}, coefficients={sorted(missing_coefficients)}"
        )
    target_year = int(target_season[:4])
    rookies = panel.loc[panel["is_rookie"].astype(bool)].copy()
    labels = historical_coefficients.loc[:, list(required_coefficients)].copy()
    labels["season"] = labels["season"].astype(str)
    labels["player_id"] = pd.to_numeric(labels["player_id"], errors="raise").astype(int)
    if labels.duplicated(["season", "player_id"]).any():
        raise ValueError("Historical O/D coefficients must be unique by season and player")
    historical_raw = rookies.loc[rookies["season_start_year"].lt(target_year)].merge(
        labels,
        on=["season", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    target_raw = rookies.loc[rookies["season_start_year"].eq(target_year)].copy()
    if historical_raw.empty or target_raw.empty:
        raise ValueError("O/D draft-rate study requires historical and target first-year cohorts")
    training = _prepare_features(historical_raw)
    reference = _feature_reference(_feature_base(historical_raw))
    target = _prepare_features(target_raw, reference=reference)
    if not training["rapm_possessions"].gt(0).all():
        raise ValueError("O/D draft-rate study requires positive reconstructed possession weights")
    return (
        training.sort_values(["season_start_year", "player_id"], kind="stable").reset_index(
            drop=True
        ),
        target.sort_values("player_id", kind="stable").reset_index(drop=True),
    )


def fit_od_rate_model(
    training: pd.DataFrame,
    *,
    target_column: str,
    regularization: float,
) -> Pipeline:
    """Fit one possession-weighted ridge prior for an O/D rate component."""

    if target_column not in {"offense_rapm", "defense_rapm"}:
        raise ValueError("O/D draft rate target must be offense_rapm or defense_rapm")
    if regularization < 0:
        raise ValueError("O/D draft-rate regularization must be non-negative")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=regularization * len(training), solver="lsqr", tol=1e-8)),
        ]
    )
    weights = training["rapm_possessions"].to_numpy(dtype=float)
    model.fit(
        training.loc[:, FEATURE_COLUMNS],
        training[target_column].to_numpy(dtype=float),
        ridge__sample_weight=weights / np.mean(weights),
    )
    return model


def select_od_rate_regularization(
    training: pd.DataFrame,
    *,
    regularization_grid: tuple[float, ...],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Select O/D ridge penalties with six strictly earlier expanding folds."""

    if len(regularization_grid) < 2 or any(value < 0 for value in regularization_grid):
        raise ValueError("O/D rate regularization grid must have two non-negative values")
    seasons = tuple(sorted(training["season"].unique(), key=lambda value: int(value[:4])))
    if len(seasons) < 8:
        raise ValueError("O/D draft-rate study requires at least eight historical seasons")
    rows: list[dict[str, float | int | str]] = []
    for validation_season in seasons[-6:]:
        fit = training.loc[training["season"].lt(validation_season)]
        validation = training.loc[training["season"].eq(validation_season)]
        weights = validation["rapm_possessions"].to_numpy(dtype=float)
        for component in ("offense", "defense"):
            target_column = f"{component}_rapm"
            for regularization in regularization_grid:
                prediction = fit_od_rate_model(
                    fit, target_column=target_column, regularization=regularization
                ).predict(validation.loc[:, FEATURE_COLUMNS])
                mse = mean_squared_error(
                    validation[target_column].to_numpy(dtype=float), prediction, weights
                )
                rows.append(
                    {
                        "component": component,
                        "validation_season": validation_season,
                        "regularization": regularization,
                        "training_player_count": len(fit),
                        "validation_player_count": len(validation),
                        "validation_possessions": float(weights.sum()),
                        "squared_error_sum": float(mse * weights.sum()),
                        "weighted_mse": mse,
                    }
                )
    results = pd.DataFrame(rows)
    summary = results.groupby(["component", "regularization"], as_index=False).agg(
        squared_error_sum=("squared_error_sum", "sum"),
        validation_possessions=("validation_possessions", "sum"),
    )
    summary["weighted_mse"] = summary["squared_error_sum"] / summary["validation_possessions"]
    selected = {
        component: float(
            summary.loc[summary["component"].eq(component)]
            .sort_values(["weighted_mse", "regularization"], kind="stable")
            .iloc[0]["regularization"]
        )
        for component in ("offense", "defense")
    }
    return selected, results.sort_values(
        ["component", "regularization", "validation_season"], kind="stable"
    ).reset_index(drop=True)


def score_od_draft_rates(
    offense_model: Pipeline,
    defense_model: Pipeline,
    target_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Score outcome-free target first-year draft profiles with two ridge models."""

    output = target_profiles.loc[
        :,
        [
            "player_id",
            "player_name",
            "listed_position",
            "draft_status",
            "draft_number",
            "draft_age",
        ],
    ].copy()
    output["draft_offense_rapm"] = offense_model.predict(target_profiles.loc[:, FEATURE_COLUMNS])
    output["draft_defense_rapm"] = defense_model.predict(target_profiles.loc[:, FEATURE_COLUMNS])
    return output


def fit_od_replacement_tokens(
    panel: pd.DataFrame,
    historical_coefficients: pd.DataFrame,
    *,
    through_season: str,
    analytical_dir: Path | str,
) -> pd.DataFrame:
    """Estimate pooled offense and defense low-exposure tokens in each completed season."""

    cutoff_year = int(validate_season(through_season)[:4])
    required = {"season", "season_start_year", "player_id"}
    if required - set(panel):
        raise ValueError("Player panel is missing O/D replacement membership columns")
    lambdas = historical_coefficients.loc[
        :, ["season", "selected_lambda"]
    ].drop_duplicates()
    if lambdas.duplicated("season").any():
        raise ValueError("O/D historical artifact has multiple lambdas for a season")
    rows: list[dict[str, float | int | str]] = []
    for season, season_panel in panel.loc[
        panel["season_start_year"].le(cutoff_year)
    ].groupby("season", sort=True):
        stints = read_rapm_stints(str(season), analytical_dir=analytical_dir)
        exposure = player_exposure_shares(stints)
        known_ids = set(season_panel["player_id"].astype(int))
        replacement_ids = set(
            exposure.loc[
                exposure["exposure_share"].lt(DEFAULT_REPLACEMENT_SHARE_CUTOFF)
                & exposure["player_id"].astype(int).isin(known_ids),
                "player_id",
            ].astype(int)
        )
        if not replacement_ids:
            raise ValueError(f"No O/D replacement candidates in {season}")
        lambda_rows = lambdas.loc[lambdas["season"].astype(str).eq(str(season))]
        if len(lambda_rows) != 1:
            raise ValueError(f"O/D historical lambda missing for {season}")
        tokenized = tokenize_replacement_lineups(stints, replacement_ids)
        design = build_side_design(tokenized)
        if REPLACEMENT_TOKEN_ID not in design.player_ids:
            raise ValueError(f"Replacement token has no O/D design column in {season}")
        model = PriorCenteredRidgeLineupModel(float(lambda_rows["selected_lambda"].item())).fit(
            design.features,
            design.target,
            design.weights,
            np.zeros(design.features.shape[1], dtype=float),
        )
        index = design.player_ids.index(REPLACEMENT_TOKEN_ID)
        player_count = len(design.player_ids)
        rows.append(
            {
                "season": str(season),
                "season_start_year": int(season_panel["season_start_year"].iloc[0]),
                "selected_lambda": float(lambda_rows["selected_lambda"].item()),
                "replacement_player_count": len(replacement_ids),
                "offense_replacement_rapm": float(model.coef_[index]),
                "defense_replacement_rapm": float(model.coef_[player_count + index]),
            }
        )
    return pd.DataFrame(rows).sort_values("season_start_year", kind="stable").reset_index(drop=True)


def aggregate_od_replacement_tokens(season_tokens: pd.DataFrame) -> dict[str, float | int]:
    """Use equal-season pooled-token means so no target season can enter the prior."""

    if season_tokens.empty:
        raise ValueError("O/D replacement aggregation requires historical season estimates")
    columns = ["offense_replacement_rapm", "defense_replacement_rapm"]
    if not np.isfinite(season_tokens.loc[:, columns].to_numpy(dtype=float)).all():
        raise ValueError("O/D replacement estimates must be finite")
    offense = float(season_tokens["offense_replacement_rapm"].mean())
    defense = float(season_tokens["defense_replacement_rapm"].mean())
    return {
        "season_count": int(len(season_tokens)),
        "offense_replacement_rapm": offense,
        "defense_replacement_rapm": defense,
        "net_replacement_rapm": offense + defense,
        "estimator": "equal-season mean of separately pooled O/D replacement tokens",
    }


def blend_od_cold_start_priors(
    target_rates: pd.DataFrame,
    exposure_predictions: pd.DataFrame,
    *,
    replacement_offense_rapm: float,
    replacement_defense_rapm: float,
) -> pd.DataFrame:
    """Form a continuous O/D rookie prior from rate and exposure components."""

    required_rates = {"player_id", "draft_offense_rapm", "draft_defense_rapm"}
    required_exposure = {"player_id", "predicted_replacement_probability"}
    if required_rates - set(target_rates) or required_exposure - set(exposure_predictions):
        raise ValueError("O/D cold-start blend inputs are incomplete")
    output = target_rates.merge(
        exposure_predictions.loc[:, ["player_id", "predicted_replacement_probability"]],
        on="player_id",
        how="inner",
        validate="one_to_one",
    )
    if len(output) != len(target_rates) or len(output) != len(exposure_predictions):
        raise ValueError("O/D draft-rate and exposure target cohorts do not match")
    probability = output["predicted_replacement_probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or (probability < 0.0).any() or (probability > 1.0).any():
        raise ValueError("O/D exposure probabilities must lie in [0, 1]")
    output["replacement_offense_rapm"] = replacement_offense_rapm
    output["replacement_defense_rapm"] = replacement_defense_rapm
    output["offense_rapm"] = probability * replacement_offense_rapm + (1.0 - probability) * output[
        "draft_offense_rapm"
    ]
    output["defense_rapm"] = probability * replacement_defense_rapm + (1.0 - probability) * output[
        "draft_defense_rapm"
    ]
    output["net_rapm"] = output["offense_rapm"] + output["defense_rapm"]
    output["rank"] = output["net_rapm"].rank(method="first", ascending=False).astype(int)
    return output.sort_values("rank", kind="stable").reset_index(drop=True)


def combine_frozen_od_coefficients(
    returning_coefficients: pd.DataFrame,
    rookie_priors: pd.DataFrame,
) -> pd.DataFrame:
    """Append first-year O/D priors to the completed prior-season O/D state."""

    returning = returning_coefficients.loc[:, ["player_id", "offense_rapm", "defense_rapm"]].copy()
    rookies = rookie_priors.loc[:, ["player_id", "offense_rapm", "defense_rapm"]].copy()
    for frame in (returning, rookies):
        frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype(int)
    if returning["player_id"].duplicated().any() or rookies["player_id"].duplicated().any():
        raise ValueError("Frozen O/D component inputs have duplicate player IDs")
    if set(returning["player_id"]) & set(rookies["player_id"]):
        raise ValueError("First-year O/D priors overlap completed returning O/D state")
    output = pd.concat([returning, rookies], ignore_index=True)
    if not np.isfinite(output[["offense_rapm", "defense_rapm"]].to_numpy(dtype=float)).all():
        raise ValueError("Frozen O/D coefficients must be finite")
    return output.sort_values("player_id", kind="stable").reset_index(drop=True)


def _write_study(
    *,
    target_season: str,
    source_season: str,
    source_root: Path,
    exposure_root: Path,
    target_rates: pd.DataFrame,
    rookie_priors: pd.DataFrame,
    frozen_coefficients: pd.DataFrame,
    replacement_seasons: pd.DataFrame,
    replacement: dict[str, float | int],
    rate_cv: pd.DataFrame,
    selected_rates: dict[str, float],
    offense_model: Pipeline,
    defense_model: Pipeline,
    evaluation: dict[str, object],
    artifacts_dir: Path,
) -> ExposureGatedOffenseDefenseStudy:
    now = datetime.now(UTC)
    run_id = f"exposure-gated-od-{target_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "exposure_gated_offense_defense" / target_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        tables = {
            "rookie_od_draft_rates.parquet": target_rates,
            "revised_rookie_od_rankings.parquet": rookie_priors,
            "frozen_player_priors.parquet": frozen_coefficients,
            "season_od_replacement_tokens.parquet": replacement_seasons,
            "od_rate_cross_validation.parquet": rate_cv,
            "cohort_metrics.parquet": _with_model_name(evaluation["cohort_metrics"]),
            "possession_predictions.parquet": evaluation["possession_predictions"],
            "game_predictions.parquet": evaluation["game_predictions"],
            "regular_game_predictions.parquet": evaluation["regular_game_predictions"],
            "team_net_rating_predictions.parquet": evaluation["team_net_rating_predictions"],
            "team_net_rating_metrics.parquet": _with_model_name(
                evaluation["team_net_rating_metrics"]
            ),
            "team_win_predictions.parquet": evaluation["team_win_predictions"],
            "team_win_metrics.parquet": _with_model_name(evaluation["team_win_metrics"]),
            "pythagorean_calibration_team_seasons.parquet": evaluation[
                "pythagorean_calibration_team_seasons"
            ],
        }
        for filename, table in tables.items():
            table.to_parquet(temporary / filename, index=False)
        joblib.dump(offense_model, temporary / "offense_draft_rate_model.joblib")
        joblib.dump(defense_model, temporary / "defense_draft_rate_model.joblib")
        source_state = dict(evaluation["source_state"])
        source_state.update(
            {
                "player_prior_method": (
                    "exposure-gated O/D draft-rate and pooled replacement priors "
                    "for first-year players"
                ),
                "source_od_run_id": json.loads(
                    (source_root / "manifest.json").read_text()
                )["run_id"],
                "source_od_manifest_sha256": _sha256_file(source_root / "manifest.json"),
                "exposure_gate_run_id": json.loads(
                    (exposure_root / "manifest.json").read_text()
                )["run_id"],
                "exposure_gate_manifest_sha256": _sha256_file(exposure_root / "manifest.json"),
                "replacement_tokens": replacement,
                "rate_regularization": selected_rates,
                "first_year_player_count": len(rookie_priors),
            }
        )
        (temporary / "source_state.json").write_text(json.dumps(source_state, indent=2) + "\n")
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "season": target_season,
            "source_season": source_season,
            "target_outcomes_used_for_fit": False,
            "historical_training_variant": "regular_only",
            "formula": "p_low * replacement_side + (1-p_low) * draft_rate_side",
            "replacement_share_cutoff": DEFAULT_REPLACEMENT_SHARE_CUTOFF,
            "evaluation_code_version": offense_defense_code_fingerprint(),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
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
        validate_exposure_gated_offense_defense_study(output)
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return ExposureGatedOffenseDefenseStudy(
            run_dir=output,
            run_id=run_id,
            offense_replacement_rapm=float(replacement["offense_replacement_rapm"]),
            defense_replacement_rapm=float(replacement["defense_replacement_rapm"]),
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_exposure_gated_offense_defense_study(run_dir: Path | str) -> dict[str, object]:
    """Validate O/D artifact integrity and the frozen target-outcome boundary."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("target_outcomes_used_for_fit") is not False:
        raise ValueError("Exposure-gated O/D study uses target outcomes")
    required = {
        "revised_rookie_od_rankings.parquet",
        "frozen_player_priors.parquet",
        "season_od_replacement_tokens.parquet",
        "cohort_metrics.parquet",
        "team_net_rating_metrics.parquet",
        "team_win_metrics.parquet",
        "source_state.json",
        "metadata.json",
    }
    records = {record["filename"]: record for record in manifest["artifacts"]}
    if not required <= set(records):
        raise ValueError("Exposure-gated O/D artifact is incomplete")
    for filename, record in records.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["byte_count"]:
            raise ValueError(f"Exposure-gated O/D artifact changed: {filename}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Exposure-gated O/D artifact hash changed: {filename}")
        if record["row_count"] is not None and len(pd.read_parquet(path)) != record["row_count"]:
            raise ValueError(f"Exposure-gated O/D artifact row count changed: {filename}")
    return manifest


def _with_model_name(table: pd.DataFrame) -> pd.DataFrame:
    """Mark shared-evaluator metric tables with this composite model's identity."""

    output = table.copy()
    if "model" in output:
        output["model"] = MODEL_NAME
    return output


def _latest_run(parent: Path) -> Path:
    run_id = str(json.loads((parent / "latest.json").read_text())["run_id"])
    output = parent / run_id
    if not output.is_dir():
        raise ValueError(f"Model run not found: {output}")
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen exposure-gated O/D cold-start RAPM")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--curated-dir", default=str(DEFAULT_CURATED_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    args = parser.parse_args()
    study = build_exposure_gated_offense_defense_study(
        season=args.season,
        player_season_panel_path=args.player_season_panel_path,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(study.run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(f"Exposure-gated O/D study: run={study.run_dir}{tracking_text}")


if __name__ == "__main__":
    main()
