"""Materialize frozen preseason NAIL-RAPM ranking states for the web API.

The cache is deliberately an offline artifact.  It prevents the public runtime
from quietly substituting a completed-season rating for a next-season forecast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.aging import (
    VALUE_CONDITIONED_AGING_FEATURE_COLUMNS,
    fit_aging_pipeline,
    prepare_aging_prior_features,
    prepare_aging_transitions,
)
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN, ForwardLaggedRapmSeason
from nba_lineup_model.web_api.inference import (
    DEFAULT_ARTIFACTS_DIR,
    DISPLAY_SEASON,
    MODEL_ARTIFACT,
    PRESEASON_PREVIEW_SEASON,
    _compiled_linear_x3_coefficients,
    _player_rating_center,
    _published_profile_padding_contract,
    build_contextual_player_profiles,
    compiled_linear_x3_additive_profile_breakdown,
    exposure_cohort_path,
    forward_draft_cold_start_rankings_path,
    preseason_profiles_path,
    preseason_rankings_path,
    team_roster_path,
)

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_REPLACEMENT_ARTIFACT = "forward_exposure_gated_rapm"


def materialize_preseason_rankings(
    *,
    target_season: str = PRESEASON_PREVIEW_SEASON,
    completed_season: str = DISPLAY_SEASON,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
) -> Path:
    """Write a true frozen next-season catalog for the published NAIL run.

    Returners receive the value-conditioned aging prior fit through the
    completed season.  Every rostered player then receives the frozen previous
    season's additive profile adjustment.  No target-season RAPM update or
    non-additive lineup observation is available yet.
    """

    run_dir, run_id = _published_run(completed_season)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    padding_contract = _published_profile_padding_contract(metadata)
    use_last_observed_profile = (
        metadata.get("profile_padding_contract", {}).get("gap_returner_profile_method")
        == "last_observed_padded_profile"
    )
    panel = pd.read_parquet(panel_path)
    roster = _load_roster(target_season)
    target_panel = _append_target_bios(panel, roster, target_season=target_season)
    roster = roster.merge(
        target_panel.loc[
            target_panel["season"].eq(target_season),
            ["player_id", "draft_year", "draft_round", "draft_number", "is_undrafted"],
        ],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    historical = pd.read_parquet(run_dir / "historical_player_coefficients.parquet")
    results = _completed_results(historical)
    exposure_history = _exposure_history(run_id, historical, panel)
    print("Fitting frozen forward prior", flush=True)
    priors = _forward_returner_prior(
        target_season=target_season,
        panel=target_panel,
        results=results,
        exposure_history=exposure_history,
        regularization=_aging_regularization(run_dir),
    )
    prior_by_player = dict(
        zip(priors["player_id"].astype(int), priors[PRIOR_MEAN_COLUMN].astype(float), strict=True)
    )
    draft = _draft_cold_start(target_season)
    completed_ids = set(
        historical.loc[
            historical["season"].eq(completed_season), "player_id"
        ].astype(int)
    )
    replacement = float(draft["replacement_rapm"].dropna().iloc[0])
    roster["player_id"] = roster["player_id"].astype(int)
    roster["base_prior"] = roster["player_id"].map(prior_by_player)
    roster["forecast_source"] = np.where(
        roster["player_id"].isin(completed_ids),
        "forward_value_conditioned_aging_prior",
        "replacement_cold_start_prior",
    )
    draft_prior = dict(
        zip(
            draft["player_id"].astype(int),
            draft["cold_start_rapm_prior"].astype(float),
            strict=True,
        )
    )
    is_drafted_cold_start = roster["base_prior"].isna() & roster["player_id"].isin(draft_prior)
    roster.loc[is_drafted_cold_start, "base_prior"] = roster.loc[
        is_drafted_cold_start, "player_id"
    ].map(draft_prior)
    roster.loc[is_drafted_cold_start, "forecast_source"] = "draft_cold_start_prior"
    roster["base_prior"] = roster["base_prior"].fillna(replacement).astype(float)

    print("Scoring frozen additive profiles", flush=True)
    models = joblib.load(run_dir / "season_context_models.joblib")
    context_model = models[completed_season]
    profiles = build_contextual_player_profiles(
        target_panel,
        target_season=target_season,
        target_player_ids=roster["player_id"].tolist(),
        exposure_cohort=pd.read_parquet(exposure_cohort_path(MODEL_ARTIFACT, run_id)),
        padding_contract=padding_contract,
        use_last_observed_profile=use_last_observed_profile,
    )
    base = roster.loc[:, ["player_id", "base_prior"]].rename(columns={"base_prior": "rapm"})
    uncentered = _compiled_linear_x3_coefficients(base, profiles, context_model)
    profile_adjustment = (
        uncentered.set_index("player_id")["rapm"]
        - base.set_index("player_id")["rapm"]
    )
    completed_exposure = exposure_history[-1].loc[:, ["player_id", "on_court_possessions"]]
    center = _player_rating_center(
        uncentered,
        completed_exposure.rename(columns={"on_court_possessions": "possessions"}),
    )
    profile_breakdown = compiled_linear_x3_additive_profile_breakdown(
        profiles,
        context_model,
        completed_exposure.rename(columns={"on_court_possessions": "possessions"}),
    )
    breakdown_json = (
        profile_breakdown.loc[
            :, ["player_id", "feature", "player_value", "reference_value", "contribution"]
        ]
        .groupby("player_id", sort=False)
        .apply(
            lambda frame: json.dumps(
                [
                    {
                        "feature": str(row.feature),
                        "player_value": float(row.player_value),
                        "reference_value": float(row.reference_value),
                        "contribution": float(row.contribution),
                    }
                    for row in frame.itertuples(index=False)
                ]
            ),
            include_groups=False,
        )
        .rename("additive_profile_breakdown_json")
        .reset_index()
    )

    output = roster.copy()
    output = output.merge(breakdown_json, on="player_id", how="left", validate="one_to_one")
    # Preserve the published coordinate while folding an imputed profile into
    # a cold-start prior. A projected cohort profile is not an observed player
    # additive contribution and should not be displayed as one.
    output["additive_profile_adjustment"] = (
        output["player_id"].map(profile_adjustment).astype(float) - center
    )
    output["prior_rating"] = output["base_prior"]
    output["rapm"] = output["prior_rating"] + output["additive_profile_adjustment"]
    cold_start = output["forecast_source"].isin(
        ["draft_cold_start_prior", "replacement_cold_start_prior"]
    )
    output.loc[cold_start, "prior_rating"] = output.loc[cold_start, "rapm"]
    output.loc[cold_start, "additive_profile_adjustment"] = np.nan
    output.loc[cold_start, "additive_profile_breakdown_json"] = None
    output["season_update"] = np.nan
    output["observed_context_exposure"] = np.nan
    output["possessions"] = 0.0
    output["games"] = 0
    output["season"] = target_season
    output["team"] = output["team_abbreviation"].astype(str)
    output["position"] = output["listed_position"].astype(str)
    output["is_undrafted"] = output["is_undrafted"].fillna(output["draft_number"].isna())
    is_new = _is_new_player(output["experience"])
    output["draft_class_year"] = output["draft_year"].where(
        output["draft_year"].notna(),
        pd.Series(np.where(is_new, int(target_season[:4]), np.nan), index=output.index),
    )
    output["rookie_season"] = np.where(
        is_new, target_season, pd.NA
    )
    output["profile_source"] = output["forecast_source"]

    columns = [
        "season", "player_id", "player_name", "team", "position", "draft_year",
        "draft_round", "draft_number", "is_undrafted", "draft_class_year", "rapm",
        "prior_rating", "season_update", "additive_profile_adjustment",
        "observed_context_exposure", "possessions", "games", "age", "rookie_season",
        "profile_source", "forecast_source", "additive_profile_breakdown_json",
    ]
    output = output.loc[:, columns].sort_values(
        ["rapm", "player_name", "player_id"], ascending=[False, True, True], kind="stable"
    ).reset_index(drop=True)
    path = preseason_rankings_path(MODEL_ARTIFACT, run_id, target_season)
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Writing preseason ranking cache", flush=True)
    output.to_parquet(path, index=False)
    profiles_path = preseason_profiles_path(MODEL_ARTIFACT, run_id, target_season)
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    profiles.assign(season=target_season).to_parquet(profiles_path, index=False)
    metadata = {
        "target_season": target_season,
        "completed_season": completed_season,
        "model_artifact": MODEL_ARTIFACT,
        "run_id": run_id,
        "contract": (
            "NAIL forecast = forward prior + frozen additive profile; cold starts fold "
            "the imputed profile into their prior; no season update or non-additive "
            "lineup edge"
        ),
        "display_center": center,
        "player_count": len(output),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def _published_run(completed_season: str) -> tuple[Path, str]:
    root = DEFAULT_ARTIFACTS_DIR / MODEL_ARTIFACT / completed_season
    latest = json.loads((root / "latest.json").read_text())
    run_id = str(latest["run_id"])
    return root / run_id, run_id


def _completed_results(coefficients: pd.DataFrame) -> list[ForwardLaggedRapmSeason]:
    results: list[ForwardLaggedRapmSeason] = []
    for season, frame in coefficients.groupby("season", sort=True):
        estimates = frame.loc[:, ["player_id", "rapm"]].copy()
        estimates.insert(0, "season", str(season))
        estimates["player_id"] = estimates["player_id"].astype(int)
        selected_lambda = float(frame["selected_lambda"].iloc[0])
        results.append(
            ForwardLaggedRapmSeason(
                season=str(season),
                selected_lambda=selected_lambda,
                cv_results=pd.DataFrame(),
                player_estimates=estimates,
                player_priors=pd.DataFrame(),
            )
        )
    return results


def _exposure_history(
    run_id: str, coefficients: pd.DataFrame, panel: pd.DataFrame
) -> list[pd.DataFrame]:
    cohort = pd.read_parquet(exposure_cohort_path(MODEL_ARTIFACT, run_id))
    histories: list[pd.DataFrame] = []
    for season in sorted(coefficients["season"].astype(str).unique()):
        exposure = cohort.loc[
            cohort["season"].eq(season),
            ["player_id", "on_court_possessions", "exposure_share"],
        ].copy()
        frame = panel.loc[panel["season"].eq(season)].merge(
            exposure,
            on="player_id",
            how="inner",
            validate="one_to_one",
        )
        if frame.empty:
            raise ValueError(f"Missing exposure state for {season}")
        frame["player_id"] = frame["player_id"].astype(int)
        histories.append(frame)
    return histories


def _aging_regularization(run_dir: Path) -> float:
    metadata = pd.read_parquet(run_dir / "season_player_prior_metadata.parquet")
    value = metadata.loc[metadata["season"].eq(DISPLAY_SEASON), "aging_selected_regularization"]
    if value.empty or pd.isna(value.iloc[0]):
        raise ValueError("Published model lacks an aging regularization selection")
    return float(value.iloc[0])


def _forward_returner_prior(
    *,
    target_season: str,
    panel: pd.DataFrame,
    results: list[ForwardLaggedRapmSeason],
    exposure_history: list[pd.DataFrame],
    regularization: float,
) -> pd.DataFrame:
    """Fit only the completed-state returner aging branch, without cold starts."""

    exposures = {
        result.season: frame.loc[:, ["player_id", "on_court_possessions"]].copy()
        for result, frame in zip(results, exposure_history, strict=True)
    }
    bio_columns = [
        "player_id", "player_name", "age", "nba_experience_years", "is_rookie",
        "draft_year", "draft_number", "height_inches", "weight_pounds", "is_undrafted",
        "rapm_seconds", "rapm_exposure_eligible",
    ]
    transitions: list[pd.DataFrame] = []
    for prior, target in zip(results, results[1:], strict=False):
        if int(target.season[:4]) != int(prior.season[:4]) + 1:
            continue
        target_bios = panel.loc[panel["season"].eq(target.season), bio_columns].copy()
        transition = (
            target_bios.merge(
                target.player_estimates.loc[:, ["player_id", "rapm"]].rename(
                    columns={"rapm": "target_rapm"}
                ),
                on="player_id", how="inner", validate="one_to_one",
            )
            .merge(exposures[target.season], on="player_id", how="inner", validate="one_to_one")
            .merge(
                prior.player_estimates.loc[:, ["player_id", "rapm"]].rename(
                    columns={"rapm": "prior_rapm"}
                ),
                on="player_id", how="inner", validate="one_to_one",
            )
            .merge(
                exposures[prior.season].rename(
                    columns={"on_court_possessions": "prior_rapm_possessions"}
                ),
                on="player_id", how="inner", validate="one_to_one",
            )
            .rename(
                columns={
                    "age": "target_age",
                    "nba_experience_years": "target_nba_experience_years",
                    "on_court_possessions": "target_rapm_possessions",
                    "rapm_seconds": "target_rapm_seconds",
                    "rapm_exposure_eligible": "target_rapm_exposure_eligible",
                }
            )
        )
        transition.insert(0, "target_season", target.season)
        transition.insert(1, "prior_season", prior.season)
        transition["has_prior_season"] = True
        transitions.append(transition)
    training = prepare_aging_transitions(pd.concat(transitions, ignore_index=True))
    model = fit_aging_pipeline(
        training,
        regularization=regularization,
        age_spline_knots=5,
        age_spline_degree=2,
        feature_columns=VALUE_CONDITIONED_AGING_FEATURE_COLUMNS,
    )
    latest = results[-1].player_estimates.loc[:, ["player_id", "rapm"]].rename(
        columns={"rapm": "prior_rapm"}
    )
    target_bios = panel.loc[panel["season"].eq(target_season), bio_columns].copy()
    target = (
        target_bios.merge(latest, on="player_id", how="inner", validate="one_to_one")
        .merge(
            exposures[results[-1].season].rename(
                columns={"on_court_possessions": "prior_rapm_possessions"}
            ),
            on="player_id", how="inner", validate="one_to_one",
        )
        .rename(
            columns={
                "age": "target_age",
                "nba_experience_years": "target_nba_experience_years",
            }
        )
    )
    target["target_season"] = target_season
    target["has_prior_season"] = True
    features = prepare_aging_prior_features(target)
    output = features.loc[:, ["player_id"]].copy()
    output[PRIOR_MEAN_COLUMN] = model.predict(
        features.loc[:, VALUE_CONDITIONED_AGING_FEATURE_COLUMNS]
    )
    weights = output.merge(
        exposures[results[-1].season], on="player_id", how="left", validate="one_to_one"
    )["on_court_possessions"].fillna(0.0)
    center = float(np.average(output[PRIOR_MEAN_COLUMN], weights=weights))
    output[PRIOR_MEAN_COLUMN] -= center
    return output


def _load_roster(target_season: str) -> pd.DataFrame:
    roster = pd.read_parquet(team_roster_path(target_season)).copy()
    roster["player_id"] = pd.to_numeric(roster["player_id"], errors="raise").astype(int)
    roster = roster.sort_values(["player_id", "team_abbreviation"], kind="stable").drop_duplicates(
        "player_id", keep="first"
    )
    if roster["player_id"].duplicated().any() or roster.empty:
        raise ValueError("Preseason roster must contain unique players")
    return roster


def _draft_cold_start(target_season: str) -> pd.DataFrame:
    path = forward_draft_cold_start_rankings_path(target_season)
    draft = pd.read_parquet(path).copy()
    draft["player_id"] = pd.to_numeric(draft["player_id"], errors="raise").astype(int)
    return draft.drop_duplicates("player_id", keep="first")


def _append_target_bios(
    panel: pd.DataFrame, roster: pd.DataFrame, *, target_season: str
) -> pd.DataFrame:
    """Append roster-only target rows; no target box-score or RAPM field is read."""

    if panel["season"].eq(target_season).any():
        raise ValueError(f"Panel unexpectedly already contains {target_season}")
    latest = (
        panel.sort_values(["player_id", "season_start_year"], kind="stable")
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")
    )
    draft_path = Path("data/curated/draft_history") / target_season / "part-00000.parquet"
    draft = _normalize_draft_history_ids(pd.read_parquet(draft_path))
    draft = draft.drop_duplicates("player_id", keep="first").set_index("player_id")
    rows: list[dict[str, object]] = []
    for roster_row in roster.itertuples(index=False):
        player_id = int(roster_row.player_id)
        prior = latest.loc[player_id] if player_id in latest.index else pd.Series(dtype=object)
        current_draft = (
            draft.loc[player_id] if player_id in draft.index else pd.Series(dtype=object)
        )
        row = {column: np.nan for column in panel.columns}
        row.update(
            {
                "season": target_season,
                "season_start_year": int(target_season[:4]),
                "player_id": player_id,
                "player_name": str(roster_row.player_name),
                "age": _number_or_nan(roster_row.age),
                "listed_position": str(roster_row.listed_position),
                "height_inches": _number_or_nan(roster_row.height_inches),
                "weight_pounds": _number_or_nan(roster_row.weight_pounds),
                "nba_experience_years": _experience_years(roster_row.experience),
                "is_rookie": _experience_years(roster_row.experience) == 0,
                "rapm_seconds": 0.0,
                "rapm_exposure_eligible": False,
                "rapm_possessions": 0.0,
            }
        )
        for column in ("draft_year", "draft_round", "draft_number", "is_undrafted"):
            if column in prior and pd.notna(prior[column]):
                row[column] = prior[column]
        if player_id in draft.index:
            row["draft_year"] = current_draft["draft_year"]
            row["draft_round"] = current_draft["draft_round"]
            row["draft_number"] = current_draft["draft_number"]
            row["is_undrafted"] = False
        elif pd.isna(row["draft_number"]):
            row["is_undrafted"] = True
        rows.append(row)
    target_rows = pd.DataFrame(rows, columns=panel.columns)
    return pd.concat([panel, target_rows], ignore_index=True)


def _normalize_draft_history_ids(draft: pd.DataFrame) -> pd.DataFrame:
    """Align source draft IDs with the integer player IDs used elsewhere."""

    output = draft.copy()
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    return output


def _number_or_nan(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else np.nan


def _experience_years(value: object) -> int:
    experience = _number_or_nan(value)
    return int(experience) if np.isfinite(experience) else 0


def _is_new_player(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.fillna(0).eq(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize frozen NBA GESTALT preseason rankings"
    )
    parser.add_argument("--target-season", default=PRESEASON_PREVIEW_SEASON)
    parser.add_argument("--completed-season", default=DISPLAY_SEASON)
    args = parser.parse_args()
    path = materialize_preseason_rankings(
        target_season=args.target_season,
        completed_season=args.completed_season,
    )
    print(f"Materialized preseason rankings: {path}", flush=True)


if __name__ == "__main__":
    main()
