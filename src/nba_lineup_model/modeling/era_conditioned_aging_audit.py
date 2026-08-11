"""Diagnose the era-conditioned aging prior against its value-conditioned parent."""

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

from nba_lineup_model.modeling.aging import (
    VALUE_CONDITIONED_AGING_FEATURE_COLUMNS,
    fit_aging_pipeline,
    materialize_aging_curve_grid,
    prepare_aging_transitions,
)
from nba_lineup_model.modeling.forward_aging_player_prior import _aging_transition_history
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_PANEL_PATH, _previous_season
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import _latest_run
from nba_lineup_model.modeling.prior_rapm import ForwardLaggedRapmSeason
from nba_lineup_model.modeling.replacement_level import player_exposure_shares
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints

DEFAULT_SEASON = "2025-26"
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_AUDITS_DIR = Path("artifacts/analysis/era_conditioned_aging")
DEFAULT_DOCS_ASSETS_DIR = Path("docs/assets/images/era-conditioned-aging")
DEFAULT_DOCS_PAGE = Path(
    "docs/models/forward-centered-era-conditioned-aging-"
    "bounded-hierarchical-portable-matchup-contextual-rapm.md"
)
VALUE_MODEL = (
    "forward_centered_value_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm"
)
ERA_MODEL = (
    "forward_centered_era_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm"
)
SECTION_START = "<!-- era-conditioned-aging-audit:start -->"
SECTION_END = "<!-- era-conditioned-aging-audit:end -->"


@dataclass(frozen=True)
class EraConditionedAgingAuditRun:
    """One immutable diagnostic for the era-conditioned aging candidate."""

    run_dir: Path
    run_id: str


def build_era_conditioned_aging_audit(
    *,
    season: str = DEFAULT_SEASON,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    audits_dir: Path | str = DEFAULT_AUDITS_DIR,
    docs_assets_dir: Path | str = DEFAULT_DOCS_ASSETS_DIR,
    docs_page: Path | str = DEFAULT_DOCS_PAGE,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = "data/analytical",
) -> EraConditionedAgingAuditRun:
    """Compare terminal age curves and veteran prior errors without refitting RAPM.

    The parent value-conditioned aging pipeline is reconstructed from its immutable
    completed seasonal coefficients and the selected terminal aging penalty. The
    era-conditioned pipeline is loaded from its published terminal-season state.
    Both use only seasons completed before ``season``.
    """

    artifacts_root = Path(artifacts_dir)
    value_run = _latest_run(artifacts_root / VALUE_MODEL / season)
    era_run = _latest_run(artifacts_root / ERA_MODEL / season)
    panel = pd.read_parquet(player_season_panel_path)
    value_model, value_training = _reconstruct_value_aging_model(
        value_run,
        season=season,
        panel=panel,
        analytical_dir=Path(analytical_dir),
    )
    era_models = joblib.load(era_run / "season_aging_models.joblib")
    era_aging_model = era_models[season]
    era_training = _reconstruct_aging_training(
        era_run,
        season=season,
        panel=panel,
        analytical_dir=Path(analytical_dir),
    )
    curve_comparison = _curve_comparison(
        value_model,
        value_training,
        era_aging_model,
        era_training,
        season=season,
    )
    cohort_metrics = _veteran_cohort_metrics(
        value_run,
        era_run,
        season=season,
        panel=panel,
        analytical_dir=Path(analytical_dir),
    )

    now = datetime.now(UTC)
    run_id = f"era-conditioned-aging-audit-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = Path(audits_dir) / season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        curve_comparison.to_parquet(temporary / "curve_comparison.parquet", index=False)
        cohort_metrics.to_parquet(temporary / "veteran_cohort_prior_errors.parquet", index=False)
        _render_curve_delta(curve_comparison, temporary / "curve-delta.png")
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "season": season,
            "value_model": VALUE_MODEL,
            "value_run_id": value_run.name,
            "era_model": ERA_MODEL,
            "era_run_id": era_run.name,
            "curve_contract": (
                "Population partial-age-effect curves, each anchored at its recorded "
                "reference age and evaluated at the model's p25, p50, and p75 prior-RAPM profiles"
            ),
            "veteran_error_contract": (
                "Possession-weighted difference between frozen 2025-26 player prior and "
                "that model's completed 2025-26 refit coefficient, restricted to players "
                "present in the completed 2024-25 coefficient state"
            ),
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {
                "filename": path.name,
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
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    docs_asset = Path(docs_assets_dir) / f"{season}-curve-delta.png"
    docs_asset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / "curve-delta.png", docs_asset)
    _update_docs_page(
        Path(docs_page),
        curve_comparison,
        cohort_metrics,
        output=output,
        docs_asset=docs_asset,
    )
    return EraConditionedAgingAuditRun(run_dir=output, run_id=run_id)


def _reconstruct_value_aging_model(
    run_dir: Path,
    *,
    season: str,
    panel: pd.DataFrame,
    analytical_dir: Path,
) -> tuple[object, pd.DataFrame]:
    training = _reconstruct_aging_training(
        run_dir,
        season=season,
        panel=panel,
        analytical_dir=analytical_dir,
    )
    metadata = pd.read_parquet(run_dir / "season_player_prior_metadata.parquet")
    regularization = float(
        metadata.loc[metadata["season"].eq(season), "aging_selected_regularization"].item()
    )
    model = fit_aging_pipeline(
        training,
        regularization=regularization,
        age_spline_knots=5,
        age_spline_degree=2,
        feature_columns=VALUE_CONDITIONED_AGING_FEATURE_COLUMNS,
    )
    return model, training


def _reconstruct_aging_training(
    run_dir: Path,
    *,
    season: str,
    panel: pd.DataFrame,
    analytical_dir: Path,
) -> pd.DataFrame:
    coefficients = pd.read_parquet(run_dir / "historical_player_coefficients.parquet")
    completed_seasons = sorted(
        (value for value in coefficients["season"].unique() if value < season),
        key=lambda value: int(value[:4]),
    )
    results = [
        ForwardLaggedRapmSeason(
            season=value,
            selected_lambda=float(
                coefficients.loc[coefficients["season"].eq(value), "selected_lambda"].iloc[0]
            ),
            cv_results=pd.DataFrame(),
            player_estimates=coefficients.loc[coefficients["season"].eq(value)].copy(),
            player_priors=pd.DataFrame(),
        )
        for value in completed_seasons
    ]
    exposure_history = [
        player_exposure_shares(read_rapm_stints(value, analytical_dir=analytical_dir))
        for value in completed_seasons
    ]
    transitions = _aging_transition_history(panel, results, exposure_history)
    return prepare_aging_transitions(transitions)


def _curve_comparison(
    value_model: object,
    value_training: pd.DataFrame,
    era_model: object,
    era_training: pd.DataFrame,
    *,
    season: str,
) -> pd.DataFrame:
    value = materialize_aging_curve_grid(
        value_model,
        value_training,
        feature_columns=VALUE_CONDITIONED_AGING_FEATURE_COLUMNS,
        fitted_season=season,
    ).rename(
        columns={
            "predicted_rapm": "value_predicted_rapm",
            "partial_age_effect": "value_partial_age_effect",
            "prior_rapm_reference": "value_prior_rapm_reference",
            "reference_age": "value_reference_age",
        }
    )
    era = materialize_aging_curve_grid(
        era_model,
        era_training,
        feature_columns=tuple(era_model.feature_names_in_),
        fitted_season=season,
    ).rename(
        columns={
            "predicted_rapm": "era_predicted_rapm",
            "partial_age_effect": "era_partial_age_effect",
            "prior_rapm_reference": "era_prior_rapm_reference",
            "reference_age": "era_reference_age",
        }
    )
    keep = ["age", "prior_rapm_profile"]
    output = value.merge(era, on=keep, suffixes=("", "_era"), validate="one_to_one")
    output["era_minus_value_partial_age_effect"] = (
        output["era_partial_age_effect"] - output["value_partial_age_effect"]
    )
    return output.sort_values(["prior_rapm_profile", "age"], kind="stable").reset_index(drop=True)


def _veteran_cohort_metrics(
    value_run: Path,
    era_run: Path,
    *,
    season: str,
    panel: pd.DataFrame,
    analytical_dir: Path,
) -> pd.DataFrame:
    exposure = player_exposure_shares(read_rapm_stints(season, analytical_dir=analytical_dir))
    previous = pd.read_parquet(value_run / "historical_player_coefficients.parquet")
    returning_ids = set(
        previous.loc[previous["season"].eq(_previous_season(season)), "player_id"].astype(int)
    )
    frames = []
    for label, run_dir in (
        ("Value-Conditioned Aging HPM", value_run),
        ("Era-Conditioned Aging HPM", era_run),
    ):
        priors = pd.read_parquet(run_dir / "frozen_2025_26_player_priors.parquet").rename(
            columns={"prior_rapm_mean": "prior_rapm"}
        )
        coefficients = pd.read_parquet(run_dir / "historical_player_coefficients.parquet")
        completed = coefficients.loc[coefficients["season"].eq(season), ["player_id", "rapm"]]
        frame = priors.loc[:, ["player_id", "prior_rapm"]].merge(
            completed, on="player_id", how="inner", validate="one_to_one"
        )
        frame = frame.merge(
            panel.loc[panel["season"].eq(season), ["player_id", "age"]],
            on="player_id",
            how="inner",
            validate="one_to_one",
        ).merge(exposure.loc[:, ["player_id", "on_court_possessions"]], on="player_id", how="inner")
        frame = frame.loc[frame["player_id"].isin(returning_ids)].copy()
        frame["model"] = label
        frames.append(frame)
    paired = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for threshold in (30, 34, 36):
        for model, frame in paired.groupby("model", sort=False):
            cohort = frame.loc[frame["age"].ge(threshold)].copy()
            weights = cohort["on_court_possessions"].to_numpy(dtype=float)
            error = (cohort["rapm"] - cohort["prior_rapm"]).to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "age_cohort": f"Age {threshold}+",
                    "player_count": len(cohort),
                    "on_court_possessions": float(weights.sum()),
                    "weighted_prior_error_mae": float(np.average(np.abs(error), weights=weights)),
                    "weighted_prior_error_rmse": float(
                        np.sqrt(np.average(error**2, weights=weights))
                    ),
                    "weighted_mean_prior_rapm": float(
                        np.average(cohort["prior_rapm"], weights=weights)
                    ),
                    "weighted_mean_completed_rapm": float(
                        np.average(cohort["rapm"], weights=weights)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _render_curve_delta(comparison: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2), layout="constrained")
    labels = {
        "prior_p25": "Prior RAPM p25",
        "prior_p50": "Prior RAPM p50",
        "prior_p75": "Prior RAPM p75",
    }
    colors = {"prior_p25": "#b8422b", "prior_p50": "#1f5c4d", "prior_p75": "#1f4d78"}
    for profile, frame in comparison.groupby("prior_rapm_profile", sort=False):
        axis.plot(
            frame["age"],
            frame["era_minus_value_partial_age_effect"],
            label=labels[profile],
            color=colors[profile],
            linewidth=2.3,
        )
    axis.axhline(0.0, color="#7a817c", linewidth=1.0, linestyle="--")
    axis.axvline(27, color="#7a817c", linewidth=1.0, linestyle="--")
    axis.set_title("2025 era-model age adjustment versus its value-conditioned parent")
    axis.set_xlabel("Age")
    axis.set_ylabel("Net-rating points per 100 possessions")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.savefig(output, dpi=180, transparent=False)
    plt.close(figure)


def _update_docs_page(
    path: Path,
    comparison: pd.DataFrame,
    cohort_metrics: pd.DataFrame,
    *,
    output: Path,
    docs_asset: Path,
) -> None:
    delta = comparison["era_minus_value_partial_age_effect"]
    max_row = comparison.loc[delta.abs().idxmax()]
    p50_young = comparison.loc[
        comparison["prior_rapm_profile"].eq("prior_p50")
        & comparison["age"].eq(comparison["age"].min())
    ].iloc[0]
    cohort = cohort_metrics.set_index(["model", "age_cohort"])
    value_30 = cohort.loc[("Value-Conditioned Aging HPM", "Age 30+")]
    era_30 = cohort.loc[("Era-Conditioned Aging HPM", "Age 30+")]
    value_34 = cohort.loc[("Value-Conditioned Aging HPM", "Age 34+")]
    era_34 = cohort.loc[("Era-Conditioned Aging HPM", "Age 34+")]
    value_36 = cohort.loc[("Value-Conditioned Aging HPM", "Age 36+")]
    era_36 = cohort.loc[("Era-Conditioned Aging HPM", "Age 36+")]
    lines = [
        SECTION_START,
        "## Why Era Conditioning Did Not Help",
        "",
        "The figure compares the two terminal population age curves. Each line is the",
        "era-conditioned curve minus its value-conditioned parent, after both curves",
        "are anchored at their own reference age. Positive values favor the era model's",
        "age effect. This is not a longitudinal estimate that NBA 19-year-olds improved",
        "by the displayed amount: it is the difference between two 2025 model",
        "specifications, with their own recursive coefficient histories and reference",
        "profiles. The largest visible divergence is "
        f"{float(max_row['era_minus_value_partial_age_effect']):+.2f} at age {int(max_row['age'])} "
        f"for the {str(max_row['prior_rapm_profile']).replace('_', ' ')} profile.",
        "",
        f"![Era-conditioned curve delta](../assets/images/era-conditioned-aging/{docs_asset.name})",
        "",
        "The cohort table is a diagnostic rather than a leaderboard metric. It compares",
        "each frozen 2025-26 prior with that model's completed 2025-26 coefficient for",
        "returning players, weighted by their realized on-court possessions. It tests",
        "whether the additional age flexibility reduced the subsequent rating update for",
        "older players. It does not use target-season outcomes to construct either prior.",
        "",
        (
            "| Model | Cohort | Players | Possessions | Prior-error MAE | Prior-error RMSE | "
            "Mean prior | Mean completed rating |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in cohort_metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.age_cohort} | {row.player_count:,} | "
            f"{row.on_court_possessions:,.0f} | {row.weighted_prior_error_mae:.3f} | "
            f"{row.weighted_prior_error_rmse:.3f} | {row.weighted_mean_prior_rapm:+.3f} | "
            f"{row.weighted_mean_completed_rapm:+.3f} |"
        )
    lines.extend(
        [
            "",
            (
                f"The extra flexibility is concentrated at the early-career boundary: at age "
                f"{int(p50_young['age'])}, the median-prior partial age effect is "
                f"{float(p50_young['era_minus_value_partial_age_effect']):+.2f} points higher "
                "under era conditioning. That is a large model-specification shift, not evidence "
                "of a causal improvement in 19-year-old talent, and it is not the veteran-specific "
                "correction this ablation was intended to test."
            ),
            "",
            (
                "For returning players age 30+, prior RMSE rose from "
                f"{float(value_30['weighted_prior_error_rmse']):.3f} to "
                f"{float(era_30['weighted_prior_error_rmse']):.3f}; for age 34+, it rose from "
                f"{float(value_34['weighted_prior_error_rmse']):.3f} to "
                f"{float(era_34['weighted_prior_error_rmse']):.3f}. Age 36+ is the only "
                "cohort with a small RMSE reduction "
                f"({float(value_36['weighted_prior_error_rmse']):.3f} to "
                f"{float(era_36['weighted_prior_error_rmse']):.3f}), but its MAE still rises "
                f"({float(value_36['weighted_prior_error_mae']):.3f} to "
                f"{float(era_36['weighted_prior_error_mae']):.3f}). The evidence therefore "
                "does not support retaining the extra era interaction."
            ),
            "",
            (
                f"Audit artifact: `{output}`. It retains the plotted curve comparison and "
                "cohort calculations."
            ),
            SECTION_END,
        ]
    )
    replacement = "\n".join(lines)
    original = path.read_text()
    if SECTION_START in original:
        start = original.index(SECTION_START)
        end = original.index(SECTION_END, start) + len(SECTION_END)
        updated = original[:start] + replacement + original[end:]
    else:
        updated = original.rstrip() + "\n\n" + replacement + "\n"
    path.write_text(updated)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit era-conditioned aging against its value-conditioned parent"
    )
    parser.add_argument("--season", default=DEFAULT_SEASON)
    args = parser.parse_args()
    run = build_era_conditioned_aging_audit(season=args.season)
    print(f"Era-conditioned aging audit: run={run.run_dir}")


if __name__ == "__main__":
    main()
