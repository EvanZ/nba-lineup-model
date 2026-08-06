"""Validate frozen forward cold-start rookie priors against post-season RAPM."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from nba_lineup_model.modeling.replacement_level import player_exposure_shares
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.season.schema import validate_season

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_DOCS_ASSET_DIR = Path("docs/assets/images/forward-cold-start-validation")
DEFAULT_DOCS_PAGE = Path("docs/models/forward-cold-start-validation.md")
DEFAULT_REPLACEMENT_SHARE_CUTOFF = 0.05


@dataclass(frozen=True)
class ForwardColdStartValidation:
    """Immutable forward cold-start validation outputs."""

    run_dir: Path
    run_id: str
    rookie_count: int


def build_forward_cold_start_validation(
    *,
    season: str = "2025-26",
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    docs_asset_dir: Path | str = DEFAULT_DOCS_ASSET_DIR,
) -> ForwardColdStartValidation:
    """Compare frozen first-year priors with same-season forward RAPM refits."""

    target_season = validate_season(season)
    artifacts_root = Path(artifacts_dir)
    forward_root = _latest_run(
        artifacts_root / "forward_exposure_gated_rapm" / target_season
    )
    validation = assemble_rookie_validation(
        forward_root=forward_root,
        season=target_season,
        panel=pd.read_parquet(player_season_panel_path),
        analytical_dir=Path(analytical_dir),
    )
    metrics = summarize_rookie_validation(validation)
    return _write_validation(
        season=target_season,
        forward_root=forward_root,
        validation=validation,
        metrics=metrics,
        artifacts_dir=artifacts_root,
        docs_asset_dir=Path(docs_asset_dir),
    )


def assemble_rookie_validation(
    *,
    forward_root: Path | str,
    season: str,
    panel: pd.DataFrame,
    analytical_dir: Path | str,
) -> pd.DataFrame:
    """Join a season's frozen cold-start priors to post-fit forward coefficients."""

    season = validate_season(season)
    root = Path(forward_root)
    priors = pd.read_parquet(root / "season_player_priors.parquet")
    estimates = pd.read_parquet(root / "historical_player_coefficients.parquet")
    required_panel = {
        "season",
        "player_id",
        "player_name",
        "listed_position",
        "primary_team_tricode",
        "draft_number",
        "is_undrafted",
        "is_rookie",
    }
    missing = required_panel - set(panel)
    if missing:
        raise ValueError(f"Player-season panel missing validation columns: {sorted(missing)}")
    predicted = priors.loc[
        priors["season"].eq(season)
        & priors["prior_branch"].eq("exposure_gated_cold_start"),
        ["player_id", "lagged_rapm_prior"],
    ].rename(columns={"lagged_rapm_prior": "cold_start_rapm_prior"})
    actual = estimates.loc[
        estimates["season"].eq(season),
        ["player_id", "rapm", "rapm_adjustment_from_prior", "selected_lambda"],
    ].rename(columns={"rapm": "refit_forward_rapm"})
    rookies = panel.loc[
        panel["season"].eq(season) & panel["is_rookie"].astype(bool),
        [
            "player_id",
            "player_name",
            "listed_position",
            "primary_team_tricode",
            "draft_number",
            "is_undrafted",
        ],
    ].copy()
    exposure = player_exposure_shares(read_rapm_stints(season, analytical_dir=analytical_dir))
    output = predicted.merge(rookies, on="player_id", how="inner", validate="one_to_one")
    output = output.merge(actual, on="player_id", how="inner", validate="one_to_one")
    output = output.merge(exposure, on="player_id", how="inner", validate="one_to_one")
    if len(output) != len(predicted):
        raise ValueError("Forward cold-start priors do not map one-to-one to rookie outcomes")
    output["prediction_error"] = output["refit_forward_rapm"] - output["cold_start_rapm_prior"]
    output["actual_low_exposure"] = output["exposure_share"].lt(
        DEFAULT_REPLACEMENT_SHARE_CUTOFF
    )
    output["draft_status"] = np.where(
        output["is_undrafted"].astype(bool),
        "Undrafted",
        "Drafted 1-60",
    )
    output["prior_rank"] = output["cold_start_rapm_prior"].rank(
        method="first", ascending=False
    ).astype(int)
    output["refit_rank"] = output["refit_forward_rapm"].rank(
        method="first", ascending=False
    ).astype(int)
    return output.sort_values("prior_rank", kind="stable").reset_index(drop=True)


def summarize_rookie_validation(validation: pd.DataFrame) -> pd.DataFrame:
    """Compute transparent unweighted and possession-weighted rookie metrics."""

    required = {
        "cold_start_rapm_prior",
        "refit_forward_rapm",
        "on_court_possessions",
        "draft_status",
        "actual_low_exposure",
    }
    missing = required - set(validation)
    if missing:
        raise ValueError(f"Rookie validation missing columns: {sorted(missing)}")
    groups = {
        "All first-year players": validation,
        "Drafted 1-60": validation.loc[validation["draft_status"].eq("Drafted 1-60")],
        "Undrafted": validation.loc[validation["draft_status"].eq("Undrafted")],
        "Realized low exposure (<5%)": validation.loc[validation["actual_low_exposure"]],
        "Realized rotation exposure (>=5%)": validation.loc[
            ~validation["actual_low_exposure"]
        ],
    }
    return pd.DataFrame(
        [_metric_row(name, frame) for name, frame in groups.items() if len(frame) >= 3]
    )


def render_forward_cold_start_validation_page(
    run_dir: Path | str,
    *,
    page_path: Path | str = DEFAULT_DOCS_PAGE,
) -> Path:
    """Render methodology, metrics, and a sortable rookie-validation table."""

    root = Path(run_dir)
    metadata = validate_forward_cold_start_validation(root)
    metrics = pd.read_parquet(root / "metrics.parquet")
    rookies = pd.read_parquet(root / "rookie_validation.parquet")
    lines = [
        "# Forward Cold-Start Validation",
        "",
        "This study tests the first-year branch of the forward exposure-gated RAPM model. "
        "It compares every 2025-26 rookie's frozen preseason prior to the coefficient "
        "from the completed 2025-26 forward RAPM refit.",
        "",
        "The outcome is not an independent ground-truth player value. It is the model's "
        "post-season, regular-only estimate and remains noisy for low-exposure players. "
        "That is why the report presents possession-weighted metrics and the realized "
        "exposure split alongside the raw correlation.",
        "",
        "## Definition",
        "",
        r"For rookie \(i\), the frozen preseason prediction is \(\widehat R_i^{pre}\) "
        r"and the post-season forward-RAPM refit is \(\widehat R_i^{post}\). The table "
        r"reports the error \(\widehat R_i^{post} - \widehat R_i^{pre}\). Possession-"
        r"weighted statistics use each player's on-court possessions as \(w_i\).",
        "",
        "## 2025-26 Metrics",
        "",
        (
            "| Cohort | Players | On-court possessions | Pearson | Weighted Pearson | Spearman | "
            "Weighted Spearman | MAE | Weighted MAE | RMSE | Weighted RMSE |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.player_count} | {row.on_court_possessions:,.0f} | "
            f"{row.pearson_correlation:.3f} | {row.weighted_pearson_correlation:.3f} | "
            f"{row.spearman_correlation:.3f} | {row.weighted_spearman_correlation:.3f} | "
            f"{row.mae:.3f} | {row.weighted_mae:.3f} | {row.rmse:.3f} | "
            f"{row.weighted_rmse:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Prior Versus Refit",
            "",
            "Each point is a first-year player. Point area scales with on-court possessions; "
            "orange points finished below 5% of team possession opportunities.",
            "",
            (
                "![Frozen prior versus refitted forward RAPM]"
                "(../assets/images/forward-cold-start-validation/"
                f"{metadata['season']}-prior-vs-refit.svg)"
            ),
            "",
            "## Rookie Detail",
            "",
            "The table is sortable by every column. `Prior rank` orders the pre-season "
            "forecast; `Refit rank` orders the completed regular-season coefficient.",
            "",
            (
                "| Prior rank | Prior | Refit rank | Refit RAPM | Adjustment | Player | Team | "
                "Pos. | "
                "Draft status | Pick | Possessions | Exposure | Low exposure |"
            ),
            (
                "| ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | "
                "---: | --- |"
            ),
        ]
    )
    for row in rookies.itertuples(index=False):
        pick = "" if pd.isna(row.draft_number) else str(int(row.draft_number))
        lines.append(
            f"| {row.prior_rank} | {row.cold_start_rapm_prior:+.2f} | {row.refit_rank} | "
            f"{row.refit_forward_rapm:+.2f} | {row.prediction_error:+.2f} | "
            f"{row.player_name} | {row.primary_team_tricode} | {row.listed_position} | "
            f"{row.draft_status} | {pick} | {row.on_court_possessions:,.0f} | "
            f"{row.exposure_share:.1%} | {'Yes' if row.actual_low_exposure else 'No'} |"
        )
    destination = Path(page_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return destination


def validate_forward_cold_start_validation(run_dir: Path | str) -> dict[str, object]:
    """Validate hashes and the frozen-prior/post-refit temporal boundary."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("prediction_timing") != "frozen before target-season fit":
        raise ValueError("Forward cold-start validation does not use frozen predictions")
    records = {record["filename"]: record for record in manifest["artifacts"]}
    required = {
        "rookie_validation.parquet",
        "metrics.parquet",
        "prior-vs-refit.svg",
        "metadata.json",
    }
    if not required <= set(records):
        raise ValueError("Forward cold-start validation is missing required files")
    for filename, record in records.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["byte_count"]:
            raise ValueError(f"Forward cold-start validation changed: {filename}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Forward cold-start validation hash changed: {filename}")
        if record["row_count"] is not None and len(pd.read_parquet(path)) != record["row_count"]:
            raise ValueError(f"Forward cold-start validation row count changed: {filename}")
    return manifest


def _metric_row(cohort: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    predicted = frame["cold_start_rapm_prior"].to_numpy(dtype=float)
    actual = frame["refit_forward_rapm"].to_numpy(dtype=float)
    weights = frame["on_court_possessions"].to_numpy(dtype=float)
    error = actual - predicted
    return {
        "cohort": cohort,
        "player_count": len(frame),
        "on_court_possessions": float(weights.sum()),
        "pearson_correlation": _correlation(predicted, actual),
        "weighted_pearson_correlation": _weighted_correlation(predicted, actual, weights),
        "spearman_correlation": _correlation(rankdata(predicted), rankdata(actual)),
        "weighted_spearman_correlation": _weighted_correlation(
            rankdata(predicted), rankdata(actual), weights
        ),
        "mae": float(np.mean(np.abs(error))),
        "weighted_mae": float(np.average(np.abs(error), weights=weights)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "weighted_rmse": float(np.sqrt(np.average(error**2, weights=weights))),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _weighted_correlation(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Possession weights must be finite and positive")
    left_mean = np.average(left, weights=weights)
    right_mean = np.average(right, weights=weights)
    covariance = np.average((left - left_mean) * (right - right_mean), weights=weights)
    left_scale = np.sqrt(np.average((left - left_mean) ** 2, weights=weights))
    right_scale = np.sqrt(np.average((right - right_mean) ** 2, weights=weights))
    if left_scale <= 1e-12 or right_scale <= 1e-12:
        return np.nan
    return float(covariance / (left_scale * right_scale))


def _latest_run(root: Path) -> Path:
    latest = json.loads((root / "latest.json").read_text())
    output = root / str(latest["run_id"])
    if not output.is_dir():
        raise FileNotFoundError(f"Forward RAPM artifact does not exist: {output}")
    return output


def _write_validation(
    *,
    season: str,
    forward_root: Path,
    validation: pd.DataFrame,
    metrics: pd.DataFrame,
    artifacts_dir: Path,
    docs_asset_dir: Path,
) -> ForwardColdStartValidation:
    now = datetime.now(UTC)
    run_id = f"forward-cold-start-validation-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "forward_cold_start_validation" / season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        validation.to_parquet(temporary / "rookie_validation.parquet", index=False)
        metrics.to_parquet(temporary / "metrics.parquet", index=False)
        _render_scatter(validation, temporary / "prior-vs-refit.svg")
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": "forward_exposure_gated_cold_start_validation",
            "season": season,
            "rookie_count": len(validation),
            "prediction_timing": "frozen before target-season fit",
            "outcome": "same-season post-fit regular-only forward RAPM coefficient",
            "exposure_cutoff": DEFAULT_REPLACEMENT_SHARE_CUTOFF,
            "forward_run": str(forward_root),
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {
                "filename": path.name,
                "row_count": (
                    len(validation)
                    if path.name == "rookie_validation.parquet"
                    else len(metrics)
                    if path.name == "metrics.parquet"
                    else None
                ),
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
        validate_forward_cold_start_validation(output)
        docs_asset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "prior-vs-refit.svg", docs_asset_dir / f"{season}-prior-vs-refit.svg")
        latest_tmp = root / "latest.json.tmp"
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(root / "latest.json")
        return ForwardColdStartValidation(output, run_id, len(validation))
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _render_scatter(validation: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    for low_exposure, color, label in (
        (False, "#176b87", "At least 5% exposure"),
        (True, "#d26a27", "Below 5% exposure"),
    ):
        group = validation.loc[validation["actual_low_exposure"].eq(low_exposure)]
        axis.scatter(
            group["cold_start_rapm_prior"],
            group["refit_forward_rapm"],
            s=18 + 3.2 * np.sqrt(group["on_court_possessions"]),
            alpha=0.72,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            label=label,
        )
    limits = np.array(
        [
            validation[["cold_start_rapm_prior", "refit_forward_rapm"]].min().min(),
            validation[["cold_start_rapm_prior", "refit_forward_rapm"]].max().max(),
        ]
    )
    padding = 0.08 * max(limits[1] - limits[0], 1.0)
    limits += np.array([-padding, padding])
    axis.plot(limits, limits, color="#555555", linestyle="--", linewidth=1.2, label="Equal")
    axis.set(
        xlim=limits,
        ylim=limits,
        xlabel="Frozen preseason cold-start RAPM prior",
        ylabel="Post-season refitted forward RAPM",
        title="2025-26 first-year players",
    )
    axis.legend(frameon=False, loc="upper left")
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen forward cold-start rookie priors")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--docs-asset-dir", default=str(DEFAULT_DOCS_ASSET_DIR))
    parser.add_argument("--render-docs-page", action="store_true")
    args = parser.parse_args()
    result = build_forward_cold_start_validation(
        season=args.season,
        player_season_panel_path=args.player_season_panel_path,
        analytical_dir=args.analytical_dir,
        artifacts_dir=args.artifacts_dir,
        docs_asset_dir=args.docs_asset_dir,
    )
    if args.render_docs_page:
        render_forward_cold_start_validation_page(result.run_dir)
    print(f"run={result.run_dir}; rookies={result.rookie_count}")


if __name__ == "__main__":
    main()
