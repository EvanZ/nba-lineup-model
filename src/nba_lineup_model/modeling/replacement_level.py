"""Historical low-exposure study for a candidate replacement-level prior."""

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

from nba_lineup_model.modeling.player_history import validate_player_season_panel
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.season.schema import validate_season

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_DOCS_ASSET_DIR = Path("docs/assets/images/replacement-level")
DEFAULT_REPLACEMENT_SHARE_CUTOFF = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 1_000
DEFAULT_BOOTSTRAP_SEED = 20260806
EXPOSURE_BANDS = (
    (0.0, 0.05, "0-5%"),
    (0.05, 0.15, "5-15%"),
    (0.15, 0.30, "15-30%"),
    (0.30, np.inf, "30%+"),
)


@dataclass(frozen=True)
class ReplacementLevelStudy:
    """Immutable outputs for a descriptive replacement-level analysis."""

    run_dir: Path
    run_id: str
    cohort_player_count: int
    candidate_replacement_prior: float


def build_replacement_level_study(
    *,
    through_season: str = "2025-26",
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    docs_asset_dir: Path | str = DEFAULT_DOCS_ASSET_DIR,
    replacement_share_cutoff: float = DEFAULT_REPLACEMENT_SHARE_CUTOFF,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> ReplacementLevelStudy:
    """Build a low-exposure report without changing a predictive prior."""

    cutoff = validate_season(through_season)
    if not 0.0 < replacement_share_cutoff < 1.0:
        raise ValueError("Replacement share cutoff must lie strictly between zero and one")
    if bootstrap_samples < 1:
        raise ValueError("Bootstrap samples must be positive")

    panel_path = Path(player_season_panel_path)
    validate_player_season_panel(panel_path.parent)
    panel = pd.read_parquet(panel_path)
    cohort = prepare_player_exposure_cohort(
        panel,
        through_season=cutoff,
        analytical_dir=analytical_dir,
    )
    band_summary = summarize_exposure_bands(
        cohort,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    season_estimates = summarize_low_exposure_seasons(
        cohort,
        replacement_share_cutoff=replacement_share_cutoff,
    )
    experience_summary = summarize_low_exposure_experience(
        cohort,
        replacement_share_cutoff=replacement_share_cutoff,
    )
    candidate_prior = candidate_replacement_prior(
        season_estimates,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return _write_study(
        cutoff=cutoff,
        panel_path=panel_path,
        cohort=cohort,
        band_summary=band_summary,
        season_estimates=season_estimates,
        experience_summary=experience_summary,
        candidate_prior=candidate_prior,
        replacement_share_cutoff=replacement_share_cutoff,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        artifacts_dir=Path(artifacts_dir),
        docs_asset_dir=Path(docs_asset_dir),
    )


def prepare_player_exposure_cohort(
    panel: pd.DataFrame,
    *,
    through_season: str,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
) -> pd.DataFrame:
    """Attach player-seasons to their season-long team possession opportunities."""

    required = {
        "season",
        "season_start_year",
        "player_id",
        "player_name",
        "is_rookie",
        "nba_experience_years",
        "rapm",
        "rapm_possessions",
        "listed_position",
    }
    missing = required - set(panel)
    if missing:
        raise ValueError(
            f"Player-season panel missing replacement-study columns: {sorted(missing)}"
        )

    cutoff_year = int(validate_season(through_season)[:4])
    players = panel.loc[
        panel["season_start_year"].le(cutoff_year),
        [
            "season",
            "season_start_year",
            "player_id",
            "player_name",
            "listed_position",
            "is_rookie",
            "nba_experience_years",
            "rapm",
            "rapm_possessions",
        ],
    ].copy()
    if players.empty:
        raise ValueError("Replacement study requires at least one player-season")
    if players.duplicated(["season", "player_id"]).any():
        raise ValueError("Player exposure cohort must be unique by season and player")

    season_frames: list[pd.DataFrame] = []
    for season, season_players in players.groupby("season", sort=True):
        exposure = player_exposure_shares(read_rapm_stints(str(season), analytical_dir))
        merged = season_players.merge(exposure, on="player_id", how="left", validate="one_to_one")
        if merged["exposure_share"].isna().any():
            missing_ids = merged.loc[merged["exposure_share"].isna(), "player_id"].tolist()
            raise ValueError(f"Player cohort has no RAPM exposure rows: {missing_ids[:5]}")
        if not np.allclose(
            merged["rapm_possessions"].to_numpy(dtype=float),
            merged["on_court_possessions"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"RAPM possession totals disagree with stint exposure for {season}")
        season_frames.append(merged)

    cohort = pd.concat(season_frames, ignore_index=True)
    cohort["exposure_band"] = exposure_band(cohort["exposure_share"])
    cohort["experience_band"] = experience_band(cohort["nba_experience_years"])
    return cohort.sort_values(["season_start_year", "player_id"], kind="stable").reset_index(
        drop=True
    )


def player_exposure_shares(stints: pd.DataFrame) -> pd.DataFrame:
    """Return each player's share of their teams' season-long possession opportunities."""

    required = {
        "possessions",
        "home_team_id",
        "away_team_id",
        "home_player_ids",
        "away_player_ids",
    }
    missing = required - set(stints)
    if missing:
        raise ValueError(f"RAPM stints missing exposure columns: {sorted(missing)}")
    if stints.empty or not stints["possessions"].gt(0).all():
        raise ValueError("RAPM stints require positive possession exposure")

    team_rows: list[pd.DataFrame] = []
    player_rows: list[pd.DataFrame] = []
    for side in ("home", "away"):
        team_id = f"{side}_team_id"
        player_id = f"{side}_player_ids"
        team_rows.append(
            stints.loc[:, [team_id, "possessions"]].rename(columns={team_id: "team_id"})
        )
        player_rows.append(
            stints.loc[:, [team_id, player_id, "possessions"]]
            .explode(player_id, ignore_index=True)
            .rename(columns={team_id: "team_id", player_id: "player_id"})
        )

    team_opportunities = (
        pd.concat(team_rows, ignore_index=True)
        .groupby("team_id", as_index=False, sort=True)["possessions"]
        .sum()
        .rename(columns={"possessions": "team_possessions"})
    )
    player_team = (
        pd.concat(player_rows, ignore_index=True)
        .astype({"player_id": "int64"})
        .groupby(["player_id", "team_id"], as_index=False, sort=True)["possessions"]
        .sum()
        .rename(columns={"possessions": "player_team_possessions"})
        .merge(team_opportunities, on="team_id", validate="many_to_one")
    )
    player_team["team_exposure_share"] = (
        player_team["player_team_possessions"] / player_team["team_possessions"]
    )
    if player_team["team_exposure_share"].gt(1.0 + 1e-9).any():
        raise ValueError("Player on-court possessions exceed a team season total")

    output = player_team.groupby("player_id", as_index=False, sort=True).agg(
        on_court_possessions=("player_team_possessions", "sum"),
        team_opportunity_possessions=("team_possessions", "sum"),
        exposure_share=("team_exposure_share", "sum"),
        team_count=("team_id", "nunique"),
    )
    if output["exposure_share"].gt(1.0 + 1e-9).any():
        raise ValueError("Player exposure share exceeds one season of opportunities")
    return output


def exposure_band(exposure_share: pd.Series) -> pd.Categorical:
    """Assign the fixed, descriptive low-exposure bands."""

    values = pd.to_numeric(exposure_share, errors="raise")
    if values.isna().any() or values.lt(0).any() or values.gt(1.0 + 1e-9).any():
        raise ValueError("Exposure share must be finite and lie between zero and one")
    return pd.cut(
        values,
        bins=[lower for lower, _, _ in EXPOSURE_BANDS] + [np.inf],
        labels=[label for _, _, label in EXPOSURE_BANDS],
        right=False,
        include_lowest=True,
    )


def experience_band(nba_experience_years: pd.Series) -> pd.Categorical:
    """Assign broad experience bands to reveal low-exposure cohort composition."""

    values = pd.to_numeric(nba_experience_years, errors="raise")
    if values.isna().any() or values.lt(0).any():
        raise ValueError("NBA experience years must be finite and non-negative")
    return pd.cut(
        values,
        bins=[-1, 0, 3, np.inf],
        labels=["First year", "Years 1-3", "Years 4+"],
    )


def summarize_exposure_bands(
    cohort: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Summarize all player-season RAPM values by fixed exposure share bands."""

    required = {
        "season",
        "player_id",
        "rapm",
        "rapm_possessions",
        "exposure_share",
        "exposure_band",
    }
    missing = required - set(cohort)
    if missing:
        raise ValueError(f"Exposure cohort missing summary columns: {sorted(missing)}")
    rows: list[dict[str, float | int | str]] = []
    for index, (_, _, label) in enumerate(EXPOSURE_BANDS):
        subset = cohort.loc[cohort["exposure_band"].astype(str).eq(label)].copy()
        if subset.empty:
            raise ValueError(f"Exposure band has no player-seasons: {label}")
        interval = _season_block_interval(
            subset,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
        )
        rows.append(
            {
                "exposure_band": label,
                "player_count": len(subset),
                "season_count": subset["season"].nunique(),
                "mean_exposure_share": float(subset["exposure_share"].mean()),
                "median_exposure_share": float(subset["exposure_share"].median()),
                "equal_player_mean_rapm": float(subset["rapm"].mean()),
                "median_rapm": float(subset["rapm"].median()),
                "possession_weighted_mean_rapm": float(
                    np.average(subset["rapm"], weights=subset["rapm_possessions"])
                ),
                "rapm_possessions": float(subset["rapm_possessions"].sum()),
                **interval,
            }
        )
    return pd.DataFrame(rows)


def summarize_low_exposure_seasons(
    cohort: pd.DataFrame,
    *,
    replacement_share_cutoff: float,
) -> pd.DataFrame:
    """Return one equal-player low-exposure estimate per historical season."""

    subset = cohort.loc[cohort["exposure_share"].lt(replacement_share_cutoff)].copy()
    if subset.empty:
        raise ValueError("Replacement share cutoff contains no player-seasons")
    summary = subset.groupby(["season", "season_start_year"], as_index=False, sort=True).agg(
        player_count=("player_id", "size"),
        mean_exposure_share=("exposure_share", "mean"),
        equal_player_mean_rapm=("rapm", "mean"),
        median_rapm=("rapm", "median"),
        possession_weighted_mean_rapm=(
            "rapm",
            lambda values: np.average(values, weights=subset.loc[values.index, "rapm_possessions"]),
        ),
        rapm_possessions=("rapm_possessions", "sum"),
    )
    return summary.sort_values("season_start_year", kind="stable").reset_index(drop=True)


def summarize_low_exposure_experience(
    cohort: pd.DataFrame,
    *,
    replacement_share_cutoff: float,
) -> pd.DataFrame:
    """Describe the low-exposure pool by career stage without defining contracts."""

    required = {"experience_band", "exposure_share", "player_id", "rapm", "rapm_possessions"}
    missing = required - set(cohort)
    if missing:
        raise ValueError(f"Exposure cohort missing experience columns: {sorted(missing)}")
    subset = cohort.loc[cohort["exposure_share"].lt(replacement_share_cutoff)].copy()
    if subset.empty:
        raise ValueError("Replacement share cutoff contains no player-seasons")
    rows: list[dict[str, float | int | str]] = []
    for label in ("First year", "Years 1-3", "Years 4+"):
        frame = subset.loc[subset["experience_band"].astype(str).eq(label)]
        if frame.empty:
            continue
        rows.append(
            {
                "experience_band": label,
                "player_season_count": len(frame),
                "mean_exposure_share": float(frame["exposure_share"].mean()),
                "equal_player_mean_rapm": float(frame["rapm"].mean()),
                "median_rapm": float(frame["rapm"].median()),
                "possession_weighted_mean_rapm": float(
                    np.average(frame["rapm"], weights=frame["rapm_possessions"])
                ),
            }
        )
    return pd.DataFrame(rows)


def candidate_replacement_prior(
    season_estimates: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    """Estimate a season-balanced candidate prior with a season-block interval."""

    values = season_estimates["equal_player_mean_rapm"].to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Replacement prior needs at least two finite season estimates")
    generator = np.random.default_rng(bootstrap_seed)
    draws = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        draws[index] = float(generator.choice(values, size=len(values), replace=True).mean())
    return {
        "candidate_replacement_prior": float(values.mean()),
        "season_balanced_median_rapm": float(np.median(values)),
        "season_block_bootstrap_lower": float(np.quantile(draws, 0.05)),
        "season_block_bootstrap_upper": float(np.quantile(draws, 0.95)),
        "season_count": int(len(values)),
    }


def _season_block_interval(
    subset: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, float]:
    season_means = subset.groupby("season", sort=True)["rapm"].mean().to_numpy(dtype=float)
    if len(season_means) < 2:
        raise ValueError("Exposure summary needs at least two seasons")
    generator = np.random.default_rng(bootstrap_seed)
    draws = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        draws[index] = float(
            generator.choice(season_means, size=len(season_means), replace=True).mean()
        )
    return {
        "season_balanced_mean_rapm": float(season_means.mean()),
        "season_block_bootstrap_lower": float(np.quantile(draws, 0.05)),
        "season_block_bootstrap_upper": float(np.quantile(draws, 0.95)),
    }


def _write_study(
    *,
    cutoff: str,
    panel_path: Path,
    cohort: pd.DataFrame,
    band_summary: pd.DataFrame,
    season_estimates: pd.DataFrame,
    experience_summary: pd.DataFrame,
    candidate_prior: dict[str, float | int],
    replacement_share_cutoff: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    artifacts_dir: Path,
    docs_asset_dir: Path,
) -> ReplacementLevelStudy:
    now = datetime.now(UTC)
    run_id = f"replacement-level-{cutoff}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "replacement_level" / cutoff
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        tables = {
            "player_exposure_cohort.parquet": cohort,
            "exposure_band_summary.parquet": band_summary,
            "low_exposure_season_estimates.parquet": season_estimates,
            "low_exposure_experience_summary.parquet": experience_summary,
        }
        for filename, frame in tables.items():
            frame.to_parquet(temporary / filename, index=False)
        prior_path = temporary / "candidate_replacement_prior.json"
        prior_payload = {
            **candidate_prior,
            "replacement_share_cutoff": replacement_share_cutoff,
            "definition": "All player-seasons below the exposure-share cutoff",
            "estimator": "Season-balanced mean of equal-player seasonal RAPM means",
            "status": "diagnostic_only_not_a_predictive_prior",
        }
        prior_path.write_text(json.dumps(prior_payload, indent=2) + "\n")
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": "low_exposure_replacement_level_diagnostic",
            "season": cutoff,
            "through_season": cutoff,
            "season_type": "regular",
            "cohort": "all regular-season player-season RAPM rows",
            "cohort_player_season_count": len(cohort),
            "cohort_season_count": int(cohort["season"].nunique()),
            "exposure_definition": (
                "sum over a player's teams of on-court RAPM possessions divided by "
                "that team's full regular-season RAPM possession opportunities"
            ),
            "replacement_share_cutoff": replacement_share_cutoff,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "candidate_prior_status": "diagnostic_only_not_a_predictive_prior",
            "source_panel_path": str(panel_path),
            "source_panel_manifest_sha256": _sha256_file(panel_path.parent / "_manifest.json"),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        chart_path = temporary / "replacement-level-study.svg"
        _render_study_chart(band_summary, season_estimates, chart_path)
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
        validate_replacement_level_study(output)
        docs_asset_dir.mkdir(parents=True, exist_ok=True)
        chart_path = output / "replacement-level-study.svg"
        shutil.copy2(chart_path, docs_asset_dir / chart_path.name)
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return ReplacementLevelStudy(
            run_dir=output,
            run_id=run_id,
            cohort_player_count=len(cohort),
            candidate_replacement_prior=float(candidate_prior["candidate_replacement_prior"]),
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_replacement_level_study(run_dir: Path | str) -> dict[str, object]:
    """Validate immutable replacement-level study artifacts and integrity hashes."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("candidate_prior_status") != "diagnostic_only_not_a_predictive_prior":
        raise ValueError("Replacement study status is not diagnostic-only")
    required = {
        "player_exposure_cohort.parquet",
        "exposure_band_summary.parquet",
        "low_exposure_season_estimates.parquet",
        "low_exposure_experience_summary.parquet",
        "candidate_replacement_prior.json",
        "metadata.json",
        "replacement-level-study.svg",
    }
    records = {record["filename"]: record for record in manifest["artifacts"]}
    if not required <= set(records):
        raise ValueError("Replacement study is missing required files")
    for filename, record in records.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["byte_count"]:
            raise ValueError(f"Replacement study changed: {filename}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Replacement study hash changed: {filename}")
        if record["row_count"] is not None and len(pd.read_parquet(path)) != record["row_count"]:
            raise ValueError(f"Replacement study row count changed: {filename}")
    return manifest


def _render_study_chart(
    band_summary: pd.DataFrame,
    season_estimates: pd.DataFrame,
    path: Path,
) -> None:
    figure, (band_axis, season_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        constrained_layout=True,
    )
    positions = np.arange(len(band_summary))
    means = band_summary["season_balanced_mean_rapm"].to_numpy(dtype=float)
    lower = band_summary["season_block_bootstrap_lower"].to_numpy(dtype=float)
    upper = band_summary["season_block_bootstrap_upper"].to_numpy(dtype=float)
    band_axis.errorbar(
        positions,
        means,
        yerr=np.vstack((means - lower, upper - means)),
        color="#1e628f",
        marker="o",
        linewidth=2,
        capsize=4,
    )
    band_axis.axhline(0.0, color="#596879", linewidth=0.8)
    band_axis.set(
        title="Player-season RAPM by exposure band",
        xlabel="Share of team possession opportunities",
        ylabel="Season-balanced mean RAPM",
        xticks=positions,
        xticklabels=band_summary["exposure_band"].tolist(),
    )
    season_axis.plot(
        season_estimates["season_start_year"],
        season_estimates["equal_player_mean_rapm"],
        color="#e66a25",
        marker="o",
        linewidth=1.6,
        markersize=3,
    )
    season_axis.axhline(0.0, color="#596879", linewidth=0.8)
    season_axis.set(
        title="Low-exposure player pool by season",
        xlabel="Season start year",
        ylabel="Equal-player mean RAPM",
    )
    for axis in (band_axis, season_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a low-exposure replacement-level study")
    parser.add_argument("--through-season", default="2025-26")
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--docs-asset-dir", default=str(DEFAULT_DOCS_ASSET_DIR))
    parser.add_argument(
        "--replacement-share-cutoff",
        type=float,
        default=DEFAULT_REPLACEMENT_SHARE_CUTOFF,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    study = build_replacement_level_study(
        through_season=args.through_season,
        player_season_panel_path=args.player_season_panel_path,
        analytical_dir=args.analytical_dir,
        artifacts_dir=args.artifacts_dir,
        docs_asset_dir=args.docs_asset_dir,
        replacement_share_cutoff=args.replacement_share_cutoff,
        bootstrap_samples=args.bootstrap_samples,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(study.run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(
        "Replacement-level study: "
        f"run={study.run_dir}; candidate_prior={study.candidate_replacement_prior:.3f}"
        f"{tracking_text}"
    )


if __name__ == "__main__":
    main()
