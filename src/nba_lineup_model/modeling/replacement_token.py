"""Estimate a pooled replacement-player token directly in the RAPM design."""

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
from nba_lineup_model.modeling.replacement_level import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_PANEL_PATH,
    player_exposure_shares,
)
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.models.baselines import (
    RidgeLineupModel,
    entity_vocabulary,
    signed_entity_matrix,
    vocabulary_mapping,
)
from nba_lineup_model.season.schema import validate_season

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_DOCS_ASSET_DIR = Path("docs/assets/images/replacement-token")
DEFAULT_REPLACEMENT_SHARE_CUTOFF = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 1_000
DEFAULT_BOOTSTRAP_SEED = 20260807
REPLACEMENT_TOKEN_ID = -1


@dataclass(frozen=True)
class ReplacementTokenStudy:
    """Immutable outputs for a pooled replacement-token RAPM study."""

    run_dir: Path
    run_id: str
    season_count: int
    season_balanced_replacement_rapm: float


def build_replacement_token_study(
    *,
    through_season: str = "2025-26",
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    docs_asset_dir: Path | str = DEFAULT_DOCS_ASSET_DIR,
    replacement_share_cutoff: float = DEFAULT_REPLACEMENT_SHARE_CUTOFF,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> ReplacementTokenStudy:
    """Fit a common replacement token for every historical regular season."""

    cutoff = validate_season(through_season)
    if not 0.0 < replacement_share_cutoff < 1.0:
        raise ValueError("Replacement share cutoff must lie strictly between zero and one")
    if bootstrap_samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    panel_path = Path(player_season_panel_path)
    validate_player_season_panel(panel_path.parent)
    panel = pd.read_parquet(panel_path)
    summary = fit_historical_replacement_tokens(
        panel,
        through_season=cutoff,
        analytical_dir=analytical_dir,
        artifacts_dir=artifacts_dir,
        replacement_share_cutoff=replacement_share_cutoff,
    )
    aggregate = aggregate_replacement_token_coefficients(
        summary,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return _write_study(
        cutoff=cutoff,
        panel_path=panel_path,
        summary=summary,
        aggregate=aggregate,
        replacement_share_cutoff=replacement_share_cutoff,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        artifacts_dir=Path(artifacts_dir),
        docs_asset_dir=Path(docs_asset_dir),
    )


def fit_historical_replacement_tokens(
    panel: pd.DataFrame,
    *,
    through_season: str,
    analytical_dir: Path | str,
    artifacts_dir: Path | str,
    replacement_share_cutoff: float,
) -> pd.DataFrame:
    """Fit one replacement-token RAPM coefficient per season through the cutoff."""

    required = {"season", "season_start_year", "player_id", "rapm", "rapm_run_id"}
    missing = required - set(panel)
    if missing:
        raise ValueError(
            f"Player-season panel missing replacement-token columns: {sorted(missing)}"
        )
    cutoff_year = int(validate_season(through_season)[:4])
    selected = panel.loc[panel["season_start_year"].le(cutoff_year)].copy()
    rows: list[dict[str, float | int | str]] = []
    for season, season_panel in selected.groupby("season", sort=True):
        stints = read_rapm_stints(str(season), analytical_dir)
        exposure = player_exposure_shares(stints)
        panel_player_ids = set(season_panel["player_id"].astype(int))
        replacement_ids = set(
            exposure.loc[
                exposure["exposure_share"].lt(replacement_share_cutoff)
                & exposure["player_id"].astype(int).isin(panel_player_ids),
                "player_id",
            ].astype(int)
        )
        if not replacement_ids:
            raise ValueError(f"No replacement candidates in {season}")
        lambda_value = canonical_rapm_lambda(
            season_panel,
            season=str(season),
            artifacts_dir=artifacts_dir,
        )
        coefficient, intercept = fit_replacement_token_season(
            stints,
            replacement_player_ids=replacement_ids,
            regularization=lambda_value,
        )
        reference = season_panel.loc[
            season_panel["player_id"].astype(int).isin(replacement_ids), "rapm"
        ]
        group_exposure = exposure.loc[
            exposure["player_id"].astype(int).isin(replacement_ids), "on_court_possessions"
        ].sum()
        rows.append(
            {
                "season": str(season),
                "season_start_year": int(season_panel["season_start_year"].iloc[0]),
                "canonical_rapm_regularization": lambda_value,
                "replacement_player_count": len(replacement_ids),
                "replacement_on_court_possessions": float(group_exposure),
                "replacement_token_rapm": coefficient,
                "intercept": intercept,
                "separately_ridged_low_exposure_mean_rapm": float(reference.mean()),
                "separately_ridged_low_exposure_median_rapm": float(reference.median()),
            }
        )
    return pd.DataFrame(rows).sort_values("season_start_year", kind="stable").reset_index(drop=True)


def canonical_rapm_lambda(
    season_panel: pd.DataFrame,
    *,
    season: str,
    artifacts_dir: Path | str,
) -> float:
    """Read the chosen canonical RAPM penalty associated with panel rows."""

    run_ids = set(season_panel["rapm_run_id"].astype(str))
    if len(run_ids) != 1:
        raise ValueError(f"Player panel has multiple RAPM run IDs for {season}")
    run_id = run_ids.pop()
    manifest_path = Path(artifacts_dir) / "rapm" / season / run_id / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Canonical RAPM manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_id") != run_id or manifest.get("season") != season:
        raise ValueError("Canonical RAPM manifest does not match the player-season panel")
    value = float(manifest["selected_rapm_lambda"])
    if value < 0:
        raise ValueError("Canonical RAPM regularization must be non-negative")
    return value


def fit_replacement_token_season(
    stints: pd.DataFrame,
    *,
    replacement_player_ids: set[int],
    regularization: float,
) -> tuple[float, float]:
    """Fit one season after mapping every replacement candidate to one token."""

    if REPLACEMENT_TOKEN_ID in replacement_player_ids:
        raise ValueError("Replacement token ID conflicts with a player ID")
    tokenized = tokenize_replacement_lineups(stints, replacement_player_ids)
    player_ids = entity_vocabulary(
        tokenized,
        "home_player_ids",
        "away_player_ids",
        multiple=True,
    )
    if REPLACEMENT_TOKEN_ID not in player_ids:
        raise ValueError("Replacement token never appears in a lineup")
    matrix = signed_entity_matrix(
        tokenized,
        "home_player_ids",
        "away_player_ids",
        vocabulary_mapping(player_ids),
        multiple=True,
    )
    model = RidgeLineupModel(regularization).fit(
        matrix,
        tokenized["target_home_net_rating"].to_numpy(dtype=float),
        tokenized["possessions"].to_numpy(dtype=float),
    )
    coefficient_map = dict(zip(player_ids, model.coef_, strict=True))
    return float(coefficient_map[REPLACEMENT_TOKEN_ID]), model.intercept_


def tokenize_replacement_lineups(
    stints: pd.DataFrame,
    replacement_player_ids: set[int],
) -> pd.DataFrame:
    """Map eligible player IDs to a shared token while retaining token counts."""

    required = {"home_player_ids", "away_player_ids", "possessions", "target_home_net_rating"}
    missing = required - set(stints)
    if missing:
        raise ValueError(f"RAPM stints missing token-model columns: {sorted(missing)}")
    output = stints.copy()
    for side in ("home", "away"):
        column = f"{side}_player_ids"
        output[column] = output[column].map(
            lambda lineup: tuple(
                REPLACEMENT_TOKEN_ID if int(player_id) in replacement_player_ids else int(player_id)
                for player_id in lineup
            )
        )
    return output


def aggregate_replacement_token_coefficients(
    summary: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    """Summarize season-level replacement coefficients without pooling eras directly."""

    values = summary["replacement_token_rapm"].to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Replacement token study needs at least two finite season estimates")
    generator = np.random.default_rng(bootstrap_seed)
    draws = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        draws[index] = float(generator.choice(values, size=len(values), replace=True).mean())
    return {
        "season_balanced_replacement_token_rapm": float(values.mean()),
        "season_balanced_median_replacement_token_rapm": float(np.median(values)),
        "season_block_bootstrap_lower": float(np.quantile(draws, 0.05)),
        "season_block_bootstrap_upper": float(np.quantile(draws, 0.95)),
        "season_count": int(len(values)),
    }


def _write_study(
    *,
    cutoff: str,
    panel_path: Path,
    summary: pd.DataFrame,
    aggregate: dict[str, float | int],
    replacement_share_cutoff: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    artifacts_dir: Path,
    docs_asset_dir: Path,
) -> ReplacementTokenStudy:
    now = datetime.now(UTC)
    run_id = f"replacement-token-{cutoff}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "replacement_token" / cutoff
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        summary_path = temporary / "season_replacement_token_coefficients.parquet"
        summary.to_parquet(summary_path, index=False)
        aggregate_path = temporary / "replacement_token_summary.json"
        aggregate_path.write_text(
            json.dumps(
                {
                    **aggregate,
                    "replacement_share_cutoff": replacement_share_cutoff,
                    "definition": "One shared lineup token for all realized low-exposure players",
                    "status": "retrospective_design_diagnostic",
                },
                indent=2,
            )
            + "\n"
        )
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": "pooled_replacement_token_rapm",
            "season": cutoff,
            "through_season": cutoff,
            "season_type": "regular",
            "replacement_share_cutoff": replacement_share_cutoff,
            "replacement_token_id": REPLACEMENT_TOKEN_ID,
            "regularization": "canonical selected per-season RAPM lambda",
            "membership": "realized same-season exposure share below cutoff",
            "status": "retrospective_design_diagnostic",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "source_panel_path": str(panel_path),
            "source_panel_manifest_sha256": _sha256_file(panel_path.parent / "_manifest.json"),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        chart_path = temporary / "replacement-token-by-season.svg"
        _render_token_chart(summary, aggregate, chart_path)
        records = [
            {
                "filename": path.name,
                "row_count": len(summary) if path.name == summary_path.name else None,
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
        validate_replacement_token_study(output)
        docs_asset_dir.mkdir(parents=True, exist_ok=True)
        chart_path = output / "replacement-token-by-season.svg"
        shutil.copy2(chart_path, docs_asset_dir / chart_path.name)
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return ReplacementTokenStudy(
            run_dir=output,
            run_id=run_id,
            season_count=int(aggregate["season_count"]),
            season_balanced_replacement_rapm=float(
                aggregate["season_balanced_replacement_token_rapm"]
            ),
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_replacement_token_study(run_dir: Path | str) -> dict[str, object]:
    """Validate immutable pooled replacement-token artifacts."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "retrospective_design_diagnostic":
        raise ValueError("Replacement-token study status is invalid")
    required = {
        "season_replacement_token_coefficients.parquet",
        "replacement_token_summary.json",
        "replacement-token-by-season.svg",
        "metadata.json",
    }
    records = {record["filename"]: record for record in manifest["artifacts"]}
    if not required <= set(records):
        raise ValueError("Replacement-token study is missing required files")
    for filename, record in records.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["byte_count"]:
            raise ValueError(f"Replacement-token study changed: {filename}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Replacement-token study hash changed: {filename}")
        if record["row_count"] is not None and len(pd.read_parquet(path)) != record["row_count"]:
            raise ValueError(f"Replacement-token study row count changed: {filename}")
    return manifest


def _render_token_chart(
    summary: pd.DataFrame,
    aggregate: dict[str, float | int],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    axis.plot(
        summary["season_start_year"],
        summary["replacement_token_rapm"],
        color="#1e628f",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Pooled replacement-token RAPM",
    )
    axis.plot(
        summary["season_start_year"],
        summary["separately_ridged_low_exposure_mean_rapm"],
        color="#e66a25",
        linewidth=1.2,
        alpha=0.85,
        label="Mean of separately ridged low-exposure players",
    )
    axis.axhline(
        float(aggregate["season_balanced_replacement_token_rapm"]),
        color="#246b4a",
        linewidth=1.2,
        linestyle="--",
        label="Season-balanced token mean",
    )
    axis.axhline(0.0, color="#596879", linewidth=0.8)
    axis.set(
        title="Replacement-token RAPM by season",
        xlabel="Season start year",
        ylabel="RAPM",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pooled replacement-token RAPM study")
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
    study = build_replacement_token_study(
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
        "Replacement-token study: "
        f"run={study.run_dir}; token_rapm={study.season_balanced_replacement_rapm:.3f}"
        f"{tracking_text}"
    )


if __name__ == "__main__":
    main()
