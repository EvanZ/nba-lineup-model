"""One-season lambda sensitivity analysis for Student-t talent-prior RAPM."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.modeling import student_t_talent_forward_rapm
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_DOCS_PAGE = Path("docs/models/student-t-talent-lambda-sensitivity.md")
MODEL_NAME = "student_t_talent_lambda_sensitivity"


@dataclass(frozen=True)
class StudentTTalentLambdaSensitivityRun:
    run_dir: Path
    run_id: str


def analyze_student_t_talent_lambda_sensitivity(
    *,
    season: str = "2025-26",
    alternative_lambda: float = 0.10,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
) -> StudentTTalentLambdaSensitivityRun:
    """Refit one completed season with a fixed alternative lambda.

    The prior entering the target season is read from the source immutable run;
    only the final-season shrinkage strength changes.
    """

    if alternative_lambda <= 0:
        raise ValueError("Alternative lambda must be positive")
    source_run = _latest_run(Path(artifacts_dir) / "student_t_talent_forward_rapm" / season)
    source_metadata = json.loads((source_run / "metadata.json").read_text())
    if source_metadata.get("model") != student_t_talent_forward_rapm.MODEL_NAME:
        raise ValueError("Lambda sensitivity requires a Student-t talent-prior source run")
    degrees_of_freedom = float(source_metadata["degrees_of_freedom"])
    prior_scale = float(source_metadata["prior_scale"])
    coefficients = pd.read_parquet(source_run / "historical_player_coefficients.parquet")
    source_lambda = float(
        coefficients.loc[coefficients["season"].eq(season), "selected_lambda"].iloc[0]
    )
    priors = pd.read_parquet(source_run / "season_player_priors.parquet")
    entering_priors = priors.loc[priors["season"].eq(season)].drop(columns="season")
    stints = read_rapm_stints(season, analytical_dir=analytical_dir)
    alternative, diagnostics = student_t_talent_forward_rapm._fit_season(
        season,
        stints,
        entering_priors,
        alternative_lambda,
        degrees_of_freedom,
        prior_scale,
        120,
    )
    panel = pd.read_parquet(player_season_panel_path)
    source_rankings = pd.read_parquet(source_run / "next_season_returning_rankings.parquet")
    alternative_rankings = _rank_coefficients(alternative.player_estimates, panel, season)
    comparison = _compare_rankings(
        source_rankings, alternative_rankings, source_lambda, alternative_lambda
    )
    summary = _summary(comparison, source_lambda, alternative_lambda, diagnostics)
    run = _write_run(
        season=season,
        source_run=source_run,
        source_metadata=source_metadata,
        source_lambda=source_lambda,
        alternative_lambda=alternative_lambda,
        source_rankings=source_rankings,
        alternative_rankings=alternative_rankings,
        comparison=comparison,
        summary=summary,
        artifacts_dir=Path(artifacts_dir),
    )
    render_student_t_talent_lambda_sensitivity_page(run.run_dir)
    return run


def _latest_run(root: Path) -> Path:
    return root / str(json.loads((root / "latest.json").read_text())["run_id"])


def _rank_coefficients(
    estimates: pd.DataFrame,
    panel: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    ranking = (
        estimates.loc[:, ["player_id", "rapm", "prior_rapm", "rapm_adjustment_from_prior"]]
        .merge(
            panel.loc[
                panel["season"].eq(season),
                ["player_id", "player_name", "listed_position", "rapm_possessions"],
            ],
            on="player_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            ["rapm", "rapm_possessions", "player_id"],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1
    return ranking


def _compare_rankings(
    source: pd.DataFrame,
    alternative: pd.DataFrame,
    source_lambda: float,
    alternative_lambda: float,
) -> pd.DataFrame:
    source_name = _lambda_label(source_lambda)
    alternative_name = _lambda_label(alternative_lambda)
    source_columns = source.loc[:, ["player_id", "rank", "rapm"]].rename(
        columns={"rank": f"rank_lambda_{source_name}", "rapm": f"rapm_lambda_{source_name}"}
    )
    alternative_columns = alternative.loc[
        :, ["player_id", "player_name", "listed_position", "rapm_possessions", "rank", "rapm"]
    ].rename(
        columns={
            "rank": f"rank_lambda_{alternative_name}",
            "rapm": f"rapm_lambda_{alternative_name}",
        }
    )
    comparison = alternative_columns.merge(
        source_columns, on="player_id", how="inner", validate="one_to_one"
    )
    comparison["rapm_difference"] = (
        comparison[f"rapm_lambda_{alternative_name}"] - comparison[f"rapm_lambda_{source_name}"]
    )
    comparison["rank_difference"] = (
        comparison[f"rank_lambda_{alternative_name}"] - comparison[f"rank_lambda_{source_name}"]
    )
    return comparison.sort_values(f"rank_lambda_{alternative_name}", kind="stable").reset_index(
        drop=True
    )


def _summary(
    comparison: pd.DataFrame,
    source_lambda: float,
    alternative_lambda: float,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    source_name = _lambda_label(source_lambda)
    alternative_name = _lambda_label(alternative_lambda)
    source_rating = comparison[f"rapm_lambda_{source_name}"]
    alternative_rating = comparison[f"rapm_lambda_{alternative_name}"]
    return {
        "source_lambda": source_lambda,
        "alternative_lambda": alternative_lambda,
        "player_count": int(len(comparison)),
        "pearson_correlation": float(source_rating.corr(alternative_rating)),
        "spearman_correlation": float(source_rating.corr(alternative_rating, method="spearman")),
        "mean_absolute_rating_difference": float((alternative_rating - source_rating).abs().mean()),
        "maximum_absolute_rating_difference": float(
            (alternative_rating - source_rating).abs().max()
        ),
        "mean_absolute_rank_difference": float(comparison["rank_difference"].abs().mean()),
        "maximum_absolute_rank_difference": int(comparison["rank_difference"].abs().max()),
        **diagnostics,
    }


def _write_run(
    *,
    season: str,
    source_run: Path,
    source_metadata: dict[str, object],
    source_lambda: float,
    alternative_lambda: float,
    source_rankings: pd.DataFrame,
    alternative_rankings: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: dict[str, object],
    artifacts_dir: Path,
) -> StudentTTalentLambdaSensitivityRun:
    now = datetime.now(UTC)
    run_id = f"student-t-talent-lambda-sensitivity-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "student_t_talent_lambda_sensitivity" / season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for name, frame in {
            "source_rankings.parquet": source_rankings,
            "alternative_rankings.parquet": alternative_rankings,
            "ranking_comparison.parquet": comparison,
        }.items():
            frame.to_parquet(temporary / name, index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "season": season,
            "source_run": str(source_run),
            "source_model": source_metadata["model"],
            "source_lambda": source_lambda,
            "alternative_lambda": alternative_lambda,
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
        }
        (temporary / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return StudentTTalentLambdaSensitivityRun(output, run_id)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def render_student_t_talent_lambda_sensitivity_page(
    run_dir: Path | str,
    *,
    page_path: Path | str = DEFAULT_DOCS_PAGE,
) -> Path:
    """Render the fixed-prior 2025-26 lambda-sensitivity report."""

    root = Path(run_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    comparison = pd.read_parquet(root / "ranking_comparison.parquet")
    source_name = _lambda_label(float(metadata["source_lambda"]))
    alternative_name = _lambda_label(float(metadata["alternative_lambda"]))
    lines = [
        "# Student-t Talent-Prior Lambda Sensitivity",
        "",
        "This controlled post-season refit holds every 2025-26 entering player prior, "
        "the Gaussian stint-error model, and the Student-t talent-prior settings fixed. "
        "It changes only the final-season ridge lambda from "
        f"`{float(metadata['source_lambda']):g}` to `{float(metadata['alternative_lambda']):g}`.",
        "",
        f"- Source run: `{metadata['source_run']}`.",
        f"- Players: {summary['player_count']:,}.",
        f"- Rating Pearson/Spearman: {summary['pearson_correlation']:.6f} / "
        f"{summary['spearman_correlation']:.6f}.",
        f"- Mean / maximum absolute rating difference: "
        f"{summary['mean_absolute_rating_difference']:.3f} / "
        f"{summary['maximum_absolute_rating_difference']:.3f} RAPM.",
        f"- Mean / maximum absolute rank movement: "
        f"{summary['mean_absolute_rank_difference']:.1f} / "
        f"{summary['maximum_absolute_rank_difference']} places.",
        "",
        "This is not a new frozen-preseason leaderboard entry: 2025-26 outcomes are "
        "used for both post-season refits. It measures ranking-scale sensitivity only.",
        "",
        "## Lambda 0.10 Ranking",
        "",
        f"| Rank ({alternative_name}) | Player | Pos. | RAPM ({alternative_name}) | "
        f"RAPM ({source_name}) | Difference | Rank ({source_name}) | Rank movement | Possessions |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.itertuples(index=False):
        position = "" if pd.isna(row.listed_position) else str(row.listed_position)
        lines.append(
            f"| {getattr(row, f'rank_lambda_{alternative_name}')} | {row.player_name} | "
            f"{position} | {getattr(row, f'rapm_lambda_{alternative_name}'):+.2f} | "
            f"{getattr(row, f'rapm_lambda_{source_name}'):+.2f} | "
            f"{row.rapm_difference:+.2f} | {getattr(row, f'rank_lambda_{source_name}')} | "
            f"{row.rank_difference:+d} | {row.rapm_possessions:,.0f} |"
        )
    page = Path(page_path)
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("\n".join(lines) + "\n")
    return page


def _lambda_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Student-t talent RAPM lambda sensitivity")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--alternative-lambda", type=float, default=0.10)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    args = parser.parse_args()
    run = analyze_student_t_talent_lambda_sensitivity(
        season=args.season,
        alternative_lambda=args.alternative_lambda,
        artifacts_dir=args.artifacts_dir,
        analytical_dir=args.analytical_dir,
        player_season_panel_path=args.player_season_panel_path,
    )
    print(f"Student-t talent lambda sensitivity: run={run.run_dir}")


if __name__ == "__main__":
    main()
