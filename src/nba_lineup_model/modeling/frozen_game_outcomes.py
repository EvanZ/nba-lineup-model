"""Aggregate full regular-season game outcomes from frozen model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.schema import ArtifactRecord
from nba_lineup_model.season.schema import validate_season

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_DOCS_PATH = Path("docs/models/preseason-leaderboard.md")
DEFAULT_SEASON = "2025-26"
REPORT_NAME = "frozen_game_outcomes"
SECTION_START = "<!-- frozen-full-game-outcomes:start -->"
SECTION_END = "<!-- frozen-full-game-outcomes:end -->"


@dataclass(frozen=True)
class FrozenOutcomeModel:
    """One promoted frozen candidate and its public-facing label."""

    model: str
    model_id: str
    display_name: str


PROMOTED_MODELS: tuple[FrozenOutcomeModel, ...] = (
    FrozenOutcomeModel(
        "frozen_one_year_rapm_no_priors",
        "frozen-one-year-no-prior-rapm",
        "Frozen 1-year no-prior RAPM",
    ),
    FrozenOutcomeModel(
        "frozen_three_year_rapm_no_priors",
        "frozen-three-year-no-prior-rapm",
        "Frozen pooled 3-year no-prior RAPM",
    ),
    FrozenOutcomeModel(
        "frozen_regular_only_lagged_rapm",
        "frozen-lagged-rapm",
        "Frozen lagged RAPM",
    ),
    FrozenOutcomeModel("frozen_aging_prior", "frozen-aging-prior", "Frozen aging prior"),
    FrozenOutcomeModel(
        "frozen_offense_defense_rapm",
        "frozen-od-rapm",
        "Frozen O/D RAPM",
    ),
    FrozenOutcomeModel(
        "frozen_draft_cold_start_prior",
        "frozen-draft-cold-start",
        "Frozen draft cold-start prior",
    ),
    FrozenOutcomeModel(
        "frozen_exposure_gated_cold_start_prior",
        "frozen-exposure-gated-cold-start",
        "Frozen exposure-gated cold-start prior",
    ),
    FrozenOutcomeModel(
        "frozen_exposure_gated_offense_defense_cold_start_prior",
        "frozen-exposure-gated-od",
        "Frozen exposure-gated O/D cold-start prior",
    ),
    FrozenOutcomeModel(
        "forward_exposure_gated_prior_centered_ridge_rapm",
        "recursive-exposure-gated-rapm",
        "Recursive exposure-gated RAPM",
    ),
    FrozenOutcomeModel(
        "student_t_forward_exposure_gated_rapm",
        "student-t-recursive-rapm",
        "Student-t recursive RAPM",
    ),
    FrozenOutcomeModel(
        "student_t_talent_forward_exposure_gated_rapm",
        "student-t-talent-prior",
        "Student-t talent-prior RAPM",
    ),
    FrozenOutcomeModel(
        "forward_contextual_offset_rapm",
        "forward-contextual-rapm",
        "Forward contextual RAPM",
    ),
    FrozenOutcomeModel(
        "forward_decomposed_contextual_rapm",
        "forward-decomposed-contextual-rapm",
        "Forward decomposed contextual RAPM",
    ),
    FrozenOutcomeModel(
        "forward_portable_matchup_contextual_rapm",
        "forward-portable-matchup-contextual-rapm",
        "Forward portable-matchup contextual RAPM",
    ),
    FrozenOutcomeModel(
        "forward_hierarchical_pspline_contextual_rapm",
        "forward-hierarchical-pspline-contextual-rapm",
        "Forward hierarchical P-spline contextual RAPM",
    ),
    FrozenOutcomeModel(
        "forward_bounded_hierarchical_pspline_contextual_rapm",
        "forward-bounded-hierarchical-pspline-contextual-rapm",
        "Forward bounded hierarchical P-spline contextual RAPM",
    ),
    FrozenOutcomeModel(
        "forward_bounded_hierarchical_portable_matchup_contextual_rapm",
        "forward-bounded-hierarchical-portable-matchup-contextual-rapm",
        "Forward bounded hierarchical portable-matchup contextual RAPM",
    ),
    FrozenOutcomeModel(
        "forward_aging_bounded_hierarchical_portable_matchup_contextual_rapm",
        "forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm",
        "Forward aging bounded hierarchical portable-matchup contextual RAPM",
    ),
    FrozenOutcomeModel(
        "forward_centered_aging_bounded_hierarchical_portable_matchup_contextual_rapm",
        "forward-centered-aging-bounded-hierarchical-portable-matchup-contextual-rapm",
        "Forward centered aging bounded hierarchical portable-matchup contextual RAPM",
    ),
    FrozenOutcomeModel(
        "forward_centered_value_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm",
        "forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm",
        "Forward centered value-conditioned aging bounded hierarchical "
        "portable-matchup contextual RAPM",
    ),
    FrozenOutcomeModel(
        "forward_centered_era_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm",
        "forward-centered-era-conditioned-aging-bounded-hierarchical-"
        "portable-matchup-contextual-rapm",
        "Forward centered era-conditioned aging bounded hierarchical "
        "portable-matchup contextual RAPM",
    ),
    FrozenOutcomeModel(
        "student_t_talent_forward_contextual_rapm",
        "student-t-talent-contextual-rapm",
        "Student-t talent-prior contextual RAPM",
    ),
)


@dataclass(frozen=True)
class OutcomeSource:
    """Validated immutable source used by an outcome-report run."""

    candidate: FrozenOutcomeModel
    run_dir: Path
    run_id: str
    created_at: datetime
    manifest_sha256: str


def score_full_game_outcomes(game_predictions: pd.DataFrame) -> dict[str, float | int]:
    """Score full-game margins and deterministic winner calls in the home frame.

    A forecast with exactly zero home margin receives half credit for winner
    accuracy. NBA games cannot end tied, but treating a tied forecast as an
    away prediction would make accuracy depend on an arbitrary implementation
    detail.
    """

    required = {
        "game_id",
        "actual_home_margin",
        "predicted_home_margin",
        "actual_home_win",
    }
    missing = required - set(game_predictions)
    if missing:
        raise ValueError(f"Game predictions missing required columns: {sorted(missing)}")
    if game_predictions["game_id"].isna().any() or game_predictions["game_id"].duplicated().any():
        raise ValueError("Game predictions must have one non-null row per game")

    actual_margin = game_predictions["actual_home_margin"].to_numpy(dtype=float)
    predicted_margin = game_predictions["predicted_home_margin"].to_numpy(dtype=float)
    if not np.isfinite(actual_margin).all() or not np.isfinite(predicted_margin).all():
        raise ValueError("Game margins must be finite")
    if np.isclose(actual_margin, 0.0).any():
        raise ValueError("Actual NBA game margins must not be tied")

    actual_home_win = game_predictions["actual_home_win"].to_numpy(dtype=bool)
    if not np.array_equal(actual_home_win, actual_margin > 0.0):
        raise ValueError("actual_home_win must agree with actual_home_margin")
    predicted_tie = np.isclose(predicted_margin, 0.0)
    predicted_home_win = predicted_margin > 0.0
    winner_credit = np.where(predicted_tie, 0.5, predicted_home_win == actual_home_win)
    residual = actual_margin - predicted_margin

    return {
        "regular_game_count": int(len(game_predictions)),
        "full_game_margin_rmse": float(np.sqrt(np.mean(residual**2))),
        "full_game_margin_mae": float(np.mean(np.abs(residual))),
        "game_winner_accuracy": float(np.mean(winner_credit)),
        "predicted_tie_count": int(predicted_tie.sum()),
    }


def discover_promoted_outcome_sources(
    *,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    season: str = DEFAULT_SEASON,
) -> tuple[OutcomeSource, ...]:
    """Find the latest valid full-game artifact for every promoted candidate."""

    target_season = validate_season(season)
    root = Path(artifacts_dir)
    candidates = {candidate.model: candidate for candidate in PROMOTED_MODELS}
    latest: dict[str, OutcomeSource] = {}
    for manifest_path in root.glob(f"**/{target_season}/**/manifest.json"):
        game_path = manifest_path.parent / "regular_game_predictions.parquet"
        if not game_path.is_file():
            continue
        metadata = json.loads(manifest_path.read_text())
        model = metadata.get("model")
        candidate = candidates.get(model)
        if candidate is None:
            continue
        created_at = _parse_timestamp(metadata.get("created_at"), manifest_path)
        source = OutcomeSource(
            candidate=candidate,
            run_dir=manifest_path.parent,
            run_id=str(metadata.get("run_id", manifest_path.parent.name)),
            created_at=created_at,
            manifest_sha256=_sha256_file(manifest_path),
        )
        previous = latest.get(candidate.model)
        if previous is None or source.created_at > previous.created_at:
            latest[candidate.model] = source
    missing = [
        candidate.display_name for candidate in PROMOTED_MODELS if candidate.model not in latest
    ]
    if missing:
        raise ValueError(f"Missing frozen full-game artifacts for: {', '.join(missing)}")
    return tuple(latest[candidate.model] for candidate in PROMOTED_MODELS)


def build_frozen_game_outcome_report(
    *,
    season: str = DEFAULT_SEASON,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    docs_path: Path | str | None = DEFAULT_DOCS_PATH,
) -> tuple[pd.DataFrame, Path]:
    """Publish common full-game outcome metrics for every frozen candidate."""

    target_season = validate_season(season)
    root = Path(artifacts_dir)
    sources = discover_promoted_outcome_sources(artifacts_dir=root, season=target_season)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    reference_games: pd.Series | None = None
    reference_actual: pd.Series | None = None

    for source in sources:
        predictions = pd.read_parquet(source.run_dir / "regular_game_predictions.parquet")
        metrics = score_full_game_outcomes(predictions)
        games = predictions["game_id"].astype(str).sort_values(kind="stable").reset_index(drop=True)
        actual = (
            predictions.assign(game_id=predictions["game_id"].astype(str))
            .sort_values("game_id", kind="stable")["actual_home_margin"]
            .reset_index(drop=True)
        )
        if reference_games is None:
            reference_games, reference_actual = games, actual
        elif not games.equals(reference_games) or not np.array_equal(
            actual.to_numpy(dtype=float), reference_actual.to_numpy(dtype=float)
        ):
            raise ValueError(
                f"{source.candidate.display_name} does not use the common regular-season schedule"
            )
        metric_rows.append(
            {
                "season": target_season,
                "model_id": source.candidate.model_id,
                "model": source.candidate.display_name,
                "source_model": source.candidate.model,
                "source_run_id": source.run_id,
                **metrics,
            }
        )
        detailed = predictions.copy()
        detailed.insert(0, "model_id", source.candidate.model_id)
        detailed.insert(1, "model", source.candidate.display_name)
        detailed.insert(2, "source_run_id", source.run_id)
        prediction_frames.append(detailed)

    outcomes = pd.DataFrame(metric_rows)
    _validate_common_outcome_metrics(outcomes)
    run_dir = _write_report(
        outcomes=outcomes,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        sources=sources,
        artifacts_dir=root,
        season=target_season,
    )
    if docs_path is not None:
        _update_docs(Path(docs_path), outcomes, run_dir)
    return outcomes, run_dir


def _validate_common_outcome_metrics(outcomes: pd.DataFrame) -> None:
    if outcomes["model_id"].duplicated().any():
        raise ValueError("Outcome report has duplicate model IDs")
    if outcomes["regular_game_count"].nunique() != 1:
        raise ValueError("Outcome report candidates cover different game counts")
    if outcomes["regular_game_count"].iat[0] <= 0:
        raise ValueError("Outcome report must cover at least one game")
    if not outcomes["game_winner_accuracy"].between(0.0, 1.0).all():
        raise ValueError("Game-winner accuracy must be between zero and one")


def _write_report(
    *,
    outcomes: pd.DataFrame,
    predictions: pd.DataFrame,
    sources: tuple[OutcomeSource, ...],
    artifacts_dir: Path,
    season: str,
) -> Path:
    now = datetime.now(UTC)
    run_id = f"{REPORT_NAME}-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    season_dir = artifacts_dir / REPORT_NAME / season
    run_dir = season_dir / run_id
    temporary = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        outcomes.to_parquet(temporary / "game_outcome_metrics.parquet", index=False)
        predictions.to_parquet(temporary / "game_outcome_predictions.parquet", index=False)
        source_rows = pd.DataFrame(
            [
                {
                    "model_id": source.candidate.model_id,
                    "model": source.candidate.display_name,
                    "source_model": source.candidate.model,
                    "source_run_id": source.run_id,
                    "source_run_dir": str(source.run_dir),
                    "source_created_at": source.created_at.isoformat(),
                    "source_manifest_sha256": source.manifest_sha256,
                }
                for source in sources
            ]
        )
        source_rows.to_parquet(temporary / "sources.parquet", index=False)
        metadata = {
            "run_id": run_id,
            "created_at": now.isoformat(),
            "season": season,
            "information_boundary": "frozen source artifacts; no model refit",
            "cohort": "regular_season_full_games",
            "winner_tie_policy": "exactly zero predicted margins receive half credit",
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        artifacts = [
            ArtifactRecord(
                filename=path.name,
                row_count=(
                    len(outcomes)
                    if path.name == "game_outcome_metrics.parquet"
                    else len(predictions)
                    if path.name == "game_outcome_predictions.parquet"
                    else len(source_rows)
                    if path.name == "sources.parquet"
                    else None
                ),
                byte_count=path.stat().st_size,
                sha256=_sha256_file(path),
            ).model_dump()
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        manifest = {**metadata, "artifacts": artifacts}
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(run_dir)
        latest = season_dir / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return run_dir
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _update_docs(path: Path, outcomes: pd.DataFrame, run_dir: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Frozen leaderboard documentation page does not exist: {path}")
    start_index = path.read_text().find(SECTION_START)
    end_index = path.read_text().find(SECTION_END)
    if start_index < 0 or end_index < 0 or start_index >= end_index:
        raise ValueError("Frozen leaderboard is missing full-game outcome section markers")
    best_margin = outcomes["full_game_margin_rmse"].min()
    best_mae = outcomes["full_game_margin_mae"].min()
    best_accuracy = outcomes["game_winner_accuracy"].max()
    lines = [
        SECTION_START,
        (
            "Full-game outcomes aggregate every allocated regular-season stint to the official "
            "final home margin. This is deliberately distinct from the eligible-possession game\n"
            "metric above, which excludes possession rows without one reconstructed lineup. "
            "Winner accuracy calls the sign of that full-game margin; an exactly zero forecast\n"
            "receives half credit rather than being arbitrarily assigned to the away team."
        ),
        "",
        (
            "| Model | Games | Full-game margin RMSE | Full-game margin MAE | "
            "Winner accuracy | Predicted ties |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in outcomes.itertuples(index=False):
        margin = _bold_if_best(row.full_game_margin_rmse, best_margin, 4)
        mae = _bold_if_best(row.full_game_margin_mae, best_mae, 4)
        accuracy = _bold_if_best(row.game_winner_accuracy, best_accuracy, 2, percent=True)
        lines.append(
            f"| {_markdown_model_link(row.model_id, row.model)} | {row.regular_game_count:,} | "
            f"{margin} | {mae} | {accuracy} | {row.predicted_tie_count:,} |"
        )
    lines.extend(
        [
            "",
            (
                f"Source report: `{run_dir}`. The retained `game_outcome_predictions.parquet`\n"
                "contains one final-margin prediction per model and game, and `sources.parquet`\n"
                "pins every upstream immutable manifest."
            ),
            SECTION_END,
        ]
    )
    replacement = "\n".join(lines)
    original = path.read_text()
    updated = original[:start_index] + replacement + original[end_index + len(SECTION_END) :]
    path.write_text(updated)


def _bold_if_best(value: float, best: float, decimals: int, *, percent: bool = False) -> str:
    rendered = f"{value * 100.0:.{decimals}f}%" if percent else f"{value:.{decimals}f}"
    return f"**{rendered}**" if np.isclose(value, best, rtol=0.0, atol=1e-12) else rendered


def _markdown_model_link(model_id: str, display_name: str) -> str:
    """Return the shared Markdown reference link used by the leaderboard."""

    return f"[{display_name}][{model_id}]"


def _parse_timestamp(value: object, path: Path) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Frozen artifact manifest has no created_at timestamp: {path}")
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError(f"Frozen artifact timestamp must be timezone-aware: {path}")
    return timestamp.astimezone(UTC)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate full regular-season outcomes for frozen RAPM candidates."
    )
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--docs-path", default=str(DEFAULT_DOCS_PATH))
    parser.add_argument("--no-docs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outcomes, run_dir = build_frozen_game_outcome_report(
        season=args.season,
        artifacts_dir=args.artifacts_dir,
        docs_path=None if args.no_docs else args.docs_path,
    )
    print(outcomes.to_string(index=False))
    print(f"Wrote full-game outcome report: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
