"""Recursive forward RAPM with a carried lineup-composition offset."""

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

from nba_lineup_model.modeling.contextual_features import lineup_context_features
from nba_lineup_model.modeling.contextual_prior import (
    _evaluate_target,
    _fit_model,
    _lineup_effects,
)
from nba_lineup_model.modeling.contextual_profiles import build_contextual_player_profiles
from nba_lineup_model.modeling.forward_exposure_gated_rapm import (
    _cold_start_priors,
    _combine_priors,
    _fit_replacement_token,
    _latest_run,
    _returning_priors,
)
from nba_lineup_model.modeling.frozen_prior_evaluation import _recover_home_intercept
from nba_lineup_model.modeling.prior_rapm import (
    HISTORICAL_SEASONS,
    PRIOR_MEAN_COLUMN,
    ForwardLaggedRapmSeason,
    fit_forward_lagged_rapm_season,
)
from nba_lineup_model.modeling.replacement_level import (
    player_exposure_shares,
    prepare_player_exposure_cohort,
)
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.season.schema import validate_season

DEFAULT_TARGET_SEASON = "2025-26"
DEFAULT_CONTEXT_ALPHA = 10_000.0
DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_RANKINGS_PAGE = Path("docs/models/forward-contextual-rapm.md")
MODEL_NAME = "forward_contextual_offset_rapm"
RUN_PREFIX = "forward-contextual-rapm"
RANKING_MODEL_NAME = "forward_contextual_player_rankings"
RANKING_RUN_PREFIX = "forward-contextual-rankings"
RANKING_SECTION_START = "<!-- forward-contextual-rankings:start -->"
RANKING_SECTION_END = "<!-- forward-contextual-rankings:end -->"


@dataclass(frozen=True)
class ForwardContextualRapmRun:
    run_dir: Path
    run_id: str


@dataclass(frozen=True)
class ForwardContextualRankingRun:
    run_dir: Path
    run_id: str
    next_season: str


def train_forward_contextual_rapm(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    context_fit: object | None = None,
    model_name: str = MODEL_NAME,
    run_prefix: str = RUN_PREFIX,
    artifact_name: str = "forward_contextual_rapm",
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardContextualRapmRun:
    """Roll player RAPM and contextual residual states forward one season at a time."""

    target = validate_season(through_season)
    if context_alpha <= 0:
        raise ValueError("Contextual alpha must be positive")
    panel = pd.read_parquet(player_season_panel_path)
    artifact_root = Path(artifacts_dir)
    reference_root = _latest_run(artifact_root / "forward_exposure_gated_rapm" / target)
    reference_coefficients = pd.read_parquet(
        reference_root / "historical_player_coefficients.parquet"
    )
    lambda_schedule = reference_coefficients.groupby("season", as_index=True)[
        "selected_lambda"
    ].agg(lambda values: float(values.iloc[0]))
    seasons = tuple(season for season in HISTORICAL_SEASONS if season <= target)
    if target not in seasons:
        seasons = (*seasons, target)
    source = _previous_season(target)
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(target)],
        through_season=target,
        analytical_dir=analytical_dir,
    )
    results: list[ForwardLaggedRapmSeason] = []
    priors_by_season: list[pd.DataFrame] = []
    exposure_history: list[pd.DataFrame] = []
    replacement_tokens: list[dict[str, object]] = []
    contextual_models: dict[str, object] = {}
    contextual_metadata: list[dict[str, object]] = []
    target_priors: pd.DataFrame | None = None
    target_profiles: pd.DataFrame | None = None

    for season in seasons:
        print(f"Fitting recursive contextual state for {season}", flush=True)
        raw_stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        participants = set().union(*raw_stints["home_player_ids"], *raw_stints["away_player_ids"])
        profiles = (
            build_contextual_player_profiles(
                panel,
                target_season=season,
                target_player_ids=participants,
                analytical_dir=str(analytical_dir),
                exposure_cohort=exposure_cohort,
            )
            if season != seasons[0]
            else None
        )
        previous_model = contextual_models.get(_previous_season(season))
        offset = (
            _context_offset(raw_stints, previous_model, profiles)
            if previous_model is not None and profiles is not None
            else np.zeros(len(raw_stints), dtype=float)
        )
        adjusted_stints = raw_stints.copy()
        adjusted_stints["target_home_net_rating"] = (
            raw_stints["target_home_net_rating"].to_numpy(dtype=float) - offset
        )
        cold, _ = _cold_start_priors(
            season=season,
            panel=panel,
            completed_results=results,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
        )
        priors = _combine_priors(_returning_priors(results), cold)
        prior_rows = priors.rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm"}).copy()
        prior_rows["season"] = season
        prior_rows["context_offset_source_season"] = (
            _previous_season(season) if previous_model is not None else pd.NA
        )
        priors_by_season.append(prior_rows)
        season_lambda = float(lambda_schedule.loc[season])
        if not np.isfinite(season_lambda) or season_lambda < 0:
            raise ValueError(f"Published RAPM lambda is invalid for {season}")
        fitted = fit_forward_lagged_rapm_season(
            season,
            adjusted_stints,
            priors,
            lambda_grid=(season_lambda,),
        )
        results.append(fitted)
        exposure = player_exposure_shares(raw_stints)
        exposure_history.append(
            panel.loc[panel["season"].eq(season)].merge(
                exposure, on="player_id", how="inner", validate="one_to_one"
            )
        )
        replacement_tokens.append(
            _fit_replacement_token(season, adjusted_stints, exposure, fitted, panel)
        )
        if profiles is not None:
            model, row = _fit_contextual_season(
                raw_stints, fitted, profiles, context_alpha, context_fit, previous_model
            )
            contextual_models[season] = model
            contextual_metadata.append(row)
        if season == target:
            target_priors = priors.rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm_mean"})
            target_profiles = profiles

    if target_priors is None or target_profiles is None:
        raise ValueError("Forward contextual RAPM did not create the target frozen state")
    forecast_model = contextual_models.get(source)
    if forecast_model is None:
        raise ValueError("Forward contextual RAPM has no source-season contextual state")
    historical_coefficients = pd.concat(
        [result.player_estimates for result in results], ignore_index=True
    )
    state_priors = pd.concat(priors_by_season, ignore_index=True)
    evaluation = _evaluate_target(
        target,
        model=forecast_model,
        profiles=target_profiles,
        priors=state_priors.rename(columns={"prior_rapm": "prior_rapm"}),
        coefficients=historical_coefficients,
        analytical_dir=Path(analytical_dir),
        curated_dir=Path(curated_dir),
        evaluation_model=model_name,
    )
    return _write_run(
        target=target,
        results=results,
        priors=state_priors,
        contextual_models=contextual_models,
        contextual_metadata=pd.DataFrame(contextual_metadata),
        target_priors=target_priors,
        target_profiles=target_profiles,
        evaluation=evaluation,
        context_alpha=context_alpha,
        model_name=model_name,
        run_prefix=run_prefix,
        artifact_name=artifact_name,
        artifacts_dir=artifact_root,
    )


def _context_offset(stints: pd.DataFrame, model: object, profiles: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        model.predict(
            lineup_context_features(
                stints["home_player_ids"].tolist(), stints["away_player_ids"].tolist(), profiles
            )
        ),
        dtype=float,
    )


def _fit_contextual_season(
    stints: pd.DataFrame,
    fitted: ForwardLaggedRapmSeason,
    profiles: pd.DataFrame,
    alpha: float,
    context_fit: object | None = None,
    previous_model: object | None = None,
) -> tuple[object, dict[str, object]]:
    coefficients = fitted.player_estimates.loc[:, ["player_id", "rapm"]]
    values = dict(zip(coefficients["player_id"].astype(int), coefficients["rapm"], strict=True))
    effects, unknown = _lineup_effects(stints, values)
    intercept = _recover_home_intercept(stints, coefficients)
    frame = lineup_context_features(
        stints["home_player_ids"].tolist(), stints["away_player_ids"].tolist(), profiles
    )
    frame["possessions"] = stints["possessions"].to_numpy(dtype=float)
    frame["target_residual_net_rating"] = (
        stints["target_home_net_rating"].to_numpy(dtype=float) - effects - intercept
    )
    model = (
        context_fit(frame, alpha, previous_model)  # type: ignore[operator]
        if context_fit is not None
        else _fit_model(frame, alpha)
    )
    return model, {
        "season": fitted.season,
        "context_alpha": alpha,
        "context_training_stint_count": len(frame),
        "context_unknown_player_exposures": int(unknown.sum()),
        "context_home_intercept": intercept,
    }


def _write_run(
    *,
    target: str,
    results: list[ForwardLaggedRapmSeason],
    priors: pd.DataFrame,
    contextual_models: dict[str, object],
    contextual_metadata: pd.DataFrame,
    target_priors: pd.DataFrame,
    target_profiles: pd.DataFrame,
    evaluation: dict[str, pd.DataFrame | dict[str, object]],
    context_alpha: float,
    model_name: str,
    run_prefix: str,
    artifact_name: str,
    artifacts_dir: Path,
) -> ForwardContextualRapmRun:
    now = datetime.now(UTC)
    run_id = f"{run_prefix}-{target}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / artifact_name / target
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        tables: dict[str, pd.DataFrame] = {
            "historical_player_coefficients.parquet": pd.concat(
                [result.player_estimates for result in results], ignore_index=True
            ),
            "season_player_priors.parquet": priors,
            "season_context_metadata.parquet": contextual_metadata,
            "frozen_2025_26_player_priors.parquet": target_priors,
            "target_player_profiles.parquet": target_profiles,
            "cohort_metrics.parquet": evaluation["cohort_metrics"],  # type: ignore[dict-item]
            "possession_predictions.parquet": evaluation["possession_predictions"],  # type: ignore[dict-item]
            "game_predictions.parquet": evaluation["game_predictions"],  # type: ignore[dict-item]
            "regular_game_predictions.parquet": evaluation["regular_game_predictions"],  # type: ignore[dict-item]
            "team_net_rating_predictions.parquet": evaluation["team_net_rating_predictions"],  # type: ignore[dict-item]
            "team_net_rating_metrics.parquet": evaluation["team_net_rating_metrics"],  # type: ignore[dict-item]
            "team_win_predictions.parquet": evaluation["team_win_predictions"],  # type: ignore[dict-item]
            "team_win_metrics.parquet": evaluation["team_win_metrics"],  # type: ignore[dict-item]
        }
        for filename, frame in tables.items():
            frame.to_parquet(temporary / filename, index=False)
        joblib.dump(contextual_models, temporary / "season_context_models.joblib")
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": model_name,
            "target_season": target,
            "context_alpha": context_alpha,
            "contextual_offset_contract": "g_(t-1) is subtracted before fitting RAPM in season t",
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
            "source_state": evaluation["source_state"],
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {"filename": path.name, "byte_count": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return ForwardContextualRapmRun(output, run_id)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_forward_contextual_rankings(
    *,
    source_run_dir: Path | str | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardContextualRankingRun:
    """Publish completed-state player priors for the following season.

    The player value is the additive component of the contextual model. The
    completed contextual function remains a separate lineup-level term and is
    not attributed to individual players in this table.
    """

    artifact_root = Path(artifacts_dir)
    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(artifact_root / "forward_contextual_rapm" / DEFAULT_TARGET_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Forward contextual rankings require a matching model artifact")
    through_season = validate_season(str(metadata["target_season"]))
    next_season = _next_season(through_season)
    coefficients = pd.read_parquet(source / "historical_player_coefficients.parquet")
    required_coefficients = {
        "season",
        "player_id",
        "rapm",
        "prior_rapm",
        "rapm_adjustment_from_prior",
    }
    if required_coefficients - set(coefficients):
        raise ValueError("Forward contextual artifact has an invalid coefficient schema")
    final = coefficients.loc[
        coefficients["season"].eq(through_season),
        ["player_id", "rapm", "prior_rapm", "rapm_adjustment_from_prior"],
    ]
    panel = pd.read_parquet(player_season_panel_path)
    required_panel = {"season", "player_id", "player_name", "listed_position", "rapm_possessions"}
    if required_panel - set(panel):
        raise ValueError("Player-season panel has an invalid ranking schema")
    season_panel = panel.loc[
        panel["season"].eq(through_season),
        ["player_id", "player_name", "listed_position", "rapm_possessions"],
    ]
    rankings = (
        final.merge(season_panel, on="player_id", how="left", validate="one_to_one")
        .sort_values(
            ["rapm", "rapm_possessions", "player_id"],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    rankings.insert(0, "rank", np.arange(1, len(rankings) + 1))
    return _write_ranking_run(
        source_run_dir=source,
        source_metadata=metadata,
        rankings=rankings,
        through_season=through_season,
        next_season=next_season,
        artifacts_dir=artifact_root,
    )


def render_forward_contextual_rankings_page(
    run_dir: Path | str,
    *,
    page_path: Path | str = DEFAULT_RANKINGS_PAGE,
) -> Path:
    """Render the current top-100 table inside the forward contextual model page."""

    root = Path(run_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("model") != RANKING_MODEL_NAME:
        raise ValueError("Forward contextual ranking page requires a ranking artifact")
    rankings = pd.read_parquet(root / "next_season_top_100_player_rankings.parquet")
    required = {
        "rank",
        "player_name",
        "listed_position",
        "rapm",
        "prior_rapm",
        "rapm_adjustment_from_prior",
        "rapm_possessions",
    }
    if required - set(rankings):
        raise ValueError("Forward contextual ranking artifact has an invalid schema")
    next_season = str(metadata["next_season"])
    through_season = str(metadata["through_season"])
    lines = [
        RANKING_SECTION_START,
        f"## {next_season} Player Rankings",
        "",
        f"These are the top 100 player priors carried from the completed {through_season} "
        "forward contextual RAPM state. They are predictions for the next regular "
        "season, not retrospective rankings.",
        "",
        "The one-number value is the model's additive player component. The completed "
        f"lineup-context function `g_{through_season}` remains a separate lineup-level "
        "term, so it is not assigned to individual players. The table covers players "
        f"who appeared in {through_season}; it does not yet add the incoming rookie class "
        "or account for offseason roster moves.",
        "",
        "The table is sortable by every column. `Adjustment` is the completed-season "
        "movement from the preseason prior that entered the fit; interpret limited "
        "exposure alongside possession count.",
        "",
        f"Immutable ranking artifact: `{root}`.",
        "",
        "| Rank | Player | Pos. | 2026-27 contextual RAPM prior | "
        "2025-26 preseason prior | Adjustment | 2025-26 possessions |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rankings.itertuples(index=False):
        position = "" if pd.isna(row.listed_position) else str(row.listed_position)
        lines.append(
            f"| {int(row.rank)} | {row.player_name} | {position} | {float(row.rapm):+.2f} | "
            f"{float(row.prior_rapm):+.2f} | {float(row.rapm_adjustment_from_prior):+.2f} | "
            f"{float(row.rapm_possessions):,.0f} |"
        )
    lines.extend([RANKING_SECTION_END, ""])
    page = Path(page_path)
    text = page.read_text()
    if RANKING_SECTION_START not in text or RANKING_SECTION_END not in text:
        raise ValueError("Forward contextual model page is missing ranking section markers")
    before, remainder = text.split(RANKING_SECTION_START, maxsplit=1)
    _, after = remainder.split(RANKING_SECTION_END, maxsplit=1)
    page.write_text(before + "\n".join(lines) + after)
    return page


def _write_ranking_run(
    *,
    source_run_dir: Path,
    source_metadata: dict[str, object],
    rankings: pd.DataFrame,
    through_season: str,
    next_season: str,
    artifacts_dir: Path,
) -> ForwardContextualRankingRun:
    now = datetime.now(UTC)
    run_id = f"{RANKING_RUN_PREFIX}-{next_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "forward_contextual_rankings" / next_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        rankings.to_parquet(temporary / "next_season_player_rankings.parquet", index=False)
        rankings.head(100).to_parquet(
            temporary / "next_season_top_100_player_rankings.parquet", index=False
        )
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": RANKING_MODEL_NAME,
            "source_model": MODEL_NAME,
            "source_run_id": source_metadata["run_id"],
            "source_run_dir": str(source_run_dir),
            "through_season": through_season,
            "next_season": next_season,
            "ranking_contract": (
                "completed additive player state; g_t remains a separate lineup-level term"
            ),
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {"filename": path.name, "byte_count": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return ForwardContextualRankingRun(output, run_id, next_season)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _next_season(season: str) -> str:
    start = int(season[:4]) + 1
    return f"{start}-{str(start + 1)[-2:]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recursive contextual-offset RAPM")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    args = parser.parse_args()
    run = train_forward_contextual_rapm(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
    )
    print(f"Forward contextual RAPM: run={run.run_dir}")


def rankings_main() -> None:
    parser = argparse.ArgumentParser(description="Build forward contextual RAPM player rankings")
    parser.add_argument("--source-run-dir")
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--page-path", default=str(DEFAULT_RANKINGS_PAGE))
    args = parser.parse_args()
    run = build_forward_contextual_rankings(
        source_run_dir=args.source_run_dir,
        player_season_panel_path=args.player_season_panel_path,
        artifacts_dir=args.artifacts_dir,
    )
    page = render_forward_contextual_rankings_page(run.run_dir, page_path=args.page_path)
    print(f"Forward contextual rankings: run={run.run_dir} page={page}")


if __name__ == "__main__":
    main()
