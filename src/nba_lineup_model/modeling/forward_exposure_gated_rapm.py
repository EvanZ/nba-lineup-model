"""Strictly forward one-number RAPM with recursive exposure-gated cold starts."""

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

from nba_lineup_model.modeling.cold_start_exposure import (
    FEATURE_COLUMNS as EXPOSURE_FEATURE_COLUMNS,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    _feature_base as exposure_feature_base,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    _feature_reference as exposure_feature_reference,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    _prepare_features as prepare_exposure_features,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    fit_exposure_model,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    select_regularization as select_exposure_regularization,
)
from nba_lineup_model.modeling.draft_prior import (
    FEATURE_COLUMNS as DRAFT_FEATURE_COLUMNS,
)
from nba_lineup_model.modeling.draft_prior import (
    _feature_base as draft_feature_base,
)
from nba_lineup_model.modeling.draft_prior import (
    _feature_reference as draft_feature_reference,
)
from nba_lineup_model.modeling.draft_prior import (
    _prepare_features as prepare_draft_features,
)
from nba_lineup_model.modeling.draft_prior import (
    fit_draft_prior_model,
)
from nba_lineup_model.modeling.draft_prior import (
    select_regularization as select_draft_regularization,
)
from nba_lineup_model.modeling.frozen_prior_evaluation import run_frozen_lagged_evaluation
from nba_lineup_model.modeling.prior_rapm import (
    HISTORICAL_SEASONS,
    PRIOR_MEAN_COLUMN,
    ForwardLaggedRapmSeason,
    fit_forward_lagged_rapm_season,
)
from nba_lineup_model.modeling.replacement_level import player_exposure_shares
from nba_lineup_model.modeling.replacement_token import fit_replacement_token_season
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.season.schema import validate_season

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_REPLACEMENT_SHARE_CUTOFF = 0.05
MODEL_NAME = "forward_exposure_gated_prior_centered_ridge_rapm"
RUN_PREFIX = "forward-exposure-gated-rapm"


@dataclass(frozen=True)
class ForwardExposureGatedRapmRun:
    """Immutable forward model artifact and the next-season rating state."""

    run_dir: Path
    run_id: str
    through_season: str
    next_season: str


def train_forward_exposure_gated_rapm(
    *,
    through_season: str = "2025-26",
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    checkpoint_path: Path | str | None = None,
    max_seasons: int | None = None,
) -> ForwardExposureGatedRapmRun | None:
    """Train a no-future-information regular-season RAPM state through a season."""

    last_season = validate_season(through_season)
    panel = pd.read_parquet(player_season_panel_path)
    _validate_panel(panel, last_season)
    seasons = tuple(season for season in HISTORICAL_SEASONS if season <= last_season)
    if last_season not in seasons:
        seasons = (*seasons, last_season)
    checkpoint = Path(checkpoint_path) if checkpoint_path else (
        Path(artifacts_dir) / "forward_exposure_gated_rapm" / last_season / ".checkpoint.joblib"
    )
    state = _load_checkpoint(checkpoint, seasons)
    results: list[ForwardLaggedRapmSeason] = state["results"]
    season_priors: list[pd.DataFrame] = state["season_priors"]
    replacement_tokens: list[dict[str, object]] = state["replacement_tokens"]
    exposure_history: list[pd.DataFrame] = state["exposure_history"]
    cold_start_metadata: list[dict[str, object]] = state["cold_start_metadata"]
    frozen_2025_priors: pd.DataFrame | None = state["frozen_2025_priors"]
    starting_season_count = len(results)

    for season in seasons[len(results) :]:
        print(f"Fitting {season}", flush=True)
        stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        cold_priors, cold_metadata = _cold_start_priors(
            season=season,
            panel=panel,
            completed_results=results,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
        )
        returning = _returning_priors(results)
        priors = _combine_priors(returning, cold_priors)
        if season == "2025-26":
            frozen_2025_priors = priors.copy()
        prior_rows = priors.copy()
        prior_rows["season"] = season
        prior_rows["prior_branch"] = np.where(
            prior_rows["player_id"].isin(set(cold_priors["player_id"])),
            "exposure_gated_cold_start",
            "lagged_rapm",
        )
        season_priors.append(prior_rows)
        cold_start_metadata.append(cold_metadata)
        result = fit_forward_lagged_rapm_season(season, stints, priors)
        results.append(result)

        exposure = player_exposure_shares(stints)
        exposure_history.append(
            panel.loc[panel["season"].eq(season)].merge(
                exposure, on="player_id", how="inner", validate="one_to_one"
            )
        )
        replacement_tokens.append(_fit_replacement_token(season, stints, exposure, result, panel))
        _write_checkpoint(
            checkpoint,
            seasons=seasons,
            results=results,
            season_priors=season_priors,
            replacement_tokens=replacement_tokens,
            exposure_history=exposure_history,
            cold_start_metadata=cold_start_metadata,
            frozen_2025_priors=frozen_2025_priors,
        )
        if max_seasons is not None and len(results) >= max_seasons + starting_season_count:
            return None

    if frozen_2025_priors is None:
        raise ValueError("The current gold-standard run requires 2025-26 priors")
    frozen_evaluation = _evaluate_2025_priors(
        frozen_2025_priors,
        artifacts_dir=Path(artifacts_dir),
        analytical_dir=Path(analytical_dir),
        curated_dir=Path(curated_dir),
    )
    next_season = _next_season(last_season)
    run = _write_run(
        through_season=last_season,
        next_season=next_season,
        results=results,
        season_priors=pd.concat(season_priors, ignore_index=True),
        replacement_tokens=pd.DataFrame(replacement_tokens),
        cold_start_metadata=pd.DataFrame(cold_start_metadata),
        frozen_2025_priors=frozen_2025_priors,
        frozen_evaluation=frozen_evaluation,
        panel=panel,
        artifacts_dir=Path(artifacts_dir),
    )
    checkpoint.unlink(missing_ok=True)
    return run


def _cold_start_priors(
    *,
    season: str,
    panel: pd.DataFrame,
    completed_results: list[ForwardLaggedRapmSeason],
    exposure_history: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build current rookies' prior using only completed earlier seasons."""

    empty = pd.DataFrame(columns=["player_id", PRIOR_MEAN_COLUMN])
    target_rookies = panel.loc[panel["season"].eq(season) & panel["is_rookie"].astype(bool)]
    metadata: dict[str, object] = {
        "season": season,
        "target_rookie_count": int(len(target_rookies)),
        "training_last_season": _previous_season(season),
        "cold_start_enabled": False,
        "reason": "insufficient completed rookie history",
    }
    if target_rookies.empty or len(completed_results) < 8 or not replacement_tokens:
        return empty, metadata

    estimates = pd.concat(
        [
            result.player_estimates.loc[:, ["season", "player_id", "rapm"]]
            for result in completed_results
        ],
        ignore_index=True,
    ).rename(columns={"rapm": "forward_rapm"})
    historical_rookies = panel.loc[
        panel["season"].isin([result.season for result in completed_results])
        & panel["is_rookie"].astype(bool)
    ].drop(columns=["rapm"], errors="ignore").merge(
        estimates, on=["season", "player_id"], how="inner", validate="one_to_one"
    ).rename(columns={"forward_rapm": "rapm"})
    historical_exposure = pd.concat(exposure_history, ignore_index=True)
    historical_exposure = historical_exposure.loc[historical_exposure["is_rookie"].astype(bool)]
    if historical_rookies["season"].nunique() < 8 or historical_exposure["season"].nunique() < 8:
        return empty, metadata
    try:
        rate_training = prepare_draft_features(historical_rookies)
        rate_reference = draft_feature_reference(draft_feature_base(historical_rookies))
        target_rate = prepare_draft_features(target_rookies, reference=rate_reference)
        draft_regularization, _ = select_draft_regularization(
            rate_training, regularization_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
        )
        draft_model = fit_draft_prior_model(rate_training, regularization=draft_regularization)

        exposure_training = prepare_exposure_features(historical_exposure)
        exposure_training["is_replacement_candidate"] = exposure_training[
            "exposure_share"
        ].lt(DEFAULT_REPLACEMENT_SHARE_CUTOFF)
        exposure_reference = exposure_feature_reference(exposure_feature_base(historical_exposure))
        target_exposure = prepare_exposure_features(target_rookies, reference=exposure_reference)
        exposure_c, _ = select_exposure_regularization(
            exposure_training, c_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
        )
        exposure_model = fit_exposure_model(exposure_training, c=exposure_c)
    except ValueError as error:
        metadata["reason"] = f"deferred: {error}"
        return empty, metadata

    replacement_rapm = float(pd.DataFrame(replacement_tokens)["replacement_token_rapm"].mean())
    probability = exposure_model.predict_proba(
        target_exposure.loc[:, EXPOSURE_FEATURE_COLUMNS]
    )[:, 1]
    draft_rate = draft_model.predict(target_rate.loc[:, DRAFT_FEATURE_COLUMNS])
    output = target_rookies.loc[:, ["player_id"]].copy()
    output[PRIOR_MEAN_COLUMN] = probability * replacement_rapm + (1.0 - probability) * draft_rate
    metadata.update(
        {
            "cold_start_enabled": True,
            "reason": "exposure-gated draft/replacement prior",
            "draft_regularization": draft_regularization,
            "exposure_c": exposure_c,
            "replacement_rapm": replacement_rapm,
            "historical_rookie_count": int(len(historical_rookies)),
        }
    )
    return output, metadata


def _returning_priors(results: list[ForwardLaggedRapmSeason]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["player_id", PRIOR_MEAN_COLUMN])
    return results[-1].player_estimates.loc[:, ["player_id", "rapm"]].rename(
        columns={"rapm": PRIOR_MEAN_COLUMN}
    )


def _combine_priors(returning: pd.DataFrame, cold: pd.DataFrame) -> pd.DataFrame:
    if set(returning["player_id"]) & set(cold["player_id"]):
        raise ValueError("Cold-start players overlap the completed prior-season state")
    frames = [frame for frame in (returning, cold) if not frame.empty]
    output = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["player_id", PRIOR_MEAN_COLUMN])
    )
    output["player_id"] = output["player_id"].astype(int)
    output[PRIOR_MEAN_COLUMN] = pd.to_numeric(
        output[PRIOR_MEAN_COLUMN], errors="raise"
    ).astype(float)
    if output["player_id"].duplicated().any() or not np.isfinite(
        output[PRIOR_MEAN_COLUMN]
    ).all():
        raise ValueError("Combined forward prior state is invalid")
    return output.sort_values("player_id", kind="stable").reset_index(drop=True)


def _fit_replacement_token(
    season: str,
    stints: pd.DataFrame,
    exposure: pd.DataFrame,
    result: ForwardLaggedRapmSeason,
    panel: pd.DataFrame,
) -> dict[str, object]:
    ids = set(panel.loc[panel["season"].eq(season), "player_id"].astype(int))
    replacement_ids = set(
        exposure.loc[
            exposure["exposure_share"].lt(DEFAULT_REPLACEMENT_SHARE_CUTOFF)
            & exposure["player_id"].astype(int).isin(ids),
            "player_id",
        ].astype(int)
    )
    coefficient, _ = fit_replacement_token_season(
        stints,
        replacement_player_ids=replacement_ids,
        regularization=result.selected_lambda,
    )
    return {
        "season": season,
        "selected_lambda": result.selected_lambda,
        "replacement_player_count": len(replacement_ids),
        "replacement_token_rapm": coefficient,
    }


def _evaluate_2025_priors(
    priors: pd.DataFrame,
    *,
    artifacts_dir: Path,
    analytical_dir: Path,
    curated_dir: Path,
) -> object:
    source_root = _latest_run(artifacts_dir / "prior_rapm" / "2025-26")
    return run_frozen_lagged_evaluation(
        season="2025-26",
        prior_run_dir=source_root,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        player_priors_override=priors.rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm_mean"}),
        source_state_overrides={
            "player_prior_method": "strictly forward exposure-gated RAPM cold-start state",
            "cold_start_prior": "exposure-gated draft and replacement prior",
        },
        evaluation_model="frozen_forward_exposure_gated_rapm",
    )


def _write_run(
    *,
    through_season: str,
    next_season: str,
    results: list[ForwardLaggedRapmSeason],
    season_priors: pd.DataFrame,
    replacement_tokens: pd.DataFrame,
    cold_start_metadata: pd.DataFrame,
    frozen_2025_priors: pd.DataFrame,
    frozen_evaluation: object,
    panel: pd.DataFrame,
    artifacts_dir: Path,
) -> ForwardExposureGatedRapmRun:
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}-{through_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "forward_exposure_gated_rapm" / through_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        final = results[-1].player_estimates.loc[
            :, ["player_id", "rapm", "prior_rapm", "rapm_adjustment_from_prior"]
        ]
        season_panel = panel.loc[
            panel["season"].eq(through_season),
            ["player_id", "player_name", "listed_position", "rapm_possessions"],
        ]
        rankings = final.merge(
            season_panel, on="player_id", how="left", validate="one_to_one"
        ).sort_values(
            ["rapm", "rapm_possessions", "player_id"],
            ascending=[False, False, True],
            kind="stable",
        )
        rankings["rank"] = np.arange(1, len(rankings) + 1)
        tables = {
            "historical_player_coefficients.parquet": pd.concat(
                [result.player_estimates for result in results], ignore_index=True
            ),
            "historical_cv_results.parquet": pd.concat(
                [result.cv_results.assign(season=result.season) for result in results],
                ignore_index=True,
            ),
            "season_player_priors.parquet": season_priors,
            "season_replacement_tokens.parquet": replacement_tokens,
            "season_cold_start_metadata.parquet": cold_start_metadata,
            "frozen_2025_26_player_priors.parquet": frozen_2025_priors,
            "next_season_returning_rankings.parquet": rankings,
            "next_season_top_100_returning_rankings.parquet": rankings.head(100),
            "cohort_metrics.parquet": frozen_evaluation.cohort_metrics,
            "possession_predictions.parquet": frozen_evaluation.possession_predictions,
            "game_predictions.parquet": frozen_evaluation.game_predictions,
            "regular_game_predictions.parquet": frozen_evaluation.regular_game_predictions,
            "team_net_rating_predictions.parquet": frozen_evaluation.team_net_rating_predictions,
            "team_net_rating_metrics.parquet": frozen_evaluation.team_net_rating_metrics,
            "team_win_predictions.parquet": frozen_evaluation.team_win_predictions,
            "team_win_metrics.parquet": frozen_evaluation.team_win_metrics,
        }
        for filename, frame in tables.items():
            frame.to_parquet(temporary / filename, index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "through_season": through_season,
            "next_season": next_season,
            "historical_training_variant": "regular_only",
            "information_boundary": (
                "all cold-start components use only earlier completed regular seasons"
            ),
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {"filename": path.name, "byte_count": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(temporary.iterdir()) if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        latest_tmp = root / "latest.json.tmp"
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(root / "latest.json")
        return ForwardExposureGatedRapmRun(output, run_id, through_season, next_season)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validate_panel(panel: pd.DataFrame, through_season: str) -> None:
    required = {
        "season",
        "player_id",
        "is_rookie",
        "rapm_possessions",
        "player_name",
        "listed_position",
    }
    if required - set(panel):
        raise ValueError(f"Player panel missing required columns: {sorted(required - set(panel))}")
    if not panel["season"].eq(through_season).any():
        raise ValueError(f"Player panel does not include {through_season}")


def _load_checkpoint(checkpoint: Path, seasons: tuple[str, ...]) -> dict[str, object]:
    if not checkpoint.is_file():
        return {
            "results": [],
            "season_priors": [],
            "replacement_tokens": [],
            "exposure_history": [],
            "cold_start_metadata": [],
            "frozen_2025_priors": None,
        }
    state = joblib.load(checkpoint)
    required = {
        "seasons",
        "results",
        "season_priors",
        "replacement_tokens",
        "exposure_history",
        "cold_start_metadata",
        "frozen_2025_priors",
    }
    if not isinstance(state, dict) or required - set(state) or tuple(state["seasons"]) != seasons:
        raise ValueError("Forward exposure-gated RAPM checkpoint is incompatible")
    lengths = [
        len(state["results"]),
        len(state["season_priors"]),
        len(state["replacement_tokens"]),
        len(state["exposure_history"]),
        len(state["cold_start_metadata"]),
    ]
    if len(set(lengths)) != 1 or lengths[0] > len(seasons):
        raise ValueError("Forward exposure-gated RAPM checkpoint is incomplete")
    return state


def _write_checkpoint(
    checkpoint: Path,
    *,
    seasons: tuple[str, ...],
    results: list[ForwardLaggedRapmSeason],
    season_priors: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
    exposure_history: list[pd.DataFrame],
    cold_start_metadata: list[dict[str, object]],
    frozen_2025_priors: pd.DataFrame | None,
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    joblib.dump(
        {
            "seasons": seasons,
            "results": results,
            "season_priors": season_priors,
            "replacement_tokens": replacement_tokens,
            "exposure_history": exposure_history,
            "cold_start_metadata": cold_start_metadata,
            "frozen_2025_priors": frozen_2025_priors,
        },
        temporary,
    )
    temporary.replace(checkpoint)


def _latest_run(parent: Path) -> Path:
    run_id = json.loads((parent / "latest.json").read_text())["run_id"]
    return parent / str(run_id)


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
    parser = argparse.ArgumentParser(description="Train forward exposure-gated RAPM")
    parser.add_argument("--through-season", default="2025-26")
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--curated-dir", default=str(DEFAULT_CURATED_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--max-seasons", type=int)
    args = parser.parse_args()
    run = train_forward_exposure_gated_rapm(
        through_season=args.through_season,
        player_season_panel_path=args.player_season_panel_path,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        checkpoint_path=args.checkpoint_path,
        max_seasons=args.max_seasons,
    )
    if run is None:
        print("Forward exposure-gated RAPM checkpoint saved")
    else:
        print(f"Forward exposure-gated RAPM: run={run.run_dir}")


if __name__ == "__main__":
    main()
