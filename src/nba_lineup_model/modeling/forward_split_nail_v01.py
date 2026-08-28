"""Split NAIL v0.1: jointly fit combined value and O/D specialization.

The player portion of a home-minus-away stint is parameterized as
``sum_home(R + s) - sum_away(R - s)``.  ``R`` is the combined rating and
``s`` is a zero-centered offense-versus-defense specialization.  Setting all
specializations to zero recovers the scalar NAIL player design exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    DEFAULT_TARGET_SEASON,
    _previous_season,
)
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import MODEL_NAME as SOURCE_MODEL
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import _Tee
from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.models.baselines import PriorPrecisionRidgeLineupModel, entity_vocabulary, signed_entity_matrix
from nba_lineup_model.season.schema import validate_season

MODEL_NAME = "forward_split_nail_v01"
RUN_PREFIX = "split-nail-v01"
SOURCE_RUN_DIR = Path(
    "artifacts/models/forward_nail_rapm_v1212_back_to_back/2025-26/"
    "forward-nail-rapm-v1212-back-to-back-2025-26-20260824T140929Z-e99e9646"
)
FIRST_SEASON = "1996-97"
DEFAULT_SPECIALIZATION_RELATIVE_PRECISION = 4.0


@dataclass(frozen=True)
class SplitNailV01State:
    """Persisted completed-season state used for the following forecast."""

    player_ids: tuple[int, ...]
    combined_rating: np.ndarray
    specialization: np.ndarray
    intercept: float
    selected_lambda: float
    specialization_relative_precision: float

    def specialization_by_player(self) -> dict[int, float]:
        return dict(zip(self.player_ids, self.specialization, strict=True))


@dataclass(frozen=True)
class ForwardSplitNailV01Run:
    run_dir: Path
    run_id: str


def train_split_nail_v01(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    source_run_dir: Path | str = SOURCE_RUN_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    specialization_relative_precision: float = DEFAULT_SPECIALIZATION_RELATIVE_PRECISION,
    game_catalog_path: Path | str = Path("data/catalog/games.parquet"),
) -> ForwardSplitNailV01Run:
    """Fit completed R/s states around the frozen scalar NAIL prior contract."""

    target = validate_season(through_season)
    if specialization_relative_precision <= 0:
        raise ValueError("specialization_relative_precision must be positive")
    source = Path(source_run_dir)
    priors, lambdas, context_models, schedule_models = _load_source_contract(source)
    panel = pd.read_parquet(player_season_panel_path)
    catalog = pd.read_parquet(game_catalog_path)
    schedule_features = build_back_to_back_game_features(catalog)
    states: dict[str, SplitNailV01State] = {}
    rows: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []

    seasons = _seasons_through(target)
    for index, season in enumerate(seasons, start=1):
        print(f"Fitting Split NAIL v0.1 for {season} ({index}/{len(seasons)})", flush=True)
        stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        adjusted, context_source, schedule_source = _source_adjusted_stints(
            season,
            stints,
            panel=panel,
            context_models=context_models,
            schedule_models=schedule_models,
            schedule_features=schedule_features,
            analytical_dir=analytical_dir,
            curated_dir=curated_dir,
        )
        state = _fit_season(
            adjusted,
            scalar_priors=priors.get(season, {}),
            previous_specialization=(
                states[_previous_season(season)].specialization_by_player()
                if _previous_season(season) in states
                else {}
            ),
            selected_lambda=lambdas[season],
            specialization_relative_precision=specialization_relative_precision,
        )
        states[season] = state
        rows.append(_state_ratings(state, season, panel))
        metadata.append(
            {
                "season": season,
                "selected_lambda": state.selected_lambda,
                "specialization_relative_precision": specialization_relative_precision,
                "specialization_regularization": (
                    state.selected_lambda * specialization_relative_precision
                ),
                "context_source_season": context_source,
                "schedule_source_season": schedule_source,
                "player_count": len(state.player_ids),
                "training_stint_count": len(adjusted),
                "intercept_home_court": state.intercept,
            }
        )
        print(f"Completed Split NAIL v0.1 {season}", flush=True)

    return _write_run(
        target=target,
        source_run_dir=source,
        states=states,
        ratings=pd.concat(rows, ignore_index=True),
        metadata=pd.DataFrame(metadata),
        artifacts_dir=Path(artifacts_dir),
        specialization_relative_precision=specialization_relative_precision,
    )


def _source_adjusted_stints(
    season: str,
    stints: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    context_models: dict[str, object],
    schedule_models: dict[str, object],
    schedule_features: pd.DataFrame,
    analytical_dir: Path | str,
    curated_dir: Path | str,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """Apply the exact scalar NAIL t-1 context and schedule offsets."""

    source = _previous_season(season)
    offset = np.zeros(len(stints), dtype=float)
    context = context_models.get(source)
    if context is not None:
        participants = set().union(*stints["home_player_ids"], *stints["away_player_ids"])
        profiles = build_contextual_player_profiles(
            panel,
            target_season=season,
            target_player_ids=participants,
            analytical_dir=str(analytical_dir),
            curated_dir=str(curated_dir),
            padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
            use_last_observed_profile=True,
        )
        offset += np.asarray(
            context.predict_lineups(
                stints["home_player_ids"].tolist(),
                stints["away_player_ids"].tolist(),
                profiles,
            ),
            dtype=float,
        )
    schedule = schedule_models.get(source)
    if schedule is not None:
        offset += np.asarray(schedule.predict_games(stints, schedule_features), dtype=float)
    adjusted = stints.copy()
    adjusted["target_home_net_rating"] = (
        stints["target_home_net_rating"].to_numpy(dtype=float) - offset
    )
    return adjusted, source if context is not None else None, source if schedule is not None else None


def _fit_season(
    stints: pd.DataFrame,
    *,
    scalar_priors: dict[int, float],
    previous_specialization: dict[int, float],
    selected_lambda: float,
    specialization_relative_precision: float,
) -> SplitNailV01State:
    player_ids = tuple(
        entity_vocabulary(stints, "home_player_ids", "away_player_ids", multiple=True)
    )
    scalar = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        {player_id: index for index, player_id in enumerate(player_ids)},
        multiple=True,
    ).tocsr()
    specialization = abs(scalar).tocsr()
    matrix = sparse.hstack([scalar, specialization], format="csr")
    prior = np.concatenate(
        [
            np.asarray([scalar_priors.get(player_id, 0.0) for player_id in player_ids]),
            np.asarray([previous_specialization.get(player_id, 0.0) for player_id in player_ids]),
        ]
    )
    precision = np.concatenate(
        [
            np.ones(len(player_ids), dtype=float),
            np.full(len(player_ids), specialization_relative_precision, dtype=float),
        ]
    )
    model = PriorPrecisionRidgeLineupModel(selected_lambda).fit(
        matrix,
        stints["target_home_net_rating"].to_numpy(dtype=float),
        stints["possessions"].to_numpy(dtype=float),
        prior,
        precision,
    )
    coefficients = model.coef_
    return SplitNailV01State(
        player_ids=player_ids,
        combined_rating=coefficients[: len(player_ids)],
        specialization=coefficients[len(player_ids) :],
        intercept=model.intercept_,
        selected_lambda=float(selected_lambda),
        specialization_relative_precision=float(specialization_relative_precision),
    )


def _state_ratings(state: SplitNailV01State, season: str, panel: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "season": season,
            "player_id": state.player_ids,
            "nail_rating": state.combined_rating,
            "specialization": state.specialization,
        }
    )
    output["offense_rating"] = output["nail_rating"] + output["specialization"]
    output["defense_rating"] = output["nail_rating"] - output["specialization"]
    names = panel.loc[
        panel["season"].eq(season),
        [column for column in ("player_id", "player_name", "age") if column in panel],
    ].drop_duplicates("player_id")
    return output.merge(names, on="player_id", how="left", validate="one_to_one")


def _load_source_contract(
    source_run_dir: Path,
) -> tuple[dict[str, dict[int, float]], dict[str, float], dict[str, object], dict[str, object]]:
    priors = pd.read_parquet(source_run_dir / "season_player_priors.parquet")
    coefficients = pd.read_parquet(source_run_dir / "historical_player_coefficients.parquet")
    return (
        {
            str(season): dict(zip(group["player_id"].astype(int), group["prior_rapm"], strict=True))
            for season, group in priors.groupby("season", sort=False)
        },
        {
            str(season): float(group["selected_lambda"].iloc[0])
            for season, group in coefficients.groupby("season", sort=False)
        },
        joblib.load(source_run_dir / "season_context_models.joblib"),
        joblib.load(source_run_dir / "season_schedule_models.joblib"),
    )


def _seasons_through(target: str) -> tuple[str, ...]:
    return tuple(
        f"{year}-{str(year + 1)[-2:]}"
        for year in range(int(FIRST_SEASON[:4]), int(target[:4]) + 1)
    )


def _write_run(
    *,
    target: str,
    source_run_dir: Path,
    states: dict[str, SplitNailV01State],
    ratings: pd.DataFrame,
    metadata: pd.DataFrame,
    artifacts_dir: Path,
    specialization_relative_precision: float,
) -> ForwardSplitNailV01Run:
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}-{target}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / MODEL_NAME / target
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        ratings.to_parquet(temporary / "player_season_ratings.parquet", index=False)
        metadata.to_parquet(temporary / "season_model_metadata.parquet", index=False)
        joblib.dump(states, temporary / "season_split_nail_states.joblib")
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "target_season": target,
            "created_at": now.isoformat(),
            "source_scalar_model": SOURCE_MODEL,
            "source_scalar_run_dir": str(source_run_dir),
            "parameterization": "offense=R+s; defense=R-s; displayed_combined=R",
            "specialization_relative_precision": specialization_relative_precision,
            "code_version": modeling_code_fingerprint((Path(__file__),)),
        }
        (temporary / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
        records = [
            {"filename": path.name, "byte_count": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(json.dumps({**payload, "artifacts": records}, indent=2) + "\n")
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return ForwardSplitNailV01Run(run_dir=output, run_id=run_id)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Split NAIL v0.1")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument(
        "--specialization-relative-precision",
        type=float,
        default=DEFAULT_SPECIALIZATION_RELATIVE_PRECISION,
    )
    parser.add_argument("--log-path")
    args = parser.parse_args()
    kwargs = {
        "through_season": args.through_season,
        "specialization_relative_precision": args.specialization_relative_precision,
    }
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_split_nail_v01(**kwargs)
                print(f"Split NAIL v0.1: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_split_nail_v01(**kwargs)
    print(f"Split NAIL v0.1: run={run.run_dir}")


if __name__ == "__main__":
    main()
