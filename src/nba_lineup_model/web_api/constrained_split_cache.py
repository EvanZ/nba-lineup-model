"""Materialize a descriptive O/D companion cache for a published NAIL release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from nba_lineup_model.modeling.constrained_split_nail import (
    DISPLAY_SEASON as SPLIT_DISPLAY_SEASON,
)
from nba_lineup_model.modeling.constrained_split_nail import (
    MODEL_NAME as CONSTRAINED_SPLIT_MODEL_NAME,
)
from nba_lineup_model.modeling.constrained_split_nail import (
    ConstrainedSplitState,
    _default_design_cache_path,
    _load_production_state,
    _materialize_display_ratings,
    _materialize_feature_allocations,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_PANEL_PATH
from nba_lineup_model.web_api.inference import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_LINEUP_RANKINGS_CACHE_DIR,
    DISPLAY_SEASON,
    MODEL_ARTIFACT,
    constrained_split_ratings_root,
)


def materialize_constrained_split_cache(
    *,
    production_run_id: str | None = None,
    constrained_run_dir: Path | str | None = None,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
) -> Path:
    """Write all completed O/D display ratings tied to one scalar NAIL release."""

    production_root = DEFAULT_ARTIFACTS_DIR / MODEL_ARTIFACT / DISPLAY_SEASON
    selected_production_run_id = production_run_id or str(
        json.loads((production_root / "latest.json").read_text())["run_id"]
    )
    production_run_dir = production_root / selected_production_run_id
    split_root = DEFAULT_ARTIFACTS_DIR / CONSTRAINED_SPLIT_MODEL_NAME / SPLIT_DISPLAY_SEASON
    selected_split_run_dir = (
        Path(constrained_run_dir)
        if constrained_run_dir is not None
        else split_root / str(json.loads((split_root / "latest.json").read_text())["run_id"])
    )
    metadata = json.loads((selected_split_run_dir / "metadata.json").read_text())
    if metadata.get("model") != CONSTRAINED_SPLIT_MODEL_NAME:
        raise ValueError("The constrained companion artifact has an unexpected model identity")
    if Path(str(metadata.get("source_production_run"))).name != selected_production_run_id:
        raise ValueError(
            "The constrained companion artifact was fit against a different production NAIL run"
        )

    designs = joblib.load(_default_design_cache_path(str(metadata["through_season"])))
    states = joblib.load(selected_split_run_dir / "season_states.joblib")
    if set(designs) != set(states):
        raise ValueError("Constrained companion designs and states cover different seasons")
    production = _load_production_state(production_run_dir)
    panel = pd.read_parquet(panel_path)

    frames: list[pd.DataFrame] = []
    context_frames: list[pd.DataFrame] = []
    for index, season in enumerate(sorted(states), start=1):
        print(f"Materializing constrained Split NAIL {season} ({index}/{len(states)})", flush=True)
        if season == min(states):
            ratings = _materialize_initial_season_ratings(
                season,
                states[season],
                production_run_dir=production_run_dir,
            )
        else:
            ratings, _, _ = _materialize_display_ratings(
                season,
                states[season],
                design=designs[season],
                source_dir=production_run_dir,
                production=production,
                panel=panel,
            )
        context_frames.append(
            _materialize_feature_allocations(
                season,
                states[season],
                design=designs[season],
                production=production,
            ).query("feature_layer == 'nonadditive_lineup'")
        )
        frames.append(
            ratings.loc[
                :,
                [
                    "season",
                    "player_id",
                    "total_nail",
                    "offense_rating",
                    "defense_rating",
                    "od_sum_error",
                ],
            ]
        )
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["season", "player_id"]).any():
        raise ValueError("Constrained O/D cache must be unique by season and player")
    if float(output["od_sum_error"].abs().max()) > 1e-9:
        raise ValueError("Constrained O/D cache does not reconcile to production NAIL")
    context_allocations = pd.concat(context_frames, ignore_index=True)
    if context_allocations.duplicated(["season", "feature"]).any():
        raise ValueError("Constrained O/D context allocations must be unique by season and feature")
    if float(context_allocations["od_sum_error"].abs().max()) > 1e-9:
        raise ValueError("Constrained O/D context allocations do not reconstruct scalar weights")

    cache_root = constrained_split_ratings_root(MODEL_ARTIFACT, selected_production_run_id)
    run_root = cache_root / selected_split_run_dir.name
    run_root.mkdir(parents=True, exist_ok=True)
    output.to_parquet(run_root / "player_ratings.parquet", index=False)
    context_allocations.to_parquet(run_root / "context_feature_allocations.parquet", index=False)
    (run_root / "metadata.json").write_text(
        json.dumps(
            {
                "production_model_artifact": MODEL_ARTIFACT,
                "production_run_id": selected_production_run_id,
                "constrained_split_model": CONSTRAINED_SPLIT_MODEL_NAME,
                "constrained_split_run_id": selected_split_run_dir.name,
                "selected_r_player": metadata["selected_r_player"],
                "selected_r_context": metadata["selected_r_context"],
                "season_count": int(output["season"].nunique()),
                "player_season_count": int(len(output)),
                "context_feature_allocation_count": int(len(context_allocations)),
                "max_od_sum_error": float(output["od_sum_error"].abs().max()),
            },
            indent=2,
        )
        + "\n"
    )
    (cache_root / "latest.json").write_text(
        json.dumps({"run_id": selected_split_run_dir.name}, indent=2) + "\n"
    )
    print(f"Wrote constrained Split NAIL cache: {run_root / 'player_ratings.parquet'}")
    return run_root / "player_ratings.parquet"


def _materialize_initial_season_ratings(
    season: str,
    state: ConstrainedSplitState,
    *,
    production_run_dir: Path,
) -> pd.DataFrame:
    """Split the initialization year before a prior-season profile can exist."""

    published_path = (
        DEFAULT_LINEUP_RANKINGS_CACHE_DIR
        / MODEL_ARTIFACT
        / production_run_dir.name
        / "player_ratings.parquet"
    )
    ratings = pd.read_parquet(published_path)
    ratings = ratings.loc[ratings["season"].astype(str).eq(season), ["season", "player_id", "rapm"]]
    ratings = ratings.rename(columns={"rapm": "total_nail"}).copy()
    ratings["difference"] = ratings["player_id"].map(state.player_difference).fillna(0.0)
    ratings["offense_rating"] = 0.5 * (ratings["total_nail"] + ratings["difference"])
    ratings["defense_rating"] = 0.5 * (ratings["total_nail"] - ratings["difference"])
    ratings["od_sum_error"] = (
        ratings["offense_rating"] + ratings["defense_rating"] - ratings["total_nail"]
    )
    return ratings.drop(columns="difference")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the web constrained Split NAIL companion cache"
    )
    parser.add_argument("--production-run-id")
    parser.add_argument("--constrained-run-dir")
    args = parser.parse_args()
    materialize_constrained_split_cache(
        production_run_id=args.production_run_id,
        constrained_run_dir=args.constrained_run_dir,
    )


if __name__ == "__main__":
    main()
