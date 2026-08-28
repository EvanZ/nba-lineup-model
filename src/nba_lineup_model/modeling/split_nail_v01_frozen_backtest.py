"""Three-season frozen evaluation for Split NAIL v0.1.

The scalar NAIL v1.2.1.2 frozen forecast is the base prediction.  A source
season's fitted specialization adds ``sum(s_home) + sum(s_away)`` to the home
net-rating estimate.  This keeps the baseline contract and evaluation support
identical while testing whether O/D specialization generalizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.forward_contextual_rapm import _previous_season
from nba_lineup_model.modeling.forward_split_nail_v01 import (
    MODEL_NAME,
    SplitNailV01State,
)
from nba_lineup_model.modeling.frozen_prior_evaluation import (
    _game_prediction_frame,
    _historical_team_seasons,
    _read_playoff_possessions,
    _team_net_rating_metrics,
    _team_win_evaluation,
    fit_pythagorean_win_model,
    score_possession_cohort,
)
from nba_lineup_model.modeling.neural_data import read_neural_possessions
from nba_lineup_model.modeling.stints import read_rapm_stints

SEASONS = ("2023-24", "2024-25", "2025-26")
BASELINE_MODEL = "forward_nail_rapm_v1212_back_to_back"
DEFAULT_SPLIT_RUN_DIR = Path(
    "artifacts/models/forward_split_nail_v01/2025-26/"
    "split-nail-v01-2025-26-REPLACE_AFTER_TRAINING"
)
DEFAULT_BASELINE_RUN_DIR = Path(
    "artifacts/models/nail_v1212_back_to_back_frozen_backtest/"
    "frozen_multiseason_backtest/2023-24_to_2025-26/"
    "frozen_multiseason_backtest-2023-24-to-2025-26-20260824T141701Z-1b7f1e15"
)


def run_split_nail_v01_frozen_backtest(
    *,
    split_run_dir: Path | str,
    baseline_run_dir: Path | str = DEFAULT_BASELINE_RUN_DIR,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    output_root: Path | str = Path("artifacts/models/split_nail_v01_frozen_backtest"),
    seasons: tuple[str, ...] = SEASONS,
) -> Path:
    """Replay source specialization states on the scalar NAIL frozen support."""

    split_dir = Path(split_run_dir)
    baseline_dir = Path(baseline_run_dir)
    states: dict[str, SplitNailV01State] = joblib.load(
        split_dir / "season_split_nail_states.joblib"
    )
    base_predictions = pd.read_parquet(baseline_dir / "possession_predictions.parquet")
    base_games = pd.read_parquet(baseline_dir / "regular_game_predictions.parquet")
    base_source_states = pd.read_parquet(baseline_dir / "source_states.parquet")

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[pd.DataFrame] = []
    full_game_rows: list[pd.DataFrame] = []
    team_rows: list[pd.DataFrame] = []
    win_rows: list[pd.DataFrame] = []
    win_metric_rows: list[pd.DataFrame] = []
    support_rows: list[dict[str, object]] = []
    for season in seasons:
        source = _previous_season(season)
        if source not in states:
            raise ValueError(f"Split NAIL v0.1 lacks source state for {season}")
        specialization = states[source].specialization_by_player()
        print(f"Scoring Split NAIL v0.1 state {source} onto {season}", flush=True)
        source_mean = _source_mean(base_source_states, season)
        for cohort, possessions in (
            ("regular_season", read_neural_possessions(season, analytical_dir=analytical_dir)),
            ("playoffs", _read_playoff_possessions(season, curated_dir)[0]),
        ):
            baseline = _baseline_predictions(base_predictions, season, cohort)
            predictions = _add_specialization(possessions, baseline, specialization, cohort)
            support_rows.append(_assert_support(predictions, baseline, season, cohort))
            prediction_rows.append(predictions)
            metric_rows.append(
                score_possession_cohort(
                    predictions,
                    source_mean=source_mean,
                    model=MODEL_NAME,
                ).assign(season=season)
            )

        target_stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        baseline_games = _baseline_games(base_games, season)
        games, teams = _add_specialization_to_full_games(
            target_stints,
            baseline_games,
            specialization,
        )
        full_game_rows.append(games.assign(season=season, model=MODEL_NAME))
        team_rows.append(teams.assign(season=season, model=MODEL_NAME))
        pythagorean = fit_pythagorean_win_model(
            _historical_team_seasons(analytical_dir=Path(analytical_dir), through_season=source)
        )
        wins, win_metrics = _team_win_evaluation(games, teams, pythagorean, model=MODEL_NAME)
        win_rows.append(wins.assign(season=season, model=MODEL_NAME))
        win_metric_rows.append(win_metrics.assign(season=season))

    support = pd.DataFrame(support_rows)
    if not support["support_matches"].all():
        raise AssertionError("Split NAIL v0.1 support differs from scalar NAIL")
    return _write_run(
        split_run_dir=split_dir,
        predictions=pd.concat(prediction_rows, ignore_index=True),
        cohort_metrics=pd.concat(metric_rows, ignore_index=True),
        games=pd.concat(full_game_rows, ignore_index=True),
        teams=pd.concat(team_rows, ignore_index=True),
        wins=pd.concat(win_rows, ignore_index=True),
        win_metrics=pd.concat(win_metric_rows, ignore_index=True),
        support=support,
        output_root=Path(output_root),
    )


def _add_specialization(
    possessions: pd.DataFrame,
    baseline: pd.DataFrame,
    specialization: dict[int, float],
    cohort: str,
) -> pd.DataFrame:
    base = baseline.set_index(["game_id", "possession_id"], verify_integrity=True)
    key = pd.MultiIndex.from_frame(possessions.loc[:, ["game_id", "possession_id"]])
    if not key.isin(base.index).all():
        raise ValueError("Scalar NAIL frozen predictions are missing possession keys")
    output = base.loc[key].reset_index()
    home_lineups = [
        offense if home else defense
        for offense, defense, home in zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            possessions["home_offense"],
            strict=True,
        )
    ]
    away_lineups = [
        defense if home else offense
        for offense, defense, home in zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            possessions["home_offense"],
            strict=True,
        )
    ]
    correction = np.asarray(
        [
            sum(specialization.get(int(player), 0.0) for player in home)
            + sum(specialization.get(int(player), 0.0) for player in away)
            for home, away in zip(home_lineups, away_lineups, strict=True)
        ],
        dtype=float,
    )
    output["prediction_home_margin"] += correction / 100.0
    output["prediction_offense_margin"] = (
        output["prediction_home_margin"] * output["home_offense_sign"]
    )
    output["residual_offense_margin"] = (
        output["target_offense_margin"] - output["prediction_offense_margin"]
    )
    output["cohort"] = cohort
    output["model"] = MODEL_NAME
    output["label"] = "Split NAIL v0.1 R/s specialization"
    return output


def _add_specialization_to_full_games(
    stints: pd.DataFrame,
    baseline_games: pd.DataFrame,
    specialization: dict[int, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = stints.loc[
        :, [
            "game_id",
            "home_team_id",
            "away_team_id",
            "home_team_tricode",
            "away_team_tricode",
            "possessions",
            "home_player_ids",
            "away_player_ids",
        ]
    ].copy()
    rows["specialization_rating"] = [
        sum(specialization.get(int(player), 0.0) for player in home)
        + sum(specialization.get(int(player), 0.0) for player in away)
        for home, away in zip(rows["home_player_ids"], rows["away_player_ids"], strict=True)
    ]
    rows["specialization_margin"] = rows["specialization_rating"] * rows["possessions"] / 100.0
    correction = rows.groupby("game_id", as_index=False, sort=False).agg(
        specialization_margin=("specialization_margin", "sum"),
        possessions=("possessions", "sum"),
    )
    games = baseline_games.merge(correction, on="game_id", how="inner", validate="one_to_one").copy()
    games["predicted_home_margin"] += games.pop("specialization_margin")
    games["margin_error"] = games["predicted_home_margin"] - games["actual_home_margin"]
    games["predicted_tie"] = np.isclose(games["predicted_home_margin"], 0.0)
    games["predicted_home_win"] = games["predicted_home_margin"].gt(0.0)
    games["actual_home_win"] = games["actual_home_margin"].gt(0.0)
    teams = _teams_from_games(games)
    return games, teams


def _teams_from_games(games: pd.DataFrame) -> pd.DataFrame:
    base = games.loc[
        :, [
            "home_team_id", "away_team_id", "home_team_tricode", "away_team_tricode",
            "possessions", "actual_home_margin", "predicted_home_margin",
        ]
    ]
    home = base.rename(
        columns={
            "home_team_id": "team_id", "home_team_tricode": "team_tricode",
            "actual_home_margin": "actual_margin", "predicted_home_margin": "predicted_margin",
        }
    ).loc[:, ["team_id", "team_tricode", "possessions", "actual_margin", "predicted_margin"]]
    away = base.rename(
        columns={
            "away_team_id": "team_id", "away_team_tricode": "team_tricode",
            "actual_home_margin": "actual_margin", "predicted_home_margin": "predicted_margin",
        }
    ).loc[:, ["team_id", "team_tricode", "possessions", "actual_margin", "predicted_margin"]].copy()
    away[["actual_margin", "predicted_margin"]] *= -1.0
    teams = pd.concat([home, away], ignore_index=True).groupby(
        ["team_id", "team_tricode"], as_index=False, sort=True
    ).agg(
        possessions=("possessions", "sum"),
        actual_total_margin=("actual_margin", "sum"),
        predicted_total_margin=("predicted_margin", "sum"),
    )
    teams["actual_net_rating"] = 100.0 * teams["actual_total_margin"] / teams["possessions"]
    teams["predicted_net_rating"] = 100.0 * teams["predicted_total_margin"] / teams["possessions"]
    teams["net_rating_error"] = teams["predicted_net_rating"] - teams["actual_net_rating"]
    return teams


def _baseline_predictions(predictions: pd.DataFrame, season: str, cohort: str) -> pd.DataFrame:
    rows = predictions.loc[
        predictions["season"].eq(season)
        & predictions["cohort"].eq(cohort)
        & predictions["model"].eq(BASELINE_MODEL)
    ].copy()
    if rows.empty:
        raise ValueError(f"Scalar NAIL frozen artifact lacks {season} {cohort} predictions")
    return rows


def _baseline_games(games: pd.DataFrame, season: str) -> pd.DataFrame:
    rows = games.loc[
        games["season"].eq(season) & games["model"].eq(BASELINE_MODEL)
    ].copy()
    if rows.empty:
        raise ValueError(f"Scalar NAIL frozen artifact lacks {season} full games")
    return rows


def _source_mean(states: pd.DataFrame, season: str) -> float:
    values = states.loc[
        states["season"].eq(season) & states["model"].eq(BASELINE_MODEL),
        "source_offense_margin_mean",
    ]
    if len(values) != 1:
        raise ValueError(f"Scalar NAIL frozen artifact lacks a source mean for {season}")
    return float(values.iloc[0])


def _assert_support(predictions: pd.DataFrame, baseline: pd.DataFrame, season: str, cohort: str) -> dict[str, object]:
    left = predictions.loc[:, ["game_id", "possession_id", "target_offense_margin"]].sort_values(
        ["game_id", "possession_id"], kind="stable"
    )
    right = baseline.loc[:, ["game_id", "possession_id", "target_offense_margin"]].sort_values(
        ["game_id", "possession_id"], kind="stable"
    )
    matches = left.reset_index(drop=True).equals(right.reset_index(drop=True))
    return {"season": season, "cohort": cohort, "possession_count": len(left), "support_matches": matches}


def _write_run(
    *,
    split_run_dir: Path,
    predictions: pd.DataFrame,
    cohort_metrics: pd.DataFrame,
    games: pd.DataFrame,
    teams: pd.DataFrame,
    wins: pd.DataFrame,
    win_metrics: pd.DataFrame,
    support: pd.DataFrame,
    output_root: Path,
) -> Path:
    now = datetime.now(UTC)
    run_id = f"split-nail-v01-frozen-backtest-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    output = output_root / run_id
    temporary = output_root / f".{run_id}.tmp"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        team_metrics = pd.concat(
            [
                _team_net_rating_metrics(
                    teams.loc[teams["season"].eq(season)], model=MODEL_NAME
                ).assign(season=season)
                for season in sorted(teams["season"].unique())
            ],
            ignore_index=True,
        )
        tables = {
            "possession_predictions.parquet": predictions,
            "cohort_metrics.parquet": cohort_metrics,
            "regular_game_predictions.parquet": games,
            "team_net_rating_predictions.parquet": teams,
            "team_net_rating_metrics.parquet": team_metrics,
            "team_win_predictions.parquet": wins,
            "team_win_metrics.parquet": win_metrics,
            "support_audit.parquet": support,
        }
        for filename, frame in tables.items():
            frame.to_parquet(temporary / filename, index=False)
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "source_split_run_dir": str(split_run_dir),
            "base_model": BASELINE_MODEL,
            "state_contract": "scalar NAIL frozen prediction plus prior-season specialization",
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
        records = [
            {"filename": p.name, "byte_count": p.stat().st_size, "sha256": _sha256(p)}
            for p in sorted(temporary.iterdir()) if p.is_file()
        ]
        (temporary / "manifest.json").write_text(json.dumps({**payload, "artifacts": records}, indent=2) + "\n")
        temporary.replace(output)
        (output_root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return output
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
    parser = argparse.ArgumentParser(description="Evaluate Split NAIL v0.1 on frozen seasons")
    parser.add_argument("--split-run-dir", required=True)
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.log_path:
        path = Path(args.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            from contextlib import redirect_stdout
            with redirect_stdout(handle):
                output = run_split_nail_v01_frozen_backtest(split_run_dir=args.split_run_dir)
                print(f"Split NAIL v0.1 frozen backtest: run={output}")
        return
    output = run_split_nail_v01_frozen_backtest(split_run_dir=args.split_run_dir)
    print(f"Split NAIL v0.1 frozen backtest: run={output}")


if __name__ == "__main__":
    main()
