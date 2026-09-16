"""Materialize historical team wins and Gestalt PyWins for the web release."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nba_lineup_model.web_api.inference import (
    MODEL_ARTIFACT,
    LineupEvaluator,
    team_win_history_path,
)

_OUTPUT_COLUMNS = (
    "team",
    "season",
    "games",
    "actual_wins",
    "gestalt_pywins",
    "gestalt_rating",
)


def build_team_win_history_cache(
    *,
    evaluator: LineupEvaluator | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Write the schedule-independent Team page history for one NAIL release."""

    state = evaluator or LineupEvaluator.from_latest_artifact()
    records = [
        {"team": team, **record}
        for team, history in state.team_win_histories.items()
        for record in history
    ]
    if not records:
        raise ValueError("Cannot materialize an empty team win-history cache")
    frame = pd.DataFrame.from_records(records)
    missing = sorted(set(_OUTPUT_COLUMNS) - set(frame))
    if missing:
        raise ValueError("Team win history lacks " + ", ".join(missing))
    if frame.duplicated(["team", "season"]).any():
        raise ValueError("Team win history has duplicate team-season rows")
    path = (
        Path(output_path)
        if output_path is not None
        else team_win_history_path(MODEL_ARTIFACT, state.run_id)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(_OUTPUT_COLUMNS)].sort_values(
        ["team", "season"], kind="stable"
    ).to_parquet(path, index=False)
    return path


def main() -> int:
    """Build the release-scoped historical Team page cache."""

    parser = argparse.ArgumentParser(description="Build the GESTALT team win-history cache")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    path = build_team_win_history_cache(output_path=args.output)
    print(f"Materialized team win-history cache: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
