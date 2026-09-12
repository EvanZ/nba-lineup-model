"""Materialize the FCM v0.3 team-strength Win Projections cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nba_lineup_model.web_api.inference import MODEL_ARTIFACT, LineupEvaluator
from nba_lineup_model.web_api.preseason_minutes import (
    build_incumbent_team_strength_minutes_payload,
)
from nba_lineup_model.web_api.win_projections import (
    build_win_projection_payload,
    read_win_projection_cache,
    win_projection_cache_path,
)

DEFAULT_TEAM_STRENGTH_HISTORY_PATH = Path(
    "artifacts/rotation/forward_conditional_team_strength_roster_squash/"
    "forward-conditional-team-strength-roster-squash-20260912T195229Z-0881199/"
    "team_strength_player_season_summary.parquet"
)
DEFAULT_OUTPUT_PATH = Path(
    "artifacts/web/local_previews/forward_nail_rapm_v1212_residualized_lambda/"
    "forward-conditional-team-strength-2026-27.json"
)


def materialize_team_strength_preview(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    team_strength_history_path: Path | str = DEFAULT_TEAM_STRENGTH_HISTORY_PATH,
) -> Path:
    """Write one self-contained local preview payload."""

    return materialize_team_strength_win_projection_cache(
        output_path=output_path,
        team_strength_history_path=team_strength_history_path,
    )


def materialize_team_strength_win_projection_cache(
    *,
    output_path: Path | str | None = None,
    team_strength_history_path: Path | str = DEFAULT_TEAM_STRENGTH_HISTORY_PATH,
) -> Path:
    """Build the versioned FCM v0.3 release cache from the baseline cache."""

    evaluator = LineupEvaluator.from_latest_artifact()
    baseline = read_win_projection_cache(
        model_artifact=MODEL_ARTIFACT,
        run_id=evaluator.run_id,
    )
    minutes = baseline.get("minutes")
    if not isinstance(minutes, dict):
        raise ValueError("Published win projection cache lacks a minutes payload")
    history_path = Path(team_strength_history_path)
    if not history_path.is_file():
        raise FileNotFoundError(f"Team-strength history is missing: {history_path}")
    candidate_minutes = build_incumbent_team_strength_minutes_payload(
        baseline_minutes_payload=minutes,
        team_strength_history=pd.read_parquet(history_path),
    )
    payload = {
        "minutes": candidate_minutes,
        "win_projection": build_win_projection_payload(
            evaluator=evaluator,
            minutes_payload=candidate_minutes,
        ),
    }
    output = (
        Path(output_path)
        if output_path is not None
        else win_projection_cache_path(MODEL_ARTIFACT, evaluator.run_id)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    """Write the local preview cache used by ``GESTALT_WIN_PROJECTION_CACHE_PATH``."""

    parser = argparse.ArgumentParser(
        description="Build the local incumbent-team-strength Win Projections preview"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--team-strength-history", type=Path, default=DEFAULT_TEAM_STRENGTH_HISTORY_PATH
    )
    args = parser.parse_args()
    print(
        materialize_team_strength_preview(
            output_path=args.output,
            team_strength_history_path=args.team_strength_history,
        )
    )


def main_release() -> None:
    """Overwrite the materialized baseline cache with the FCM v0.3 cache."""

    parser = argparse.ArgumentParser(
        description="Build the FCM v0.3 team-strength Win Projections release cache"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--team-strength-history", type=Path, default=DEFAULT_TEAM_STRENGTH_HISTORY_PATH
    )
    args = parser.parse_args()
    print(
        materialize_team_strength_win_projection_cache(
            output_path=args.output,
            team_strength_history_path=args.team_strength_history,
        )
    )


if __name__ == "__main__":
    main()
