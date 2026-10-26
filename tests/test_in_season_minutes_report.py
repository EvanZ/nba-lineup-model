from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_lineup_model.modeling.in_season_minutes_report import (
    render_minutes_shadow_report,
)


def test_render_minutes_shadow_report_is_self_contained(tmp_path: Path) -> None:
    summary = {
        "season": "2026-27",
        "through_date": "2026-10-25",
        "generated_at": "2026-10-26T04:00:00Z",
        "games": 2,
        "player_game_rows": 2,
        "excluded_games": [],
        "config": {"conditional_prior_games": 0.1},
        "player_metrics": {
            "frozen": {
                "conditional_minutes_mae": 5.5,
                "unconditional_minutes_mae": 5.2,
            },
            "updated": {
                "conditional_minutes_mae": 4.8,
                "unconditional_minutes_mae": 4.7,
                "conditional_minutes_n": 2,
            },
        },
        "four_cell_game_metrics": {
            key: {
                "games": 2,
                "margin_mae": 8.0,
                "margin_rmse": 9.0,
                "brier_score": 0.18,
                "log_loss": 0.54,
                "winner_accuracy": 0.75,
                "ats_wins": 1,
                "ats_losses": 1,
                "ats_pushes": 0,
            }
            for key in (
                "frozen_frozen",
                "frozen_updated",
                "updated_frozen",
                "updated_updated",
            )
        },
    }
    evaluation = pd.DataFrame(
        {
            "player_id": [30, 30],
            "player_name": ["Stephen Curry", "Stephen Curry"],
            "team": ["GSW", "GSW"],
            "game_id": ["1", "2"],
            "played": [True, True],
            "outcome_classification": ["played", "played"],
            "regulation_equivalent_minutes": [34.0, 36.0],
            "frozen_conditional_minutes": [32.0, 32.0],
            "updated_conditional_minutes": [33.0, 34.0],
        }
    )
    state = pd.DataFrame(
        {
            "player_id": [30],
            "preseason_conditional_minutes_prior": [32.0],
            "current_conditional_minutes_estimate": [34.0],
            "conditional_observations": [2],
        }
    )
    destination = tmp_path / "report.html"

    rendered = render_minutes_shadow_report(
        summary=summary,
        evaluation=evaluation,
        games=pd.DataFrame(),
        state=state,
        output_path=destination,
    )

    page = destination.read_text()
    assert rendered == destination
    assert "In-season minutes updater" in page
    assert "Stephen Curry" in page
    assert "Log loss" in page
    assert "through 2026-10-25 Pacific" in page
    assert "conditional prior 0.1 equivalent games" in page
    assert "0.18000" in page
    assert "https://" not in page
