from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.forward_cold_start_validation import summarize_rookie_validation


def test_rookie_validation_reports_weighted_and_unweighted_metrics() -> None:
    metrics = summarize_rookie_validation(
        pd.DataFrame(
            {
                "cold_start_rapm_prior": [-1.0, 0.0, 1.0, -2.0],
                "refit_forward_rapm": [-0.5, 0.2, 1.4, -1.8],
                "on_court_possessions": [100.0, 200.0, 300.0, 50.0],
                "draft_status": ["Drafted 1-60", "Drafted 1-60", "Undrafted", "Undrafted"],
                "actual_low_exposure": [False, False, False, True],
            }
        )
    )

    all_players = metrics.loc[metrics["cohort"].eq("All first-year players")].iloc[0]
    assert all_players["player_count"] == 4
    assert all_players["pearson_correlation"] > 0.9
    assert all_players["weighted_pearson_correlation"] > 0.9
    assert all_players["weighted_rmse"] > 0
