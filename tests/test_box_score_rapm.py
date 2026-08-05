from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.box_score_rapm import (
    BOX_SCORE_FEATURE_COLUMNS,
    prepare_returning_player_rows,
    run_box_score_rapm_experiment,
)
from nba_lineup_model.modeling.cold_start_rapm import prepare_cold_start_rows


def test_box_score_selection_is_forward_only_and_excludes_cold_starts():
    features = synthetic_box_score_features()
    original = run_box_score_rapm_experiment(
        features,
        regularization_grid=(0.001, 0.1, 1.0),
    )
    perturbed = features.copy()
    holdout = perturbed["target_season"].eq("2023-24")
    perturbed.loc[holdout, "target_rapm"] += 100.0
    changed = run_box_score_rapm_experiment(
        perturbed,
        regularization_grid=(0.001, 0.1, 1.0),
    )

    assert original.holdout_target_season == "2023-24"
    assert original.selected_regularization == changed.selected_regularization
    assert original.feature_coefficients["coefficient"].tolist() == (
        changed.feature_coefficients["coefficient"].tolist()
    )
    assert len(original.holdout_predictions) == 20
    assert original.holdout_predictions["has_prior_season"].astype(bool).all()
    assert set(original.holdout_metrics["cohort"]) == {
        "returning_all",
        "low_exposure",
        "developing",
        "established",
    }
    returning_all = original.holdout_metrics.loc[
        original.holdout_metrics["cohort"].eq("returning_all")
    ].set_index("model")
    assert (
        returning_all.loc["box_score", "weighted_rmse"]
        < returning_all.loc["persistence", "weighted_rmse"]
    )
    assert not original.holdout_predictions["target_rapm"].equals(
        changed.holdout_predictions["target_rapm"]
    )


def test_returning_rows_require_complete_prior_profile():
    rows = synthetic_box_score_features()
    rows.loc[rows.index[0], "prior_boxscore_features_available"] = False

    eligible = prepare_returning_player_rows(rows)

    assert len(eligible) == 79
    assert eligible["prior_rapm"].notna().all()


def test_cold_start_rows_cannot_include_returning_players():
    cold = prepare_cold_start_rows(synthetic_box_score_features())

    assert len(cold) == 20
    assert not cold["has_prior_season"].astype(bool).any()


def synthetic_box_score_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target_seasons = ("2020-21", "2021-22", "2022-23", "2023-24")
    for season_index, target_season in enumerate(target_seasons):
        prior_season = f"{int(target_season[:4]) - 1}-{target_season[2:4]}"
        for player_id in range(1, 26):
            cold_start = player_id > 20
            prior_possessions = float(300 + player_id * 80)
            prior_rapm = -3.0 + player_id * 0.3 + season_index * 0.1
            assist_rate = 1.5 + (player_id % 7) * 0.7
            rebound_rate = 2.0 + (player_id % 5) * 0.9
            target_rapm = (
                0.46 * prior_rapm
                + 0.4 * assist_rate
                + 0.25 * rebound_rate
                + season_index * 0.05
            )
            row: dict[str, object] = {
                "target_season": target_season,
                "prior_source_season": prior_season if not cold_start else None,
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "target_rapm": target_rapm,
                "target_rapm_possessions": prior_possessions + 100.0,
                "prior_exposure_cohort": (
                    "low_exposure"
                    if player_id <= 6
                    else "developing" if player_id <= 12 else "established"
                ),
                "has_prior_season": not cold_start,
                "prior_rapm_available": not cold_start,
                "prior_boxscore_features_available": not cold_start,
            }
            for column in BOX_SCORE_FEATURE_COLUMNS:
                row[column] = 0.0
            row.update(
                {
                    "preseason_age": 20.0 + player_id % 12 + season_index,
                    "preseason_nba_experience_years": max(0, player_id % 8),
                    "preseason_is_rookie": cold_start,
                    "preseason_years_since_draft": max(0, player_id % 8),
                    "draft_year": 2010 + player_id % 10,
                    "draft_round": 1.0,
                    "draft_number": float(player_id),
                    "is_undrafted": False,
                    "height_inches": 72.0 + player_id % 10,
                    "weight_pounds": 180.0 + player_id * 2,
                    "listed_position": "G" if player_id % 2 else "F",
                    "prior_log_on_court_possessions": np.log1p(prior_possessions),
                    "prior_rapm": prior_rapm if not cold_start else np.nan,
                    "prior_assists_per_100_on_court_possessions": assist_rate,
                    "prior_defensive_rebounds_per_100_on_court_possessions": rebound_rate,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
