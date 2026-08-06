from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from nba_lineup_model.modeling.cold_start_exposure import (
    TARGET_OUTCOME_COLUMNS,
    adjusted_draft_exposure_curve,
    fit_exposure_model,
    prepare_cold_start_exposure_data,
    select_regularization,
)


def test_cold_start_gate_uses_only_historical_first_year_outcomes() -> None:
    cohort = _cohort()
    training, target_profiles = prepare_cold_start_exposure_data(
        cohort,
        target_season="2025-26",
        replacement_share_cutoff=0.05,
    )
    changed = cohort.copy()
    changed.loc[changed["season"].eq("2025-26"), "exposure_share"] = 0.99
    changed.loc[changed["season"].eq("2025-26"), "rapm"] = 10_000.0
    changed_training, changed_target_profiles = prepare_cold_start_exposure_data(
        changed,
        target_season="2025-26",
        replacement_share_cutoff=0.05,
    )

    assert training["season"].max() == "2024-25"
    assert not (TARGET_OUTCOME_COLUMNS & set(target_profiles))
    pdt.assert_frame_equal(training, changed_training)
    pdt.assert_frame_equal(target_profiles, changed_target_profiles)


def test_cold_start_gate_excludes_returning_players() -> None:
    cohort = _cohort()
    cohort.loc[cohort.index[0], "is_rookie"] = False

    training, _ = prepare_cold_start_exposure_data(
        cohort,
        target_season="2025-26",
        replacement_share_cutoff=0.05,
    )

    assert cohort.loc[cohort.index[0], "player_id"] not in set(training["player_id"])


def test_cold_start_gate_has_expected_draft_pick_direction() -> None:
    training, _ = prepare_cold_start_exposure_data(
        _cohort(), target_season="2025-26", replacement_share_cutoff=0.05
    )
    selected, folds = select_regularization(training, c_grid=(0.03, 0.3, 3.0))
    curve = adjusted_draft_exposure_curve(fit_exposure_model(training, c=selected), training)

    assert set(folds["validation_season"]) == {
        "2019-20",
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
    }
    first_pick = curve.loc[
        curve["draft_pick"].eq(1), "predicted_replacement_probability"
    ].item()
    last_pick = curve.loc[
        curve["draft_pick"].eq(60), "predicted_replacement_probability"
    ].item()
    assert first_pick < last_pick


def _cohort() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season_start in range(2017, 2026):
        season = f"{season_start}-{str(season_start + 1)[-2:]}"
        for draft_number in (2, 16, 42, 55):
            player_id = season_start * 100 + draft_number
            rows.append(
                {
                    "season": season,
                    "season_start_year": season_start,
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "is_rookie": True,
                    "exposure_share": 0.20 if draft_number <= 16 else 0.02,
                    "rapm": 4.0 - 0.08 * draft_number,
                    "rapm_possessions": 500.0,
                    "on_court_possessions": 500.0,
                    "team_opportunity_possessions": 5_000.0,
                    "draft_year": season_start,
                    "draft_round": 1 if draft_number <= 30 else 2,
                    "draft_number": draft_number,
                    "is_undrafted": False,
                    "age": 20.0 + (draft_number % 3),
                    "height_inches": 76.0 + (draft_number % 4),
                    "weight_pounds": 205.0 + draft_number,
                    "listed_position": "G",
                }
            )
    return pd.DataFrame(rows)
