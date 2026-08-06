from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from nba_lineup_model.modeling.draft_prior import (
    adjusted_draft_curve,
    empirical_draft_curve,
    fit_draft_prior_model,
    prepare_draft_prior_data,
    rolling_regularization_stability,
    select_regularization,
)


def test_draft_prior_uses_only_historical_first_nba_season_outcomes() -> None:
    panel = _rookie_panel()
    training, target_profiles = prepare_draft_prior_data(panel, target_season="2025-26")
    changed = panel.copy()
    changed.loc[changed["season"].eq("2025-26"), "rapm"] = 10_000.0
    changed.loc[changed["season"].eq("2025-26"), "rapm_possessions"] = 10_000_000.0
    changed_training, changed_target_profiles = prepare_draft_prior_data(
        changed,
        target_season="2025-26",
    )

    assert training["season"].max() == "2024-25"
    assert "rapm" not in target_profiles
    assert "rapm_possessions" not in target_profiles
    pdt.assert_frame_equal(training, changed_training)
    pdt.assert_frame_equal(target_profiles, changed_target_profiles)


def test_draft_prior_curve_has_expected_pick_direction() -> None:
    training, _ = prepare_draft_prior_data(_rookie_panel(), target_season="2025-26")
    selected, folds = select_regularization(
        training,
        regularization_grid=(0.03, 0.3, 3.0),
    )
    curve = adjusted_draft_curve(
        fit_draft_prior_model(training, regularization=selected),
        training,
    )

    assert set(folds["validation_season"]) == {
        "2019-20",
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
    }
    assert curve.loc[curve["draft_pick"].eq(1), "draft_prior"].item() > curve.loc[
        curve["draft_pick"].eq(60), "draft_prior"
    ].item()


def test_empirical_curve_excludes_imputed_undrafted_pick_values() -> None:
    training, _ = prepare_draft_prior_data(_rookie_panel(), target_season="2025-26")
    training.loc[training.index[0], "draft_status"] = "undrafted"
    curve = empirical_draft_curve(training)

    expected = (
        training["draft_status"].eq("drafted_1_60")
        & training["draft_number"].between(1, 3)
    ).sum()
    assert curve.loc[curve["draft_tier"].eq("1-3"), "player_count"].item() == expected


def test_draft_age_interaction_is_zero_for_undrafted_profiles() -> None:
    panel = _rookie_panel()
    panel.loc[panel.index[0], ["is_undrafted", "draft_round", "draft_number"]] = [True, None, None]

    training, _ = prepare_draft_prior_data(panel, target_season="2025-26")

    assert (
        training.loc[training["draft_status"].eq("undrafted"), "draft_pick_draft_age_interaction"]
        == 0.0
    ).all()


def test_regularization_stability_uses_only_expanding_historical_snapshots() -> None:
    training, _ = prepare_draft_prior_data(_rookie_panel(), target_season="2025-26")

    stability = rolling_regularization_stability(
        training,
        regularization_grid=(0.03, 0.3, 3.0),
        minimum_seasons=8,
    )

    assert stability["training_last_season"].tolist() == ["2024-25"]
    assert stability["validation_last_season"].tolist() == ["2024-25"]


def _rookie_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season_start in range(2017, 2026):
        season = f"{season_start}-{str(season_start + 1)[-2:]}"
        for draft_number in (2, 16, 42, 55):
            player_id = season_start * 100 + draft_number
            rows.append(
                {
                    "season": season,
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "is_rookie": True,
                    "rapm": 4.0 - 0.08 * draft_number + 0.02 * (season_start - 2017),
                    "rapm_possessions": 500.0 + 5.0 * draft_number,
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
