from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.aging_prior_rapm import aging_prior_frame
from nba_lineup_model.modeling.blended_prior_rapm import blended_prior_frame
from nba_lineup_model.modeling.prior_rapm import (
    fit_forward_lagged_rapm_history,
    fit_prior_rapm_experiment,
)
from nba_lineup_model.modeling.schema import ChronologicalSplitConfig
from nba_lineup_model.models.baselines import (
    PriorCenteredRidgeLineupModel,
    signed_entity_matrix,
    vocabulary_mapping,
)


def test_prior_centered_ridge_matches_residualized_objective() -> None:
    frame = pd.DataFrame({"home": [[1], [2]], "away": [[2], [1]]})
    matrix = signed_entity_matrix(frame, "home", "away", vocabulary_mapping((1, 2)), multiple=True)
    model = PriorCenteredRidgeLineupModel(0.01).fit(
        matrix,
        np.array([4.0, -4.0]),
        np.ones(2),
        np.array([1.5, -1.5]),
    )

    assert model.predict(matrix).tolist() == pytest.approx([4.0, -4.0], abs=0.05)
    assert model.coef_.tolist() == pytest.approx(
        (np.array([1.5, -1.5]) + model.adjustment_).tolist()
    )


def test_prior_rapm_uses_frozen_prior_and_zero_for_unseen_players() -> None:
    stints = _stints(game_count=20)
    priors = pd.DataFrame(
        {
            "player_id": list(range(1, 20)),
            "lagged_rapm_prior": np.linspace(-1.0, 1.0, 19),
        }
    )
    result = fit_prior_rapm_experiment(
        stints,
        priors,
        lambda_grid=(0.001, 0.1),
        split_config=ChronologicalSplitConfig(
            cv_folds=2, validation_fraction=0.15, test_fraction=0.15
        ),
        minimum_ranking_possessions=1.0,
    )

    assert result.selected_lambda in {0.001, 0.1}
    assert set(result.test_metrics["model"]) == {"mean", "prior_rapm"}
    unseen = result.player_priors.loc[result.player_priors["player_id"].eq(20)].iloc[0]
    assert unseen.prior_rapm_mean == 0.0
    assert not bool(unseen.prior_available)
    assert np.isfinite(result.test_predictions["prediction_prior_rapm"]).all()


def test_ranked_adjustments_remain_aligned_to_player_ids() -> None:
    stints = _stints(game_count=20)
    priors = pd.DataFrame(
        {
            "player_id": list(range(1, 21)),
            "lagged_rapm_prior": np.linspace(-1.0, 1.0, 20),
        }
    )
    result = fit_prior_rapm_experiment(
        stints,
        priors,
        lambda_grid=(0.001, 0.1),
        split_config=ChronologicalSplitConfig(
            cv_folds=2, validation_fraction=0.15, test_fraction=0.15
        ),
        minimum_ranking_possessions=1.0,
    )
    rankings = result.player_rankings
    assert np.allclose(
        rankings["rapm"].to_numpy(dtype=float),
        rankings["prior_rapm_mean"].to_numpy(dtype=float)
        + rankings["rapm_adjustment_from_prior"].to_numpy(dtype=float),
    )


def test_forward_history_uses_only_previous_season_estimates_as_priors() -> None:
    first = _stints(game_count=20)
    second = _stints(game_count=20).assign(
        game_id=lambda frame: frame["game_id"].str.replace("002250", "002260"),
        game_date=lambda frame: frame["game_date"] + timedelta(days=365),
        game_time_utc=lambda frame: frame["game_time_utc"] + timedelta(days=365),
    )

    results = fit_forward_lagged_rapm_history(
        {"2019-20": first, "2020-21": second},
        lambda_grid=(0.001, 0.1),
        split_config=ChronologicalSplitConfig(
            cv_folds=2, validation_fraction=0.15, test_fraction=0.15
        ),
    )

    first_estimates = results[0].player_estimates.set_index("player_id")["rapm"]
    second_priors = results[1].player_estimates.set_index("player_id")["prior_rapm"]
    assert results[0].player_estimates["prior_available"].sum() == 0
    assert second_priors.to_dict() == pytest.approx(first_estimates.to_dict())


def test_aging_prior_frame_is_label_free_and_preseason() -> None:
    source = pd.DataFrame(
        {
            "target_season": ["2025-26", "2025-26"],
            "player_id": [1, 2],
            "aging_prior_mean": [1.5, -0.5],
            "model_train_last_target_season": ["2024-25", "2024-25"],
        }
    )

    result = aging_prior_frame(source, target_season="2025-26")

    assert result.to_dict("list") == {
        "player_id": [1, 2],
        "lagged_rapm_prior": [1.5, -0.5],
    }
    with pytest.raises(ValueError, match="target-season outcomes"):
        aging_prior_frame(
            source.assign(target_rapm=[1.0, 2.0]),
            target_season="2025-26",
        )


def test_blended_prior_preserves_endpoint_models() -> None:
    aging = pd.DataFrame({"player_id": [1, 2], "lagged_rapm_prior": [1.0, -2.0]})
    lagged = pd.DataFrame({"player_id": [1, 2], "lagged_rapm_prior": [3.0, 4.0]})

    assert blended_prior_frame(aging, lagged, lagged_weight=0.0)[
        "lagged_rapm_prior"
    ].tolist() == pytest.approx([1.0, -2.0])
    assert blended_prior_frame(aging, lagged, lagged_weight=1.0)[
        "lagged_rapm_prior"
    ].tolist() == pytest.approx([3.0, 4.0])
    assert blended_prior_frame(aging, lagged, lagged_weight=0.25)[
        "lagged_rapm_prior"
    ].tolist() == pytest.approx([1.5, -0.5])


def _stints(game_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2025, 10, 21, tzinfo=UTC)
    for index in range(game_count):
        home_players = [1, 2, 3, 4, 5]
        away_players = [6, 7, 8, 9, 10 if index % 2 else 20]
        rating = 2.0 + index * 0.04
        rows.append(
            {
                "game_id": f"002250{index:04d}",
                "game_date": date(2025, 10, 21) + timedelta(days=index),
                "game_time_utc": start + timedelta(days=index),
                "stint_index": 0,
                "home_player_ids": home_players,
                "away_player_ids": away_players,
                "possessions": 10.0,
                "home_margin": rating,
                "target_home_net_rating": rating,
                "duration_seconds": 120.0,
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_tricode": "HOM",
                "away_team_tricode": "AWY",
            }
        )
    return pd.DataFrame(rows)
