from __future__ import annotations

import math

import pandas as pd
import pytest

from nba_lineup_model.rotation.l20_preseason_minute_share_persistence import (
    summarize_l20_metrics,
)
from nba_lineup_model.rotation.l20_source_aware_preseason_minute_share import (
    SourceAwareConfig,
    SourceAwareSeasonInputs,
    _build_role_state,
    _predict_allocation,
    build_source_aware_features,
    evaluate_source_aware_minute_share,
    fit_source_aware_minute_model,
)


def _role_state() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "team": ["MIN", "MIN", "MIN"],
            "player_id": [1, 2, 3],
            "player_name": ["Continuous", "Returner", "Rookie"],
            "player_state": ["continuous", "gap_returner", "cold_start"],
            "gap_seasons": [0.0, 1.0, 0.0],
            "source_minutes": [1600.0, 1200.0, 0.0],
            "source_games": [64.0, 60.0, 0.0],
            "source_starts": [52.0, 40.0, 0.0],
            "preseason_nail": [2.0, 1.0, 0.5],
            "draft_capital": [0.0, 0.0, 0.9],
            "is_undrafted": [False, False, False],
            "age_offset": [4.0, 3.0, -2.0],
            "is_guard": [1.0, 0.0, 1.0],
            "is_forward": [0.0, 1.0, 0.0],
            "is_center": [0.0, 0.0, 0.0],
        }
    )


def _target_games() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for player_id, share in {1: 0.55, 2: 0.35, 3: 0.10}.items():
        rows.append(
            {
                "game_id": "g1",
                "team_id": 1,
                "team_tricode": "MIN",
                "player_id": player_id,
                "player_name": str(player_id),
                "minutes": share * 240.0,
                "game_time_utc": "2025-10-21T00:00:00Z",
            }
        )
    return pd.DataFrame(rows)


def test_feature_construction_keeps_low_exposure_players_off_the_league_mean() -> None:
    config = SourceAwareConfig(role_shrinkage_games=20.0, gap_decay=0.5, alpha=1.0)
    features = build_source_aware_features(_role_state(), config=config).set_index("player_id")

    assert features.loc[1, "log_total_minutes"] > features.loc[2, "log_total_minutes"]
    assert features.loc[2, "log_total_minutes"] == pytest.approx(
        0.5 * math.log1p(1200.0)
    )
    assert features.loc[3, "log_total_minutes"] == 0.0
    assert features.loc[3, "cold_draft_capital"] == pytest.approx(0.9)
    assert features.loc[3, "shrunken_mpg"] == 0.0


def test_gap_returner_uses_last_observed_role_and_rookie_is_a_separate_state() -> None:
    candidates = _role_state().loc[:, ["team_id", "team", "player_id", "player_name"]]
    immediate = pd.DataFrame(
        {
            "player_id": [1],
            "source_minutes": [1600.0],
            "source_games": [64.0],
            "source_starts": [52.0],
        }
    )
    history = [
        pd.DataFrame(
            {
                "player_id": [2],
                "source_minutes": [1200.0],
                "source_games": [60.0],
                "source_starts": [40.0],
            }
        )
    ]
    panel = pd.DataFrame(
        {
            "season": ["2025-26", "2025-26", "2025-26"],
            "player_id": [1, 2, 3],
            "draft_number": [10.0, 12.0, 6.0],
            "is_undrafted": [False, False, False],
            "age": [24.0, 25.0, 20.0],
            "listed_position": ["G", "F", "G"],
        }
    )

    states = _build_role_state(
        "2025-26",
        candidates=candidates,
        immediate=immediate,
        history=history,
        panel=panel,
        preseason_nail=pd.Series({1: 2.0, 2: 1.0, 3: 0.5}),
    ).set_index("player_id")

    assert states.loc[1, "player_state"] == "continuous"
    assert states.loc[2, "player_state"] == "gap_returner"
    assert states.loc[2, "source_minutes"] == pytest.approx(1200.0)
    assert states.loc[2, "gap_seasons"] == pytest.approx(1.0)
    assert states.loc[3, "player_state"] == "cold_start"


def test_fitted_source_aware_model_returns_a_valid_team_distribution() -> None:
    inputs = SourceAwareSeasonInputs(
        season="2025-26",
        target_game_minutes=_target_games(),
        opening_candidates=_role_state().loc[:, ["team_id", "team", "player_id", "player_name"]],
        role_state=_role_state(),
        preseason_nail_ratings=pd.Series({1: 2.0, 2: 1.0, 3: 0.5}),
    )
    config = SourceAwareConfig(role_shrinkage_games=20.0, gap_decay=0.5, alpha=1.0)

    model = fit_source_aware_minute_model([inputs], config=config, team_games=1)
    predictions, metrics = evaluate_source_aware_minute_share(inputs, model=model, team_games=1)

    assert predictions["predicted_minute_share"].sum() == pytest.approx(1.0)
    assert metrics.loc[0, "allocation_total_variation"] >= 0.0
    assert not metrics.loc[0, "has_zero_probability_active_player"]
    assert summarize_l20_metrics(metrics, season="2025-26").loc[0, "evaluated_teams"] == 1


def test_replacement_token_models_outside_opening_roster_minutes() -> None:
    target = _target_games()
    target.loc[target["player_id"].eq(3), "minutes"] = 0.0
    target = pd.concat(
        [
            target,
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "team_id": 1,
                        "team_tricode": "MIN",
                        "player_id": 99,
                        "player_name": "Arrival",
                        "minutes": 24.0,
                        "game_time_utc": "2025-10-21T00:00:00Z",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    inputs = SourceAwareSeasonInputs(
        season="2025-26",
        target_game_minutes=target,
        opening_candidates=_role_state().loc[:, ["team_id", "team", "player_id", "player_name"]],
        role_state=_role_state(),
        preseason_nail_ratings=pd.Series({1: 2.0, 2: 1.0, 3: 0.5}),
    )
    config = SourceAwareConfig(role_shrinkage_games=20.0, gap_decay=0.5, alpha=1.0)

    model = fit_source_aware_minute_model(
        [inputs], config=config, team_games=1, replacement_token=True
    )
    allocation, replacement_share = _predict_allocation(
        build_source_aware_features(_role_state(), config=config), model
    )
    predictions, metrics = evaluate_source_aware_minute_share(inputs, model=model, team_games=1)

    assert sum(allocation.values()) + replacement_share == pytest.approx(1.0)
    assert metrics.loc[0, "replacement_token_actual_share"] == pytest.approx(0.1)
    assert metrics.loc[0, "replacement_token_predicted_share"] > 0.0
    token = predictions.loc[predictions["entity_type"].eq("replacement_token")].iloc[0]
    assert token["actual_minute_share"] == pytest.approx(0.1)
