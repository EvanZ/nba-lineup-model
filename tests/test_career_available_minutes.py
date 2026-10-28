from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.career_available_minutes import (
    CareerWeightConfig,
    build_career_available_feature,
    career_available_totals,
    career_prior_weight,
    chronological_components,
    score_components,
    score_current_season_rows,
)
from nba_lineup_model.modeling.in_season_minutes import (
    COACH_DNP,
    G_LEAGUE_ASSIGNMENT,
    INACTIVE,
    MEDICAL_ABSENCE,
    PLAYED,
    UNKNOWN,
)


def test_career_available_feature_counts_all_healthy_opportunities() -> None:
    history = pd.DataFrame(
        {
            "player_id": [1] * 7,
            "game_id": [
                "played",
                "coach",
                "g-league",
                "medical",
                "inactive",
                "unknown",
                "next",
            ],
            "game_date": [
                "2025-10-20",
                "2025-10-21",
                "2025-10-22",
                "2025-10-23",
                "2025-10-24",
                "2025-10-25",
                "2025-10-26",
            ],
            "availability_state": [
                PLAYED,
                COACH_DNP,
                G_LEAGUE_ASSIGNMENT,
                MEDICAL_ABSENCE,
                INACTIVE,
                UNKNOWN,
                PLAYED,
            ],
        }
    )

    result = build_career_available_feature(history).set_index("game_id")

    assert result["career_available_games"].to_dict() == {
        "played": 0,
        "coach": 1,
        "g-league": 2,
        "medical": 3,
        "inactive": 3,
        "unknown": 3,
        "next": 3,
    }


def test_medically_lost_rookie_season_adds_zero_exposure() -> None:
    history = pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "game_id": ["lost-1", "lost-2", "debut"],
            "game_date": ["2024-10-20", "2025-04-10", "2025-10-20"],
            "availability_state": [MEDICAL_ABSENCE, MEDICAL_ABSENCE, PLAYED],
        }
    )

    result = build_career_available_feature(history).set_index("game_id")

    assert result.loc["debut", "career_available_games"] == 0


def test_same_date_games_share_pregame_exposure() -> None:
    history = pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "game_id": ["same-a", "same-b", "tomorrow"],
            "game_date": ["2025-10-20", "2025-10-20", "2025-10-21"],
            "availability_state": [PLAYED, COACH_DNP, PLAYED],
        }
    )

    result = build_career_available_feature(history).set_index("game_id")

    assert result.loc["same-a", "career_available_games"] == 0
    assert result.loc["same-b", "career_available_games"] == 0
    assert result.loc["tomorrow", "career_available_games"] == 2


def test_career_available_totals_share_the_same_classification_contract() -> None:
    history = pd.DataFrame(
        {
            "player_id": [1] * 6,
            "game_id": ["played", "coach", "g-league", "medical", "inactive", "unknown"],
            "availability_state": [
                PLAYED,
                COACH_DNP,
                "available_g_league_assignment",
                MEDICAL_ABSENCE,
                INACTIVE,
                UNKNOWN,
            ],
        }
    )

    assert career_available_totals(history).to_dict() == {1: 3}


def test_weight_curve_is_log_monotonic_and_clipped() -> None:
    config = CareerWeightConfig(w_min=0.1, beta=0.5, w_max=1.0)
    games = np.array([0, 10, 25, 50, 82, 164, 410, 820], dtype=float)

    weights = career_prior_weight(games, config)

    assert weights[0] == pytest.approx(0.1)
    assert weights[4] == pytest.approx(0.1 + 0.5 * np.log(2.0))
    assert weights[-1] == pytest.approx(1.0)
    assert np.all(np.diff(weights) >= 0.0)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (CareerWeightConfig(0.0, 0.5, 1.0), "w_min"),
        (CareerWeightConfig(0.1, -0.1, 1.0), "beta"),
        (CareerWeightConfig(0.2, 0.1, 0.1), "w_max"),
    ],
)
def test_weight_curve_rejects_invalid_parameters(config: CareerWeightConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        career_prior_weight(10, config)


def test_chronological_components_do_not_leak_same_date_roles() -> None:
    panel = pd.DataFrame(
        {
            "game_id": ["g1", "g1b", "g2"],
            "game_date": pd.to_datetime(["2025-10-20", "2025-10-20", "2025-10-21"], utc=True),
            "team": ["AAA"] * 3,
            "player_id": [1, 1, 1],
            "player_name": ["One"] * 3,
            "availability_state": [PLAYED] * 3,
            "target_minutes": [30.0] * 3,
            "cutoff_minutes": [44] * 3,
            "observed_role_minutes": [20.0, 40.0, 30.0],
        }
    )
    priors = pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["One"],
            "predicted_minutes_per_available_game": [25.0],
        }
    )
    career = pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "game_id": ["g1", "g1b", "g2"],
            "career_available_games": [0, 0, 2],
        }
    )

    components = chronological_components(panel, priors, career)
    scored = score_components(components, CareerWeightConfig(w_min=0.1, beta=0.5, w_max=2.0))

    same_day = scored.loc[scored["game_date"].eq(pd.Timestamp("2025-10-20", tz="UTC"))]
    assert same_day["candidate_role_minutes"].tolist() == pytest.approx([25.0, 25.0])
    next_day = scored.loc[scored["game_id"].eq("g2")].iloc[0]
    weight = career_prior_weight(2, CareerWeightConfig(0.1, 0.5, 2.0))
    assert next_day["candidate_role_minutes"] == pytest.approx(
        (weight * 25.0 + 20.0 + 40.0) / (weight + 2.0)
    )
    assert next_day["prior_observations"] == 2


def test_current_season_replay_reproduces_control_and_normalizes_candidate() -> None:
    rows: list[dict[str, object]] = []
    for game_id, game_date, observed in (
        ("g1", "2026-10-20", 20.0),
        ("g2", "2026-10-21", 40.0),
    ):
        for player_id in range(1, 6):
            history = [] if game_id == "g1" else [20.0]
            control = (0.2 * 30.0 + sum(history)) / (0.2 + len(history))
            rows.append(
                {
                    "game_id": game_id,
                    "game_date": game_date,
                    "team": "AAA",
                    "player_id": player_id,
                    "outcome_classification": PLAYED,
                    "early_game_role_minutes": observed,
                    "frozen_conditional_minutes": 30.0,
                    "updated_conditional_minutes": control,
                    "updated_projected_minutes": 48.0,
                }
            )
    evaluation = pd.DataFrame(rows)

    scored, audit = score_current_season_rows(
        evaluation,
        pd.Series({player_id: 0 for player_id in range(1, 6)}),
        CareerWeightConfig(0.05, 0.1, 0.5),
    )

    assert audit["control_reproduction_max_absolute_error"] == pytest.approx(0.0)
    totals = scored.groupby(["game_id", "team"])["candidate_projected_minutes"].sum()
    assert totals.to_dict() == pytest.approx({("g1", "AAA"): 240.0, ("g2", "AAA"): 240.0})
