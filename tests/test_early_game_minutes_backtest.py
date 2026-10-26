from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.modeling.early_game_minutes_backtest import (
    chronological_evaluation,
    cutoff_role_observations,
)
from nba_lineup_model.modeling.in_season_minutes import (
    COACH_DNP,
    MEDICAL_ABSENCE,
    PLAYED,
)


def _availability() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game_id, game_date in (("g1", "2025-10-20"), ("g2", "2025-10-21")):
        for team, offset in (("AAA", 0), ("BBB", 10)):
            for index in range(6):
                minutes = 36.0 if index < 5 else 60.0
                if index == 5:
                    minutes = 60.0
                rows.append(
                    {
                        "game_id": game_id,
                        "game_date": game_date,
                        "team": team,
                        "player_id": offset + index + 1,
                        "player_name": f"P{offset + index + 1}",
                        "minutes": minutes,
                        "availability_state": PLAYED,
                    }
                )
    # Each team's source minutes must total 240; five starters play 36 and the
    # replacement plays 60 here only to make the normalization target distinct.
    return pd.DataFrame(rows)


def _stints() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game_id in ("g1", "g2"):
        rows.extend(
            [
                {
                    "game_id": game_id,
                    "start_elapsed_game_seconds": 0.0,
                    "end_elapsed_game_seconds": 36.0 * 60.0,
                    "home_player_ids": [1, 2, 3, 4, 5],
                    "away_player_ids": [11, 12, 13, 14, 15],
                    "catalog_home_team_tricode": "AAA",
                    "catalog_away_team_tricode": "BBB",
                },
                {
                    "game_id": game_id,
                    "start_elapsed_game_seconds": 36.0 * 60.0,
                    "end_elapsed_game_seconds": 48.0 * 60.0,
                    "home_player_ids": [2, 3, 4, 5, 6],
                    "away_player_ids": [12, 13, 14, 15, 16],
                    "catalog_home_team_tricode": "AAA",
                    "catalog_away_team_tricode": "BBB",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_cutoff_roles_normalize_early_minutes_and_conserve_team_total() -> None:
    panel, audit = cutoff_role_observations(
        _stints(), _availability(), cutoffs=(36, 48)
    )

    first_game = panel.loc[panel["game_id"].eq("g1") & panel["team"].eq("AAA")]
    starter = first_game.loc[first_game["player_id"].eq(1)].set_index("cutoff_minutes")
    replacement = first_game.loc[first_game["player_id"].eq(6)].set_index(
        "cutoff_minutes"
    )
    assert starter.loc[36, "observed_role_minutes"] == pytest.approx(48.0)
    assert starter.loc[48, "observed_role_minutes"] == pytest.approx(36.0)
    assert replacement.loc[36, "observed_role_minutes"] == pytest.approx(0.0)
    assert replacement.loc[48, "observed_role_minutes"] == pytest.approx(12.0)
    totals = first_game.groupby("cutoff_minutes")["observed_role_minutes"].sum()
    assert totals.to_dict() == pytest.approx({36: 240.0, 48: 240.0})
    assert audit["common_complete_games"] == 2


def test_stint_participation_overrides_incorrect_absence_label() -> None:
    availability = _availability()
    availability.loc[
        availability["player_id"].eq(1), "availability_state"
    ] = MEDICAL_ABSENCE

    panel, audit = cutoff_role_observations(
        _stints(), availability, cutoffs=(36, 48)
    )

    player = panel.loc[panel["player_id"].eq(1)]
    assert player["availability_state"].eq(PLAYED).all()
    assert audit["played_state_overrides_from_stints"] == 2


def test_chronological_evaluation_uses_only_prior_dates() -> None:
    panel = pd.DataFrame(
        {
            "game_id": ["g1", "g1b", "g2"],
            "game_date": pd.to_datetime(
                ["2025-10-20", "2025-10-20", "2025-10-21"], utc=True
            ),
            "team": ["AAA", "AAA", "AAA"],
            "player_id": [1, 1, 1],
            "player_name": ["One", "One", "One"],
            "availability_state": [PLAYED, PLAYED, PLAYED],
            "target_minutes": [36.0, 36.0, 36.0],
            "cutoff_minutes": [48, 48, 48],
            "observed_role_minutes": [36.0, 24.0, 48.0],
        }
    )
    priors = pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["One"],
            "predicted_minutes_per_available_game": [30.0],
        }
    )

    result = chronological_evaluation(
        panel,
        priors,
        cutoff_minutes=48,
        lookback_games=2,
        prior_weight=0.2,
    )

    same_day = result.loc[result["game_date"].eq(pd.Timestamp("2025-10-20", tz="UTC"))]
    assert same_day["updated_role_minutes"].tolist() == pytest.approx([30.0, 30.0])
    next_day = result.loc[result["game_id"].eq("g2")].iloc[0]
    assert next_day["updated_role_minutes"] == pytest.approx(
        (0.2 * 30.0 + 36.0 + 24.0) / 2.2
    )
    assert next_day["prior_observations"] == 2


def test_chronological_evaluation_excludes_medical_absence_but_keeps_coach_dnp() -> None:
    panel = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "game_date": pd.to_datetime(
                ["2025-10-20", "2025-10-20", "2025-10-21", "2025-10-21"],
                utc=True,
            ),
            "team": ["AAA"] * 4,
            "player_id": [1, 2, 1, 2],
            "player_name": ["One", "Two", "One", "Two"],
            "availability_state": [MEDICAL_ABSENCE, COACH_DNP, PLAYED, PLAYED],
            "target_minutes": [0.0, 0.0, 20.0, 20.0],
            "cutoff_minutes": [48] * 4,
            "observed_role_minutes": [0.0, 0.0, 20.0, 20.0],
        }
    )
    priors = pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["One", "Two"],
            "predicted_minutes_per_available_game": [10.0, 10.0],
        }
    )

    result = chronological_evaluation(
        panel,
        priors,
        cutoff_minutes=48,
        lookback_games=2,
        prior_weight=1.0,
    )

    assert not ((result["game_id"] == "g1") & (result["player_id"] == 1)).any()
    coach = result.loc[(result["game_id"] == "g1") & (result["player_id"] == 2)]
    assert coach.iloc[0]["observed_role_minutes"] == 0.0
    next_game = result.loc[(result["game_id"] == "g2") & (result["player_id"] == 2)]
    assert next_game.iloc[0]["updated_role_minutes"] == pytest.approx(5.0)
