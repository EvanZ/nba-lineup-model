from __future__ import annotations

import pandas as pd

from nba_lineup_model.players.availability import (
    STATE_AVAILABLE_DNP_COACH,
    STATE_PLAYED,
    STATE_UNAVAILABLE_INJURY,
    STATE_UNAVAILABLE_REST,
)
from nba_lineup_model.rotation.availability_episode_mart import (
    build_availability_episode_frame,
    build_availability_episode_season_summary,
)


def _player_games(states: list[str]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(states), freq="2D", tz="UTC")
    return pd.DataFrame(
        {
            "season": "2024-25",
            "game_id": [f"g{index}" for index in range(len(states))],
            "game_date": dates,
            "team_id": 1,
            "team": "IND",
            "player_id": 202691,
            "player_name": "Tyrese Haliburton",
            "availability_state": states,
            "available": [
                state not in {STATE_UNAVAILABLE_INJURY, STATE_UNAVAILABLE_REST}
                for state in states
            ],
            "availability_state_known": True,
            "minutes": [30.0 if state == STATE_PLAYED else 0.0 for state in states],
        }
    )


def test_bridges_isolated_coach_dnp_inside_availability_episode() -> None:
    games = _player_games(
        [
            STATE_PLAYED,
            STATE_UNAVAILABLE_INJURY,
            STATE_AVAILABLE_DNP_COACH,
            STATE_UNAVAILABLE_INJURY,
            STATE_PLAYED,
        ]
    )

    episodes = build_availability_episode_frame(games)

    assert len(episodes) == 1
    episode = episodes.iloc[0]
    assert episode["unavailable_games"] == 3
    assert episode["reconciled_coach_dnp_games"] == 1
    assert not episode["right_censored"]


def test_distinct_medical_absences_remain_separate_episodes() -> None:
    games = _player_games(
        [STATE_UNAVAILABLE_INJURY, STATE_PLAYED, STATE_UNAVAILABLE_INJURY]
    )

    episodes = build_availability_episode_frame(games)

    assert len(episodes) == 2
    assert episodes["unavailable_games"].tolist() == [1, 1]
    assert episodes["right_censored"].tolist() == [False, True]


def test_binary_unavailable_contract_includes_non_injury_states() -> None:
    games = _player_games([STATE_PLAYED, STATE_UNAVAILABLE_REST, STATE_PLAYED])

    episodes = build_availability_episode_frame(games)

    assert len(episodes) == 1
    assert episodes.iloc[0]["unavailable_games"] == 1


def test_availability_episode_continues_across_seasons() -> None:
    games = _player_games([STATE_UNAVAILABLE_INJURY, STATE_UNAVAILABLE_INJURY, STATE_PLAYED])
    games.loc[1, "season"] = "2025-26"
    games.loc[2, "season"] = "2025-26"

    episodes = build_availability_episode_frame(games)
    summary = build_availability_episode_season_summary(games, episodes)

    assert len(episodes) == 1
    episode = episodes.iloc[0]
    assert episode["episode_start_season"] == "2024-25"
    assert episode["episode_end_season"] == "2025-26"
    assert episode["unavailable_games"] == 2
    values = summary.set_index("season")
    assert values.loc["2024-25", "availability_episode_count"] == 1
    assert values.loc["2025-26", "availability_episode_count"] == 0
    assert values.loc["2025-26", "unavailable_games"] == 1


def test_player_season_summary_uses_episode_counts_and_panel_gp_gs() -> None:
    games = _player_games(
        [
            STATE_PLAYED,
            STATE_UNAVAILABLE_INJURY,
            STATE_AVAILABLE_DNP_COACH,
            STATE_UNAVAILABLE_INJURY,
            STATE_PLAYED,
        ]
    )
    episodes = build_availability_episode_frame(games)
    panel = pd.DataFrame(
        {
            "season": ["2024-25"],
            "player_id": [202691],
            "games": [70],
            "games_started": [69],
            "minutes": [2300.0],
        }
    )

    summary = build_availability_episode_season_summary(games, episodes, player_panel=panel).iloc[0]

    assert summary["availability_episode_count"] == 1
    assert summary["unavailable_games"] == 3
    assert summary["reconciled_coach_dnp_games"] == 1
    assert summary["panel_gp"] == 70
    assert summary["panel_gs"] == 69
