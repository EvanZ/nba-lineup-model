from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.rotation.forward_plackett_luce_rotation import (
    MAX_ROTATION_PLAYERS,
    ForwardPlackettLuceConfig,
    predict_forward_conditional_minutes_control,
    predict_forward_plackett_luce,
    predict_forward_plackett_luce_sequence,
    prepare_forward_plackett_luce_panel,
    summarize_rotation_metrics,
)


def _available_minutes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    games = (
        ("A-1", "2025-01-01", 1, ((1, "Mover", 180.0), (2, "Home One", 60.0))),
        ("A-2", "2025-01-03", 1, ((1, "Mover", 170.0), (2, "Home One", 70.0))),
        ("B-1", "2025-01-05", 2, ((1, "Mover", 12.0), (3, "New Star", 228.0))),
    )
    for game_id, date, team_id, players in games:
        for player_id, name, minutes in players:
            rows.append(
                {
                    "season": "2024-25",
                    "game_id": game_id,
                    "game_date": pd.Timestamp(date, tz="UTC"),
                    "team_id": team_id,
                    "team": "AAA" if team_id == 1 else "BBB",
                    "player_id": player_id,
                    "player_name": name,
                    "minutes": minutes,
                    "actual_minute_share": minutes / 240.0,
                }
            )
    return pd.DataFrame(rows)


def _prior() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "predicted_minutes_per_available_game": [24.0, 20.0, 32.0],
        }
    )


def test_destination_team_keeps_player_role_residual() -> None:
    panel = prepare_forward_plackett_luce_panel(_available_minutes(), _prior())
    predictions = predict_forward_plackett_luce(
        panel, config=ForwardPlackettLuceConfig(role_prior_precision=1.0)
    )
    destination_first_game = predictions.loc[
        predictions["game_id"].eq("B-1") & predictions["player_id"].eq(1)
    ].iloc[0]
    assert destination_first_game["player_role_games_observed"] == 2
    assert destination_first_game["player_role_adjustment"] > 0.0
    assert destination_first_game["predicted_minutes_per_available_game"] == pytest.approx(24.0)


def test_current_game_outcome_cannot_change_its_pregame_prediction() -> None:
    panel = prepare_forward_plackett_luce_panel(_available_minutes(), _prior())
    changed = panel.copy()
    mask = changed["game_id"].eq("A-2")
    changed.loc[mask, "minutes"] = [20.0, 220.0]
    changed.loc[mask, "actual_minute_share"] = [20.0 / 240.0, 220.0 / 240.0]
    original = predict_forward_plackett_luce(
        panel, config=ForwardPlackettLuceConfig(role_prior_precision=1.0)
    )
    revised = predict_forward_plackett_luce(
        changed, config=ForwardPlackettLuceConfig(role_prior_precision=1.0)
    )
    before = original.loc[original["game_id"].eq("A-2"), "predicted_minutes"].to_numpy()
    after = revised.loc[revised["game_id"].eq("A-2"), "predicted_minutes"].to_numpy()
    assert np.allclose(before, after)


def test_allocations_obey_current_roster_and_top_15_cap() -> None:
    rows = []
    for player_id in range(1, 18):
        rows.append(
            {
                "season": "2024-25",
                "game_id": "wide",
                "game_date": pd.Timestamp("2025-01-01", tz="UTC"),
                "team_id": 1,
                "team": "AAA",
                "player_id": player_id,
                "player_name": str(player_id),
                "minutes": 240.0 / 17.0,
                "actual_minute_share": 1.0 / 17.0,
            }
        )
    prior = pd.DataFrame(
        {
            "player_id": list(range(1, 18)),
            "predicted_minutes_per_available_game": list(range(17, 0, -1)),
        }
    )
    panel = prepare_forward_plackett_luce_panel(pd.DataFrame(rows), prior)
    for predictions in (
        predict_forward_conditional_minutes_control(panel),
        predict_forward_plackett_luce(panel, config=ForwardPlackettLuceConfig(1.0)),
    ):
        assert predictions["predicted_minutes"].sum() == pytest.approx(240.0)
        assert predictions["selected_top_15"].sum() == MAX_ROTATION_PLAYERS
        assert predictions["predicted_minutes"].gt(0.0).sum() == MAX_ROTATION_PLAYERS
        metrics = summarize_rotation_metrics(predictions, model="test", season="2024-25")
        assert metrics.loc[0, "evaluated_team_games"] == 1


def test_prior_season_player_role_carries_to_new_team() -> None:
    source = pd.DataFrame(
        [
            {
                "season": "2023-24",
                "game_id": "source",
                "game_date": pd.Timestamp("2024-04-01", tz="UTC"),
                "team_id": 1,
                "team": "AAA",
                "player_id": player_id,
                "player_name": name,
                "minutes": minutes,
                "actual_minute_share": minutes / 240.0,
            }
            for player_id, name, minutes in (
                (1, "Returner", 180.0),
                (2, "Mate", 30.0),
                (3, "Mover", 30.0),
            )
        ]
    )
    target_rows: list[dict[str, object]] = []
    for game_id, team_id, team, players in (
        ("same-team", 1, "AAA", ((1, "Returner", 180.0), (2, "Mate", 60.0))),
        ("new-team", 2, "BBB", ((3, "Mover", 30.0), (4, "New Mate", 210.0))),
    ):
        for player_id, name, minutes in players:
            target_rows.append(
                {
                    "season": "2024-25",
                    "game_id": game_id,
                    "game_date": pd.Timestamp("2025-01-01", tz="UTC"),
                    "team_id": team_id,
                    "team": team,
                    "player_id": player_id,
                    "player_name": name,
                    "minutes": minutes,
                    "actual_minute_share": minutes / 240.0,
                }
            )
    prior = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "predicted_minutes_per_available_game": [24.0, 20.0, 22.0, 22.0],
        }
    )
    predictions = predict_forward_plackett_luce_sequence(
        {
            "2023-24": prepare_forward_plackett_luce_panel(source, prior),
            "2024-25": prepare_forward_plackett_luce_panel(pd.DataFrame(target_rows), prior),
        },
        config=ForwardPlackettLuceConfig(role_prior_precision=1.0, season_role_retention=1.0),
    )
    target = predictions["2024-25"]
    returner = target.loc[target["player_id"].eq(1)].iloc[0]
    mover = target.loc[target["player_id"].eq(3)].iloc[0]
    assert bool(returner["has_prior_season_player_role"])
    assert returner["prior_season_player_role_adjustment"] > 0.0
    assert returner["player_role_adjustment"] == pytest.approx(
        returner["prior_season_player_role_adjustment"]
    )
    assert bool(mover["has_prior_season_player_role"])
    assert mover["prior_season_player_role_adjustment"] < 0.0
    assert mover["player_role_adjustment"] == pytest.approx(
        mover["prior_season_player_role_adjustment"]
    )


def test_target_game_outcome_cannot_change_carried_opening_prediction() -> None:
    source = _available_minutes().loc[lambda frame: frame["team_id"].eq(1)].copy()
    source["season"] = "2023-24"
    target = _available_minutes().loc[lambda frame: frame["team_id"].eq(1)].copy()
    target["season"] = "2024-25"
    target["game_id"] = target["game_id"].str.replace("A-", "target-", regex=False)
    panels = {
        "2023-24": prepare_forward_plackett_luce_panel(source, _prior()),
        "2024-25": prepare_forward_plackett_luce_panel(target, _prior()),
    }
    changed_panels = {season: panel.copy() for season, panel in panels.items()}
    changed = changed_panels["2024-25"]
    first_target = changed["game_id"].eq("target-1")
    changed.loc[first_target, "minutes"] = [10.0, 230.0]
    changed.loc[first_target, "actual_minute_share"] = [10.0 / 240.0, 230.0 / 240.0]
    config = ForwardPlackettLuceConfig(role_prior_precision=1.0, season_role_retention=0.75)
    original = predict_forward_plackett_luce_sequence(panels, config=config)["2024-25"]
    revised = predict_forward_plackett_luce_sequence(changed_panels, config=config)["2024-25"]
    original_first = original.loc[
        original["game_id"].eq("target-1"), "predicted_minutes"
    ].to_numpy()
    revised_first = revised.loc[
        revised["game_id"].eq("target-1"), "predicted_minutes"
    ].to_numpy()
    assert np.allclose(original_first, revised_first)
