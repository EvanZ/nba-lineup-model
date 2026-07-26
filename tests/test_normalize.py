from __future__ import annotations

import json
from pathlib import Path

from nba_lineup_model.events.normalize import events_frame
from nba_lineup_model.normalize.boxscore import boxscore_players_frame
from nba_lineup_model.normalize.play_by_play import play_by_play_actions_frame

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_play_by_play_actions_frame():
    frame = play_by_play_actions_frame(load_fixture("playbyplay_minimal.json"))

    assert list(frame["game_id"].unique()) == ["0022000180"]
    assert len(frame) == 2
    assert frame.loc[1, "actionType"] == "3pt"
    assert frame.loc[1, "teamTricode"] == "BOS"


def test_events_frame_uses_canonical_event_names():
    frame = events_frame(load_fixture("playbyplay_minimal.json"))

    assert list(frame["event_type"]) == ["period", "3pt"]
    assert list(frame["score_home_delta"]) == [0, 3]
    assert "source_order_number" in frame.columns


def test_boxscore_players_frame():
    frame = boxscore_players_frame(load_fixture("boxscore_minimal.json"))

    assert len(frame) == 2
    assert set(frame["team_side"]) == {"home", "away"}
    assert "statistics_points" in frame.columns
    assert frame.loc[frame["team_side"] == "home", "team_tricode"].item() == "BOS"
