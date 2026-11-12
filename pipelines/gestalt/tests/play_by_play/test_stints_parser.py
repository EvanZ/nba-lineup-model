import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from gestalt_pipelines.play_by_play.utils import StintsError, parse_stints

FIXTURES = Path(__file__).parent/"fixtures"


def full_game():
    pbp = json.loads((FIXTURES/"playbyplay_lineup_scenario.json").read_text())
    box = json.loads((FIXTURES/"boxscore_lineup_scenario.json").read_text())
    extra = copy.deepcopy(pbp["game"]["actions"])
    for action in extra:
        action["period"] += 2
        action["actionNumber"] += 13
        action["orderNumber"] += 130000
        for key in ("scoreHome", "scoreAway"):
            action[key] = str(int(action[key])+2)
    pbp["game"]["actions"].extend(extra)
    box["game"].update({"gameStatus": 3, "period": 4})
    for side in ("home", "away"):
        box["game"][side+"Team"]["score"] = 4
        box["game"][side+"Team"]["teamTricode"] = side.upper()
    game = {
        "game_id": box["game"]["gameId"], "home_team_id": 100, "away_team_id": 200,
        "home_team_tricode": "HOME", "away_team_tricode": "AWAY",
        "game_time_utc": datetime(2026, 10, 21, 0, tzinfo=UTC),
    }
    return pbp, box, game


def test_shared_engine_reconstructs_full_timeline_and_fractional_possessions():
    pbp, box, game = full_game()
    rows, quality = parse_stints(json.dumps(pbp).encode(), json.dumps(box).encode(), game)
    assert sum(row["duration_seconds"] for row in rows) == 2880
    assert sum(row["points_home"] for row in rows) == 4
    assert sum(row["points_away"] for row in rows) == 4
    assert sum(row["possessions"] for row in rows)*2 == quality["team_possession_count"]
    assert any(row["home_offensive_possessions"] == 0.5 for row in rows)
    assert all(len(row["home_player_ids"]) == len(row["away_player_ids"]) == 5 for row in rows)


def test_game_identity_and_real_score_mismatches_fail():
    pbp, box, game = full_game()
    with pytest.raises(StintsError, match="game IDs"):
        parse_stints(json.dumps(pbp).encode(), json.dumps(box).encode(), {**game, "game_id": "0020000002"})
    box["game"]["homeTeam"]["score"] = 20
    with pytest.raises(StintsError, match="scoring"):
        parse_stints(json.dumps(pbp).encode(), json.dumps(box).encode(), game)


def test_zero_exposure_interval_is_retained_with_null_rate(monkeypatch):
    from gestalt_pipelines.play_by_play.utils import reconstruction as build_game
    pbp, box, game = full_game()
    home, away = (1,2,3,4,5), (6,7,8,9,10)
    def stint(index, start, end, first, last, points):
        return SimpleNamespace(
            stint_index=index, period=4, start_event_index=first, end_event_index=last,
            start_elapsed_game_seconds=start, end_elapsed_game_seconds=end, duration_seconds=end-start,
            home_player_ids=home, away_player_ids=away, points_home=points, points_away=0,
        )
    segment = SimpleNamespace(
        possession_id="p1", start_event_index=0, end_event_index=8,
        home_player_ids=home, away_player_ids=away, offense_team_id=100,
        points_home=2, points_away=0,
    )
    reconstructed = SimpleNamespace(
        lineups=SimpleNamespace(stints=[stint(0,0,2879,0,8,2),stint(1,2879,2880,9,10,0)],issues=[]),
        possessions=SimpleNamespace(possessions=[object()],issues=[]),
        possession_segments=SimpleNamespace(segments=[segment],issues=[]),
    )
    monkeypatch.setattr(build_game, "reconstruct_game_payloads", lambda *args:reconstructed)
    box["game"]["homeTeam"]["score"],box["game"]["awayTeam"]["score"] = 2,0
    rows,_=parse_stints(json.dumps(pbp).encode(),json.dumps(box).encode(),game)
    assert len(rows)==2 and rows[-1]["duration_seconds"]==1
    assert rows[-1]["possessions"]==0 and rows[-1]["target_home_net_rating"] is None


def test_overtime_keeps_the_additional_five_minutes():
    pbp,box,game=full_game()
    extra=copy.deepcopy(pbp["game"]["actions"][8:13])
    for a in extra:
        a["period"]=5
        a["periodType"]="OVERTIME"
        a["actionNumber"]+=26
        a["orderNumber"]+=260000
        if a["clock"]=="PT12M00.00S":a["clock"]="PT05M00.00S"
        if a["clock"]=="PT11M00.00S":a["clock"]="PT04M00.00S"
        a["scoreHome"]=str(int(a["scoreHome"])+2)
        a["scoreAway"]=str(int(a["scoreAway"])+4)
    pbp["game"]["actions"].extend(extra)
    box["game"]["period"]=5
    box["game"]["awayTeam"]["score"]=6
    rows,_=parse_stints(json.dumps(pbp).encode(),json.dumps(box).encode(),game)
    assert sum(r["duration_seconds"] for r in rows)==3180
    assert any(r["period"]==5 for r in rows)
