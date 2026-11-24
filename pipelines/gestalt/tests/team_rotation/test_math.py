"""Team-only math and the legacy Pearson/undefined-value contract."""
from copy import deepcopy
import numpy as np
import pytest
from gestalt_pipelines.team_rotation.utils import summarize_team, validate_rows

A, B, C = 1610612737, 1610612738, 1610612739


def game(gid="0022600001", home=A, away=B, day="2026-10-20"):
    return {"game_id": gid, "home_team_id": home, "away_team_id": away, "game_date_pacific": day}


def row(index, ids=None, p=10, duration=120, gid="0022600001", home=A, away=B):
    return {"game_id": gid, "stint_index": index, "home_team_id": home, "away_team_id": away,
            "home_player_ids": ids or [1,2,3,4,5], "away_player_ids": [6,7,8,9,10],
            "possessions": p, "duration_seconds": duration}


def summary(rows, games=None, team=A, day="2026-10-20"):
    return summarize_team(season="2026-27", day=day, team_id=team, team="AAA",
                          games=games or [game()], rows=rows, names={i:f"Player {i}" for i in range(1,100)}, inputs={})


def test_team_only_pearson_shared_and_exposure():
    rows = [row(0,p=5), row(1,[1,2,3,4,11],p=10), row(2,p=15)]
    out = summary(rows)
    order = out["player_order"]
    signal = np.array([[r["possessions"] if pid in r["home_player_ids"] else 0 for pid in order] for r in rows])
    expected = np.corrcoef(signal,rowvar=False)
    assert np.allclose(out["floor_correlation"], expected)
    i,j=order.index(5),order.index(11)
    assert out["shared_possessions"][i][j]==0
    assert out["shared_court_similarity"][i][j]==0
    assert out["floor_correlation"][i][j]<0
    assert out["players"][i]["on_court_possessions"]==20
    # An unrelated club's added stints must have no numerical effect.
    other = row(0,[21,22,23,24,25],p=100,gid="0022600002",home=C,away=B)
    again=summary(rows+[other],games=[game(),game("0022600002",C,B)])
    for key in ("players","player_order","floor_correlation","shared_possessions","shared_court_similarity"):
        assert again[key]==out[key]


def test_zero_possessions_degenerate_diagonal_and_no_games():
    rows=[row(0),row(1),row(2,[1,2,3,4,99],p=0,duration=999)]
    out=summary(rows)
    assert 99 not in out["player_order"]
    assert np.allclose(out["floor_correlation"],np.eye(5))
    assert np.allclose(out["shared_court_similarity"],np.ones((5,5)))
    assert summary([row(0)])["floor_correlation"]==np.eye(5).tolist()
    empty=summary(rows,team=C)
    assert empty["players"]==empty["floor_correlation"]==[]
    assert empty["last_team_game_date_pacific"] is None


def test_jaccard_minutes_games_trades_ties_and_off_day():
    rows=[row(0),row(1,[1,2,3,4,11],duration=0),row(0,[1,2,3,4,12],gid="0022600002")]
    games=[game(),game("0022600002",A,C,"2026-10-22")]
    out=summary(rows,games)
    by_id={p["player_id"]:p for p in out["players"]}
    assert set(by_id)=={1,2,3,4,5,11,12}  # Retain departed player 5.
    assert by_id[11]["games"]==0 and by_id[1]["games"]==2
    assert by_id[1]["on_court_minutes"]==4
    i,j=out["player_order"].index(1),out["player_order"].index(5)
    assert out["shared_court_similarity"][i][j]==pytest.approx(1/3)
    off=summary(rows,games,day="2026-10-23")
    for key in ("players","floor_correlation","shared_possessions","shared_court_similarity"):
        assert off[key]==out[key]
    # Player's appearance after a trade only counts for the new club.
    moved=row(0,[1,31,32,33,34],gid="0022600003",home=B,away=C)
    old=summary(rows+[moved],games+[game("0022600003",B,C)])
    assert old["players"]==out["players"]


@pytest.mark.parametrize("change",[
    lambda r:r.update(home_player_ids=[1,1,2,3,4]),
    lambda r:r.update(possessions=-1), lambda r:r.update(duration_seconds=float("nan")),
    lambda r:r.update(home_team_id=C), lambda r:r.update(game_id="missing"),
    lambda r:r.update(possessions=0),
])
def test_invalid_rows_rejected(change):
    r=row(0);change(r)
    with pytest.raises(ValueError):validate_rows([r],[game()])


def test_duplicate_or_missing_games_rejected():
    with pytest.raises(ValueError):validate_rows([row(0),row(0)],[game()])
    with pytest.raises(ValueError):validate_rows([row(0)],[game(),game("0022600002")])
