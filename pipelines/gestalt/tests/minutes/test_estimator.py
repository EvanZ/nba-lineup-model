from copy import deepcopy
from datetime import timedelta

import pytest

from gestalt_pipelines.minutes.utils import (
    MinutesError, classify, initialize_player, role_minutes, update_state, validate_state,
)
from .helpers import AWAY, FIRST, HOME, SEASON, priors, reference, slate


@pytest.mark.parametrize("status,reason,expected", [
    ("Available", "Injury/Illness - Ankle", "available_zero"),
    ("Available", "", "available_zero"),
    ("Probable", "Injury/Illness - Ankle", "medical_exclusion"),
    ("Questionable", "Injury/Illness - Ankle", "medical_exclusion"),
    ("Doubtful", "Injury/Illness - Ankle", "medical_exclusion"),
    ("Out", "Injury/Illness - Ankle", "medical_exclusion"),
    ("Out", "G League - Two- Way", "g_league_zero"),
    ("Available", "G League - On Assignment", "g_league_zero"),
])
def test_zero_classifications(status, reason, expected):
    assert classify(0, {"status": status, "reason": reason}) == expected
    assert classify(1, {"status": status, "reason": reason}) == "played"
    assert classify(0, None) == "available_zero"


def test_first_44_minutes_scales_and_excludes_late_play_and_overtime():
    data = slate()
    game = data["games"][0]
    roles = role_minutes(data["stints"]["rows"], game)
    assert roles[1] == 48
    assert roles[5] == 24
    assert roles[7] == 0
    data["stints"]["rows"].append({
        **data["stints"]["rows"][-1], "stint_index": 3,
        "start_elapsed_game_seconds": 2880, "end_elapsed_game_seconds": 3180,
    })
    assert role_minutes(data["stints"]["rows"], game) == roles
    data["stints"]["rows"][1]["start_elapsed_game_seconds"] += 1
    with pytest.raises(MinutesError, match="gap"):
        role_minutes(data["stints"]["rows"], game)


def test_four_dates_keep_only_three_observations_and_count_true_zeros():
    prior = priors()
    parent = None
    for n in range(4):
        day = FIRST+timedelta(days=n)
        data = slate(day, f"002260000{n+1}")
        data["inputs"]["parent"] = reference(f"parent{n}") if parent else None
        state = update_state(priors=prior, parent=parent, **data)
        count = min(n+1, 3)
        p = state["players"]["5"]
        assert len(p["history"]) == count
        assert p["estimate"] == pytest.approx((0.2*30+count*24)/(0.2+count))
        assert [x["game_id"] for x in p["history"]] == [
            f"002260000{i+1}" for i in range(max(0,n-2),n+1)
        ]
        assert state["players"]["1"]["history"][-1]["minutes"] == 42  # played clipping
        assert state["players"]["7"]["history"][-1]["minutes"] == 0   # late-only appearance
        for pid in ("15", "16", "19"):
            assert state["players"][pid]["history"][-1]["minutes"] == 0
        for pid in ("17", "18"):
            assert state["players"][pid]["history"] == []
            assert state["players"][pid]["estimate"] == 30
        assert len(state["players"]) == sum(len(v) for v in data["rosters"]["teams"].values())
        parent = state


def test_no_game_date_carries_every_window_without_restarting():
    prior = priors()
    parent = update_state(priors=prior, parent=None, **slate())
    data = slate(FIRST+timedelta(days=1), "0022600002")
    data["games"] = []
    data["injuries"]["games"] = {}
    data["stints"]["rows"] = []
    data["stints"]["validation"].update(expected_game_ids=[], processed_game_ids=[])
    state = update_state(priors=prior, parent=parent, **data)
    assert state["players"] == parent["players"]
    assert state["included_game_ids"] == parent["included_game_ids"]
    assert state["observations"] == []


def test_off_roster_gap_return_and_same_day_trade_preserve_identity():
    prior = priors()
    first = update_state(priors=prior, parent=None, **slate())
    data = slate(FIRST+timedelta(days=1), "0022600002")
    data["rosters"]["teams"][str(HOME)].remove(19)
    data["rosters"]["teams"][str(HOME)].remove(5)
    data["rosters"]["teams"][str(AWAY)].append(5)
    second = update_state(priors=prior, parent=first, **data)
    assert second["players"]["19"]["current_team_id"] is None
    assert second["players"]["19"]["history"] == first["players"]["19"]["history"]
    assert second["players"]["5"]["current_team_id"] == AWAY
    assert second["players"]["5"]["history"][-1]["team_id"] == HOME
    third = update_state(priors=prior, parent=second, **slate(FIRST+timedelta(days=2), "0022600003"))
    assert third["players"]["19"]["introduced_on"] == str(FIRST)
    assert len(third["players"]["19"]["history"]) == 2
    assert third["players"]["5"]["starting_minutes"] == first["players"]["5"]["starting_minutes"]


def test_cold_start_and_historical_initialization_keep_frozen_values():
    prior = priors()
    known = initialize_player(6, prior, day=str(FIRST))
    assert known["starting_minutes"] == 10
    assert known["initialization"]["branch"] == "frozen_cold_start"
    returned = initialize_player(999, prior, day=str(FIRST))
    assert returned["starting_minutes"] == 9
    assert returned["initialization"]["branch"] == "last_historical_available_game_role"
    new = initialize_player(998, prior, day=str(FIRST))
    assert new["starting_minutes"] == 10
    assert new["initialization"]["branch"] == "frozen_cohort_cold_start"


def test_missing_or_wrong_parent_duplicate_games_and_changed_priors_fail():
    prior = priors()
    parent = update_state(priors=prior, parent=None, **slate())
    data = slate(FIRST+timedelta(days=1), "0022600002")
    with pytest.raises(MinutesError, match="previous date"):
        update_state(priors=prior, parent=None, **data)
    data["inputs"]["priors"] = reference("changed")
    with pytest.raises(MinutesError, match="starting inputs changed"):
        update_state(priors=prior, parent=parent, **data)
    data["inputs"]["priors"] = reference("priors")
    bad = deepcopy(parent)
    bad["included_game_ids"].append("0022600002")
    with pytest.raises(MinutesError, match="already been applied"):
        update_state(priors=prior, parent=bad, **data)
    with pytest.raises(MinutesError, match="incompatible"):
        validate_state(parent, SEASON, "v2")


def test_corrupt_estimate_missing_slate_and_duplicate_roster_fail():
    prior = priors()
    state = update_state(priors=prior, parent=None, **slate())
    state["players"]["5"]["estimate"] += 1
    with pytest.raises(MinutesError, match="does not match"):
        validate_state(state, SEASON)
    data = slate()
    data["stints"]["validation"]["processed_game_ids"] = []
    with pytest.raises(MinutesError, match="full official slate"):
        update_state(priors=prior, parent=None, **data)
    data = slate()
    data["rosters"]["teams"][str(AWAY)].append(1)
    with pytest.raises(MinutesError, match="multiple current"):
        update_state(priors=prior, parent=None, **data)


def test_prior_source_ignores_forecasts_but_rejects_current_observed_games():
    import json
    from gestalt_pipelines.minutes.utils import build_priors
    frozen = json.dumps({"minutes": {
        "season": SEASON, "completed_season": "2025-26", "players": [{
            "player_id": 1, "player_name": "Player", "conditional_minutes_per_game": 30,
        }],
    }}).encode()
    historical = {"player_id": 9, "player_name": "Returner", "season": "2025-26",
                  "season_start_year": 2025, "actual_minutes_per_available_game": 17}
    forecast = {**historical, "season": SEASON, "season_start_year": 2026,
                "actual_minutes_per_available_game": None, "is_preseason_forecast": True}
    result = build_priors(frozen, [historical, forecast], season=SEASON,
                          history_reference=reference("history"), draft_reference=reference("draft"))
    assert result["historical_roles"]["9"]["starting_minutes"] == 17
    assert result["cold_start_minutes"] == 8
    assert result["sources"]["historical_roles"]["ignored_forecast_rows"] == 1
    with pytest.raises(MinutesError, match="Current/future-season"):
        build_priors(frozen, [historical, {**forecast, "is_preseason_forecast": False}], season=SEASON,
                     history_reference=reference("history"), draft_reference=reference("draft"))


@pytest.mark.parametrize("report,expected", [
    (None, "available_zero"),
    ({"status": "Available", "reason": "Illness"}, "available_zero"),
    ({"status": "Out", "reason": "G League - Two-Way"}, "g_league_zero"),
    ({"status": "Probable", "reason": "Ankle"}, "medical_exclusion"),
    ({"status": "Questionable", "reason": "Ankle"}, "medical_exclusion"),
    ({"status": "Doubtful", "reason": "Ankle"}, "medical_exclusion"),
    ({"status": "Out", "reason": "Ankle"}, "medical_exclusion"),
])
def test_ticket_74_rostered_player_absent_from_stints_uses_injury_rule(report, expected):
    data = slate()
    if report is not None:
        data["injuries"]["games"]["0022600001"]["teams"][str(HOME)]["players"]["19"] = report
    state = update_state(priors=priors(), parent=None, **data)
    observation = next(row for row in state["observations"] if row["player_id"] == 19)
    assert observation["classification"] == expected
    assert observation["actual_minutes"] == 0 and observation["role_minutes"] == 0
    if expected == "medical_exclusion":
        assert observation["minutes"] is None and not state["players"]["19"]["history"]
    else:
        assert observation["minutes"] == 0
        assert state["players"]["19"]["history"][-1]["minutes"] == 0


def test_ticket_74_only_rostered_players_get_observations():
    data = slate()
    # Player 7 appears in valid stints but is not on the selected roster.
    data["rosters"]["teams"][str(HOME)].remove(7)
    # Player 999 appears only in an injury report, also outside the roster.
    data["injuries"]["games"]["0022600001"]["teams"][str(HOME)]["players"]["999"] = {
        "status": "Available", "reason": "",
    }
    state = update_state(priors=priors(), parent=None, **data)
    expected = set(data["rosters"]["teams"][str(HOME)] + data["rosters"]["teams"][str(AWAY)])
    assert {row["player_id"] for row in state["observations"]} == expected
    assert "7" not in state["players"] and "999" not in state["players"]


@pytest.mark.parametrize("overtime", [False, True])
def test_played_outside_first_44_is_still_played_and_legacy_clipping_is_preserved(overtime):
    data = slate()
    rows = data["stints"]["rows"]
    if overtime:
        rows.append({**rows[-1], "stint_index": 3, "start_elapsed_game_seconds": 2880,
                     "end_elapsed_game_seconds": 3180})
    rows[-1]["home_player_ids"] = [2, 3, 4, 5, 19]
    data["injuries"]["games"]["0022600001"]["teams"][str(HOME)]["players"]["19"] = {
        "status": "Out", "reason": "Injury",
    }
    state = update_state(priors=priors(), parent=None, **data)
    observation = next(row for row in state["observations"] if row["player_id"] == 19)
    assert observation["actual_minutes"] == (5 if overtime else 4)
    assert observation["role_minutes"] == 0 and observation["classification"] == "played"
    assert observation["minutes"] == 18  # Existing prior 30 minus the agreed played clipping of 12.
