"""Numerical, date and downstream JSON contracts."""
import json
import pytest
from gestalt_pipelines.predictions.utils import (
    PredictionError, build_prediction, json_bytes, validate_prediction, win_probability,
)


def test_margin_uses_exact_minutes_weights_hca_and_fixed_b2b(data):
    r = build_prediction(**data)
    values = {p["player_id"]: p["nail"] for p in data["ratings"]["players"]}
    totals = {side: sum(p["projected_minutes"]/48*values[p["player_id"]]
              for p in r[side]["players"]) for side in ("home", "away")}
    expected = totals["home"]-totals["away"]+data["ratings"]["hca"]["estimate"]-1.5087106592522943
    assert r["predicted_home_margin"] == expected
    assert r["home_win_probability"] == min(1, max(0, .499583+.030250*expected))
    assert r["assumed_possessions"] == 100
    data["b2b"] = {"home": False, "away": True}
    assert build_prediction(**data)["components"]["b2b_adjustment"] == 1.5087106592522943
    data["b2b"] = {"home": True, "away": True}
    assert build_prediction(**data)["components"]["b2b_adjustment"] == 0


def test_full_precision_and_every_upstream_display_field_survives(data):
    data["ratings"]["players"][0]["nail"] = 1.2345678901234567
    record = json.loads(json_bytes(build_prediction(**data)))
    assert record["home"]["players"][0]["nail"] == 1.2345678901234567
    for side in ("home", "away"):
        source = data["minutes"]["teams"][str(record[side]["team_id"])]
        assert record[side]["total_projected_minutes"] == source["total_projected_minutes"]
        assert record[side]["submission_status"] == source["submission_status"]
        for incoming, outgoing in zip(source["players"], record[side]["players"]):
            assert all(outgoing[k] == incoming[k] for k in (
                "player_id", "name", "conditional_minutes", "projected_minutes", "injury_status",
                "injury_reason", "play_probability", "selected_rotation"))
    assert record["injury_report"] == data["report"]


def test_saved_prior_then_replacement_and_daily_rating_precedence(data):
    players = data["ratings"]["players"]
    removed = [players.pop(0), players.pop(0)]
    data["priors"] = {"players": [{"player_id": removed[0]["player_id"], "rapm": 4.123456789012345},
                                   {"player_id": players[0]["player_id"], "rapm": 99}]}
    record = build_prediction(**data)
    rows = {p["player_id"]: p for side in ("home", "away") for p in record[side]["players"]}
    assert rows[removed[0]["player_id"]]["nail"] == 4.123456789012345
    assert rows[removed[0]["player_id"]]["rating_source"] == "frozen_prior"
    assert rows[removed[1]["player_id"]]["nail"] == -5.109488431706889
    assert rows[removed[1]["player_id"]]["rating_source"] == "replacement"
    assert rows[players[0]["player_id"]]["nail"] == players[0]["nail"]


@pytest.mark.parametrize("margin, expected", [(0, .499583), (100, 1), (-100, 0), (5, .650833)])
def test_probability(margin, expected):
    assert win_probability(margin) == expected


@pytest.mark.parametrize("bad", ["future_ratings", "future_minutes", "duplicate", "margin", "source", "nan"])
def test_invalid_inputs_or_changed_outputs_are_rejected(data, bad):
    if bad == "future_ratings":
        data["ratings"]["observed_through_game_date_pacific"] = data["day"]
        with pytest.raises(PredictionError):
            build_prediction(**data)
        return
    r = build_prediction(**data)
    if bad == "future_minutes": r["minutes_observed_through_game_date_pacific"] = data["day"]
    if bad == "duplicate": r["away"]["players"][0]["player_id"] = r["home"]["players"][0]["player_id"]
    if bad == "margin": r["predicted_home_margin"] += 1
    if bad == "source": r["injury_report"]["source_url"] = "https://espn.com"
    if bad == "nan": r["home"]["players"][0]["nail"] = float("nan")
    with pytest.raises(ValueError):
        validate_prediction(r, data["season"])
