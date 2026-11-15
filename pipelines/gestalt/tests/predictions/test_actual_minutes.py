"""Full-game counterfactual arithmetic and separate immutable publication."""
import json
from copy import deepcopy
from datetime import timedelta

import pytest
from botocore.exceptions import ClientError
from gestalt_pipelines.predictions.utils import (
    PredictionError, actual_minutes_fingerprint, build_actual_minutes_prediction,
    json_bytes, validate_actual_minutes_prediction,
)
from .conftest import NOW


@pytest.fixture
def box(record):
    value = {"gameId": record["game_id"], "gameStatus": 3}
    for side, score in (("home", 112), ("away", 101)):
        team = record[side]
        value[side+"Team"] = {"teamId": team["team_id"], "score": score, "players": [
            {"personId": p["player_id"], "name": p["name"], "statistics": {"minutes": "PT48M00.00S"}}
            for p in team["players"][:5]]}
    return value


@pytest.fixture
def counter_inputs(record):
    return {"prediction": {"uri": "s3://nba-gestalt-data/test/forecast.json", "sha256": "a"*64},
            "box_score": {"uri": "s3://nba-gestalt-data/test/box.json", "sha256": "b"*64}}


@pytest.fixture
def counter(record, box, counter_inputs):
    return build_actual_minutes_prediction(prediction=record, box=box, inputs=counter_inputs,
                                           generated_at=NOW+timedelta(days=1))


def test_regulation_preserves_ratings_context_and_full_precision(record, box, counter_inputs):
    before = json_bytes(record)
    value = build_actual_minutes_prediction(prediction=record, box=box, inputs=counter_inputs, generated_at=NOW+timedelta(days=1))
    assert json_bytes(record) == before
    assert json.loads(json_bytes(value)) == value
    for side in ("home", "away"):
        lookup = {p["player_id"]: p for p in record[side]["players"]}
        assert value[side]["total_actual_minutes"] == 240
        for p in value[side]["players"]:
            assert p["nail"] == lookup[p["player_id"]]["nail"]
            assert p["projected_rating_contribution"] == lookup[p["player_id"]]["rating_contribution"]
            assert p["actual_rating_contribution"] == p["actual_minutes"]/48*p["nail"]
            assert p["contribution_delta"] == p["actual_rating_contribution"]-p["projected_rating_contribution"]
    for key in ("hca", "b2b_adjustment", "b2b_coefficient", "home_is_b2b", "away_is_b2b"):
        assert value["components"][key] == record["components"][key]
    assert value["original_predicted_home_margin"] == record["predicted_home_margin"]
    assert value["actual_home_margin"] == 11
    assert value["baseline_prediction_purpose"] == record["purpose"]


def test_overtime_normalizes_team_to_five_players(record, box, counter_inputs):
    regulation = build_actual_minutes_prediction(prediction=record, box=box, inputs=counter_inputs, generated_at=NOW+timedelta(days=1))
    for side in ("home", "away"):
        for p in box[side+"Team"]["players"]:
            p["statistics"]["minutes"] = "PT53M00.00S"
    overtime = build_actual_minutes_prediction(prediction=record, box=box, inputs=counter_inputs, generated_at=NOW+timedelta(days=1))
    assert overtime["home"]["total_actual_minutes"] == 265
    assert overtime["counterfactual_home_margin"] == regulation["counterfactual_home_margin"]
    assert overtime["home"]["actual_player_total"] == regulation["home"]["actual_player_total"]


def test_additional_actual_player_uses_pinned_state_then_prior_then_replacement(record, data, box, counter_inputs):
    pid = 99999
    box["homeTeam"]["players"][0]["personId"] = pid
    counter_inputs["player_ratings"] = record["inputs"]["player_ratings"]
    state = deepcopy(data["ratings"])
    state["players"].append({"player_id": pid, "nail": 7.123456789012345})
    args = dict(prediction=record, box=box, inputs=counter_inputs, ratings=state, generated_at=NOW+timedelta(days=1))
    value = build_actual_minutes_prediction(**args)
    p = next(p for p in value["home"]["players"] if p["player_id"] == pid)
    assert p["nail"] == 7.123456789012345 and p["projected_minutes"] == 0
    state["players"].pop()
    args["priors"] = {"players": [{"player_id": pid, "rapm": 3.123456789012345}]}
    value = build_actual_minutes_prediction(**args)
    p = next(p for p in value["home"]["players"] if p["player_id"] == pid)
    assert p["rating_source"] == "frozen_prior" and p["nail"] == 3.123456789012345
    args["priors"] = {"players": []}
    value = build_actual_minutes_prediction(**args)
    p = next(p for p in value["home"]["players"] if p["player_id"] == pid)
    assert p["rating_source"] == "replacement" and p["nail"] == -5.109488431706889
    state["game_date_pacific"] = "2026-10-21"
    state["observed_through_game_date_pacific"] = "2026-10-21"
    state["effective_from_game_date_pacific"] = "2026-10-22"
    with pytest.raises(PredictionError, match="rating input"):
        build_actual_minutes_prediction(**args)


@pytest.mark.parametrize("case", ["not_final", "wrong_game", "wrong_team", "duplicate", "bad_minutes", "zero_minutes"])
def test_invalid_final_inputs_fail(record, box, counter_inputs, case):
    if case == "not_final": box["gameStatus"] = 2
    if case == "wrong_game": box["gameId"] = "0022609999"
    if case == "wrong_team": box["homeTeam"]["teamId"] = box["awayTeam"]["teamId"]
    if case == "duplicate": box["homeTeam"]["players"].append(box["homeTeam"]["players"][0])
    if case == "bad_minutes": box["homeTeam"]["players"][0]["statistics"]["minutes"] = "garbage"
    if case == "zero_minutes":
        for p in box["homeTeam"]["players"]: p["statistics"]["minutes"] = "PT0M0S"
    with pytest.raises(PredictionError):
        build_actual_minutes_prediction(prediction=record, box=box, inputs=counter_inputs, generated_at=NOW+timedelta(days=1))


def test_separate_archive_reuses_and_versions_without_changing_baseline(archive, record, counter, clock):
    store, s3 = archive
    store.save(record, expected_etag=None)
    baseline = dict(s3.objects)
    first = store.save_actual_minutes(counter, expected_etag=None)
    pointer, etag = store.selection(counter["game_date_pacific"], counter["game_id"], actual_minutes=True)
    assert pointer["uri"].startswith("s3://nba-gestalt-data/lake/artifacts/game_predictions_actual_minutes/season=2026-27/model_version=v1/")
    assert store.load(pointer, actual_minutes=True) == counter
    before = dict(s3.objects)
    counter["generated_at_utc"] = (NOW+timedelta(days=1, minutes=1)).isoformat()
    assert store.save_actual_minutes(counter, expected_etag=etag)["reused"]
    assert s3.objects == before
    counter["inputs"]["box_score"]["sha256"] = "e"*64
    counter["input_fingerprint"] = actual_minutes_fingerprint(counter["inputs"], implementation=counter["implementation"])
    second = store.save_actual_minutes(counter, expected_etag=etag)
    assert first["sha256"] != second["sha256"]
    assert all(s3.objects[k] == value for k, value in baseline.items())
    assert first["uri"].removeprefix("s3://nba-gestalt-data/") in s3.objects


def test_counterfactual_stale_writer_and_s3_race_are_rejected(archive, counter):
    store, s3 = archive
    store.save_actual_minutes(counter, expected_etag=None)
    with pytest.raises(PredictionError, match="Another run"):
        store.save_actual_minutes(counter, expected_etag=None)
    _, etag = store.selection(counter["game_date_pacific"], counter["game_id"], actual_minutes=True)
    counter["inputs"]["box_score"]["sha256"] = "e"*64
    counter["input_fingerprint"] = actual_minutes_fingerprint(counter["inputs"], implementation=counter["implementation"])
    s3.race = True
    with pytest.raises(ClientError):
        store.save_actual_minutes(counter, expected_etag=etag)
