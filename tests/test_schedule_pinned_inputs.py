"""Published report inputs remain pinned when the raw capture archive advances."""

import hashlib
import json
from copy import deepcopy
from datetime import timedelta

import pytest
from test_saved_schedule_api import DAY, GID, NOW, SEASON, START, encoded
from test_schedule_details_api import actual_fixture, saved_market, select_actual, setup_game

from nba_lineup_model.web_api.schedule import ScheduleReader
from nba_lineup_model.web_api.schedule_details import DetailsReader


def save_box(s3, game, actual, *, revision=0, scores=(100, 90)):
    box = {"gameId": GID, "gameStatus": 3}
    metadata = {"season": SEASON, "game_id": GID}
    for side, score in zip(("home", "away"), scores, strict=True):
        players = [
            {
                "personId": p["player_id"],
                "name": p["name"],
                "statistics": {
                    "minutes": f"PT{p['actual_minutes'] * 60:.8f}S",
                    "unrelatedStat": revision,
                },
            }
            for p in actual[side]["players"]
        ]
        box[side + "Team"] = {
            "teamId": game[side + "_team_id"],
            "teamTricode": game[side + "_team"],
            "score": score,
            "players": players,
            "statistics": {"unrelatedStat": revision},
        }
        roster = sorted(
            [{"player_id": p["personId"], "player_name": p["name"]} for p in players],
            key=lambda p: p["player_id"],
        )
        metadata[side + "_team_id"] = game[side + "_team_id"]
        metadata[side + "_roster_sha256"] = hashlib.sha256(encoded(roster)).hexdigest()
    document = {"meta": {"refresh": revision}, "game": box}
    sha = hashlib.sha256(encoded(document)).hexdigest()
    root = f"lake/raw/cdn/boxscore/season={SEASON}/game_id={GID}/"
    key = root + sha + ".json"
    s3.put_fixture(key, document)
    metadata["box_score_sha256"] = sha
    meta_key = root + sha + ".meta.json"
    meta_sha = s3.put_fixture(meta_key, metadata)
    if not hasattr(s3, "modified"):
        s3.modified = {}

        def paginate(*, Bucket, Prefix):
            s3.calls.append(("list", Prefix))
            yield {
                "Contents": [
                    {"Key": k, "LastModified": s3.modified.get(k, NOW)}
                    for k in s3.objects
                    if k.startswith(Prefix)
                ]
            }

        s3.paginate = paginate
    s3.modified[key] = NOW + timedelta(minutes=revision)
    return {
        "uri": "s3://nba-gestalt-data/" + key,
        "sha256": sha,
        "metadata_uri": "s3://nba-gestalt-data/" + meta_key,
        "metadata_sha256": meta_sha,
    }


def row_for(game, reference, *, scores=(100, 90), won=True):
    return {
        "game_id": GID,
        "scheduled_start_utc": START,
        "home": {"team_id": 1, "team_abbreviation": "HOM", "final_score": scores[0]},
        "away": {"team_id": 2, "team_abbreviation": "AWY", "final_score": scores[1]},
        "inputs": {
            "prediction": deepcopy(game["sources"]["prediction"]),
            "final_box_score": reference,
        },
        "markets": saved_market(game),
        "comparisons": {
            "model": {
                "predicted_home_margin": game["projection"]["home_margin"],
                "home_win_probability": game["projection"]["home_win_probability"],
                "winner_correct": won,
                "ats": {"result": "win" if won else "loss"},
            }
        },
    }


def publish(s3, row, actual, box_ref):
    actual["inputs"]["box_score"] = box_ref
    select_actual(s3, actual)
    root = (
        f"lake/artifacts/market_performance/season={SEASON}/model_version=v1/"
        f"game_date_pacific={DAY}/"
    )
    doc = {
        "schema_version": 1,
        "artifact_kind": "market_performance",
        "season": SEASON,
        "model_version": "v1",
        "game_date_pacific": DAY,
        "games": [row],
    }
    sha = s3.put_fixture(root + hashlib.sha256(encoded(doc)).hexdigest() + ".json", doc)
    s3.put_fixture(
        root + "latest.json",
        {k: doc[k] for k in ["schema_version", "season", "model_version", "game_date_pacific"]}
        | {"sha256": sha, "uri": "s3://nba-gestalt-data/" + root + sha + ".json"},
    )


def fixture():
    s3, reader, game, baseline = setup_game()
    actual, _ = actual_fixture(s3, game, baseline)
    reference = save_box(s3, game, actual)
    row = row_for(game, reference)
    game["status"].update(state="post", completed=True)
    publish(s3, row, actual, reference)
    return s3, reader, game, actual, row, reference


def serve(reader, game):
    g = deepcopy(game)
    d = DetailsReader(reader, SEASON, DAY, NOW)
    d.enrich_game(g, d.historical([g]))
    return g


def test_newer_unrelated_box_refresh_does_not_invalidate_published_reports():
    s3, reader, game, actual, row, original = fixture()
    before = serve(reader, game)
    newer = save_box(s3, game, actual, revision=1)
    assert original != newer
    assert reader.final_box(SEASON, game)[1]["sha256"] == newer["sha256"]
    s3.calls.clear()
    after = serve(reader, game)
    assert after["market"] == before["market"]
    assert after["final_audit"] == before["final_audit"]
    assert after["retrospective"] == before["retrospective"]
    assert after["projection"] == game["projection"]
    assert (
        after["availability"]["grading"] == after["availability"]["postgame_details"] == "available"
    )
    assert after["final_audit"]["inputs"]["final_box_score"] == original
    assert after["retrospective"]["inputs"]["box_score"] == original
    assert not [key for op, key in s3.calls if op == "list"]


def test_score_correction_changes_only_newly_published_outcomes_and_grades():
    s3, reader, game, actual, row, original = fixture()
    before = serve(reader, game)
    newer = save_box(s3, game, actual, revision=1, scores=(95, 100))
    still_published = serve(reader, game)
    assert still_published["final_audit"] == before["final_audit"]
    assert still_published["retrospective"]["recorded_outcome"] == {
        "home_score": 100,
        "away_score": 90,
    }
    actual.update(actual_home_score=95, actual_away_score=100)
    replacement = row_for(game, newer, scores=(95, 100), won=False)
    publish(s3, replacement, actual, newer)
    after = serve(reader, game)
    assert after["final_audit"]["grading"]["model_favorite_won"] is False
    assert after["final_audit"]["grading"]["market"]["model_ats_result"] == "loss"
    assert after["final_audit"]["recorded_outcome"] == {"home_score": 95, "away_score": 100}
    assert after["retrospective"]["recorded_outcome"] == after["final_audit"]["recorded_outcome"]
    assert after["retrospective"]["home_margin"] == before["retrospective"]["home_margin"]
    assert after["market"] == before["market"] and after["projection"] == game["projection"]


def test_minutes_correction_requires_replacement_and_keeps_pregame_prediction():
    s3, reader, game, actual, row, original = fixture()
    before = serve(reader, game)
    actual["home"]["players"][0].update(
        actual_minutes=5.125, actual_rating_contribution=0.375, contribution_delta=-0.280321
    )
    actual["counterfactual_home_margin"] = 1.987654321
    actual["counterfactual_home_win_probability"] = 0.6123456789
    newer = save_box(s3, game, actual, revision=1)
    assert serve(reader, game)["retrospective"] == before["retrospective"]
    row["inputs"]["final_box_score"] = newer
    publish(s3, row, actual, newer)
    after = serve(reader, game)
    assert after["retrospective"]["home_margin"] == 1.987654321
    assert after["retrospective"]["home_win_probability"] == 0.6123456789
    assert after["retrospective"]["actual_minutes"]["home"][0]["actual_minutes"] == 5.125
    assert after["market"] == before["market"] and after["projection"] == game["projection"]


@pytest.mark.parametrize("damage", ["box_bytes", "metadata_bytes", "missing_box", "wrong_identity"])
def test_corrupt_pinned_inputs_fail_postgame_but_do_not_hide_valid_quotes(damage):
    s3, reader, game, actual, row, original = fixture()
    key = original["uri"].split("/", 3)[3]
    if damage == "box_bytes":
        s3.put_fixture(key, {})
    elif damage == "metadata_bytes":
        s3.put_fixture(original["metadata_uri"].split("/", 3)[3], {})
    elif damage == "missing_box":
        del s3.objects[key]
    else:
        document = json.loads(s3.objects[key][0])
        document["game"]["gameId"] = "0022600999"
        sha = hashlib.sha256(encoded(document)).hexdigest()
        root = key.rsplit("/", 1)[0] + "/"
        s3.put_fixture(root + sha + ".json", document)
        bad = {
            **original,
            "uri": "s3://nba-gestalt-data/" + root + sha + ".json",
            "sha256": sha,
            "metadata_uri": "s3://nba-gestalt-data/" + root + sha + ".meta.json",
        }
        row["inputs"]["final_box_score"] = bad
        publish(s3, row, actual, bad)
    g = serve(ScheduleReader(s3), game)
    assert g["availability"]["market"] == "available" and g["market"] is not None
    assert g["availability"]["grading"] == g["availability"]["postgame_details"] == "invalid"
    assert not g.get("final_audit") and not g.get("retrospective")
    assert g["projection"] == game["projection"]


def test_baseline_change_blocks_grades_and_counterfactual_but_keeps_pregame_quotes():
    s3, reader, game, actual, row, original = fixture()
    game["sources"]["prediction"] = {**game["sources"]["prediction"], "sha256": "0" * 64}
    g = serve(reader, game)
    assert g["availability"]["market"] == "available"
    assert g["availability"]["grading"] == g["availability"]["postgame_details"] == "stale"
    assert g["market"]["home_spread"] == -2.1256789
    assert not g.get("retrospective") and not g.get("final_audit")


def test_pinned_reference_cannot_escape_its_game_prefix():
    s3, reader, game, actual, row, original = fixture()
    escaped = {**original, "uri": original["uri"].replace(GID, "0022600999")}
    with pytest.raises(ValueError, match="escapes"):
        reader.pinned_final_box(SEASON, game, escaped)
