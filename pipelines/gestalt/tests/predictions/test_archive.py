"""S3 immutability, repeat runs, concurrent writers and tipoff protection."""
from datetime import timedelta
import pytest
from botocore.exceptions import ClientError
from gestalt_pipelines.predictions.utils import PredictionError


def test_rerun_keeps_original_generation_time_and_pointer(archive, record, clock):
    store, s3 = archive
    first = store.save(record, expected_etag=None)
    before = dict(s3.objects)
    selected, etag = store.selection(record["game_date_pacific"], record["game_id"])
    record["generated_at_utc"] = (clock.value+timedelta(minutes=1)).isoformat()
    again = store.save(record, expected_etag=etag)
    assert again["reused"] and again["sha256"] == first["sha256"]
    assert s3.objects == before
    assert store.load(selected)["generated_at_utc"] != record["generated_at_utc"]


def test_replay_preserves_pregame_even_without_purpose_on_pointer(archive, record, clock):
    from gestalt_pipelines.predictions.utils import json_bytes
    store, s3 = archive
    store.save(record, expected_etag=None)
    pointer, etag = store.selection(record["game_date_pacific"], record["game_id"])
    pointer.pop("purpose")
    key = store._game_prefix(record["game_date_pacific"], record["game_id"])+"latest.json"
    s3.objects[key] = json_bytes(pointer)
    before = dict(s3.objects)
    record["purpose"] = "historical_replay"
    clock.value += timedelta(days=2)
    assert store.save(record, expected_etag=etag) == {"frozen": True}
    assert s3.objects == before


def test_replay_can_advance_and_retains_previous_versions(archive, record, clock):
    store, s3 = archive
    record["purpose"] = "historical_replay"
    first = store.save(record, expected_etag=None)
    _, etag = store.selection(record["game_date_pacific"], record["game_id"])
    record["inputs"]["player_ratings"]["sha256"] = "e"*64
    from gestalt_pipelines.predictions.utils import prediction_fingerprint
    game = {"game_id": record["game_id"], "game_time_utc": clock.value+timedelta(hours=2),
            "home_team_id": record["home"]["team_id"], "away_team_id": record["away"]["team_id"]}
    record["input_fingerprint"] = prediction_fingerprint(game, record["inputs"], implementation=record["implementation"])
    second = store.save(record, expected_etag=etag)
    assert first["sha256"] != second["sha256"]
    assert first["uri"].split("nba-gestalt-data/")[1] in s3.objects


def test_stale_writer_and_corrupt_immutable_are_rejected(archive, record, clock):
    store, s3 = archive
    store.save(record, expected_etag=None)
    with pytest.raises(PredictionError, match="Another run"):
        store.save(record, expected_etag=None)
    selected, _ = store.selection(record["game_date_pacific"], record["game_id"])
    s3.objects[selected["uri"].split("nba-gestalt-data/")[1]] = b"{}"
    with pytest.raises(PredictionError, match="checksum"):
        store.load(selected)


def test_tipoff_during_s3_write_leaves_pointer_untouched(archive, record, clock, monkeypatch):
    store, s3 = archive
    original = s3.put_object
    def put(**kwargs):
        result = original(**kwargs)
        clock.value += timedelta(hours=3)
        return result
    monkeypatch.setattr(s3, "put_object", put)
    assert store.save(record, expected_etag=None) == {"frozen": True}
    assert not any(k.endswith("latest.json") for k in s3.objects)


def test_conditional_s3_write_cannot_overwrite_competing_publication(archive, record, clock):
    store, s3 = archive
    s3.race = True
    with pytest.raises(ClientError):
        store.save(record, expected_etag=None)
    assert not any(k.endswith("latest.json") for k in s3.objects)
