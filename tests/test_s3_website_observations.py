"""Permanent observations must remain available without any local data files."""
import hashlib
import io
import json
from pathlib import Path

import boto3
import pytest

from nba_lineup_model.season.serving_selection import ObservationSnapshot

DAY = "2026-11-04"
PREFIX = "schedule/season=2026-27/game_date=" + DAY

class S3:
    def __init__(self, objects):
        self.objects = objects
        self.reads = []
    def head_object(self, *, Bucket, Key):
        return {"ContentLength": len(self.objects[Key])}
    def get_paginator(self, name):
        return self
    def paginate(self, *, Bucket, Prefix):
        return [{"Contents": [{"Key": key} for key in self.objects if key.startswith(Prefix)]}]
    def get_object(self, *, Bucket, Key, **kwargs):
        self.reads.append(Key)
        raw = self.objects[Key]
        return {"Body": io.BytesIO(raw), "ContentLength": len(raw),
                "ETag": '"' + hashlib.md5(raw).hexdigest() + '"'}

def source(monkeypatch):
    games = [{"game_id": f"0022600{n}", "home_team": "BOS", "away_team": "NYK",
              "home_score": 110, "away_score": 100,
              "status": {"state": "post", "completed": True}}
             for n in range(173,187)]
    objects = {PREFIX + "/latest.json": json.dumps({"games": games,
                 "collected_at": "2026-11-05T06:00:00Z"}).encode()}
    for game in games:
        objects[PREFIX + "/final/" + game["game_id"] + ".nba.json"] = json.dumps(
            {"game": {"gameId": game["game_id"], "gameStatus": 3,
                      "homeTeam": {"teamTricode": "BOS", "score": 110},
                      "awayTeam": {"teamTricode": "NYK", "score": 100}}}).encode()
        objects[PREFIX + '/final/' + game['game_id'] + '.summary.json'] = json.dumps({
            'game_id': game['game_id'], 'home_team':'BOS', 'away_team':'NYK',
            'final':{'home_score':110,'away_score':100,'status':{'state':'post','completed':True}}}).encode()
    client = S3(objects)
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: client)
    monkeypatch.setenv("GESTALT_OBSERVATION_S3_URI", "s3://test-bucket/schedule")
    monkeypatch.setenv("GESTALT_OBSERVATION_SEASON", "2026-27")
    return client

def test_completed_results_are_read_from_s3_without_local_files(monkeypatch):
    client = source(monkeypatch)
    def forbidden(*a, **kw):
        raise AssertionError("Permanent observations may not read or write local files")
    monkeypatch.setattr(Path, "open", forbidden)
    snapshot = ObservationSnapshot.capture(None, [DAY])
    assert len(snapshot.boards.get(DAY, {}).get("games", [])) == 14
    assert len(snapshot.boxes) == 14
    assert snapshot.errors == {}
    assert len(client.reads) == 29  # Board, 14 official boxes, and 14 final summaries.

def test_missing_s3_box_fails_without_using_a_local_copy(monkeypatch):
    client = source(monkeypatch)
    del client.objects[PREFIX + "/final/0022600180.nba.json"]
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        ObservationSnapshot.capture(None, [DAY])


def test_corrupt_s3_bytes_fail(monkeypatch):
    client = source(monkeypatch)
    original = client.get_object
    def corrupt(**kwargs):
        response = original(**kwargs)
        response["ETag"] = '"' + '0' * 32 + '"'
        return response
    client.get_object = corrupt
    with pytest.raises(ValueError, match="checksum mismatch"):
        ObservationSnapshot.capture(None, [DAY])

def test_serving_inventory_reads_s3_and_rejects_local_fallback(tmp_path, monkeypatch):
    from nba_lineup_model.storage.website_data import ServingData
    raw = b'{"durable": true}'
    client = S3({"data/object.json": raw})
    session = ServingData(tmp_path, lake_uri="s3://test-bucket/data", client=client)
    session.history = tmp_path
    session.register(tmp_path, {"object.json": {"s3_uri": "s3://test-bucket/data/object.json",
                     "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}})
    (tmp_path / "object.json").write_text('wrong local data')
    def forbidden(*a, **kw):
        raise AssertionError("Local data access forbidden")
    monkeypatch.setattr(Path, "open", forbidden)
    assert session.read(tmp_path / "object.json") == raw
    with pytest.raises(FileNotFoundError, match="No selected S3 object"):
        session.read(tmp_path / "unselected.json")
    session.objects[str(tmp_path / "object.json")]["sha256"] = '0' * 64
    with pytest.raises(ValueError, match="Pinned S3 serving object changed"):
        session.read(tmp_path / "object.json")


def test_published_final_box_advances_a_stale_s3_board(monkeypatch):
    client = source(monkeypatch)
    key = PREFIX + "/latest.json"
    board = json.loads(client.objects[key])
    for game in board["games"]:
        game.update(status={"state":"in", "completed":False}, home_score=1, away_score=2)
    client.objects[key] = json.dumps(board).encode()
    snapshot = ObservationSnapshot.capture(None, [DAY])
    assert all(g["status"]["completed"] and g["home_score"] == 110
               for g in snapshot.boards[DAY]["games"])


def test_scoreboard_final_and_complete_box_may_have_lagging_status(monkeypatch):
    client = source(monkeypatch)
    key = PREFIX + '/final/0022600182.nba.json'
    box = json.loads(client.objects[key])
    box['game']['gameStatus'] = 2
    client.objects[key] = json.dumps(box).encode()
    snapshot = ObservationSnapshot.capture(None, [DAY])
    assert len(snapshot.boxes) == 14
    assert all(g['status']['completed'] for g in snapshot.boards[DAY]['games'])


def test_final_summary_and_box_must_agree(monkeypatch):
    client = source(monkeypatch)
    key = PREFIX + '/final/0022600182.summary.json'
    summary = json.loads(client.objects[key])
    summary['final']['home_score'] = 111
    client.objects[key] = json.dumps(summary).encode()
    with pytest.raises(ValueError, match='summary/box score mismatch'):
        ObservationSnapshot.capture(None, [DAY])


def test_accepted_official_results_are_explicitly_pinned(monkeypatch, tmp_path):
    from nba_lineup_model.season.durable_observations import select_sources, capture
    from nba_lineup_model.storage.website_data import ServingData
    from nba_lineup_model.season import durable_release_storage
    client = source(monkeypatch)
    (tmp_path / 'runtime-manifest.json').write_text('{}')
    entries = {}
    for key, raw in client.objects.items():
        name = key.removeprefix(PREFIX + '/')
        if name == 'latest.json':
            name = 'latest/' + DAY + '.json'
        entries['selected-schedule/' + name] = {'s3_uri':'s3://test-bucket/'+key,
            'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
    monkeypatch.setattr(durable_release_storage, 'production_release_reference', lambda manifest: 'accepted')
    monkeypatch.setattr(ServingData, 'manifest', lambda self, ref: {'objects':entries})
    selected = select_sources([DAY], uri='s3://absent/current', season='2026-27',
                              client=client, release_root=tmp_path)
    assert set(selected[DAY]['boxes']) == {f'0022600{n}' for n in range(173,187)}
    snapshot = capture(None, [DAY], uri='s3://absent/current', season='2026-27', sources=selected)
    assert len(snapshot.boxes) == 14
    entries.pop('selected-schedule/final/0022600182.nba.json')
    with pytest.raises(ValueError, match='lacks official NBA result'):
        select_sources([DAY], uri='s3://absent/current', season='2026-27', client=client, release_root=tmp_path)
