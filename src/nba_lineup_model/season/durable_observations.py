"""Read official observations from their existing S3 archives, without disk I/O."""
from __future__ import annotations

import base64
import copy
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from threading import RLock
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import time

import boto3

@lru_cache(maxsize=4)
def _client(factory):
    return factory("s3")

_BYTES = OrderedDict()
_BYTES_LOCK = RLock()
_BYTE_BUDGET = 32 * 1024 * 1024

def _read(client, uri: str, generation: int, expected_sha=None, expected_size=None):
    key = (client, uri, generation, expected_sha, expected_size)
    with _BYTES_LOCK:
        if key in _BYTES:
            _BYTES.move_to_end(key)
            return _BYTES[key]
    result = _download(client, uri, expected_sha, expected_size)
    if len(result[0]) <= _BYTE_BUDGET:
        with _BYTES_LOCK:
            _BYTES[key] = result
            while sum(len(item[0]) for item in _BYTES.values()) > _BYTE_BUDGET:
                _BYTES.popitem(last=False)
    return result

def _download(client, uri: str, expected_sha=None, expected_size=None):
    # Completed boxes are immutable; moving daily boards refresh every 30s.
    # This bounded cache contains S3 response bytes in process memory only.
    if not uri.startswith("s3://"):
        raise ValueError("Official observations require an S3 URI")
    bucket, separator, key = uri[5:].partition("/")
    if not bucket or not separator or not key:
        raise ValueError("Invalid official observation URI")
    try:
        response = client.get_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    except Exception as error:
        raise ValueError(f"Cannot read required S3 object: {uri}") from error
    stream = response["Body"]
    try:
        raw = stream.read()
    finally:
        stream.close()
    if len(raw) != response.get("ContentLength") or (expected_size is not None and len(raw) != expected_size):
        raise ValueError(f"S3 observation size mismatch: {uri}")
    sha = hashlib.sha256(raw).hexdigest()
    expected = expected_sha or response.get("Metadata", {}).get("sha256")
    checksum = response.get("ChecksumSHA256")
    etag = response.get("ETag", "").strip('"')
    if expected:
        valid = sha == expected
    elif checksum:
        valid = base64.b64encode(hashlib.sha256(raw).digest()).decode() == checksum
    elif re.fullmatch(r"[0-9a-f]{32}", etag) and response.get("ServerSideEncryption") in (None, "AES256"):
        valid = hashlib.md5(raw).hexdigest() == etag
    else:
        raise ValueError(f"S3 observation has no supported integrity checksum: {uri}")
    if not valid:
        raise ValueError(f"S3 observation checksum mismatch: {uri}")
    return raw, sha

@lru_cache(maxsize=64)
def _final_keys(client, prefix, generation):
    bucket, _, key = prefix[5:].partition('/')
    found = set()
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=key+'/final/'):
        for item in page.get('Contents', []):
            name = item['Key'].rsplit('/',1)[-1]
            if name.endswith('.nba.json'):
                found.add(name.removesuffix('.nba.json'))
    return found

def select_sources(dates, *, uri, season, client=None, release_root=None):
    """Bind accepted historical NBA results or the current official collector."""
    client = client or _client(boto3.client)
    objects = {}
    if release_root is not None:
        from nba_lineup_model.storage.website_data import ServingData
        from nba_lineup_model.storage.duckdb_lake import DEFAULT_LAKE_URI
        from nba_lineup_model.season.durable_release_storage import production_release_reference
        root = Path(release_root)
        # This local file chooses a deployment identity, never basketball data.
        bootstrap = json.loads((root / 'runtime-manifest.json').read_text())
        store = ServingData(root, lake_uri=DEFAULT_LAKE_URI, client=client)
        objects = store.manifest(production_release_reference(bootstrap))['objects']
    selected = {}
    for day in sorted(dates):
        board_entry = objects.get(f'selected-schedule/latest/{day}.json')
        if board_entry is not None:
            raw, _ = _read(client, board_entry['s3_uri'], 0,
                           board_entry['sha256'], board_entry['size_bytes'])
            games = json.loads(raw)['games']
            if all(g.get('status', {}).get('completed') for g in games):
                boxes, summaries = {}, {}
                for game in games:
                    gid = str(game['game_id'])
                    entry = objects.get(f'selected-schedule/final/{gid}.nba.json')
                    if not re.fullmatch(r'00\d{8}', gid) or entry is None:
                        raise ValueError(f'Accepted release lacks official NBA result: {day}/{gid}')
                    boxes[gid] = entry
                    summaries[gid] = objects[f'selected-schedule/final/{gid}.summary.json']
                selected[day] = {'board': board_entry, 'boxes': boxes, 'summaries': summaries}
                continue
        selected[day] = {'prefix': uri.rstrip('/')}
    return selected


def capture(root: Path | None, dates: list[str], *, uri: str, season: str, sources: dict | None = None):
    from nba_lineup_model.season.serving_selection import ObservationSnapshot, digest
    client = _client(boto3.client)
    generation = int(time.monotonic() // 30)
    fields = ("game_id", "away_team", "home_team", "start_time", "away_score",
              "home_score", "status", "injury_report", "market", "prediction_market")
    def day(day):
        source = sources[day] if sources is not None else {'prefix': uri}
        prefix = f"{source.get('prefix', uri).rstrip('/')}/season={season}/game_date={day}"
        pinned = source.get('board')
        location = pinned['s3_uri'] if pinned else prefix + '/latest.json'
        raw, sha = _read(client, location, 0 if pinned else generation,
                        pinned['sha256'] if pinned else None, pinned['size_bytes'] if pinned else None)
        board = json.loads(raw)
        if not isinstance(board.get("games"), list):
            raise ValueError(f"Invalid S3 observation board: {day}")
        clean = [{key: copy.deepcopy(game.get(key)) for key in fields} for game in board["games"]]
        ids = [str(game["game_id"]) for game in clean]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate official observation game: {day}")
        boxes, hashes = {}, {location: sha}
        published = set(source['boxes']) if pinned else _final_keys(client, prefix, generation)
        if published - set(ids):
            raise ValueError(f"S3 final games absent from official board: {day}: {sorted(published-set(ids))}")
        for game in clean:
            if str(game['game_id']) not in published and not (game.get("status", {}).get("completed") or game.get("status", {}).get("state") == "post"):
                continue
            gid = str(game["game_id"])
            entry = source['boxes'][gid] if pinned else None
            box_uri = entry['s3_uri'] if entry else prefix + f"/final/{gid}.nba.json"
            raw, box_sha = _read(client, box_uri, 0,
                entry['sha256'] if entry else None, entry['size_bytes'] if entry else None)
            box = json.loads(raw)
            if str(box.get("game", {}).get("gameId")) != gid:
                raise ValueError(f"Official S3 box identity mismatch: {gid}")
            official = box['game']
            summary_entry = source['summaries'][gid] if pinned else None
            summary_uri = summary_entry['s3_uri'] if summary_entry else prefix + f'/final/{gid}.summary.json'
            summary_raw, summary_sha = _read(client, summary_uri, 0,
                summary_entry['sha256'] if summary_entry else None,
                summary_entry['size_bytes'] if summary_entry else None)
            summary = json.loads(summary_raw)
            final = summary.get('final', {})
            if (summary.get('game_id') != gid or not final.get('status', {}).get('completed')
                    or final.get('status', {}).get('state') != 'post'):
                raise ValueError(f'Official final summary is not completed: {gid}')
            hashes[summary_uri] = summary_sha
            for side in ('home', 'away'):
                team = official.get(side+'Team', {})
                if team.get('teamTricode') != game[side+'_team'] or not isinstance(team.get('score'), (int, float)):
                    raise ValueError(f"Official S3 final team/score mismatch: {gid}")
                if summary.get(side+'_team') != game[side+'_team'] or final.get(side+'_score') != team['score']:
                    raise ValueError(f'Official final summary/box score mismatch: {gid}')
                game[side+'_score'] = team['score']
            game['status'] = copy.deepcopy(final['status'])
            boxes[gid], hashes[box_uri] = box, box_sha
        # Only in-progress scores may enter from the local ephemeral source.
        active = {str(g["game_id"]): g for g in clean if g.get("status", {}).get("state") == "in"}
        if active and root is not None:
            path = root / "latest" / f"{day}.json"
            if path.is_file():
                live = json.loads(path.read_bytes())
                for update in live.get("games", []):
                    game = active.get(str(update.get("game_id")))
                    if game is not None and update.get("status", {}).get("state") == "in" and not update.get("status", {}).get("completed"):
                        if any(update.get(k) != game.get(k) for k in ("home_team", "away_team")):
                            raise ValueError("Ephemeral score team identity mismatch")
                        for key in ("home_score", "away_score", "status"):
                            game[key] = copy.deepcopy(update.get(key))
        result = {"games": clean, "collected_at": board.get("live_score_collected_at") or board.get("collected_at")}
        hashes[f"board:{day}"] = digest(result)
        return day, result, boxes, hashes
    boards, boxes, hashes = {}, {}, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for day, board, day_boxes, day_hashes in pool.map(day, sorted(set(dates))):
            boards[day] = board
            boxes.update(day_boxes)
            hashes.update(day_hashes)
    return ObservationSnapshot(digest({"files": hashes, "errors": {}}), boards, boxes, hashes, {})
