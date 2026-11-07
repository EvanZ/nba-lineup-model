"""Immutable S3 report snapshots with a last-published pointer per environment."""
from __future__ import annotations

import hashlib
import json
import time

import boto3
from botocore.exceptions import ClientError

from nba_lineup_model.season.durable_observations import _client, _read


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def _location(uri, kind):
    if not uri.startswith('s3://') or kind not in ('data-quality', 'daily-pipeline'):
        raise ValueError('Invalid operational report S3 location')
    return uri.rstrip('/') + '/' + kind


def publish(uri, kind, files, *, client=None):
    """Publish exact report text, verifying the snapshot before moving its pointer."""
    client = client or _client(boto3.client)
    prefix = _location(uri, kind)
    raw = _bytes({'schema_version': 1, 'files': files})
    sha = hashlib.sha256(raw).hexdigest()
    location = prefix + '/snapshot_sha256=' + sha + '.json'
    bucket, _, key = location[5:].partition('/')
    try:
        client.put_object(Bucket=bucket, Key=key, Body=raw, IfNoneMatch='*',
                          ContentType='application/json', Metadata={'sha256': sha})
    except ClientError as error:
        if error.response['Error']['Code'] not in ('PreconditionFailed', '412'):
            raise
    verified, _ = _read(client, location, 0, sha, len(raw))
    if verified != raw:
        raise ValueError('Operational report snapshot verification failed')
    pointer = _bytes({'schema_version': 1, 's3_uri': location, 'sha256': sha, 'size_bytes': len(raw)})
    pointer_key = prefix[5:].partition('/')[2] + '/latest.json'
    client.put_object(Bucket=bucket, Key=pointer_key, Body=pointer,
                      ContentType='application/json', Metadata={'sha256': hashlib.sha256(pointer).hexdigest()})
    return location


def read(uri, kind, *, client=None):
    """Read a verified report snapshot directly from S3; never use disk files."""
    client = client or _client(boto3.client)
    prefix = _location(uri, kind)
    pointer_raw, _ = _read(client, prefix + '/latest.json', int(time.monotonic() // 10))
    pointer = json.loads(pointer_raw)
    if (pointer.get('schema_version') != 1
            or pointer.get('s3_uri') != prefix + '/snapshot_sha256=' + pointer.get('sha256', '') + '.json'):
        raise ValueError('Invalid operational report selection')
    raw, _ = _read(client, pointer['s3_uri'], 0, pointer['sha256'], pointer['size_bytes'])
    snapshot = json.loads(raw)
    if snapshot.get('schema_version') != 1 or not isinstance(snapshot.get('files'), dict):
        raise ValueError('Invalid operational report snapshot')
    return snapshot['files']
